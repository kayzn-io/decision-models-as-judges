# Running Laya locally

Laya is an open-weight (Apache 2.0) typed decision model by Convai Innovations.
It reads a **state** (text or JSON) plus a set of **typed questions** and returns
typed answers with calibrated probabilities in a single, non-autoregressive
forward pass. It never generates free text, so its output is parsed directly
rather than extracted from prose. This repository uses the English checkpoint
`convaiinnovations/laya` (ModernBERT-large backbone, 421M parameters, 512-token
context) as a local judge answering the same `choice` / `score` / `noul`
questions the hosted Jev judge answers.

All facts below cite a source that was read directly: the Hugging Face model
card, the loader/runtime source shipped in the model repository
(`rl_agent_api.py`, `rl_common.py`, `rl_agent_config.json`), the project GitHub
README, or the two community runtime ports. Anything the sources do not state is
labelled **Fallback** and describes what this repository does instead.

## Loading

The model repository ships its own inference code; the encoder is a stock
`AutoModel` (ModernBERT-large) whose weights are replaced from a saved state
dict, and a from-scratch decision head is added on top. There is **no custom
`AutoModelForSequenceClassification` and no `trust_remote_code=True`** — the head
lives in `rl_common.py`, not in the encoder config. Two supported paths exist.

**1. The `laya` PyPI package (highest level).** The model card and GitHub README
document a one-call loader:

```python
import laya

agent = laya.load("convaiinnovations/laya")  # English root; downloads on first use
result = agent.predict(state, questions)  # or agent.system_one(state, questions)
```

`laya` is a separate PyPI package (`pip install laya`, Python 3.10+). It is **not**
a dependency of this repository — the `laya` extra installs only `torch` and
`transformers` (see `pyproject.toml`).
(Sources: model card "Single-Model Mode"; GitHub README "Single-Model Mode".)

**2. Transformers + safetensors (the path this repository uses).** This
repository vendors the model repository's own inference code — `rl_common.py` and
`rl_agent_config.json`, copied unmodified at revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` under
`decision_judges/judges/laya_vendor/` (Apache-2.0, Convai Innovations) — and
delegates to it. `LayaDecisionModel.from_pretrained` reads a checkpoint directory
containing `rl_agent_config.json`, `tokenizer/`, `encoder/`, and
`model.safetensors`:

```python
from safetensors.torch import load_file
from transformers import AutoTokenizer

from decision_judges.judges.laya_vendor import rl_common

tok = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
model = rl_common.build_model(cfg, encoder_dir=f"{model_dir}/encoder")
model.load_state_dict(load_file(f"{model_dir}/model.safetensors"), strict=True)
model.to(device).eval()
```

`build_model` (in the vendored `rl_common.py`) constructs the encoder
architecture from `encoder/config.json` with `AutoModel.from_config(...)` and
wraps it in the `DecisionModel` head; weights then come from the state dict.
Loading is `strict=True`, so any drift between the head's module attribute names
and the checkpoint keys fails loudly. A directory argument is loaded as-is;
otherwise the checkpoint is fetched with `huggingface_hub.snapshot_download` at
the revision recorded in `config/study.toml`.
(Sources: `rl_agent_api.py`, vendored `rl_common.py` `build_model`, HF file listing.)

> If `transformers` probes for TensorFlow at import and hangs, run with
> `USE_TF=0`. (Source: model card note under "Single-Model Mode".)

## Input encoding

Each question is encoded into one sequence, and every option is scored at its own
`[MASK]` marker token. `build_sequence` (in `rl_common.py`) lays the sequence out
as:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

- **Option rendering** (`render_options`): a `choice` option is `key` or
  `key: description`; a `score` level is `level i: description`; a `noul` question
  always renders exactly two options, `false: ...` and `true: ...`, in that
  order, so the "true" slot is index 1.
- **Question-type signal**: the type name is written into the text (`choice
  question:` / `score question:` / `noul question:`) and, in the head, a learned
  `type_emb` embedding for the type index (`choice=0, score=1, noul=2`) is added
  to every position.
- **Length budget** (English checkpoint, from `rl_agent_config.json`):
  `max_len = 512` total, split into an option-prompt budget `head_max_len = 192`
  and the remaining ~320 tokens for the serialized state.
- **When input exceeds the budget**: the state is truncated to the room left
  after the head (`state[:room]`, or `state[-room:]` for conversation prefixes),
  and each option text is capped at 48 tokens. If the options cannot fit within
  `head_max_len` at all, the reference API raises
  `ValueError("options do not fit in head_max_len=...")`; the Jev-compatible
  server returns HTTP 422.
(Sources: `rl_common.py` `build_sequence` / `render_options`; `rl_agent_config.json`;
model card "Architecture" and "Honest Limits".)

## Output decoding

The forward pass returns per-option `logits` and an act head output. Logits are
divided by the fitted temperature for the question's `(type, option-count)`
bucket, then softmaxed over that question's options (`rl_agent_api.py`
`system_one`).

- **`choice`**: `probabilities` is the softmax over option keys; `choice` is the
  arg-max key.
- **`score`**: per-level `probabilities` (keyed by level index, with a `legend`
  mapping index to description) and the reported `score` is the **expected value
  over ordered levels**, `sum(i * p_i)`.
- **`noul`**: the returned value is `p[1]` — the probability of the "true" slot,
  i.e. **P(true)**.
- **Confidence** (`choice` and `score` only): `confidence_from_probs` returns
  `1 - normalized_entropy`, i.e. `1 - entropy(p) / log(k)`. This measures how
  concentrated the distribution is, not the probability of the chosen answer.
  (The `laya` PyPI package additionally exposes `answer_confidence`, the
  probability of the reported answer.)
- **Act / escalate head**: the model has an `act_head` producing act logits;
  after softmax, `act_probability = act[:, 0]` is attached to each answer under an
  `rl_agent` key. The project states this head **carries no usable signal yet**
  (it reads ~1.0 for almost every input); gate on `confidence` instead.
(Sources: `rl_agent_api.py` `system_one`; `rl_common.py` `confidence_from_probs`,
`DecisionModel`; model card "Honest Limits" on `act_probability`.)

## Calibration

Calibration is temperature scaling applied to the **option logits before the
softmax**. The parameter is a per-question-type temperature plus optional
per-cardinality overrides, stored in `rl_agent_config.json`:

- `temperature`: a list of three values indexed by type (`choice, score, noul`).
- `temperature_by_options`: a mapping keyed by buckets such as `choice:3-5`,
  `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2` (`temp_bucket` in
  `rl_common.py` maps an option count `k` to `2 / 3-5 / 6-10 / 11+`).

At inference, `logits / temperature_by_options.get(bucket, temperature[type])` is
softmaxed. The documented procedure is to refit one temperature per
`(question type, option count)` bucket on held-out data — this moves the English
checkpoint's mean ECE from **0.466 to 0.081**. Numeric temperatures are clamped
to `[0.5, 5.0]` at load; the fine-tuning notebook fits one temperature per type
and clears inherited bucket values so they cannot mask the new fit.
(Sources: `rl_agent_config.json`; `rl_common.py` `temp_bucket`; model card
"Ships over-confident"; GitHub README "Calibration" and "Fine-Tuning".)

## Fine-tuning

**Entry point.** The published entry point is the Kaggle notebook
`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, which runs the whole
loop: build the dataset, train, fit calibration temperatures, evaluate, and push
to the Hub. There is no standalone training CLI in the model repository; the
reusable training logic lives in `rl_common.py`.

**What is trained.** The backbone encoder is **fully fine-tuned** (not LoRA, not
frozen) together with the from-scratch decision head. `rl_agent_config.json`
records the shipped checkpoint at 7,313 updates, 1 epoch, ~1.96 hours,
`world_size = 1`, `fine_tuned_from_checkpoint = true`.

**Training method.** RLCD — reinforcement learning against strictly proper
scoring rules. The reward (`proper_reward` in `rl_common.py`) is a log score plus
a spherical score for all types, plus a ranked probability score for ordinal
`score` questions; updates are REINFORCE with a group-mean baseline (GRPO-style),
and multi-turn conversations use TD(λ=1.0) over prefix slices.

**Training-record format.** `encode_record` (in `rl_common.py`) reads records of
this shape:

```json
{
  "state": {"subject": "...", "body": "..."},
  "src": "optional-source-tag",
  "qs": [
    {"t": "choice", "ins": "Which department?",
     "crit": {"billing": "invoices, refunds", "technical": "bugs"}, "y": 0},
    {"t": "score", "ins": "How urgent?",
     "crit": ["not urgent", "soon", "blocking"], "y": 2},
    {"t": "noul", "ins": "Does the user threaten to cancel?", "y": 1}
  ]
}
```

`y` is the correct label index (a `choice` key position, a `score` level, or 0/1
for `noul`); an optional `soft` list supplies a soft target distribution instead
of a one-hot label. Multi-turn records use `{"kind": "episode", "ep": {"ctx":
..., "turns": [...], "y": ...}, "qs": [<one noul question>]}`.

**Hyperparameters / hardware.** Gradient checkpointing is available on both the
encoder and the head (`model.head_checkpointing = True`). The notebook runs on
Kaggle's free 2×T4 GPUs; a full run is roughly 4–5 hours for 4 epochs over ~30k
questions. `amp_dtype` is `bf16` on Ampere+ and falls back to `fp16` on a T4.
(Sources: GitHub README "Fine-Tuning"; `rl_common.py` `encode_record`,
`proper_reward`, `DecisionModel`; `rl_agent_config.json` `training` block; model
card "Architecture" and "Training".)

## Runtime alternatives

- **MLX port — `laya-mlx` (github.com/mizorewww/laya-mlx).** Native Apple Silicon
  inference with no PyTorch or Transformers runtime; `pip install laya-mlx`, then
  `laya_mlx.load(...).predict(state, questions)` with the same typed-question
  shape. Reports ~13.4 ms median for a short English decision (~7.4 ms
  multilingual). Optional for this repository.
  (Source: laya-mlx README.)
- **ONNX port — `@receptron/laya` (github.com/receptron/laya).** A Node.js /
  TypeScript runtime on ONNX Runtime, no Python at runtime. Same request/response
  shape as `RLAgent.system_one`, matching the Python output to four decimal
  places; ~1.7 GB fp32 weights. Not used here (this repository is Python).
  (Source: @receptron/laya README.)

The **Transformers path is sufficient for this repository**: it runs on the
`torch` + `transformers` the `laya` extra already installs, produces the same
typed answers, and needs no extra runtime. MLX and ONNX are optional.

## Known limits stated by the project

- **Base checkpoints are near chance zero-shot on typed decisions** (0.362 for
  `laya`, against a 0.318 random and 0.461 majority-class baseline). Laya is a
  fast base to specialise; fine-tuning is where accuracy jumps (0.766 for the
  fine-tuned checkpoint).
- **Choice cardinality.** Options share the `head_max_len` budget (192 tokens on
  the English checkpoint). Beyond ~20 options accuracy falls off sharply
  (Banking77: 0.425 vs Jev's 0.870). For large label sets, raise `head_max_len`
  and `max_len`, shortlist first, or split into coarse-to-fine questions. Avoid
  boolean-word labels (`true`/`false`, `yes`/`no`) — the checkpoint can follow
  the label instead of its description.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- **`noul` can follow its `false:` / `true:` labels instead of the state**, most
  strongly on this English checkpoint; validate on your own data, or ask the same
  question as a two-option `choice` with neutral keys.
- **`act_probability` carries no usable signal yet**; gate on `confidence`.
- **English only on the root checkpoint**; use `laya-multilingual` for other
  languages.
(Source: model card / GitHub README "Honest Limits" and "Where Jev leads".)

## How this repository uses it

- **Local judge.** The English checkpoint `convaiinnovations/laya` answers the
  same `choice` / `score` / `noul` questions as the other judges, mapped to the
  Jev-style question shape (`type`, `instructions`, `criteria`) the model expects.
- **Compact state.** Because the English checkpoint reads only 512 tokens
  (~320 for the state after the option budget), states are serialized compactly,
  under 450 tokens, so the rubric and options are never crowded out.
- **Same rubric text.** Judges share the rubric verbatim; its text becomes each
  question's `instructions`, exactly as the Jev judge sends it, so the local and
  hosted judges answer identical questions.
- **Zero-shot and fine-tuned variants.** The shipped checkpoint is used
  zero-shot; a domain-fine-tuned checkpoint (trained with the notebook loop and
  its own fitted temperatures) can be swapped in behind the same interface.
- **Recorded provenance.** Every verdict records the checkpoint's pinned revision
  hash (`config/study.toml`, `convaiinnovations/laya` at
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`) so results are reproducible.

**Fallback (loader).** This repository loads the checkpoint with `transformers` +
`safetensors`, delegating to the model repository's own `rl_common.py` vendored
under `decision_judges/judges/laya_vendor/` (stock `AutoModel` encoder plus the
`DecisionModel` head), rather than depending on the `laya` PyPI package. This
keeps the dependency set to the `torch` + `transformers` the `laya` extra
installs.

**Fallback (fine-tuning entry point).** There is no standalone training CLI; the
published loop is the Kaggle notebook. When a custom loop is needed, this
repository drives the reusable pieces in `rl_common.py` (`encode_record`,
`build_model`, `proper_reward`, `collate_items`) directly, using the record
format documented above.
