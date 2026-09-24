"""Model adapter that reproduces the Laya typed-decision forward pass.

The adapter isolates every dependency on ``torch``/``transformers`` so the judge
that consumes it stays thin and testable. It wraps three pieces:

* a tokenizer loaded with ``transformers.AutoTokenizer``,
* a stock encoder loaded with ``transformers.AutoModel`` (the real checkpoint
  uses a ModernBERT backbone; the adapter accepts any ``AutoModel`` encoder), and
* a from-scratch :class:`DecisionHead` that maps each option's ``[MASK]`` hidden
  state to one logit.

Head reproduction. ``docs/laya.md`` documents that the real checkpoint's head
lives in the model repository's ``rl_common.py`` (the ``DecisionModel`` head) and
that its weights ship inside ``model.safetensors``; the exact layer sizes and
state-dict key names are not reproduced there. This adapter defines a two-layer
MLP head and loads the checkpoint's head weights by matching the keys named in
:data:`HEAD_WEIGHT_KEYS`, which is the single place to correct once the real
checkpoint's state-dict keys are confirmed against ``docs/laya.md``.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Maps this head's local state-dict keys to the keys carried in the checkpoint's
# ``model.safetensors``. Correct the right-hand values in one place if the real
# checkpoint names its head weights differently.
HEAD_WEIGHT_KEYS: dict[str, str] = {
    "dense.weight": "decision_head.dense.weight",
    "dense.bias": "decision_head.dense.bias",
    "out.weight": "decision_head.out.weight",
    "out.bias": "decision_head.out.bias",
}


class DecisionHead(nn.Module):
    """Two-layer MLP mapping a ``[MASK]`` hidden state to one option logit."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return one logit per input row of ``[MASK]`` hidden states."""
        return self.out(torch.tanh(self.dense(hidden))).squeeze(-1)


def _hash_bytes(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayaDecisionModel:
    """Runs the documented Laya template and per-``[MASK]`` option scoring."""

    input_limit = 512
    head_limit = 192

    def __init__(
        self,
        tokenizer: Any,
        encoder: nn.Module,
        head: DecisionHead,
        *,
        checkpoint_hash: str,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.head = head
        self.checkpoint_hash = checkpoint_hash
        self.device = device

    @classmethod
    def from_pretrained(
        cls, repo_or_path: str, *, revision: str | None = None, device: str = "cpu"
    ) -> "LayaDecisionModel":
        """Load the tokenizer, encoder, and head from a checkpoint directory.

        The directory holds ``tokenizer/`` and ``encoder/`` sub-directories and,
        when present, a ``model.safetensors`` carrying the head weights. The
        checkpoint hash is the sha256 of ``model.safetensors`` when it exists,
        otherwise of the encoder's ``config.json``.
        """
        root = Path(repo_or_path)
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), revision=revision)
        encoder = AutoModel.from_pretrained(str(root / "encoder"), revision=revision)
        hidden_size = int(encoder.config.hidden_size)
        head = DecisionHead(hidden_size)

        weights_path = root / "model.safetensors"
        if weights_path.exists():
            checkpoint = load_file(str(weights_path))
            local_state = {
                local: checkpoint[remote]
                for local, remote in HEAD_WEIGHT_KEYS.items()
                if remote in checkpoint
            }
            head.load_state_dict(local_state, strict=False)
            checkpoint_hash = _hash_bytes(weights_path)
        else:
            checkpoint_hash = _hash_bytes(root / "encoder" / "config.json")

        encoder.to(device).eval()
        head.to(device).eval()
        return cls(tokenizer, encoder, head, checkpoint_hash=checkpoint_hash, device=device)

    def _build_input(
        self, question_type: str, instructions: str, options: Sequence[str], state: str
    ) -> str:
        """Render the documented input template with one ``[MASK]`` per option."""
        mask = self.tokenizer.mask_token
        sep = self.tokenizer.sep_token
        cls = self.tokenizer.cls_token
        option_block = " ".join(f"{mask} {option}" for option in options)
        return (
            f"{cls} {question_type} question: {instructions} {sep} "
            f"{option_block} {sep} {state} {sep}"
        )

    def tokenizer_len(self, text: str) -> int:
        """Return the token count of ``text`` without added special tokens."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.no_grad()
    def option_logits(
        self,
        state: str,
        question_type: str,
        instructions: str,
        options: Sequence[str],
    ) -> list[float]:
        """Return one logit per option, scored at the option's ``[MASK]`` marker."""
        text = self._build_input(question_type, instructions, options, state)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.input_limit,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"][0]
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if int(mask_positions.numel()) < len(options):
            raise ValueError("options do not fit within the input limit")

        outputs = self.encoder(**encoded)
        hidden = outputs.last_hidden_state[0]
        selected = hidden[mask_positions[: len(options)]]
        logits = self.head(selected)
        return [float(value) for value in logits.reshape(-1).tolist()]
