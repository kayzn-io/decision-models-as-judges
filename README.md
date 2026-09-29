# Decision models as judges

Typed decision models such as Jev and Laya are used as evaluation judges at each step of an agent evaluation loop on tau-bench. Runs are compared with and without these judges against deterministic ground truth, and every result is reproducible from committed caches.

## Results

<!-- results:start -->
Results are published here as evaluation gates complete.
<!-- results:end -->

## Run it

The repository ships the 230 conversations, so you never have to generate them to test the judges. Two commands install the study and open the app.

```
uv sync --extra laya
make ui
```

Open the Run page and work through the seven steps that test the judges on the shipped conversations. Each step shows what it reads and writes, what it costs against the spend caps, and a live view while it runs. One `OPENROUTER_API_KEY` is needed only for the paid judges; set it in the app sidebar or export it before `make ui`. Regenerating the conversations with your own agent is optional and costs about $10. The same steps are available as commands for scripting; see the [runbook](docs/runbook.md).

## What the study does

### The conversations

The study ships 230 conversations between a customer and an AI support agent. Half come from a careful agent and half from a rushed agent, and each conversation carries a ground truth of pass or fail. The conversations are drawn from the retail part of the public tau-bench test set, where the ground truth is computed from the final state of the data rather than from anyone's opinion.

### Two reading copies

Each conversation has two reading copies. The full text carries every turn and every tool result whole, so a judge can check what the agent said against what its tools returned; it runs about 3,000 tokens. The short text runs about 400 tokens and keeps only the request, one policy line, the tool calls, and the final message. The short copy exists because the local model Laya reads at most 512 tokens, so every judge is also run on the short copy for one fair comparison.

### What a judge is asked

A judge reads one conversation and answers two questions in a single call. The first states "The agent completed the user request in line with the policy." and asks how strongly it holds. The second asks for a pass or fail choice against the outcome rubric. Each judge answers five times per conversation so the study can measure how consistent its verdicts are.

### The four batches

The judging step runs four batches: two agents by two reading copies. Each batch judges 115 conversations with 4 judges, 5 times each, for 2,300 calls, and the four batches together make 9,200 calls. The code runs them in this order:

| Order | Agent | Reading copy |
| --- | --- | --- |
| 1 | careful | full |
| 2 | careful | short |
| 3 | rushed | full |
| 4 | rushed | short |

### The judges

- Rule-based check: a deterministic check that reads the conversation, free.
- Fast text model: a small hosted model.
- Strong text model: a large hosted model.
- Jev: a hosted decision model.
- Laya, as published: the local decision model, free.
- Laya, trained on these conversations: the same local model after training on them, free.

### Resuming

Every verdict is cached the moment it completes, under a key of judge, model, rubric version, text hash, and repeat. Interrupting the run loses nothing, and rerunning does only the missing calls. Errored verdicts are retried rather than kept.

## Development

```
uv sync
make check
```

### Keys

One `OPENROUTER_API_KEY` covers the agent runs, the LLM judges, and Jev: TypeSafe paused direct signups on 2026-09-22 and serves Jev through OpenRouter. To call TypeSafe directly, set `TYPESAFE_API_KEY` and `[jev_route] provider = "typesafe"` in `config/study.toml`.
