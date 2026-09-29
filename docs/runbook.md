# Running the study end to end

The app's Run page (`make ui`, then Run) performs every step below with a live
view of progress and spend; this runbook lists the equivalent commands for
scripting or for running a single stage by hand, the spend cap each stage runs
under, and the files each command writes. Spend figures are the
caps enforced before each paid call, taken from `config/study.toml`; a stage
stops rather than exceed its cap.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) on the path.
- One `OPENROUTER_API_KEY` exported. It covers the agent runs, the LLM judges,
  and Jev, which is served through OpenRouter.
- For the local Laya model, install the optional extra and allow about 1 GB of
  disk for the checkpoint:

  ```
  uv sync --extra laya
  ```

  The default `uv sync` is enough for every stage except the two Laya judges and
  fine-tuning. Laya runs on CPU; a GPU only speeds up fine-tuning. Pass
  `--device cuda` to the `judge` and `finetune-laya` commands to use it.

## Command sequence

Run the stages in this order. Each `judge` and `analyze` command writes result
tables and charts under `results/` and the gate's prose to
`results/<gate>_findings.md`; each paid call is metered into the ledger at
`results/spend.json`.

1. `judges run-agent --variant baseline` — stage `agent`, cap $25.00 (shared
   across both variants). Writes agent records to `cache/agent/baseline/*.json`.
2. `judges run-agent --variant degraded` — stage `agent`, same $25.00 cap.
   Writes `cache/agent/degraded/*.json`.
3. `judges serialize` — no spend. Reads the agent records and writes
   judge-visible states to `cache/state/<variant>/<profile>/*.json`, plus
   per-step states under `steps/` and injected copies under `injected/`.
4. `judges judge --gate g3 --profile full --judges code,llm_cheap,llm_strong,jev`
   — stage `g3`, cap $50.00. Writes verdicts to `cache/judge/` and
   `results/g3_summary.{md,csv}`, `results/g3_accuracy.{png,svg}`,
   `results/g3_findings.md`.
5. `judges judge --gate g3 --profile compact --judges code,llm_cheap,llm_strong,jev`
   — stage `g3`, same $50.00 cap. Adds the compact-profile verdicts.
6. `judges judge --gate g3 --profile compact --judges laya_base` — stage `g3`.
   Laya runs locally and is priced at zero, so it draws nothing from the cap.
7. `judges finetune-laya` — no API spend; local training only. Writes one
   checkpoint per fold to `models/laya-g3-fold{f}/` with `model.safetensors` and
   `manifest.json`. Add `--device cuda` on a GPU host.
8. `judges judge --gate g3 --profile compact --judges laya_ft` — stage `g3`.
   Routes each task to its held-out fold under `models/` (`--models-root`).
9. `judges judge --gate g2` — stage `g2`, cap $25.00. Writes
   `results/g2_summary.md` and `results/g2_auroc.png`.
10. `judges judge --gate g4` — stage `g4`, cap $10.00. Writes
    `results/g4_summary.md` and `results/g4_auroc_by_aggregator.png`.
11. `judges judge --gate g7` — stage `g7`, cap $15.00. Writes
    `results/g7_summary.md` and `results/g7_flip_rates.png`.
12. `judges judge --gate g1` — stage `g1`, cap $5.00. Writes
    `results/g1_summary.md` and `results/g1_difficulty_vs_actions.png`.
13. `judges judge --gate g9` — stage `g9`, cap $5.00. Reads hand labels from
    `data/labels/taxonomy.jsonl`; writes `results/g9_summary.md` and
    `results/g9_accuracy.png`.
14. `judges judge --gate g10` — stage `g10`, cap $5.00. Writes
    `results/g10_summary.md`, `results/g10_folds.md`, and
    `results/g10_zero_shot_vs_finetuned.png`.
15. `judges analyze --gate g3 --profile full --profile compact` — stage `g3`,
    no spend. Rebuilds `results/g3_summary.{md,csv}`,
    `results/g3_accuracy.{png,svg}`, and `results/g3_findings.md` from every
    cached g3 verdict, covering all judges and both profiles. Each `judge`
    command above writes those files from only the batch it just ran, so run
    this once after the last g3 judge to get the complete table.
16. `judges analyze --gate g5` — stage `g5`, cap $0.00; reuses the cached g3
    verdicts. Writes `results/g5_frontier.md`, `results/g5_reference.md`, and
    `results/g5_frontier.png`.
17. `judges analyze --gate g6` — stage `g6`, cap $0.00. Writes
    `results/g6_summary.md` and `results/g6_reliability.png`.
18. `judges analyze --gate g8` — stage `g8`, cap $0.00. Writes
    `results/g8_summary.md` and `results/g8_intervals.png`.
19. `make results` — no spend. Writes `results/summary.md` and refreshes the
    results block in `README.md`.

Each `judge` command runs the judge set named by `--judges` (default `code`).
Extend that list to include `llm_cheap`, `llm_strong`, `jev`, `laya_base`, and
`laya_ft` for the gates you want those judges to cover.

## Resuming after an interruption

Rerun the same command. Verdicts and agent records are content-addressed and
cached, so completed work is read from `cache/` rather than recomputed and only
the missing calls are made. Serialization is deterministic and overwrites
identical files, so re-running `judges serialize` is safe.

## Checking spend

There is no separate spend command. Read the ledger at `results/spend.json`: it
records settled cost per stage, per model, and per call, so you can see how much
of each stage's cap has been used.

## Labeling failures for the taxonomy gate

The g9 taxonomy gate scores judges against hand labels. Produce them locally:

```
make ui
```

`make ui` starts the app with `JUDGES_LOCAL=1`, which unlocks the Label page.
Open that page and label failing runs until the progress bar reaches its target
of 50. Labels are written to `data/labels/taxonomy.jsonl`, which `judges judge
--gate g9` then reads.

## Regenerating the README

- `make results` rewrites `results/summary.md` and the results block in
  `README.md` from whatever is cached.

## Deploying the app

The app entry point is `decision_judges/ui/app.py`. On Streamlit Community
Cloud, point the app at that file. The deployment needs no secrets: without
`JUDGES_LOCAL=1` the Label and Live pages stay read-only and the app serves only
the committed cache, so the public site never makes a paid call.
