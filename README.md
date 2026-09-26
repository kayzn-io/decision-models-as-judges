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

## Development

```
uv sync
make check
```

### Keys

One `OPENROUTER_API_KEY` covers the agent runs, the LLM judges, and Jev: TypeSafe paused direct signups on 2026-09-22 and serves Jev through OpenRouter. To call TypeSafe directly, set `TYPESAFE_API_KEY` and `[jev_route] provider = "typesafe"` in `config/study.toml`.
