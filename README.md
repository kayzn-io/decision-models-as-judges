# Decision models as judges

Typed decision models such as Jev and Laya are used as evaluation judges at each step of an agent evaluation loop on tau-bench. Runs are compared with and without these judges against deterministic ground truth, and every result is reproducible from committed caches.

## Results

<!-- results:start -->
Results are published here as evaluation gates complete.
<!-- results:end -->

## Development

```
uv sync
make check
```

### Keys

One `OPENROUTER_API_KEY` covers the agent runs, the LLM judges, and Jev: TypeSafe paused direct signups on 2026-09-22 and serves Jev through OpenRouter. To call TypeSafe directly, set `TYPESAFE_API_KEY` and `[jev_route] provider = "typesafe"` in `config/study.toml`.
