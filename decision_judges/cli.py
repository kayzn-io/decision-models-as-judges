"""Command-line interface for decision_judges."""

from pathlib import Path
from typing import Annotated

import typer

from decision_judges import __version__
from decision_judges.bench import load as bench_load
from decision_judges.bench import run_agent as run_agent_mod
from decision_judges.config import load_pricing, load_study
from decision_judges.report import render_summary, update_readme
from decision_judges.spend import Spend

app = typer.Typer(help="Decision models as judges.")


@app.callback()
def main() -> None:
    """Decision models as judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def results(
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("cache"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    readme: Annotated[Path, typer.Option("--readme")] = Path("README.md"),
) -> None:
    """Render a results summary from the cache and refresh the README."""
    summary = render_summary(cache_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    if readme.is_file():
        update_readme(readme, summary)
        typer.echo(f"Wrote {summary_path} and updated {readme}")
    else:
        typer.echo(f"Wrote {summary_path}")


@app.command(name="run-agent")
def run_agent(
    variant: Annotated[str, typer.Option("--variant")],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("cache/agent"),
    study: Annotated[Path, typer.Option("--study")] = Path("config/study.toml"),
    pricing: Annotated[Path, typer.Option("--pricing")] = Path("config/pricing.toml"),
    ledger: Annotated[Path, typer.Option("--ledger")] = Path("results/spend.json"),
) -> None:
    """Run tau-bench retail tasks with the agent under one policy variant."""
    if variant not in ("baseline", "degraded"):
        raise typer.BadParameter("variant must be 'baseline' or 'degraded'")
    checked_variant: run_agent_mod.Variant = "baseline" if variant == "baseline" else "degraded"
    study_config = load_study(study)
    pricing_table = load_pricing(pricing)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(pricing_table, study_config.spend_caps, ledger)
    tasks = bench_load.load_tasks()
    runner = run_agent_mod.build_tau_runner(study_config, checked_variant)
    concurrency = study_config.concurrency.get("llm", 4)
    summary = run_agent_mod.run_variant(
        checked_variant, tasks, runner, spend, out_dir, concurrency=concurrency
    )
    typer.echo(summary.model_dump_json())
