"""Command-line interface for decision_judges."""

import json
from pathlib import Path
from typing import Annotated

import typer

from decision_judges import __version__, pipeline
from decision_judges.bench import load as bench_load
from decision_judges.bench import run_agent as run_agent_mod
from decision_judges.bench.load import Task
from decision_judges.cache import Cache
from decision_judges.config import load_pricing, load_study
from decision_judges.report import render_summary, update_readme
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Spend

app = typer.Typer(help="Decision models as judges.")

_VARIANTS = ("baseline", "degraded")


@app.callback()
def main() -> None:
    """Decision models as judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


def _load_tasks(fixture: Path | None) -> list[Task]:
    """Load tasks from a JSON fixture when given, else from tau-bench."""
    if fixture is not None:
        source = json.loads(fixture.read_text(encoding="utf-8"))
        return bench_load.load_tasks(source=source)
    return bench_load.load_tasks()


def _parse_profiles(values: list[str] | None) -> list[StateProfile]:
    """Resolve profile option values, defaulting to both profiles."""
    names = values or ["full", "compact"]
    try:
        return [StateProfile(name) for name in names]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_variants(values: list[str] | None) -> list[str]:
    """Resolve variant option values, defaulting to both variants."""
    names = values or list(_VARIANTS)
    for name in names:
        if name not in _VARIANTS:
            raise typer.BadParameter("variant must be 'baseline' or 'degraded'")
    return names


def _read_states(state_dir: Path, variant: str, profile: StateProfile) -> list[StateRecord]:
    """Read serialized states for a variant and profile, or fail clearly."""
    directory = state_dir / variant / profile.value
    states: list[StateRecord] = []
    if directory.is_dir():
        states = [
            StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]
    if not states:
        raise typer.BadParameter(f"no serialized states under {directory}; run 'serialize' first")
    return states


@app.command()
def results(
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("cache"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    readme: Annotated[Path, typer.Option("--readme")] = Path("README.md"),
) -> None:
    """Render a results summary from the cache and refresh the README."""
    summary = render_summary(cache_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables = pipeline.list_result_tables(results_dir)
    body = summary
    if tables:
        links = "\n".join(f"- [{name}]({name}.md)" for name in tables)
        body = f"{summary}\n\n**Tables:**\n\n{links}"
    summary_path = results_dir / "summary.md"
    summary_path.write_text(body, encoding="utf-8")
    if readme.is_file():
        update_readme(readme, body)
        typer.echo(f"Wrote {summary_path} and updated {readme}")
    else:
        typer.echo(f"Wrote {summary_path}")


@app.command()
def serialize(
    agent_dir: Annotated[Path, typer.Option("--agent-dir")] = Path("cache/agent"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path("cache/state"),
    profile: Annotated[list[str] | None, typer.Option("--profile")] = None,
    variant: Annotated[list[str] | None, typer.Option("--variant")] = None,
    tasks_fixture: Annotated[Path | None, typer.Option("--tasks-fixture")] = None,
) -> None:
    """Serialize recorded agent runs into judge-visible states per profile."""
    profiles = _parse_profiles(profile)
    variants = _parse_variants(variant)
    tasks = {task.task_id: task for task in _load_tasks(tasks_fixture)}
    for name in variants:
        records, warnings = pipeline.load_agent_records(agent_dir, name)
        for warning in warnings:
            typer.echo(f"skipped {name}/{warning}")
        count = pipeline.serialize_all(records, tasks, state_dir, profiles)
        typer.echo(f"{name}: {count} states")


@app.command()
def judge(
    gate: Annotated[str, typer.Option("--gate")],
    profile: Annotated[str, typer.Option("--profile")] = "full",
    variant: Annotated[str, typer.Option("--variant")] = "baseline",
    judges: Annotated[str, typer.Option("--judges")] = "code",
    repeats: Annotated[list[str] | None, typer.Option("--repeats")] = None,
    agent_dir: Annotated[Path, typer.Option("--agent-dir")] = Path("cache/agent"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path("cache/state"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("cache/judge"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    study: Annotated[Path, typer.Option("--study")] = Path("config/study.toml"),
    pricing: Annotated[Path, typer.Option("--pricing")] = Path("config/pricing.toml"),
    ledger: Annotated[Path, typer.Option("--ledger")] = Path("results/spend.json"),
    tasks_fixture: Annotated[Path | None, typer.Option("--tasks-fixture")] = None,
) -> None:
    """Run a gate's judges over serialized states and analyze the verdicts."""
    registry = pipeline.gate_registry()
    if gate not in registry:
        raise typer.BadParameter(f"unknown gate {gate!r}; known: {', '.join(sorted(registry))}")
    if variant not in _VARIANTS:
        raise typer.BadParameter("variant must be 'baseline' or 'degraded'")
    try:
        state_profile = StateProfile(profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    repeat_plan = pipeline.parse_repeats(repeats or [])

    study_config = load_study(study)
    pricing_table = load_pricing(pricing)
    gate_impl = registry[gate]()

    records, warnings = pipeline.load_agent_records(agent_dir, variant)
    for warning in warnings:
        typer.echo(f"skipped {variant}/{warning}")
    if not records:
        raise typer.BadParameter(
            f"no agent records under {agent_dir / variant}; run 'run-agent' first"
        )
    tasks = {task.task_id: task for task in _load_tasks(tasks_fixture)}
    states = _read_states(state_dir, variant, state_profile)
    items = pipeline.items_from_states(states, records)

    names = [name.strip() for name in judges.split(",") if name.strip()]
    specs = pipeline.judge_specs_from(names, study_config)
    judge_list = pipeline.build_judges(
        specs,
        study=study_config,
        tasks=tasks,
        records=records,
        rubric_path=pipeline.gate_rubric_path(gate_impl),
        prompt_version=pipeline.gate_prompt_version(gate_impl),
    )

    ledger.parent.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_dir)
    spend = Spend(pricing_table, study_config.spend_caps, ledger)
    verdicts = pipeline.run_gate(gate_impl, items, judge_list, cache, spend, repeat_plan)
    findings = pipeline.analyze_gate(gate_impl, verdicts, items, results_dir)
    typer.echo(findings)
    typer.echo(f"{gate} spend: ${spend.spent(gate_impl.stage):.6f}")


@app.command(name="run-agent")
def run_agent(
    variant: Annotated[str, typer.Option("--variant")],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("cache/agent"),
    study: Annotated[Path, typer.Option("--study")] = Path("config/study.toml"),
    pricing: Annotated[Path, typer.Option("--pricing")] = Path("config/pricing.toml"),
    ledger: Annotated[Path, typer.Option("--ledger")] = Path("results/spend.json"),
) -> None:
    """Run tau-bench retail tasks with the agent under one policy variant."""
    if variant not in _VARIANTS:
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
