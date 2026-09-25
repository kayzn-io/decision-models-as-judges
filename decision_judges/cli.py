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
        raise typer.BadParameter(
            f"unknown gate {gate!r}; known: {', '.join(sorted(registry))}. "
            "Analysis-only gates g5, g6, and g8 are run with 'judges analyze'."
        )
    if variant not in _VARIANTS:
        raise typer.BadParameter("variant must be 'baseline' or 'degraded'")
    try:
        state_profile = StateProfile(profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

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
    items = pipeline.items_for_gate(gate, state_dir, agent_dir, state_profile, variant, tasks)
    if not items:
        raise typer.BadParameter(
            f"no serialized states for gate {gate!r} under {state_dir}; run 'serialize' first"
        )

    names = [name.strip() for name in judges.split(",") if name.strip()]
    specs = pipeline.judge_specs_from(names, study_config)
    repeat_plan = (
        pipeline.parse_repeats(repeats)
        if repeats
        else pipeline.default_repeats(gate, study_config, names)
    )
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


@app.command()
def analyze(
    gate: Annotated[str, typer.Option("--gate")],
    profile: Annotated[str, typer.Option("--profile")] = "full",
    variant: Annotated[list[str] | None, typer.Option("--variant")] = None,
    agent_dir: Annotated[Path, typer.Option("--agent-dir")] = Path("cache/agent"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path("cache/state"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("cache/judge"),
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    study: Annotated[Path, typer.Option("--study")] = Path("config/study.toml"),
    pricing: Annotated[Path, typer.Option("--pricing")] = Path("config/pricing.toml"),
) -> None:
    """Analyze cached verdicts through a gate that reuses them, writing results."""
    study_config = load_study(study)
    pricing_table = load_pricing(pricing)
    registry = pipeline.analysis_registry(study_config, pricing_table)
    if gate not in registry:
        raise typer.BadParameter(
            f"unknown analysis gate {gate!r}; known: {', '.join(sorted(registry))}"
        )
    try:
        state_profile = StateProfile(profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    present: list[str] = []
    for name in _parse_variants(variant):
        if (agent_dir / name).is_dir():
            present.append(name)
        else:
            typer.echo(f"skipping {name}: no agent records under {agent_dir / name}")

    gate_impl = registry[gate]
    items = pipeline.items_for_variants(state_dir, agent_dir, state_profile, present)
    all_verdicts, warnings = pipeline.load_verdicts(cache_dir)
    for warning in warnings:
        typer.echo(f"skipped verdict {warning}")
    verdicts = pipeline.filter_verdicts_to_items(all_verdicts, items)
    if not verdicts:
        typer.echo(
            f"no cached verdicts match under {cache_dir}; "
            "run 'judges judge --gate g3' first to populate the verdict cache"
        )
        return
    findings = pipeline.analyze_gate(gate_impl, verdicts, items, results_dir)
    typer.echo(findings)


def _read_compact_states(state_dir: Path, variant: str) -> dict[str, StateRecord]:
    """Read compact serialized states for a variant, keyed by task id."""
    directory = state_dir / variant / StateProfile.compact.value
    if not directory.is_dir():
        raise typer.BadParameter(f"no compact states under {directory}; run 'serialize' first")
    states: dict[str, StateRecord] = {}
    for path in sorted(directory.glob("*.json")):
        record = StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
        states[record.task_id] = record
    if not states:
        raise typer.BadParameter(f"no compact states under {directory}; run 'serialize' first")
    return states


def _read_rewards(agent_dir: Path, variant: str) -> dict[str, float]:
    """Read run rewards for a variant, keyed by task id."""
    directory = agent_dir / variant
    if not directory.is_dir():
        raise typer.BadParameter(f"no agent records under {directory}; run 'run-agent' first")
    rewards: dict[str, float] = {}
    for path in sorted(directory.glob("*.json")):
        record = run_agent_mod.AgentRecord.model_validate_json(path.read_text(encoding="utf-8"))
        rewards[record.task_id] = record.reward
    return rewards


def _resolve_base(base: str | None, study_config: object) -> Path:
    """Resolve the base checkpoint to a local directory, downloading if needed."""
    if base is not None and Path(base).is_dir():
        return Path(base)
    from huggingface_hub import snapshot_download

    laya = study_config.models.laya  # type: ignore[attr-defined]
    repo_id = base or laya.repo_id
    revision = None if base is not None else laya.revision
    patterns = ["rl_agent_config.json", "encoder/config.json", "tokenizer/*", "model.safetensors"]
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=patterns))


@app.command(name="finetune-laya")
def finetune_laya(
    base: Annotated[str | None, typer.Option("--base")] = None,
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path("cache/state"),
    agent_dir: Annotated[Path, typer.Option("--agent-dir")] = Path("cache/agent"),
    variant: Annotated[str, typer.Option("--variant")] = "baseline",
    out: Annotated[Path, typer.Option("--out")] = Path("models"),
    folds: Annotated[int, typer.Option("--folds")] = 5,
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    lr: Annotated[float, typer.Option("--lr")] = 2e-5,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    study: Annotated[Path, typer.Option("--study")] = Path("config/study.toml"),
) -> None:
    """Fine-tune Laya on G3 outcomes with k-fold cross validation."""
    from decision_judges.gates.g3_outcome import G3Outcome
    from decision_judges.training import finetune_laya as training

    if variant not in _VARIANTS:
        raise typer.BadParameter("variant must be 'baseline' or 'degraded'")
    study_config = load_study(study)
    base_dir = _resolve_base(base, study_config)
    states = _read_compact_states(state_dir, variant)
    rewards = _read_rewards(agent_dir, variant)
    questions = G3Outcome().questions()
    manifests = training.run_cross_validation(
        base_dir,
        states,
        rewards,
        questions,
        out,
        k=folds,
        seed=study_config.seed,
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        device=device,
        max_steps=max_steps,
    )
    for manifest in manifests:
        directory = out / f"laya-g3-fold{manifest.fold}"
        typer.echo(
            f"{directory}: n_train={len(manifest.train_task_ids)} "
            f"n_test={len(manifest.test_task_ids)} "
            f"temperatures={manifest.temperature_by_options}"
        )


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
