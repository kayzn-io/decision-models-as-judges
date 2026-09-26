"""The study steps, each with its own unlock, status, and runner.

The first entry is a material panel describing the shipped conversations; the
seven that follow are the runnable steps, numbered 1 to 7 in the UI. A
:class:`RunStep` is a frozen description of one stage: what it reads, what it
writes, what it costs, and a callable that performs the real work by delegating
to the pipeline, benchmark, training, and report modules. The step functions
are pure of Streamlit so they run on a background thread and stay unit-testable.

Paid steps read the visitor's OpenRouter key from the run context and set it
into ``OPENROUTER_API_KEY`` only for the duration of the step, and only when the
environment does not already carry one, restoring the previous state afterwards.
"""

import contextlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from decision_judges import pipeline
from decision_judges.bench.load import Task
from decision_judges.config import PricingTable, StudyConfig
from decision_judges.progress import CancelToken, Progress, ProgressCallback, utc_now_iso
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Ledger, Spend
from decision_judges.ui.data import Paths

_VARIANTS = ("baseline", "degraded")
_RETAIL_TEST_TASKS = 115
_G3_JUDGES = ("code", "llm_cheap", "llm_strong", "jev")
_G3_REPEATS = 5
_OPENROUTER_ENV = "OPENROUTER_API_KEY"


class StepStatus(BaseModel):
    """A step's readiness and how much of its work is already on disk."""

    state: Literal["locked", "ready", "partial", "done"]
    done: int = 0
    total: int = 0
    detail: str = ""


class RunContext(BaseModel):
    """The live study context a step runs against, including the visitor's key."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    study: StudyConfig
    pricing: PricingTable
    key: str | None = None
    on_progress: ProgressCallback | None = None
    cancel: CancelToken | None = None


class RunStep(BaseModel):
    """One study step: what it moves, what it costs, and how it runs."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str
    title: str
    purpose: str
    pipe: str | None
    input_station: str
    output_station: str
    stages: tuple[str, ...]
    paid: bool
    learn: str
    unlock: Callable[[Paths], str | None]
    status: Callable[[Paths], StepStatus]
    example_input: Callable[[Paths], str]
    example_output: Callable[[Paths], str]
    run: Callable[[Paths, RunContext], str | None]
    material: bool = False

    def cap_usd(self, study: StudyConfig) -> float | None:
        """Return the summed spend cap across the step's stages, or None when free."""
        if not self.paid:
            return None
        return sum(study.spend_caps.get(stage, 0.0) for stage in self.stages)

    def spent_usd(self, ledger: Ledger | None) -> float:
        """Return the settled spend across the step's stages, zero when no ledger."""
        if ledger is None:
            return 0.0
        return sum(ledger.per_stage.get(stage, 0.0) for stage in self.stages)


# --- environment ------------------------------------------------------------


@contextlib.contextmanager
def _openrouter_key(key: str | None) -> Iterator[None]:
    """Install the session's OpenRouter key for the block, then restore the environment.

    tau_bench calls litellm directly and can only take credentials from the
    environment, so the key is installed under both the OpenRouter name and
    ``OPENAI_API_KEY``, which litellm's ``openai`` provider reads. A key supplied
    from the app wins over whatever the shell exported, since a stale or empty
    shell value is the usual reason a run fails with missing credentials. Without
    a supplied key the environment is left as it is.
    """
    if not key:
        yield
        return
    names = (_OPENROUTER_ENV, "OPENAI_API_KEY")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = key
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --- disk readers -----------------------------------------------------------


def _agent_records(paths: Paths) -> dict[tuple[str, str], object]:
    """Return every agent record across variants keyed by variant and task id."""
    records: dict[tuple[str, str], object] = {}
    for variant in _VARIANTS:
        loaded, _ = pipeline.load_agent_records(paths.cache_dir / "agent", variant)
        for task_id, record in loaded.items():
            records[(variant, task_id)] = record
    return records


def _variants_present(paths: Paths) -> list[str]:
    """Return the variants that have an agent-run directory, in canonical order."""
    return [name for name in _VARIANTS if (paths.cache_dir / "agent" / name).is_dir()]


def _whole_state_files(paths: Paths) -> list[Path]:
    """Return the whole-trajectory state files, excluding step and injected copies."""
    state_dir = paths.cache_dir / "state"
    if not state_dir.is_dir():
        return []
    return [
        path
        for path in sorted(state_dir.rglob("*.json"))
        if "steps" not in path.parts and "injected" not in path.parts
    ]


def _verdict_count(paths: Paths) -> int:
    """Return the number of cached verdicts under the judge cache."""
    verdicts, _ = pipeline.load_verdicts(paths.cache_dir / "judge")
    return len(verdicts)


def _failing_count(paths: Paths) -> int:
    """Return how many recorded runs did not fully pass."""
    return sum(1 for record in _agent_records(paths).values() if _reward(record) < 1.0)


def _reward(record: object) -> float:
    """Return a record's reward as a float."""
    return float(getattr(record, "reward", 0.0))


def _results_present(paths: Paths, stems: tuple[str, ...]) -> int:
    """Return how many of the named result tables exist under the results dir."""
    return sum(1 for stem in stems if (paths.results_dir / f"{stem}.md").is_file())


# --- unlock rules -----------------------------------------------------------


def _always_ready(_paths: Paths) -> str | None:
    """The first step is always runnable."""
    return None


def _needs_conversations(paths: Paths) -> str | None:
    """Require recorded agent runs before serializing."""
    if _agent_records(paths):
        return None
    return "Needs conversations: run step 1 first."


def _needs_states(paths: Paths) -> str | None:
    """Require the reading copy before judging."""
    if _whole_state_files(paths):
        return None
    return "Needs the reading copy: run step 2 first."


def _needs_verdicts(paths: Paths) -> str | None:
    """Require cached verdicts before analysis or publishing."""
    if _verdict_count(paths) > 0:
        return None
    return "Needs verdicts: run step 3 first."


def _needs_failures(paths: Paths) -> str | None:
    """Require at least one failed run before labeling."""
    if _failing_count(paths) > 0:
        return None
    return "Needs failed runs: run step 1 first."


# --- status counters --------------------------------------------------------


def _status_from(
    unlock: Callable[[Paths], str | None], counts: Callable[[Paths], StepStatus]
) -> Callable[[Paths], StepStatus]:
    """Build a status callable that reports locked first, then on-disk progress."""

    def status(paths: Paths) -> StepStatus:
        reason = unlock(paths)
        if reason is not None:
            return StepStatus(state="locked", detail=reason)
        return counts(paths)

    return status


def _state_for(done: int, total: int, detail: str) -> StepStatus:
    """Classify progress into ready, partial, or done."""
    if total > 0 and done >= total:
        return StepStatus(state="done", done=done, total=total, detail=detail)
    if done > 0:
        return StepStatus(state="partial", done=done, total=total, detail=detail)
    return StepStatus(state="ready", done=done, total=total, detail=detail)


def _agent_counts(paths: Paths) -> StepStatus:
    done = len(_agent_records(paths))
    return _state_for(done, _RETAIL_TEST_TASKS, f"{done} of {_RETAIL_TEST_TASKS} runs recorded")


def _serialize_counts(paths: Paths) -> StepStatus:
    done = len(_whole_state_files(paths))
    total = len(_agent_records(paths)) * 2
    return _state_for(done, total, f"{done} of {total} reading copies written")


def _verdict_counts(paths: Paths) -> StepStatus:
    done = _verdict_count(paths)
    return _state_for(done, done, f"{done} verdicts cached")


def _analyze_counts(paths: Paths) -> StepStatus:
    done = _results_present(paths, ("g5_frontier", "g6_summary", "g8_summary"))
    return _state_for(done, 3, f"{done} of 3 analyses written")


def _laya_counts(paths: Paths) -> StepStatus:
    done = _results_present(paths, ("g10_summary",))
    return _state_for(done, 1, "local-model results written" if done else "not run yet")


def _gates_counts(paths: Paths) -> StepStatus:
    done = _results_present(paths, ("g1_summary", "g2_summary", "g4_summary", "g7_summary"))
    return _state_for(done, 4, f"{done} of 4 probes written")


def _label_counts(paths: Paths) -> StepStatus:
    done = _results_present(paths, ("g9_summary",))
    return _state_for(done, 1, "taxonomy results written" if done else "not run yet")


def _results_counts(paths: Paths) -> StepStatus:
    done = 1 if (paths.results_dir / "summary.md").is_file() else 0
    return _state_for(done, 1, "report published" if done else "not run yet")


# --- example records --------------------------------------------------------


def _format_turns(trajectory: list[dict[str, object]], limit: int = 4) -> str:
    """Render the first turns of a trajectory as short role-prefixed lines."""
    lines: list[str] = []
    for turn in trajectory[:limit]:
        role = str(turn.get("role", "?"))
        content = str(turn.get("content", "")).strip()
        calls = turn.get("tool_calls") or []
        if isinstance(calls, list) and calls:
            names = [str(call.get("function", {}).get("name", "?")) for call in calls]
            content = (content + " " if content else "") + "calls: " + ", ".join(names)
        lines.append(f"{role}: {content}".strip()[:200])
    return "\n".join(lines)


def _first_task_instruction(paths: Paths) -> str:
    """Return the first exported task's instruction, trimmed."""
    tasks_file = paths.data_dir / "tasks.json"
    if not tasks_file.is_file():
        return ""
    try:
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    if not tasks:
        return ""
    return str(tasks[0].get("instruction", "")).strip()[:400]


def _first_conversation(paths: Paths, *, failing: bool = False) -> str:
    """Return the first conversation's opening turns, optionally a failing one."""
    for (_variant, _task_id), record in sorted(_agent_records(paths).items()):
        if failing and _reward(record) >= 1.0:
            continue
        trajectory = getattr(record, "trajectory", [])
        if trajectory:
            return _format_turns(trajectory)
    return ""


def _first_failing_conversation(paths: Paths) -> str:
    """Return the first failed run's opening turns."""
    return _first_conversation(paths, failing=True)


def _first_state_text(paths: Paths) -> str:
    """Return the first serialized state's opening text."""
    for path in _whole_state_files(paths):
        try:
            state = StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        return state.text.strip()[:400]
    return ""


def _first_verdict_json(paths: Paths) -> str:
    """Return one verdict's answers rendered as pretty JSON."""
    verdicts, _ = pipeline.load_verdicts(paths.cache_dir / "judge")
    if not verdicts:
        return ""
    answers = [answer.model_dump(exclude_none=True) for answer in verdicts[0].answers]
    return json.dumps(answers, indent=2, sort_keys=True)


def _first_findings(paths: Paths) -> str:
    """Return the first published findings paragraph, if any exists."""
    results = paths.results_dir
    if not results.is_dir():
        return ""
    for path in sorted(results.glob("*_findings.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    summary = results / "g3_summary.md"
    if summary.is_file():
        return summary.read_text(encoding="utf-8").strip()[:400]
    return ""


def _example(real: Callable[[Paths], str], sample: str) -> Callable[[Paths], str]:
    """Return an example callable that prefers a real record and falls back to a sample."""

    def example(paths: Paths) -> str:
        value = real(paths)
        return value if value.strip() else sample

    return example


_SAMPLE_CAPTION = "# example (your own appears here after the step runs)"
_SAMPLE_REQUEST = (
    f"{_SAMPLE_CAPTION}\n"
    "The customer wants to exchange a delivered keyboard for the same model in a "
    "different color, and asks to confirm the price difference before anything is "
    "charged."
)
_SAMPLE_CONVERSATION = (
    f"{_SAMPLE_CAPTION}\n"
    "customer: I want to exchange my keyboard for the black version.\n"
    "agent: Happy to help. Can you confirm your name and zip code?\n"
    'call: find_user_id_by_name_zip(name="Sam Lee", zip="94107")\n'
    'tool result: {"user_id": "sam_lee_8843"}\n'
    "agent: Found it. The black version is the same price, so there is no charge. Confirm?\n"
    "checker: pass"
)
_SAMPLE_READING_COPY = (
    f"{_SAMPLE_CAPTION}\n"
    "request: exchange a delivered keyboard for the same model in another color.\n"
    "policy summary: confirm the details with the customer before changing an order.\n"
    'call: find_user_id_by_name_zip(name="Sam Lee", zip="94107")\n'
    "reward and expected actions: not present"
)
_SAMPLE_VERDICT = f"{_SAMPLE_CAPTION}\n" + json.dumps(
    [
        {"question_id": "completed", "kind": "noul", "noul": 0.9},
        {
            "question_id": "verdict",
            "kind": "choice",
            "choice": "pass",
            "probabilities": {"pass": 0.9, "fail": 0.1},
        },
    ],
    indent=2,
    sort_keys=True,
)
_SAMPLE_FINDING = (
    f"{_SAMPLE_CAPTION}\n"
    "The strong judge agrees with the checker on N out of 115 conversations, and "
    "the cheap judge keeps most of that agreement at a fraction of the cost."
)
_SAMPLE_LABEL = (
    f"{_SAMPLE_CAPTION}\n"
    "why it failed: the agent changed the order without confirming with the "
    "customer first, so the checker marked it fail."
)
_SAMPLE_REPORT = (
    f"{_SAMPLE_CAPTION}\n"
    "README results: one short section per experiment, each with its finding, its "
    "table, and its chart, followed by the limits of the study."
)


# --- step runners -----------------------------------------------------------


def _load_tasks(paths: Paths) -> dict[str, Task]:
    """Load tasks from the exported fixture when present, else from tau-bench."""
    from decision_judges.bench import load as bench_load

    tasks_file = paths.data_dir / "tasks.json"
    if tasks_file.is_file():
        source = json.loads(tasks_file.read_text(encoding="utf-8"))
        loaded = bench_load.load_tasks(source=source)
    else:
        loaded = bench_load.load_tasks()
    return {task.task_id: task for task in loaded}


def _run_agent(paths: Paths, ctx: RunContext) -> str | None:
    """Run every retail task under both policy variants, sequentially.

    Return the first variant's ``stopped_reason`` when one reports it, stopping
    before the second variant because the same setup would fail the same way.
    Return None when both variants finish without stopping early.
    """
    from decision_judges.bench import load as bench_load
    from decision_judges.bench import run_agent as run_agent_mod

    tasks = bench_load.load_tasks()
    spend = Spend(ctx.pricing, ctx.study.spend_caps, paths.results_dir / "spend.json")
    concurrency = ctx.study.concurrency.get("llm", 4)
    with _openrouter_key(ctx.key):
        for variant in _VARIANTS:
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return None
            runner = run_agent_mod.build_tau_runner(ctx.study, variant)  # type: ignore[arg-type]
            summary = run_agent_mod.run_variant(
                variant,  # type: ignore[arg-type]
                tasks,
                runner,
                spend,
                paths.cache_dir / "agent",
                concurrency=concurrency,
                on_progress=ctx.on_progress,
                cancel=ctx.cancel,
            )
            if summary.stopped_reason is not None:
                return summary.stopped_reason
    return None


def _run_serialize(paths: Paths, ctx: RunContext) -> str | None:
    """Serialize every recorded run into the full and compact judge views.

    This step raises on failure or finishes cleanly; it has no early-stop
    reason, so it returns None.
    """
    tasks = _load_tasks(paths)
    profiles = [StateProfile.full, StateProfile.compact]
    variants = _variants_present(paths)
    started = utc_now_iso()
    for index, variant in enumerate(variants):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        records, _ = pipeline.load_agent_records(paths.cache_dir / "agent", variant)
        pipeline.serialize_all(records, tasks, paths.cache_dir / "state", profiles)
        if ctx.on_progress is not None:
            ctx.on_progress(
                Progress(
                    step_id="serialize",
                    done=index + 1,
                    total=len(variants),
                    last_item=f"{variant} · {len(records)} runs",
                    started_at=started,
                )
            )
    return None


def _run_gate(
    paths: Paths,
    ctx: RunContext,
    gate_id: str,
    profile: str,
    variant: str,
    judge_names: list[str],
    repeats: Mapping[str, int] | int | None,
) -> None:
    """Judge one gate over one variant and profile, then analyze and record it."""
    from decision_judges.cache import Cache
    from decision_judges.report import write_findings

    tasks = _load_tasks(paths)
    state_profile = StateProfile(profile)
    gate = pipeline.make_gate(gate_id, tasks)
    records, _ = pipeline.load_agent_records(paths.cache_dir / "agent", variant)
    items = pipeline.items_for_gate(
        gate_id, paths.cache_dir / "state", paths.cache_dir / "agent", state_profile, variant, tasks
    )
    if not items:
        return
    specs = pipeline.judge_specs_from(judge_names, ctx.study)
    judges = pipeline.build_judges(
        specs,
        study=ctx.study,
        tasks=tasks,
        records=records,
        rubric_path=pipeline.gate_rubric_path(gate),
        prompt_version=pipeline.gate_prompt_version(gate),
    )
    cache = Cache(paths.cache_dir / "judge")
    spend = Spend(ctx.pricing, ctx.study.spend_caps, paths.results_dir / "spend.json")
    if repeats is None:
        repeats = pipeline.default_repeats(gate_id, ctx.study, judge_names)
    with _openrouter_key(ctx.key):
        verdicts = pipeline.run_gate(
            gate,
            items,
            judges,
            cache,
            spend,
            repeats,
            on_progress=ctx.on_progress,
            cancel=ctx.cancel,
        )
    analysis_verdicts, analysis_items = pipeline.verdicts_for_analysis(
        gate_id,
        paths.cache_dir / "judge",
        verdicts,
        items,
        paths.cache_dir / "state",
        paths.cache_dir / "agent",
        state_profile,
        variant,
    )
    findings = pipeline.analyze_gate(gate, analysis_verdicts, analysis_items, paths.results_dir)
    write_findings(paths.results_dir, gate_id, findings)


def _run_judge_outcome(paths: Paths, ctx: RunContext) -> str | None:
    """Judge G3 outcomes on the full and compact views for both variants.

    Gate runs raise or finish; this step reports no early-stop reason and
    returns None.
    """
    for variant in _variants_present(paths):
        for profile in ("full", "compact"):
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return None
            _run_gate(paths, ctx, "g3", profile, variant, list(_G3_JUDGES), _G3_REPEATS)
    return None


def _run_analyze(paths: Paths, ctx: RunContext) -> str | None:
    """Reduce cached verdicts into the cascade, calibration, and regression findings.

    A read-and-write reduction with no early-stop reason; returns None.
    """
    from decision_judges.report import write_findings

    registry = pipeline.analysis_registry(ctx.study, ctx.pricing)
    variants = _variants_present(paths)
    profile = StateProfile.full
    started = utc_now_iso()
    gate_ids = ("g5", "g6", "g8")
    for index, gate_id in enumerate(gate_ids):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        items = pipeline.items_for_variants(
            paths.cache_dir / "state", paths.cache_dir / "agent", profile, variants
        )
        all_verdicts, _ = pipeline.load_verdicts(paths.cache_dir / "judge")
        verdicts = pipeline.filter_verdicts_to_items(all_verdicts, items)
        if verdicts:
            findings = pipeline.analyze_gate(registry[gate_id], verdicts, items, paths.results_dir)
            write_findings(paths.results_dir, gate_id, findings)
        if ctx.on_progress is not None:
            ctx.on_progress(
                Progress(
                    step_id="analyze",
                    done=index + 1,
                    total=len(gate_ids),
                    last_item=gate_id,
                    started_at=started,
                )
            )
    return None


def _run_laya(paths: Paths, ctx: RunContext) -> str | None:
    """Judge Laya zero-shot, fine-tune it across folds, and score the local model.

    Cross-validation returns fold manifests and reports no early-stop reason, so
    this step returns None.
    """
    from decision_judges.cli import _resolve_base
    from decision_judges.gates.g3_outcome import G3Outcome
    from decision_judges.training import finetune_laya as training

    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        _run_gate(paths, ctx, "g3", "compact", variant, ["laya_base"], 1)

    variant = "baseline"
    states = pipeline.read_states(paths.cache_dir / "state", variant, StateProfile.compact)
    records, _ = pipeline.load_agent_records(paths.cache_dir / "agent", variant)
    rewards = {task_id: record.reward for task_id, record in records.items()}
    training.run_cross_validation(
        _resolve_base(None, ctx.study),
        states,
        rewards,
        G3Outcome().questions(),
        paths.repo_root / "models",
        k=5,
        seed=ctx.study.seed,
        epochs=2,
        learning_rate=2e-5,
        batch_size=8,
        device="cpu",
        on_progress=ctx.on_progress,
        cancel=ctx.cancel,
    )

    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        _run_gate(paths, ctx, "g3", "compact", variant, ["laya_ft"], 1)
    _run_gate(paths, ctx, "g10", "compact", "baseline", ["laya_base", "laya_ft"], 1)
    return None


def _run_gates(paths: Paths, ctx: RunContext) -> str | None:
    """Probe the judges with per-step, robustness, decomposition, and triage gates.

    Gate runs raise or finish; this step reports no early-stop reason and
    returns None.
    """
    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        _run_gate(paths, ctx, "g2", "full", variant, list(_G3_JUDGES), None)
        _run_gate(paths, ctx, "g7", "full", variant, ["llm_strong"], 5)
        _run_gate(paths, ctx, "g4", "full", variant, ["llm_strong"], 1)
        _run_gate(paths, ctx, "g1", "full", variant, ["llm_strong"], 1)
    return None


def _run_label(paths: Paths, ctx: RunContext) -> str | None:
    """Score the taxonomy judge against hand labels, when any labels exist.

    Gate runs raise or finish; this step reports no early-stop reason and
    returns None.
    """
    from decision_judges.labels import LabelStore

    store = LabelStore.under(paths.data_dir)
    if not store.latest():
        return None
    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return None
        _run_gate(paths, ctx, "g9", "full", variant, ["llm_strong"], 1)
    return None


def _run_results(paths: Paths, ctx: RunContext) -> str | None:
    """Assemble every finding into the results summary and refresh the README.

    A pure assembly step with no early-stop reason; returns None.
    """
    from decision_judges.report import render_results, update_readme

    body = render_results(paths.results_dir, paths.cache_dir)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "summary.md").write_text(body, encoding="utf-8")
    readme = paths.repo_root / "README.md"
    if readme.is_file():
        update_readme(readme, body)
    if ctx.on_progress is not None:
        ctx.on_progress(Progress(step_id="results", done=1, total=1, started_at=utc_now_iso()))
    return None


# --- the study steps --------------------------------------------------------

STEPS: tuple[RunStep, ...] = (
    RunStep(
        id="run-agent",
        title="The conversations",
        purpose=(
            "An AI support agent handled 115 scripted customer problems in a fake store, "
            "twice: once following all the rules, once with one rule removed. A program graded "
            "each conversation pass or fail from the store database. These transcripts and "
            "grades are the material every judge is tested on."
        ),
        pipe="run-agent",
        input_station="tasks",
        output_station="conversations",
        stages=("agent",),
        paid=True,
        material=True,
        learn=(
            "Two language models talk to each other. One plays the store's support agent; the "
            "other plays a customer with a scripted request, one of 115 from a public "
            "benchmark. The agent can look up an order and exchange items by calling tools "
            "against a private copy of the store's database. When the talk ends, a checker "
            "compares the database to the expected result and marks the conversation pass or "
            "fail. Every conversation is a live call to a paid model, so this step costs money. "
            "The same requests run twice: once with the rule 'confirm with the customer before "
            "changing an order', and once with that rule removed, so later steps have careful "
            "and careless runs to tell apart. Nothing is judged yet; this step only produces "
            "the conversations."
        ),
        unlock=_always_ready,
        status=_status_from(_always_ready, _agent_counts),
        example_input=_example(_first_task_instruction, _SAMPLE_REQUEST),
        example_output=_example(_first_conversation, _SAMPLE_CONVERSATION),
        run=lambda paths, ctx: _run_agent(paths, ctx),
    ),
    RunStep(
        id="serialize",
        title="Prepare the reading copy",
        purpose="Make the text version each judge reads, with the answer removed.",
        pipe="serialize",
        input_station="conversations",
        output_station="judge_text",
        stages=(),
        paid=False,
        learn=(
            "A judge never reads the raw conversation. It reads a text copy with the checker's "
            "pass or fail result taken out, so it has to decide from what the agent did, not "
            "from the answer. Two copies are written: a full one with every turn, and a short "
            "one that trims the tool noise so the small local model can fit it. Taking the "
            "answer out is the whole point: if a judge could see the result, its verdict would "
            "mean nothing."
        ),
        unlock=_needs_conversations,
        status=_status_from(_needs_conversations, _serialize_counts),
        example_input=_example(_first_conversation, _SAMPLE_CONVERSATION),
        example_output=_example(_first_state_text, _SAMPLE_READING_COPY),
        run=lambda paths, ctx: _run_serialize(paths, ctx),
    ),
    RunStep(
        id="judge-outcome",
        title="Ask the judges",
        purpose=(
            "Every judge answers two questions about every conversation, five times, on the "
            "full text and on a short version."
        ),
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g3",),
        paid=True,
        learn=(
            "Each judge reads the text copy and answers two questions: did the agent finish the "
            "customer's request, and how good was the outcome. Running every judge on both the "
            "full and the short copy, five times each, lets us measure three things. Accuracy "
            "is how often a judge agrees with the checker. Agreement beyond chance discounts "
            "the agreement you would get by guessing. Agreement across the five repeats shows "
            "how steady a judge is when asked again. These answers are what every later step "
            "reuses."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _verdict_counts),
        example_input=_example(_first_state_text, _SAMPLE_READING_COPY),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_judge_outcome(paths, ctx),
    ),
    RunStep(
        id="analyze",
        title="Draw conclusions",
        purpose=(
            "From those answers: when to trust the cheap judge, whether confidence means what "
            "it says, and whether judges notice a real drop in quality."
        ),
        pipe="analyze",
        input_station="verdicts",
        output_station="findings",
        stages=("g5", "g6", "g8"),
        paid=False,
        learn=(
            "This step turns the answers into written results, with no new judging. The first "
            "result asks how much accuracy you keep if a cheap judge answers when it is sure "
            "and an expensive judge answers only when the cheap one is unsure, trading cost for "
            "accuracy. The second checks whether a judge's confidence means what it says: a "
            "judge that says it is ninety percent sure should be right about nine times in ten. "
            "The third asks whether the judges notice the drop in quality between the careful "
            "runs and the careless ones."
        ),
        unlock=_needs_verdicts,
        status=_status_from(_needs_verdicts, _analyze_counts),
        example_input=_example(_first_verdict_json, _SAMPLE_VERDICT),
        example_output=_example(_first_findings, _SAMPLE_FINDING),
        run=lambda paths, ctx: _run_analyze(paths, ctx),
    ),
    RunStep(
        id="laya",
        title="Teach the local model",
        purpose=(
            "Download Laya, judge with it as published, train it on these conversations "
            "without letting it see its own test items, and compare."
        ),
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g10",),
        paid=False,
        learn=(
            "Laya is a small model you can run on your own machine for free. It picks from "
            "fixed answers and reports how sure it is, instead of writing sentences. This step "
            "downloads Laya, judges with it as published, then trains it on these "
            "conversations. To keep the test fair, the conversations are split into groups, and "
            "each group is judged by a copy of Laya that never trained on it, so no "
            "conversation grades a model that already saw it. The trained Laya is then compared "
            "against the untrained one to show what the training bought."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _laya_counts),
        example_input=_example(_first_state_text, _SAMPLE_READING_COPY),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_laya(paths, ctx),
    ),
    RunStep(
        id="gates",
        title="Test the judges harder",
        purpose=(
            "Score every single action; plant a sentence aimed at the judge; try small "
            "questions against one big one; guess difficulty before running."
        ),
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g2", "g7", "g4", "g1"),
        paid=True,
        learn=(
            "These four checks come at the judges from angles the pass or fail question misses. "
            "The first scores every single action the agent took, not just the final result. "
            "The second plants a sentence written to fool the judge and sees whether it changes "
            "its answer. The third compares asking one broad question against asking several "
            "small ones and adding them up. The fourth guesses how hard each request is before "
            "any conversation runs, as a baseline."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _gates_counts),
        example_input=_example(_first_state_text, _SAMPLE_READING_COPY),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_gates(paths, ctx),
    ),
    RunStep(
        id="label",
        title="Be the judge yourself",
        purpose="Label why 50 conversations failed; then see how well the judges agree with you.",
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g9",),
        paid=True,
        learn=(
            "Here you are the answer key. Some failures are subtle, and this check needs human "
            "labels to score against. Open the labeling page, read a failed conversation, and "
            "pick the failure type you see. Once your labels exist, this step asks the judges "
            "to pick the same failure type and scores how well they agree with you. Without "
            "your labels there is nothing to grade the judges on."
        ),
        unlock=_needs_failures,
        status=_status_from(_needs_failures, _label_counts),
        example_input=_example(_first_failing_conversation, _SAMPLE_CONVERSATION),
        example_output=_example(_first_findings, _SAMPLE_LABEL),
        run=lambda paths, ctx: _run_label(paths, ctx),
    ),
    RunStep(
        id="results",
        title="Write it up",
        purpose="Put every finding, table, and chart into the README.",
        pipe=None,
        input_station="verdicts",
        output_station="findings",
        stages=(),
        paid=False,
        learn=(
            "This step gathers every written result into one report and refreshes the README "
            "between its result markers. It reads the tables, charts, and findings the earlier "
            "steps wrote and lays them out in order, followed by the limits of the study. "
            "Nothing is measured again here; this is the printing press, not the study."
        ),
        unlock=_needs_verdicts,
        status=_status_from(_needs_verdicts, _results_counts),
        example_input=_example(_first_findings, _SAMPLE_FINDING),
        example_output=_example(_first_findings, _SAMPLE_REPORT),
        run=lambda paths, ctx: _run_results(paths, ctx),
    ),
)


# --- material panel helpers -------------------------------------------------

MaterialStatus = Literal["shipped", "regenerating", "missing"]


class VariantCounts(BaseModel):
    """One variant's conversation count and pass rate, read from disk."""

    variant: str
    conversations: int
    pass_rate: float


class Provenance(BaseModel):
    """The models and tau-bench commit the shipped conversations came from."""

    agent_model: str
    user_model: str
    tau_bench_ref: str


def display_index(step: RunStep) -> int | None:
    """Return a step's card number, or None for the material panel.

    The material panel is unnumbered; the remaining steps are numbered 1..7 in
    their declared order, independent of the step ids the progress files use.
    """
    if step.material:
        return None
    number = 0
    for candidate in STEPS:
        if candidate.material:
            continue
        number += 1
        if candidate.id == step.id:
            return number
    return None


def material_status(paths: Paths, *, regenerating: bool = False) -> MaterialStatus:
    """Return whether the shipped conversations are present, regenerating, or missing."""
    if regenerating:
        return "regenerating"
    if _agent_records(paths):
        return "shipped"
    return "missing"


def variant_counts(paths: Paths) -> list[VariantCounts]:
    """Return each present variant's conversation count and pass rate, in order."""
    counts: list[VariantCounts] = []
    for variant in _VARIANTS:
        loaded, _ = pipeline.load_agent_records(paths.cache_dir / "agent", variant)
        rewards = [_reward(record) for record in loaded.values()]
        if not rewards:
            continue
        counts.append(
            VariantCounts(
                variant=variant,
                conversations=len(rewards),
                pass_rate=sum(rewards) / len(rewards),
            )
        )
    return counts


def provenance(paths: Paths) -> Provenance | None:
    """Return the agent model, customer model, and tau-bench commit from the records."""
    for _key, record in sorted(_agent_records(paths).items()):
        return Provenance(
            agent_model=str(getattr(record, "agent_model", "")),
            user_model=str(getattr(record, "user_model", "")),
            tau_bench_ref=str(getattr(record, "tau_bench_ref", "")),
        )
    return None
