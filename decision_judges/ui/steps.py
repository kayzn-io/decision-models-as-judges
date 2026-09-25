"""The eight steps of the study, each with its own unlock, status, and runner.

A :class:`RunStep` is a frozen description of one stage: what it reads, what it
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
    run: Callable[[Paths, RunContext], object]

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
    """Set the OpenRouter key for the block, only when one is not already present.

    A key already in the environment wins and is left untouched; a supplied key
    is installed for the duration and removed afterwards so it never outlives the
    step that used it.
    """
    if not key or _OPENROUTER_ENV in os.environ:
        yield
        return
    os.environ[_OPENROUTER_ENV] = key
    try:
        yield
    finally:
        os.environ.pop(_OPENROUTER_ENV, None)


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
    """Require serialized states before judging."""
    if _whole_state_files(paths):
        return None
    return "Needs serialized text: run step 2 first."


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
    return _state_for(done, total, f"{done} of {total} states written")


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


_SAMPLE_TASK = (
    "Sample task: the customer asks to exchange one delivered item and change the "
    "shipping address on a pending order, in a single conversation."
)
_SAMPLE_CONVERSATION = (
    "system: Retail support policy applies.\n"
    "user: I need to exchange my keyboard and update my address.\n"
    "assistant: calls: find_user_id_by_name_zip\n"
    'tool: {"user_id": "sample_user"}'
)
_SAMPLE_STATE = (
    "Sample state: a plain-text rendering of the run with the benchmark's reward "
    "removed, so the judge decides from the agent's behavior alone."
)
_SAMPLE_VERDICT = json.dumps(
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
_SAMPLE_FINDINGS = (
    "Sample finding: the strong LLM and the decision model agree with the "
    "benchmark on most runs, and the cascade keeps that accuracy at lower cost."
)
_SAMPLE_LABEL = (
    "Sample label: wrong_tool_arguments — the agent called the right tool with the "
    "wrong order id, so the change never applied."
)
_SAMPLE_REPORT = (
    "Sample report section: one block per gate with its findings, tables, and "
    "chart, followed by the threats to validity."
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


def _run_agent(paths: Paths, ctx: RunContext) -> None:
    """Run every retail task under both policy variants, sequentially."""
    from decision_judges.bench import load as bench_load
    from decision_judges.bench import run_agent as run_agent_mod

    tasks = bench_load.load_tasks()
    spend = Spend(ctx.pricing, ctx.study.spend_caps, paths.results_dir / "spend.json")
    concurrency = ctx.study.concurrency.get("llm", 4)
    with _openrouter_key(ctx.key):
        for variant in _VARIANTS:
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return
            runner = run_agent_mod.build_tau_runner(ctx.study, variant)  # type: ignore[arg-type]
            run_agent_mod.run_variant(
                variant,  # type: ignore[arg-type]
                tasks,
                runner,
                spend,
                paths.cache_dir / "agent",
                concurrency=concurrency,
                on_progress=ctx.on_progress,
                cancel=ctx.cancel,
            )


def _run_serialize(paths: Paths, ctx: RunContext) -> None:
    """Serialize every recorded run into the full and compact judge views."""
    tasks = _load_tasks(paths)
    profiles = [StateProfile.full, StateProfile.compact]
    variants = _variants_present(paths)
    started = utc_now_iso()
    for index, variant in enumerate(variants):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return
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


def _run_judge_outcome(paths: Paths, ctx: RunContext) -> None:
    """Judge G3 outcomes on the full and compact views for both variants."""
    for variant in _variants_present(paths):
        for profile in ("full", "compact"):
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return
            _run_gate(paths, ctx, "g3", profile, variant, list(_G3_JUDGES), _G3_REPEATS)


def _run_analyze(paths: Paths, ctx: RunContext) -> None:
    """Reduce cached verdicts into the cascade, calibration, and regression findings."""
    from decision_judges.report import write_findings

    registry = pipeline.analysis_registry(ctx.study, ctx.pricing)
    variants = _variants_present(paths)
    profile = StateProfile.full
    started = utc_now_iso()
    gate_ids = ("g5", "g6", "g8")
    for index, gate_id in enumerate(gate_ids):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return
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


def _run_laya(paths: Paths, ctx: RunContext) -> None:
    """Judge Laya zero-shot, fine-tune it across folds, and score the local model."""
    from decision_judges.cli import _resolve_base
    from decision_judges.gates.g3_outcome import G3Outcome
    from decision_judges.training import finetune_laya as training

    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return
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
            return
        _run_gate(paths, ctx, "g3", "compact", variant, ["laya_ft"], 1)
    _run_gate(paths, ctx, "g10", "compact", "baseline", ["laya_base", "laya_ft"], 1)


def _run_gates(paths: Paths, ctx: RunContext) -> None:
    """Probe the judges with per-step, robustness, decomposition, and triage gates."""
    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return
        _run_gate(paths, ctx, "g2", "full", variant, list(_G3_JUDGES), None)
        _run_gate(paths, ctx, "g7", "full", variant, ["llm_strong"], 5)
        _run_gate(paths, ctx, "g4", "full", variant, ["llm_strong"], 1)
        _run_gate(paths, ctx, "g1", "full", variant, ["llm_strong"], 1)


def _run_label(paths: Paths, ctx: RunContext) -> None:
    """Score the taxonomy judge against hand labels, when any labels exist."""
    from decision_judges.labels import LabelStore

    store = LabelStore.under(paths.data_dir)
    if not store.latest():
        return
    for variant in _variants_present(paths):
        if ctx.cancel is not None and ctx.cancel.is_cancelled:
            return
        _run_gate(paths, ctx, "g9", "full", variant, ["llm_strong"], 1)


def _run_results(paths: Paths, ctx: RunContext) -> None:
    """Assemble every finding into the results summary and refresh the README."""
    from decision_judges.report import render_results, update_readme

    body = render_results(paths.results_dir, paths.cache_dir)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    (paths.results_dir / "summary.md").write_text(body, encoding="utf-8")
    readme = paths.repo_root / "README.md"
    if readme.is_file():
        update_readme(readme, body)
    if ctx.on_progress is not None:
        ctx.on_progress(Progress(step_id="results", done=1, total=1, started_at=utc_now_iso()))


# --- the eight steps --------------------------------------------------------

STEPS: tuple[RunStep, ...] = (
    RunStep(
        id="run-agent",
        title="Run the agent",
        purpose="Run the retail tasks twice: under the careful policy and the careless one.",
        pipe="run-agent",
        input_station="tasks",
        output_station="conversations",
        stages=("agent",),
        paid=True,
        learn=(
            "A trajectory is the full record of one agent solving one task: every message, "
            "tool call, and tool result in order. The same tasks run twice here, once under "
            "the full retail policy and once under a policy with the confirmation rule "
            "removed, so later steps have both careful and careless runs to tell apart. The "
            "benchmark scores each run pass or fail against its own checks, and that score is "
            "the ground truth every judge is measured against. Nothing is judged yet; this "
            "step only produces the runs."
        ),
        unlock=_always_ready,
        status=_status_from(_always_ready, _agent_counts),
        example_input=_example(_first_task_instruction, _SAMPLE_TASK),
        example_output=_example(_first_conversation, _SAMPLE_CONVERSATION),
        run=lambda paths, ctx: _run_agent(paths, ctx),
    ),
    RunStep(
        id="serialize",
        title="Serialize the runs",
        purpose="Turn each run into judge-ready text with the answer key removed.",
        pipe="serialize",
        input_station="conversations",
        output_station="judge_text",
        stages=(),
        paid=False,
        learn=(
            "A judge never sees the raw trajectory. It reads a serialized state: a plain-text "
            "rendering of the run with the benchmark's own answer key removed, so the judge "
            "has to decide from the agent's behavior alone. Two views are written, a full view "
            "with every turn and a compact view that trims tool noise. Withholding the reward "
            "is the point: if the judge could see whether the task passed, its verdict would "
            "mean nothing."
        ),
        unlock=_needs_conversations,
        status=_status_from(_needs_conversations, _serialize_counts),
        example_input=_example(_first_conversation, _SAMPLE_CONVERSATION),
        example_output=_example(_first_state_text, _SAMPLE_STATE),
        run=lambda paths, ctx: _run_serialize(paths, ctx),
    ),
    RunStep(
        id="judge-outcome",
        title="Judge the outcomes",
        purpose="Ask every judge whether each run passed, on both views, five times over.",
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g3",),
        paid=True,
        learn=(
            "Each judge reads a state and answers one question: did the agent complete the "
            "task? Running every judge over both the full and compact views, five times each, "
            "lets us measure three different things. Accuracy is how often the judge agrees "
            "with the ground truth. Cohen's kappa discounts the agreement you would expect by "
            "chance. Agreement across repeats shows how stable a judge is when asked the same "
            "thing again. These raw verdicts are what every later analysis reuses."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _verdict_counts),
        example_input=_example(_first_state_text, _SAMPLE_STATE),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_judge_outcome(paths, ctx),
    ),
    RunStep(
        id="analyze",
        title="Analyze the verdicts",
        purpose="Reduce the verdicts into cascade, calibration, and regression findings.",
        pipe="analyze",
        input_station="verdicts",
        output_station="findings",
        stages=("g5", "g6", "g8"),
        paid=False,
        learn=(
            "This step reduces the verdicts into findings without any new judging. The cascade "
            "asks how much accuracy you keep if a cheap local model answers when it is "
            "confident and a strong model answers otherwise, trading cost for accuracy. "
            "Calibration checks whether a judge's stated confidence matches how often it is "
            "actually right: a judge that says ninety percent should be right about ninety "
            "percent of the time. Regression detection asks whether the judges can spot the "
            "drop in quality between the two policy variants."
        ),
        unlock=_needs_verdicts,
        status=_status_from(_needs_verdicts, _analyze_counts),
        example_input=_example(_first_verdict_json, _SAMPLE_VERDICT),
        example_output=_example(_first_findings, _SAMPLE_FINDINGS),
        run=lambda paths, ctx: _run_analyze(paths, ctx),
    ),
    RunStep(
        id="laya",
        title="Train the local model",
        purpose="Fine-tune the local model on the runs and compare it against its zero-shot self.",
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g10",),
        paid=False,
        learn=(
            "The local decision model learns from the runs themselves. Because the same runs "
            "are used to train and to test, it is trained with cross-validation: the runs are "
            "split into folds, and each fold is judged by a model that never trained on it, so "
            "no run grades a model that saw it. Fitting a calibration temperature on the "
            "held-out fold keeps its confidence honest. The fine-tuned model is compared "
            "against its own zero-shot starting point to show what the training bought."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _laya_counts),
        example_input=_example(_first_state_text, _SAMPLE_STATE),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_laya(paths, ctx),
    ),
    RunStep(
        id="gates",
        title="Probe the judges",
        purpose="Probe single steps, injected text, question shape, and task difficulty.",
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g2", "g7", "g4", "g1"),
        paid=True,
        learn=(
            "These gates probe the judges from angles a single outcome verdict misses. "
            "Per-step scoring asks whether each individual tool call was warranted, not just "
            "whether the whole task passed. Robustness splices evaluator-directed text into a "
            "run to see if a judge can be talked into flipping its verdict. Decomposition "
            "compares asking one broad question against asking several narrow ones. Triage "
            "estimates each task's difficulty before any run, as a baseline."
        ),
        unlock=_needs_states,
        status=_status_from(_needs_states, _gates_counts),
        example_input=_example(_first_state_text, _SAMPLE_STATE),
        example_output=_example(_first_verdict_json, _SAMPLE_VERDICT),
        run=lambda paths, ctx: _run_gates(paths, ctx),
    ),
    RunStep(
        id="label",
        title="Label and classify failures",
        purpose="Label failed runs by hand, then score the taxonomy judge against your labels.",
        pipe="judge",
        input_station="judge_text",
        output_station="verdicts",
        stages=("g9",),
        paid=True,
        learn=(
            "Here you are the ground truth. Some failures are subtle, and the taxonomy gate "
            "needs human labels to measure against. Open the Label page, read a failed run, "
            "and assign the failure type you see. Once labels exist, this step runs the "
            "taxonomy judge and scores its guesses against yours. Without your labels there is "
            "nothing to grade the judge on."
        ),
        unlock=_needs_failures,
        status=_status_from(_needs_failures, _label_counts),
        example_input=_example(_first_failing_conversation, _SAMPLE_CONVERSATION),
        example_output=_example(_first_findings, _SAMPLE_LABEL),
        run=lambda paths, ctx: _run_label(paths, ctx),
    ),
    RunStep(
        id="results",
        title="Publish the results",
        purpose="Assemble every finding into the README results section.",
        pipe=None,
        input_station="verdicts",
        output_station="findings",
        stages=(),
        paid=False,
        learn=(
            "This step assembles every finding into one report and refreshes the README "
            "between its result markers. It reads the tables, charts, and findings each "
            "earlier step wrote and lays them out in gate order, followed by the threats to "
            "validity. Nothing is recomputed: this is the printing press, not the study."
        ),
        unlock=_needs_verdicts,
        status=_status_from(_needs_verdicts, _results_counts),
        example_input=_example(_first_findings, _SAMPLE_FINDINGS),
        example_output=_example(_first_findings, _SAMPLE_REPORT),
        run=lambda paths, ctx: _run_results(paths, ctx),
    ),
)
