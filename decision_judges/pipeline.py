"""Injectable orchestration for the serialize and judge stages.

The command line stays thin: it parses options and calls the functions here.
Every dependency (records, tasks, judges, cache, spend, gate) is passed in, so
the whole flow runs offline in tests with a fake judge and a tasks fixture.
"""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache
from decision_judges.config import PricingTable, StudyConfig
from decision_judges.gates.base import Gate, Item
from decision_judges.gates.g2_steps import (
    G2Steps,
    enumerate_steps,
    repeats_for,
    serialize_step,
    step_items,
)
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g4_decomposition import G4Decomposition
from decision_judges.gates.g5_cascade import G5Cascade
from decision_judges.gates.g6_calibration import G6Calibration
from decision_judges.gates.g8_regression import G8Regression
from decision_judges.gates.g10_local_model import G10LocalModel
from decision_judges.judges.base import HasStateText, Judge, build_verdict, timed
from decision_judges.judges.code import CodeJudge
from decision_judges.judges.llm import LlmJudge, OpenAiClientAdapter
from decision_judges.report import write_chart, write_table
from decision_judges.serialize import StateProfile, StateRecord, serialize
from decision_judges.spend import Spend
from decision_judges.types import Answer, Question, QuestionKind, Verdict

_PASS = "pass"
_FAIL = "fail"
_UNPRICED_MODEL_ID = "none"
_KNOWN_JUDGES = ("code", "llm_cheap", "llm_strong", "jev", "fake")
_QUARANTINE = "_quarantine"


# --- record loading and serialization --------------------------------------


def load_agent_records(cache_dir: Path, variant: str) -> tuple[dict[str, AgentRecord], list[str]]:
    """Load one variant's agent records, skipping invalid files.

    Returns the records keyed by task id and a list of warnings naming files
    that failed to parse or validate.
    """
    records: dict[str, AgentRecord] = {}
    warnings: list[str] = []
    variant_dir = Path(cache_dir) / variant
    if not variant_dir.is_dir():
        return records, warnings
    for path in sorted(variant_dir.glob("*.json")):
        try:
            record = AgentRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        records[record.task_id] = record
    return records, warnings


def _write_state_json(path: Path, state: StateRecord) -> None:
    """Write a state record atomically as stable, sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _write_step_states(
    record: AgentRecord, task: Task, profile: StateProfile, out_dir: Path
) -> None:
    """Serialize each non-malformed step and write it under the ``steps`` subdirectory.

    Files are named ``<task_id>.<step_index>.json`` with the index zero-padded to
    three digits so they sort in trajectory order and never collide with the
    whole-trajectory file.
    """
    steps_dir = out_dir / record.variant / profile.value / "steps"
    for step in enumerate_steps(record):
        if step.malformed:
            continue
        state = serialize_step(record, task, step, profile)
        path = steps_dir / f"{record.task_id}.{step.step_index:03d}.json"
        _write_state_json(path, state)


def serialize_all(
    records: Mapping[str, AgentRecord],
    tasks: Mapping[str, Task],
    out_dir: Path,
    profiles: Sequence[StateProfile],
) -> int:
    """Serialize each non-excluded record under every profile and return the count.

    Whole-trajectory states are written to
    ``<out_dir>/<variant>/<profile>/<task_id>.json`` and one per-step state per
    tool call to ``<out_dir>/<variant>/<profile>/steps/<task_id>.<step_index>.json``
    (index zero-padded to three digits). The returned count is the number of
    whole-trajectory states. Serialization is deterministic, so re-running
    overwrites identical content.
    """
    out_dir = Path(out_dir)
    count = 0
    for record in records.values():
        if record.excluded:
            continue
        task = tasks[record.task_id]
        for profile in profiles:
            state = serialize(record, task, profile)
            path = out_dir / record.variant / profile.value / f"{record.task_id}.json"
            _write_state_json(path, state)
            count += 1
            _write_step_states(record, task, profile, out_dir)
    return count


def items_from_states(
    states: Sequence[StateRecord], records: Mapping[str, AgentRecord]
) -> list[Item]:
    """Pair each serialized state with the truth from its agent record.

    Truth is a pass label at reward >= 1.0 else fail, plus the raw reward.
    States without a matching record are skipped. Gates own serialization via
    ``build_items``; the CLI path prefers these pre-serialized states.
    """
    items: list[Item] = []
    for state in states:
        record = records.get(state.task_id)
        if record is None:
            continue
        label = _PASS if record.reward >= 1.0 else _FAIL
        items.append(Item(state=state, truth_label=label, truth_value=record.reward))
    return items


# --- judges -----------------------------------------------------------------


class FakeJudge:
    """A deterministic test-double judge that answers every question trivially.

    Choice questions take the first option, score questions the midpoint level,
    and noul questions 0.5. It reports the unpriced model id and zero usage so
    the serialize-and-judge CLI path runs end to end offline without a paid
    backend. It exists only for tests and local smoke runs.
    """

    model_id = _UNPRICED_MODEL_ID

    def __init__(self, *, judge_id: str = "fake", prompt_version: str = "fake-1") -> None:
        self.judge_id = judge_id
        self.prompt_version = prompt_version

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Answer every question with its trivial default and build a verdict."""
        with timed() as elapsed:
            answers = [self._answer(question) for question in questions]
            latency_ms = elapsed()
        return build_verdict(self, state, repeat, answers, latency_ms=latency_ms)

    def _answer(self, question: Question) -> Answer:
        """Return the trivial default answer for one question."""
        if question.kind is QuestionKind.choice:
            options = list(question.options or [])
            first = options[0]
            return Answer(
                question_id=question.id,
                kind=QuestionKind.choice,
                choice=first,
                probabilities={option: (1.0 if option == first else 0.0) for option in options},
                confidence=1.0,
            )
        if question.kind is QuestionKind.score:
            levels = list(question.levels or [])
            index = len(levels) // 2
            return Answer(
                question_id=question.id,
                kind=QuestionKind.score,
                score=float(index),
                probabilities={
                    level: (1.0 if position == index else 0.0)
                    for position, level in enumerate(levels)
                },
                confidence=1.0,
            )
        return Answer(question_id=question.id, kind=QuestionKind.noul, noul=0.5)


class JudgeSpec(BaseModel):
    """A resolved judge: its display name, kind, and model id."""

    name: str
    kind: Literal["code", "llm", "jev", "fake"]
    model_id: str


def judge_specs_from(names: Sequence[str], study: StudyConfig) -> list[JudgeSpec]:
    """Resolve judge names to specs, filling model ids from the study config."""
    specs: list[JudgeSpec] = []
    for name in names:
        if name == "code":
            specs.append(JudgeSpec(name="code", kind="code", model_id=_UNPRICED_MODEL_ID))
        elif name == "llm_cheap":
            specs.append(JudgeSpec(name="llm_cheap", kind="llm", model_id=study.models.llm_cheap))
        elif name == "llm_strong":
            specs.append(JudgeSpec(name="llm_strong", kind="llm", model_id=study.models.llm_strong))
        elif name == "jev":
            specs.append(JudgeSpec(name="jev", kind="jev", model_id=study.models.jev))
        elif name == "fake":
            specs.append(JudgeSpec(name="fake", kind="fake", model_id=_UNPRICED_MODEL_ID))
        else:
            raise ValueError(f"unknown judge {name!r}; known: {', '.join(_KNOWN_JUDGES)}")
    return specs


def build_judges(
    specs: Sequence[JudgeSpec],
    *,
    study: StudyConfig,
    tasks: Mapping[str, Task],
    records: Mapping[str, AgentRecord],
    rubric_path: Path,
    prompt_version: str,
) -> list[Judge]:
    """Construct judges from specs, importing the decision SDK lazily for jev."""
    judges: list[Judge] = []
    for spec in specs:
        if spec.kind == "code":
            judges.append(CodeJudge(tasks, records))
        elif spec.kind == "llm":
            judges.append(
                LlmJudge(
                    spec.name,
                    spec.model_id,
                    rubric_path,
                    OpenAiClientAdapter(study.llm_base_url, "OPENROUTER_API_KEY"),
                )
            )
        elif spec.kind == "jev":
            from typesafe_sdk import TypeSafeClient

            from decision_judges.judges.jev import JevJudge

            judges.append(
                JevJudge(
                    spec.model_id,
                    TypeSafeClient(),  # type: ignore[arg-type]
                    judge_id=spec.name,
                    prompt_version=prompt_version,
                )
            )
        elif spec.kind == "fake":
            judges.append(FakeJudge(judge_id=spec.name, prompt_version=prompt_version))
        else:
            raise ValueError(f"unknown judge kind {spec.kind!r}; known: code, llm, jev, fake")
    return judges


# --- repeats ----------------------------------------------------------------


def parse_repeats(values: Sequence[str], *, default: int = 1) -> Mapping[str, int] | int:
    """Parse repeat flags into a shared count or a per-judge mapping.

    An empty input yields the default. A single bare integer yields that shared
    count. One or more 'name=k' pairs yield a mapping. Mixing a bare count with
    pairs is rejected.
    """
    if not values:
        return default
    pairs = [value for value in values if "=" in value]
    if pairs:
        if len(pairs) != len(values):
            raise ValueError("mix of a shared count and per-judge repeats is not allowed")
        mapping: dict[str, int] = {}
        for value in values:
            name, _, count = value.partition("=")
            mapping[name.strip()] = int(count)
        return mapping
    if len(values) != 1:
        raise ValueError("provide one shared count or per-judge 'name=k' pairs")
    return int(values[0])


# --- gate execution and analysis --------------------------------------------


def gate_registry() -> dict[str, type[Gate]]:
    """Return the gates the judge command can run, keyed by gate id.

    These gates make their own judge calls and have zero-argument constructors,
    so the CLI can build them without study or pricing context.
    """
    return {"g3": G3Outcome, "g4": G4Decomposition, "g10": G10LocalModel, "g2": G2Steps}


def default_repeats(
    gate_id: str, study: StudyConfig, names: Sequence[str]
) -> Mapping[str, int] | int:
    """Return the default repeat plan for a gate when none is given on the CLI.

    G2 reads its per-judge counts from the study; every other gate defaults to a
    single repeat.
    """
    if gate_id == "g2":
        return repeats_for(study, names)
    return 1


def analysis_registry(study: StudyConfig, pricing: PricingTable) -> dict[str, Gate]:
    """Return analysis-only gates, constructed from the study and pricing.

    These gates reuse cached G3 verdicts rather than judging, so they are built
    as ready instances configured from the study seed and thresholds.
    """
    return {
        "g5": G5Cascade(pricing, study.thresholds.cascade),
        "g6": G6Calibration(seed=study.seed),
        "g8": G8Regression(seed=study.seed),
    }


def read_states(state_dir: Path, variant: str, profile: StateProfile) -> dict[str, StateRecord]:
    """Read serialized states for a variant and profile, keyed by task id.

    A missing directory yields an empty mapping so callers can decide how to
    report the absence.
    """
    directory = Path(state_dir) / variant / profile.value
    states: dict[str, StateRecord] = {}
    if not directory.is_dir():
        return states
    for path in sorted(directory.glob("*.json")):
        record = StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
        states[record.task_id] = record
    return states


def read_step_states(
    state_dir: Path, variant: str, profile: StateProfile
) -> dict[tuple[str, int], StateRecord]:
    """Read per-step states for a variant and profile, keyed by task id and step index.

    Step states live under the ``steps`` subdirectory the serialize stage writes.
    A missing directory yields an empty mapping.
    """
    directory = Path(state_dir) / variant / profile.value / "steps"
    states: dict[tuple[str, int], StateRecord] = {}
    if not directory.is_dir():
        return states
    for path in sorted(directory.glob("*.json")):
        record = StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.step_index is None:
            continue
        states[(record.task_id, record.step_index)] = record
    return states


def items_for_gate(
    gate_id: str,
    state_dir: Path,
    agent_dir: Path,
    profile: StateProfile,
    variant: str,
    tasks: Mapping[str, Task],
) -> list[Item]:
    """Build the items a gate judges for one variant and profile.

    The G2 step gate judges per-step states labeled by the task's expected
    actions; every other gate judges whole-trajectory states labeled by the
    run's reward.
    """
    records, _ = load_agent_records(agent_dir, variant)
    if gate_id == "g2":
        return step_items(read_step_states(state_dir, variant, profile), records, tasks)
    states = read_states(state_dir, variant, profile)
    return items_from_states(list(states.values()), records)


def items_for_variants(
    state_dir: Path,
    agent_dir: Path,
    profile: StateProfile,
    variants: Sequence[str],
) -> list[Item]:
    """Build items across variants by pairing each variant's states with its records."""
    items: list[Item] = []
    for variant in variants:
        records, _ = load_agent_records(agent_dir, variant)
        states = read_states(state_dir, variant, profile)
        items.extend(items_from_states(list(states.values()), records))
    return items


def load_verdicts(
    cache_dir: Path, *, judge_ids: set[str] | None = None
) -> tuple[list[Verdict], list[str]]:
    """Load every cached verdict under a directory, skipping quarantine and bad files.

    Files under the quarantine directory are ignored. Files that fail to parse
    or validate are skipped and named in the returned warnings. When
    ``judge_ids`` is given, only verdicts from those judges are kept.
    """
    cache_dir = Path(cache_dir)
    verdicts: list[Verdict] = []
    warnings: list[str] = []
    if not cache_dir.is_dir():
        return verdicts, warnings
    for path in sorted(cache_dir.rglob("*.json")):
        if _QUARANTINE in path.parts:
            continue
        try:
            verdict = Verdict.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        if judge_ids is not None and verdict.judge_id not in judge_ids:
            continue
        verdicts.append(verdict)
    return verdicts, warnings


def filter_verdicts_to_items(verdicts: Sequence[Verdict], items: Sequence[Item]) -> list[Verdict]:
    """Keep verdicts whose state hash matches one of the items' states."""
    hashes = {item.state.state_hash for item in items}
    return [verdict for verdict in verdicts if verdict.state_hash in hashes]


def gate_rubric_path(gate: Gate) -> Path:
    """Return the gate's rubric path."""
    return gate.rubric_path


def gate_prompt_version(gate: Gate) -> str:
    """Return the gate's prompt version."""
    return gate.prompt_version


def run_gate(
    gate: Gate,
    items: Sequence[Item],
    judges: Sequence[Judge],
    cache: Cache,
    spend: Spend,
    repeats: Mapping[str, int] | int,
) -> list[Verdict]:
    """Judge every item, keeping the CLI free of gate internals."""
    return gate.run(items, judges, cache, spend, repeats=repeats)


def analyze_gate(
    gate: Gate, verdicts: Sequence[Verdict], items: Sequence[Item], results_dir: Path
) -> str:
    """Analyze verdicts, write each table and chart, and return the findings."""
    result = gate.analyze(verdicts, items)
    for name, table in result.tables.items():
        write_table(results_dir, name, cast(pd.DataFrame, table))
    for name, figure in result.charts.items():
        write_chart(results_dir, name, cast(Figure, figure))
    return result.findings


def list_result_tables(results_dir: Path) -> list[str]:
    """Return the names of Markdown result tables, excluding the summary."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(path.stem for path in results_dir.glob("*.md") if path.stem != "summary")
