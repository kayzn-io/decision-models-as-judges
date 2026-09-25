"""Tests for the G1 task triage gate."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import ExpectedAction, Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g1_triage import G1Triage
from decision_judges.serialize import POLICY_SUMMARY, LeakageError, StateProfile
from decision_judges.types import Answer, QuestionKind, Usage, Verdict

_LEVELS = ["trivial", "routine", "multi_step", "hard"]
_CATEGORIES = [
    "single_lookup",
    "single_write",
    "multiple_writes",
    "cancellation_or_return",
    "ambiguous",
]


def _task(
    task_id: str, *, n_actions: int, n_writes: int = 0, outputs: list[str] | None = None
) -> Task:
    """Build a task with a chosen number of total and write actions."""
    actions: list[ExpectedAction] = []
    for _ in range(n_writes):
        actions.append(ExpectedAction(name="cancel_pending_order", kwargs={}))
    for _ in range(n_actions - n_writes):
        actions.append(ExpectedAction(name="get_order_details", kwargs={}))
    return Task(
        task_id=task_id,
        instruction=f"instruction for {task_id}",
        actions=actions,
        outputs=outputs or [],
    )


def _record(
    task_id: str, reward: float, *, variant: str = "baseline", excluded: bool = False
) -> AgentRecord:
    """Build an agent record for a task under a variant."""
    return AgentRecord(
        variant=variant,
        task_id=task_id,
        trajectory=[{"role": "user", "content": "help"}],
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
        excluded=excluded,
    )


def _difficulty_answer(level_index: float) -> Answer:
    """Build a score answer for the difficulty question."""
    probs = {level: (1.0 if i == int(level_index) else 0.0) for i, level in enumerate(_LEVELS)}
    return Answer(
        question_id="difficulty",
        kind=QuestionKind.score,
        score=float(level_index),
        probabilities=probs,
        confidence=1.0,
    )


def _category_answer(category: str) -> Answer:
    """Build a choice answer for the category question."""
    return Answer(
        question_id="category",
        kind=QuestionKind.choice,
        choice=category,
        probabilities={option: (1.0 if option == category else 0.0) for option in _CATEGORIES},
        confidence=1.0,
    )


def _verdict(state_hash: str, difficulty: float, category: str, *, judge_id: str = "j") -> Verdict:
    """Build a verdict answering both triage questions for a state."""
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=0,
        answers=[_difficulty_answer(difficulty), _category_answer(category)],
        usage=Usage(),
        latency_ms=1,
    )


# --- questions -------------------------------------------------------------


def test_questions_ids_kinds_levels_and_options() -> None:
    questions = G1Triage().questions()
    by_id = {q.id: q for q in questions}

    assert set(by_id) == {"difficulty", "category"}
    assert by_id["difficulty"].kind is QuestionKind.score
    assert by_id["difficulty"].levels == _LEVELS
    assert by_id["difficulty"].text == (
        "How hard is this task for a support agent to complete correctly under the policy?"
    )
    assert by_id["category"].kind is QuestionKind.choice
    assert by_id["category"].options == _CATEGORIES
    assert by_id["category"].text == "Which category best describes the task?"


# --- task_state ------------------------------------------------------------


def test_task_state_carries_instruction_and_policy_without_leakage() -> None:
    task = _task("retail-0", n_actions=2, outputs=["SECRET-OUTPUT-XYZ"])
    state = G1Triage().task_state(task, "baseline")

    assert state.text.startswith("Task: ")
    assert task.instruction in state.text
    assert POLICY_SUMMARY in state.text
    assert "SECRET-OUTPUT-XYZ" not in state.text
    assert state.profile is StateProfile.full
    assert state.variant == "baseline"
    assert state.task_id == "retail-0"
    assert state.token_estimate == len(state.text) // 4


def test_task_state_raises_when_output_would_leak() -> None:
    task = _task("retail-0", n_actions=1, outputs=["instruction for retail-0"])
    with pytest.raises(LeakageError):
        G1Triage().task_state(task, "baseline")


# --- build_items -----------------------------------------------------------


def test_build_items_one_per_task_with_truth_from_reward() -> None:
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    tasks = {
        "retail-0": _task("retail-0", n_actions=1),
        "retail-1": _task("retail-1", n_actions=3),
    }

    items = G1Triage(tasks=tasks).build_items(records, tasks, StateProfile.full)

    by_task = {item.state.task_id: item for item in items}
    assert set(by_task) == {"retail-0", "retail-1"}
    assert by_task["retail-0"].truth_label == "pass"
    assert by_task["retail-0"].truth_value == 1.0
    assert by_task["retail-1"].truth_label == "fail"
    assert by_task["retail-1"].truth_value == 3.0


def test_build_items_skips_excluded() -> None:
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0, excluded=True),
    }
    tasks = {tid: _task(tid, n_actions=1) for tid in records}

    items = G1Triage(tasks=tasks).build_items(records, tasks, StateProfile.full)

    assert [item.state.task_id for item in items] == ["retail-0"]


# --- analyze ---------------------------------------------------------------


def _fixture(tasks: dict[str, Task], rewards: dict[str, float]) -> tuple[list[Verdict], list[Item]]:
    """Build items and verdicts where difficulty tracks the expected action count."""
    records = {tid: _record(tid, rewards[tid]) for tid in tasks}
    gate = G1Triage(tasks=tasks)
    items = gate.build_items(records, tasks, StateProfile.full)
    verdicts: list[Verdict] = []
    for item in items:
        # difficulty grows with the expected-action count, clamped to level range.
        difficulty = min(len(_LEVELS) - 1, int(item.truth_value or 0))
        category = _CATEGORIES[0] if (item.truth_value or 0) <= 1 else _CATEGORIES[2]
        verdicts.append(_verdict(item.state.state_hash, float(difficulty), category))
    return verdicts, items


def test_analyze_difficulty_tracks_actions_and_predicts_failure() -> None:
    tasks = {
        "retail-0": _task("retail-0", n_actions=0),
        "retail-1": _task("retail-1", n_actions=1),
        "retail-2": _task("retail-2", n_actions=2, n_writes=1),
        "retail-3": _task("retail-3", n_actions=3, n_writes=2),
    }
    rewards = {"retail-0": 1.0, "retail-1": 1.0, "retail-2": 0.0, "retail-3": 0.0}
    verdicts, items = _fixture(tasks, rewards)

    result = G1Triage(tasks=tasks).analyze(verdicts, items)

    assert isinstance(result, GateResult)
    summary = result.tables["g1_summary"]
    row = summary.iloc[0]
    assert row["n"] == 4
    assert row["spearman_actions"] > 0.5
    assert row["spearman_writes"] > 0.5
    assert 0.0 <= row["auroc_fail"] <= 1.0

    categories = result.tables["g1_categories"]
    assert int(categories["count"].sum()) == 4

    assert isinstance(result.charts["g1_difficulty_vs_actions"], Figure)
    assert result.findings.startswith("This gate is exploratory:")


def test_analyze_without_tasks_reports_outcome_only() -> None:
    tasks = {
        "retail-0": _task("retail-0", n_actions=0),
        "retail-1": _task("retail-1", n_actions=2),
    }
    rewards = {"retail-0": 1.0, "retail-1": 0.0}
    verdicts, items = _fixture(tasks, rewards)

    result = G1Triage(tasks=None).analyze(verdicts, items)

    row = result.tables["g1_summary"].iloc[0]
    assert row["n"] == 2
    # Write-action correlation needs tasks; it is absent without them.
    assert pd_isna(row["spearman_writes"])
    assert result.findings.startswith("This gate is exploratory:")


def pd_isna(value: object) -> bool:
    """Return whether a value is a NaN float."""
    return isinstance(value, float) and value != value


# --- prompt version --------------------------------------------------------


def test_prompt_version_stable_and_rubric_sensitive(tmp_path: Path) -> None:
    assert G1Triage().prompt_version == G1Triage().prompt_version

    modified = tmp_path / "rubric.md"
    modified.write_text("# version: 2\n\nA different rubric body.\n", encoding="utf-8")
    assert G1Triage(rubric_path=modified).prompt_version != G1Triage().prompt_version


def _levels() -> Sequence[str]:
    """Expose the difficulty levels for reuse."""
    return _LEVELS


# --- make_gate -------------------------------------------------------------


def test_make_gate_builds_triage_gate_holding_tasks() -> None:
    from decision_judges import pipeline

    tasks = {"retail-0": _task("retail-0", n_actions=1)}
    gate = pipeline.make_gate("g1", tasks)

    assert isinstance(gate, G1Triage)
    assert gate._tasks == tasks


def test_make_gate_other_gate_ignores_tasks() -> None:
    from decision_judges import pipeline

    gate = pipeline.make_gate("g3", {"retail-0": _task("retail-0", n_actions=1)})

    assert gate.gate_id == "g3"
