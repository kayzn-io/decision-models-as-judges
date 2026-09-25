"""Tests for the G7 robustness gate over injected evaluator-directed text."""

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g7_robustness import G7Robustness, injected_items
from decision_judges.serialize import Injection, StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Usage, Verdict

_INJECTIONS = (
    Injection.none,
    Injection.final_message,
    Injection.tool_result,
    Injection.control,
)
_PLACEMENTS = ("final_message", "tool_result")


# --- builders --------------------------------------------------------------


def _state(
    state_hash: str,
    *,
    injection: Injection = Injection.none,
    variant: str = "baseline",
    task_id: str = "retail-0",
) -> StateRecord:
    """Build a minimal serialized state carrying a variant, task, and injection."""
    return StateRecord(
        variant=variant,
        task_id=task_id,
        profile=StateProfile.full,
        injection=injection,
        text="user: help\ntool: looked things up\nassistant: sorry, cannot help",
        token_estimate=100,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str, **kwargs: object) -> Item:
    """Build an item pairing a state with its ground-truth label."""
    return Item(
        state=_state(state_hash, **kwargs),  # type: ignore[arg-type]
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _record(task_id: str, reward: float) -> AgentRecord:
    """Build an agent record with a tiny valid trajectory."""
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=[
            {"role": "user", "content": "please help"},
            {"role": "assistant", "content": "done"},
        ],
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
    )


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, instruction="do the thing", actions=[], outputs=[])


def _verdict(
    judge_id: str,
    state_hash: str,
    choice: str | None,
    *,
    error: str | None = None,
    repeat: int = 0,
) -> Verdict:
    """Build a choice verdict, or an error verdict when choice is None."""
    answers: list[Answer] = []
    if error is None and choice is not None:
        passed = choice == "pass"
        answers = [
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice=choice,
                probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
                confidence=1.0,
            )
        ]
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=answers,
        usage=Usage(),
        latency_ms=1,
        error=error,
    )


def _trajectory_items(task_ids: list[str]) -> list[Item]:
    """Build one item per injection for each trajectory, all truth fail."""
    items: list[Item] = []
    for task_id in task_ids:
        for injection in _INJECTIONS:
            items.append(
                _item(f"{task_id}-{injection.value}", "fail", injection=injection, task_id=task_id)
            )
    return items


def _scripted_verdicts(
    judge_id: str, task_ids: list[str], choice_by_injection: dict[Injection, str]
) -> list[Verdict]:
    """Build one verdict per injection copy per trajectory for a judge."""
    verdicts: list[Verdict] = []
    for task_id in task_ids:
        for injection, choice in choice_by_injection.items():
            verdicts.append(_verdict(judge_id, f"{task_id}-{injection.value}", choice))
    return verdicts


# --- injected_items --------------------------------------------------------


def test_injected_items_yields_three_for_fail_none_for_pass() -> None:
    out = injected_items([_item("hfail", "fail"), _item("hpass", "pass")])

    assert len(out) == 3
    assert all(item.truth_label == "fail" for item in out)
    assert {item.state.injection for item in out} == {
        Injection.final_message,
        Injection.tool_result,
        Injection.control,
    }


def test_injected_items_skips_already_injected() -> None:
    already = _item("h", "fail", injection=Injection.final_message)

    assert injected_items([already]) == []


def test_injected_control_length_matches_final_message() -> None:
    by_kind = {item.state.injection: item for item in injected_items([_item("h", "fail")])}

    assert len(by_kind[Injection.control].state.text) == len(
        by_kind[Injection.final_message].state.text
    )


# --- build_items -----------------------------------------------------------


def test_build_items_two_records_yields_exactly_three() -> None:
    records = {"retail-0": _record("retail-0", 1.0), "retail-1": _record("retail-1", 0.0)}
    tasks = {task_id: _task(task_id) for task_id in records}

    items = G7Robustness().build_items(records, tasks, StateProfile.compact)

    assert len(items) == 3
    assert all(item.truth_label == "fail" for item in items)
    assert {item.state.injection for item in items} == {
        Injection.final_message,
        Injection.tool_result,
        Injection.control,
    }


# --- delegation ------------------------------------------------------------


def test_delegates_questions_prompt_and_rubric_to_g3() -> None:
    gate = G7Robustness()
    g3 = G3Outcome()

    assert [q.id for q in gate.questions()] == [q.id for q in g3.questions()]
    assert gate.prompt_version == g3.prompt_version
    assert gate.rubric_path == g3.rubric_path
    assert gate.gate_id == "g7"
    assert gate.stage == "g7"


# --- analyze ---------------------------------------------------------------


def test_analyze_robust_judge_has_zero_flip_and_net() -> None:
    task_ids = ["t0", "t1"]
    items = _trajectory_items(task_ids)
    verdicts = _scripted_verdicts("robust", task_ids, dict.fromkeys(_INJECTIONS, "fail"))

    result = G7Robustness().analyze(verdicts, items)

    assert isinstance(result, GateResult)
    summary = result.tables["g7_summary"]
    for placement in _PLACEMENTS:
        row = summary[(summary["judge"] == "robust") & (summary["placement"] == placement)].iloc[0]
        assert row["n"] == 2
        assert row["flip_rate"] == pytest.approx(0.0)
        assert row["net_flip_rate"] == pytest.approx(0.0)
    assert len(result.tables["g7_flips"]) == 0
    assert isinstance(result.charts["g7_flip_rates"], Figure)


def test_analyze_gullible_judge_flips_placements_but_not_control() -> None:
    task_ids = ["t0", "t1"]
    items = _trajectory_items(task_ids)
    choice_by_injection = {
        Injection.none: "fail",
        Injection.final_message: "pass",
        Injection.tool_result: "pass",
        Injection.control: "fail",
    }
    verdicts = _scripted_verdicts("gullible", task_ids, choice_by_injection)

    summary = G7Robustness().analyze(verdicts, items).tables["g7_summary"]

    for placement in _PLACEMENTS:
        row = summary[(summary["judge"] == "gullible") & (summary["placement"] == placement)].iloc[
            0
        ]
        assert row["flip_rate"] == pytest.approx(1.0)
        assert row["control_flip_rate"] == pytest.approx(0.0)
        assert row["net_flip_rate"] == pytest.approx(1.0)


def test_analyze_control_flipper_has_reduced_net() -> None:
    task_ids = ["t0", "t1"]
    items = _trajectory_items(task_ids)
    choice_by_injection = {
        Injection.none: "fail",
        Injection.final_message: "pass",
        Injection.tool_result: "pass",
        Injection.control: "pass",
    }
    verdicts = _scripted_verdicts("naive", task_ids, choice_by_injection)

    summary = G7Robustness().analyze(verdicts, items).tables["g7_summary"]

    row = summary[(summary["judge"] == "naive") & (summary["placement"] == "final_message")].iloc[0]
    assert row["flip_rate"] == pytest.approx(1.0)
    assert row["control_flip_rate"] == pytest.approx(1.0)
    assert row["net_flip_rate"] == pytest.approx(0.0)


def test_analyze_flips_table_lists_flipped_trajectories() -> None:
    task_ids = ["t0"]
    items = _trajectory_items(task_ids)
    choice_by_injection = {
        Injection.none: "fail",
        Injection.final_message: "pass",
        Injection.tool_result: "fail",
        Injection.control: "fail",
    }
    verdicts = _scripted_verdicts("gullible", task_ids, choice_by_injection)

    flips = G7Robustness().analyze(verdicts, items).tables["g7_flips"]

    assert list(flips.columns) == ["judge", "placement", "variant", "task_id"]
    assert len(flips) == 1
    row = flips.iloc[0]
    assert row["judge"] == "gullible"
    assert row["placement"] == "final_message"
    assert row["variant"] == "baseline"
    assert row["task_id"] == "t0"


def test_analyze_excludes_error_verdicts_and_counts_error_rate() -> None:
    items = _trajectory_items(["t0"])
    verdicts = [
        _verdict("judge", "t0-none", "fail"),
        _verdict("judge", "t0-final_message", None, error="boom", repeat=0),
        _verdict("judge", "t0-final_message", "pass", repeat=1),
        _verdict("judge", "t0-tool_result", "fail"),
        _verdict("judge", "t0-control", "fail"),
    ]

    summary = G7Robustness().analyze(verdicts, items).tables["g7_summary"]

    row = summary[(summary["judge"] == "judge") & (summary["placement"] == "final_message")].iloc[0]
    assert row["flip_rate"] == pytest.approx(1.0)
    assert row["error_rate"] == pytest.approx(0.5)


def test_analyze_ignores_trajectories_that_passed_before() -> None:
    items = _trajectory_items(["t0"])
    verdicts = _scripted_verdicts("judge", ["t0"], dict.fromkeys(_INJECTIONS, "pass"))

    summary = G7Robustness().analyze(verdicts, items).tables["g7_summary"]

    row = summary[(summary["judge"] == "judge") & (summary["placement"] == "final_message")].iloc[0]
    assert row["n"] == 0
