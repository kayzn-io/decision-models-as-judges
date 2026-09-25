"""Unit tests for the pure trajectory presentation helpers."""

from pathlib import Path

from decision_judges.bench.load import ExpectedAction, Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.metrics import modal_agreement
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Verdict
from decision_judges.ui import data, views

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"


def _paths() -> data.Paths:
    """Build Paths rooted at the committed UI cache fixture tree."""
    return data.Paths(
        repo_root=_UI_ROOT,
        cache_dir=_UI_ROOT / "cache",
        results_dir=_UI_ROOT / "results",
        config_dir=_UI_ROOT / "config",
        data_dir=_UI_ROOT / "data",
    )


def _record(trajectory: list[dict[str, object]], *, reward: float = 1.0) -> AgentRecord:
    """Build an agent record around a trajectory."""
    return AgentRecord(
        variant="baseline",
        task_id="retail-0",
        trajectory=trajectory,
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
    )


def _verdict_answers(choice: str, confidence: float, completed: float) -> list[Answer]:
    """Build a completed-noul and verdict-choice answer pair."""
    passed = choice == "pass"
    return [
        Answer(question_id="completed", kind=QuestionKind.noul, noul=completed),
        Answer(
            question_id="verdict",
            kind=QuestionKind.choice,
            choice=choice,
            probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
            confidence=confidence,
        ),
    ]


def _verdict(
    judge_id: str,
    repeat: int,
    *,
    choice: str | None = None,
    confidence: float = 1.0,
    completed: float = 1.0,
    error: str | None = None,
    rationale: str | None = None,
) -> Verdict:
    """Build a verdict, an error verdict when choice is None."""
    answers = _verdict_answers(choice, confidence, completed) if choice is not None else []
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash="h",
        repeat=repeat,
        answers=answers,
        rationale=rationale,
        latency_ms=1,
        error=error,
    )


def test_format_tool_call_sorts_arguments() -> None:
    call = {"function": {"name": "get_order_details", "arguments": '{"b": 2, "a": 1}'}}
    assert views.format_tool_call(call) == 'get_order_details({"a": 1, "b": 2})'


def test_format_tool_call_shows_malformed_arguments_raw() -> None:
    call = {"function": {"name": "cancel", "arguments": "{not json"}}
    assert views.format_tool_call(call) == "cancel({not json)"


def test_matched_expected_actions_matches_across_int_and_str() -> None:
    task = Task(
        task_id="retail-0",
        instruction="do it",
        actions=[
            ExpectedAction(name="get_order_details", kwargs={"order_id": 12345}),
            ExpectedAction(name="cancel_pending_order", kwargs={"order_id": "#W9"}),
        ],
        outputs=[],
    )
    record = _record(
        [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "12345"}',
                        },
                    }
                ],
            },
        ]
    )

    matched = views.matched_expected_actions(task, record)

    assert [ok for _, ok in matched] == [True, False]


def test_turns_skips_system_and_formats_tool_calls() -> None:
    record = _record(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "help me"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "#W1"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": "done"},
        ]
    )

    result = views.turns(record)

    assert [turn.role for turn in result] == ["user", "assistant", "tool", "assistant"]
    assert result[1].tool_calls == ['get_order_details({"order_id": "#W1"})']
    assert result[2].tool_call_id == "call_1"
    assert result[3].content == "done"


def test_verdict_rows_aggregates_modal_verdict_agreement_and_errors() -> None:
    verdicts = [
        _verdict("a", 0, choice="pass", confidence=0.8, completed=1.0, rationale="first"),
        _verdict("a", 1, choice="pass", confidence=0.6, completed=1.0, rationale="second"),
        _verdict("a", 2, choice="fail", confidence=1.0, completed=0.0),
        _verdict("a", 3, error="boom"),
        _verdict("b", 0, choice="fail", confidence=0.5, completed=0.0),
    ]

    frame = views.verdict_rows(verdicts)
    row_a = frame[frame["judge_id"] == "a"].iloc[0]
    row_b = frame[frame["judge_id"] == "b"].iloc[0]

    assert list(frame["judge_id"]) == ["a", "b"]
    assert row_a["verdict"] == "pass"
    assert row_a["repeats"] == 4
    assert row_a["errors"] == 1
    expected_agreement = modal_agreement([["pass", "pass", "fail"]]).per_item[0]
    assert row_a["agreement"] == expected_agreement
    assert row_a["confidence"] == (0.8 + 0.6 + 1.0) / 3
    assert row_a["completed"] == (1.0 + 1.0 + 0.0) / 3
    assert row_a["rationale"] == "first"
    assert row_b["verdict"] == "fail"
    assert row_b["repeats"] == 1


def test_verdict_rows_over_fixture_verdicts_has_two_judges() -> None:
    verdicts = data.load_verdicts(_paths())
    assert len(verdicts) == 8  # four whole-trajectory and four per-step verdicts

    grouped = data.verdicts_by_state(verdicts)
    whole = data.trajectory_states(data.load_states(_paths()))
    whole_verdicts = [v for state in whole.values() for v in grouped.get(state.state_hash, [])]
    frame = views.verdict_rows(whole_verdicts)

    assert set(frame["judge_id"]) == {"code", "fake"}
    assert list(frame["repeats"]) == [2, 2]
    assert list(frame["errors"]) == [0, 0]
    # Each judge answered both fixture states once, so the modal verdict spans two states.
    assert set(frame["verdict"]) <= {"pass", "fail"}
    assert all(value == 0.8 for value in frame["confidence"])
    completed = dict(zip(frame["judge_id"], frame["completed"], strict=True))
    assert completed["code"] == 0.5  # one pass at 0.9, one fail at 0.1
    assert completed["fake"] == 0.9  # the fake judge calls both states a pass


def _step_state(step_index: int, state_hash: str) -> StateRecord:
    """Build a per-step state record carrying an index and hash."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.full,
        text=f"step {step_index}",
        token_estimate=1,
        state_hash=state_hash,
        step_index=step_index,
    )


def _step_verdict(
    state_hash: str, repeat: int, *, necessary: float, arguments: float, error: str | None = None
) -> Verdict:
    """Build a fake-judge step verdict answering the two nouls, or an error verdict."""
    answers = (
        []
        if error is not None
        else [
            Answer(question_id="necessary", kind=QuestionKind.noul, noul=necessary),
            Answer(question_id="arguments_consistent", kind=QuestionKind.noul, noul=arguments),
        ]
    )
    return Verdict(
        judge_id="fake",
        model_id="none",
        prompt_version="g2",
        state_hash=state_hash,
        repeat=repeat,
        answers=answers,
        latency_ms=1,
        error=error,
    )


def test_step_scores_means_repeats_and_errors() -> None:
    step_states = {0: _step_state(0, "h0"), 1: _step_state(1, "h1")}
    verdicts_by_hash = {
        "h0": [
            _step_verdict("h0", 0, necessary=0.2, arguments=0.6),
            _step_verdict("h0", 1, necessary=0.4, arguments=0.4),
        ],
        "h1": [
            _step_verdict("h1", 0, necessary=0.8, arguments=0.9),
            _step_verdict("h1", 1, necessary=1.0, arguments=0.7),
            _step_verdict("h1", 2, necessary=0.0, arguments=0.0, error="boom"),
        ],
    }

    frames = views.step_scores(step_states, verdicts_by_hash)

    assert set(frames) == {0, 1}
    row0 = frames[0].iloc[0]
    assert list(frames[0]["judge_id"]) == ["fake"]
    assert row0["necessary"] == (0.2 + 0.4) / 2
    assert row0["arguments_consistent"] == (0.6 + 0.4) / 2
    assert row0["repeats"] == 2
    assert row0["errors"] == 0
    row1 = frames[1].iloc[0]
    assert row1["necessary"] == (0.8 + 1.0) / 2
    assert row1["arguments_consistent"] == (0.9 + 0.7) / 2
    assert row1["repeats"] == 3
    assert row1["errors"] == 1


def test_step_scores_empty_frame_when_no_verdicts() -> None:
    frames = views.step_scores({0: _step_state(0, "h0")}, {})
    assert frames[0].empty
    assert list(frames[0].columns) == [
        "judge_id",
        "necessary",
        "arguments_consistent",
        "repeats",
        "errors",
    ]


def test_expected_step_flags_marks_expected_calls() -> None:
    task = Task(
        task_id="retail-0",
        instruction="do it",
        actions=[ExpectedAction(name="get_product_details", kwargs={"product_id": "1656367028"})],
        outputs=[],
    )
    record = _record(
        [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "#W0000001"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "get_product_details",
                            "arguments": '{"product_id": "1656367028"}',
                        },
                    }
                ],
            },
        ]
    )

    assert views.expected_step_flags(task, record) == [False, True]


def test_load_states_separates_whole_and_step_states() -> None:
    states = data.load_states(_paths())
    keys = set(states)

    assert ("baseline", "full", "none", "retail-0", None) in keys
    assert ("baseline", "full", "none", "retail-1", None) in keys
    assert ("baseline", "full", "none", "retail-0", 0) in keys
    assert ("baseline", "full", "none", "retail-0", 1) in keys

    trajectory = data.trajectory_states(states)
    assert set(trajectory) == {
        ("baseline", "full", "none", "retail-0"),
        ("baseline", "full", "none", "retail-1"),
    }

    steps = data.step_states(states, "baseline", "full", "retail-0")
    assert set(steps) == {0, 1}
    assert steps[0].step_index == 0
    assert steps[1].step_index == 1
    assert data.step_states(states, "baseline", "full", "retail-1") == {}
