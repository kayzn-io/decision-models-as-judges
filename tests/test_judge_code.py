"""Tests for the deterministic code judge and its helpers."""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from decision_judges.bench.load import ExpectedAction, Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.judges.base import HasStateText
from decision_judges.judges.code import (
    CodeJudge,
    _assistant_text,
    _missing_actions,
    _outputs_present,
    _tool_calls,
)
from decision_judges.types import Question, QuestionKind


class StateStub(BaseModel):
    """A minimal state satisfying HasStateText for judge tests."""

    text: str = ""
    task_id: str
    state_hash: str = "hash-0"


def _state(task_id: str = "retail-0") -> HasStateText:
    return StateStub(task_id=task_id)


def _task(
    task_id: str = "retail-0",
    actions: list[ExpectedAction] | None = None,
    outputs: list[str] | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        instruction="do the thing",
        actions=actions if actions is not None else [],
        outputs=outputs if outputs is not None else [],
    )


def _tool_call(name: str, arguments: str) -> dict[str, object]:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}


def _record(
    task_id: str = "retail-0",
    trajectory: list[dict[str, object]] | None = None,
) -> AgentRecord:
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=trajectory if trajectory is not None else [],
        reward=0.0,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
    )


VERDICT_Q = Question(id="verdict", kind=QuestionKind.choice, text="pass?", options=["pass", "fail"])
COMPLETED_Q = Question(id="completed", kind=QuestionKind.noul, text="completed?")
QUESTIONS: Sequence[Question] = [VERDICT_Q, COMPLETED_Q]


def _judge(task: Task, record: AgentRecord) -> CodeJudge:
    return CodeJudge({task.task_id: task}, {record.task_id: record})


def test_pass_when_actions_and_outputs_present() -> None:
    action = ExpectedAction(name="cancel_pending_order", kwargs={"order_id": "W1", "reason": "x"})
    task = _task(actions=[action], outputs=["Cancelled", "refund"])
    trajectory = [
        {
            "role": "assistant",
            "content": "Order W1 has been CANCELLED.",
            "tool_calls": [_tool_call("cancel_pending_order", '{"reason": "x", "order_id": "W1"}')],
        },
        {"role": "assistant", "content": "Your refund is on the way."},
    ]
    verdict = _judge(task, _record(trajectory=trajectory)).judge(_state(), QUESTIONS, repeat=0)

    choice = next(a for a in verdict.answers if a.question_id == "verdict")
    noul = next(a for a in verdict.answers if a.question_id == "completed")
    assert choice.choice == "pass"
    assert choice.probabilities == {"pass": 1.0, "fail": 0.0}
    assert noul.noul == 1.0
    assert verdict.error is None


def test_fail_when_action_absent_names_it() -> None:
    action = ExpectedAction(name="cancel_pending_order", kwargs={"order_id": "W1"})
    task = _task(actions=[action], outputs=[])
    verdict = _judge(task, _record(trajectory=[])).judge(_state(), QUESTIONS, repeat=1)

    choice = next(a for a in verdict.answers if a.question_id == "verdict")
    noul = next(a for a in verdict.answers if a.question_id == "completed")
    assert choice.choice == "fail"
    assert choice.probabilities == {"pass": 0.0, "fail": 1.0}
    assert noul.noul == 0.0
    assert verdict.rationale is not None and "cancel_pending_order" in verdict.rationale


def test_fail_when_output_missing() -> None:
    action = ExpectedAction(name="get_order_details", kwargs={"order_id": "W1"})
    task = _task(actions=[action], outputs=["a required phrase"])
    trajectory = [
        {
            "role": "assistant",
            "content": "here you go",
            "tool_calls": [_tool_call("get_order_details", '{"order_id": "W1"}')],
        }
    ]
    verdict = _judge(task, _record(trajectory=trajectory)).judge(_state(), QUESTIONS, repeat=0)

    choice = next(a for a in verdict.answers if a.question_id == "verdict")
    assert choice.choice == "fail"
    assert verdict.rationale == "missing required output"


def test_kwargs_type_equivalence_counts_as_made() -> None:
    task = _task(actions=[ExpectedAction(name="tool", kwargs={"a": 1})], outputs=[])
    trajectory = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("tool", '{"a": "1"}')],
        }
    ]
    verdict = _judge(task, _record(trajectory=trajectory)).judge(_state(), QUESTIONS, repeat=0)
    choice = next(a for a in verdict.answers if a.question_id == "verdict")
    assert choice.choice == "pass"


def test_malformed_arguments_are_skipped() -> None:
    trajectory = [
        {
            "role": "assistant",
            "content": "text",
            "tool_calls": [_tool_call("tool", "{not json")],
        }
    ]
    calls = _tool_calls(_record(trajectory=trajectory))
    assert calls == []


def test_unknown_question_raises_value_error() -> None:
    task = _task()
    judge = _judge(task, _record())
    unknown = Question(id="mystery", kind=QuestionKind.noul, text="?")
    with pytest.raises(ValueError, match="mystery"):
        judge.judge(_state(), [unknown], repeat=0)


def test_verdict_fields_populated() -> None:
    task = _task(outputs=[])
    judge = _judge(task, _record())
    verdict = judge.judge(StateStub(task_id="retail-0", state_hash="h9"), QUESTIONS, repeat=3)
    assert verdict.judge_id == "code"
    assert verdict.model_id == "none"
    assert verdict.prompt_version == "code-1"
    assert verdict.state_hash == "h9"
    assert verdict.repeat == 3
    assert verdict.latency_ms >= 0


def test_empty_outputs_pass_vacuously() -> None:
    assert _outputs_present(_task(outputs=[]), "") is True


def test_assistant_text_joins_and_ignores_none() -> None:
    trajectory = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "ignored"},
        {"role": "assistant", "content": "two"},
    ]
    text = _assistant_text(_record(trajectory=trajectory))
    assert "one" in text and "two" in text and "ignored" not in text


def test_missing_actions_reports_absent_only() -> None:
    made = ExpectedAction(name="a", kwargs={})
    absent = ExpectedAction(name="b", kwargs={"k": 1})
    task = _task(actions=[made, absent])
    trajectory = [
        {"role": "assistant", "content": "x", "tool_calls": [_tool_call("a", "{}")]},
    ]
    calls = _tool_calls(_record(trajectory=trajectory))
    assert _missing_actions(task, calls) == ["b"]
