"""A deterministic baseline judge scored from expected actions and outputs."""

import json
from collections.abc import Mapping, Sequence

from decision_judges.bench.load import Task, normalize_action
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.judges.base import HasStateText, build_verdict, timed
from decision_judges.types import Answer, Question, QuestionKind, Verdict

NormalizedCall = tuple[str, tuple[tuple[str, str], ...]]


def _tool_calls(record: AgentRecord) -> list[NormalizedCall]:
    """Return normalized assistant tool calls, skipping malformed arguments."""
    calls: list[NormalizedCall] = []
    for message in record.trajectory:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, Sequence):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                continue
            try:
                kwargs = json.loads(arguments)
            except (ValueError, TypeError):
                continue
            if not isinstance(kwargs, Mapping):
                continue
            calls.append(normalize_action(name, kwargs))
    return calls


def _assistant_text(record: AgentRecord) -> str:
    """Return the concatenation of assistant message contents."""
    parts = [
        message["content"]
        for message in record.trajectory
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ]
    return "\n".join(part for part in parts if isinstance(part, str))


def _missing_actions(task: Task, calls: Sequence[NormalizedCall]) -> list[str]:
    """Return the names of expected actions absent from the made calls."""
    made = set(calls)
    return [
        action.name
        for action in task.actions
        if normalize_action(action.name, action.kwargs) not in made
    ]


def _outputs_present(task: Task, text: str) -> bool:
    """Return whether every required output appears in the text, case-insensitively."""
    haystack = text.lower()
    return all(output.lower() in haystack for output in task.outputs)


def _answer(question: Question, passed: bool) -> Answer:
    """Answer one known question, or raise for an unknown id."""
    if question.id == "verdict":
        return Answer(
            question_id="verdict",
            kind=QuestionKind.choice,
            choice="pass" if passed else "fail",
            probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
            confidence=1.0,
        )
    if question.id == "completed":
        return Answer(
            question_id="completed",
            kind=QuestionKind.noul,
            noul=1.0 if passed else 0.0,
        )
    raise ValueError(f"code judge cannot answer question {question.id!r}")


class CodeJudge:
    """A deterministic judge that compares a run against its task's actions and outputs."""

    judge_id = "code"
    model_id = "none"
    prompt_version = "code-1"
    paid = False

    def __init__(self, tasks: Mapping[str, Task], records: Mapping[str, AgentRecord]) -> None:
        self._tasks = tasks
        self._records = records

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Score the run for state.task_id and answer the known questions."""
        with timed() as elapsed:
            task = self._tasks[state.task_id]
            record = self._records[state.task_id]
            missing = _missing_actions(task, _tool_calls(record))
            outputs_ok = _outputs_present(task, _assistant_text(record))
            passed = not missing and outputs_ok
            answers = [_answer(question, passed) for question in questions]
            rationale = None
            if not passed:
                rationale = f"missing action {missing[0]}" if missing else "missing required output"
            latency_ms = elapsed()
        return build_verdict(
            self, state, repeat, answers, rationale=rationale, latency_ms=latency_ms
        )
