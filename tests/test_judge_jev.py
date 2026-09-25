"""Tests for the Jev judge and its SDK mapping helpers."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx2
import pytest
from pydantic import BaseModel
from typesafe_sdk import (
    SystemOneResponse,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from decision_judges.judges.jev import (
    MAX_STATE_TOKENS,
    JevJudge,
    SdkQuestion,
    StateTooLarge,
    from_sdk_answers,
    to_sdk_questions,
)
from decision_judges.types import Question, QuestionKind

FIXTURE = Path(__file__).parent / "fixtures" / "jev_responses" / "ok.json"
FIXTURE_WITH_COST = Path(__file__).parent / "fixtures" / "jev_responses" / "ok_with_cost.json"

VERDICT_Q = Question(
    id="verdict", kind=QuestionKind.choice, text="Did it pass?", options=["pass", "fail"]
)
URGENCY_Q = Question(
    id="urgency", kind=QuestionKind.score, text="How urgent?", levels=["low", "medium", "high"]
)
COMPLETED_Q = Question(id="completed", kind=QuestionKind.noul, text="Completed?")
QUESTIONS: Sequence[Question] = [VERDICT_Q, URGENCY_Q, COMPLETED_Q]


class StateStub(BaseModel):
    """A minimal state satisfying HasStateText for judge tests."""

    text: str = ""
    task_id: str = "task-0"
    state_hash: str = "hash-0"


def _ok_response() -> SystemOneResponse:
    return SystemOneResponse.model_validate_json(FIXTURE.read_text())


class FakeJevClient:
    """A stand-in decision client returning a canned response and recording calls."""

    def __init__(
        self,
        response: SystemOneResponse | None = None,
        *,
        errors: list[Exception] | None = None,
    ) -> None:
        self._response = response if response is not None else _ok_response()
        self._errors = list(errors or [])
        self.calls: list[dict[str, object]] = []

    def system_one(
        self, state: str, questions: Mapping[str, SdkQuestion], *, model: str
    ) -> SystemOneResponse:
        self.calls.append({"state": state, "questions": dict(questions), "model": model})
        if self._errors:
            raise self._errors.pop(0)
        return self._response


def _rate_limit_error() -> TypeSafeRateLimitError:
    return TypeSafeRateLimitError(429, {}, httpx2.Headers({"retry-after-ms": "5"}))


def test_to_sdk_questions_maps_all_kinds() -> None:
    assert to_sdk_questions(QUESTIONS) == {
        "verdict": {
            "type": "choice",
            "instructions": "Did it pass?",
            "criteria": {"pass": "pass", "fail": "fail"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "medium", "high"],
        },
        "completed": {"type": "noul", "instructions": "Completed?"},
    }


def test_response_with_id_provider_and_cost_parses() -> None:
    response = SystemOneResponse.model_validate_json(FIXTURE_WITH_COST.read_text())
    answers = from_sdk_answers(QUESTIONS, response)
    by_id = {answer.question_id: answer for answer in answers}

    assert response.usage.input_tokens == 120
    assert by_id["verdict"].choice == "pass"
    assert by_id["completed"].noul == 0.95


def test_from_sdk_answers_maps_fixture() -> None:
    answers = from_sdk_answers(QUESTIONS, _ok_response())
    by_id = {answer.question_id: answer for answer in answers}

    choice = by_id["verdict"]
    assert choice.choice == "pass"
    assert choice.probabilities == {"pass": 0.8, "fail": 0.2}
    assert choice.confidence == 0.9

    score = by_id["urgency"]
    assert score.score == 1.7
    assert score.probabilities == {"low": 0.1, "medium": 0.1, "high": 0.8}
    assert score.confidence == 0.85

    noul = by_id["completed"]
    assert noul.noul == 0.95


def test_missing_answer_id_yields_verdict_error() -> None:
    extra = Question(id="extra", kind=QuestionKind.noul, text="present?")
    client = FakeJevClient()
    judge = JevJudge("jev-latest", client, prompt_version="v1")

    verdict = judge.judge(StateStub(), [COMPLETED_Q, extra], repeat=0)

    assert verdict.error is not None
    assert "extra" in verdict.error
    assert verdict.answers == []


def test_state_too_large_raised_before_client_call() -> None:
    client = FakeJevClient()
    judge = JevJudge("jev-latest", client, prompt_version="v1")
    oversized = StateStub(text="x" * ((MAX_STATE_TOKENS + 1) * 4 + 4))

    with pytest.raises(StateTooLarge):
        judge.judge(oversized, QUESTIONS, repeat=0)
    assert client.calls == []


def test_rate_limit_retried_with_injected_sleep_then_success() -> None:
    slept: list[float] = []
    client = FakeJevClient(errors=[_rate_limit_error(), _rate_limit_error()])
    judge = JevJudge("jev-latest", client, prompt_version="v1", sleep=slept.append)

    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert verdict.error is None
    assert len(client.calls) == 3
    assert slept == [0.005, 0.005]


def test_rate_limit_exhausted_yields_verdict_error() -> None:
    client = FakeJevClient(errors=[_rate_limit_error()] * 4)
    judge = JevJudge("jev-latest", client, prompt_version="v1", sleep=lambda _: None)

    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert verdict.error is not None
    assert verdict.answers == []


def test_non_retryable_exception_yields_verdict_error() -> None:
    error = TypeSafeInternalServerError(500, {}, httpx2.Headers())
    client = FakeJevClient(errors=[error])
    judge = JevJudge("jev-latest", client, prompt_version="v1")

    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert verdict.error is not None
    assert len(client.calls) == 1


def test_verdict_fields_populated_on_success() -> None:
    client = FakeJevClient()
    judge = JevJudge("jev-latest", client, judge_id="jev", prompt_version="prompt-9")

    verdict = judge.judge(StateStub(state_hash="h7"), QUESTIONS, repeat=2)

    assert verdict.judge_id == "jev"
    assert verdict.model_id == "jev-latest"
    assert verdict.prompt_version == "prompt-9"
    assert verdict.state_hash == "h7"
    assert verdict.repeat == 2
    assert verdict.usage.input_tokens == 120
    assert verdict.usage.output_tokens == 0
    assert verdict.latency_ms >= 0
    assert len(verdict.answers) == 3


def test_unknown_score_level_index_yields_verdict_error() -> None:
    short_score = Question(
        id="urgency", kind=QuestionKind.score, text="How urgent?", levels=["low", "medium"]
    )
    client = FakeJevClient()
    judge = JevJudge("jev-latest", client, prompt_version="v1")

    verdict = judge.judge(StateStub(), [short_score], repeat=0)

    assert verdict.error is not None
