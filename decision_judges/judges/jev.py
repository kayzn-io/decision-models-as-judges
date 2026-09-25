"""A judge backed by the TypeSafe typed decision API.

SDK findings (typesafe-sdk 0.7.1, read from the installed package source):

- The synchronous client is `typesafe_sdk.TypeSafeClient`. It reads its API key from the
  `TYPESAFE_API_KEY` environment variable (or an explicit `api_key` argument).
- Decisions are requested through `TypeSafeClient.system_one(state, questions, *, model=None,
  retry=None, timeout=None, extra_headers=None, extra_body=None, response_model=None)`. `state` is
  text or JSON content; `questions` is a mapping of caller-chosen names to question objects or the
  equivalent dictionaries.
- A question dictionary carries its kind in `type` and its prompt in `instructions`. A choice
  question sets `criteria` to a mapping of label to description (a `None` description means the
  label speaks for itself). A score question sets `criteria` to an ordered list of level
  descriptions whose positions are the scores from zero. A noul question needs only `instructions`.
- `system_one` returns a `SystemOneResponse` with `model`, `usage`, and `answers` keyed by the
  question names. A `ChoiceAnswer` carries `choice`, `confidence`, and `probabilities` keyed by
  label; a `ScoreAnswer` carries `score`, `confidence`, `legend`, and `probabilities` keyed by
  integer level; a `NoulAnswer` carries `noul`. `usage.input_tokens` and `usage.output_tokens` are
  integers or None. The request id is exposed on the response as `request_id`; there is no latency
  field.
- Rate limiting surfaces as `TypeSafeRateLimitError` (HTTP 429), whose `retry_after_ms` holds the
  server's requested wait in milliseconds or None. All SDK failures derive from `TypeSafeError`.
- The client can retry natively via `RetryPolicy`. This judge instead retries rate-limit errors at
  its own layer with an injected sleep, keeping retry behavior deterministic and testable.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from pydantic import ValidationError
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeError,
    TypeSafeRateLimitError,
)

from decision_judges.judges.base import HasStateText, build_verdict, timed
from decision_judges.types import Answer, Question, QuestionKind, Usage, Verdict

MAX_STATE_TOKENS = 32_000
DEFAULT_MAX_RATE_LIMIT_RETRIES = 3

SdkQuestion = dict[str, object]


class StateTooLarge(ValueError):
    """The state text is too large to send to the decision API."""


class JevAnswerError(ValueError):
    """The response lacks a usable answer for a requested question."""


class JevClientLike(Protocol):
    """The single decision method this judge calls on the SDK client."""

    def system_one(
        self, state: str, questions: Mapping[str, SdkQuestion], *, model: str
    ) -> SystemOneResponse: ...


def _to_sdk_question(question: Question) -> SdkQuestion:
    """Express one typed question in the SDK's question shape."""
    if question.kind is QuestionKind.choice:
        options = question.options or []
        return {
            "type": "choice",
            "instructions": question.text,
            "criteria": {option: option for option in options},
        }
    if question.kind is QuestionKind.score:
        levels = question.levels or []
        return {"type": "score", "instructions": question.text, "criteria": list(levels)}
    return {"type": "noul", "instructions": question.text}


def to_sdk_questions(questions: Sequence[Question]) -> dict[str, SdkQuestion]:
    """Map typed questions to SDK question dictionaries keyed by question id."""
    return {question.id: _to_sdk_question(question) for question in questions}


def _score_probabilities(
    levels: Sequence[str], probabilities: Mapping[int, float]
) -> dict[str, float]:
    """Rekey score probabilities from integer levels to their level names."""
    result: dict[str, float] = {}
    for index, probability in probabilities.items():
        if index < 0 or index >= len(levels):
            raise JevAnswerError(f"score probability level {index} has no name")
        result[levels[index]] = probability
    return result


def _from_sdk_answer(question: Question, sdk_answer: object) -> Answer:
    """Map one SDK answer to a typed answer, matching the question's kind."""
    if question.kind is QuestionKind.choice:
        if not isinstance(sdk_answer, ChoiceAnswer):
            raise JevAnswerError(f"question {question.id!r} expected a choice answer")
        return Answer(
            question_id=question.id,
            kind=QuestionKind.choice,
            choice=sdk_answer.choice,
            probabilities=dict(sdk_answer.probabilities),
            confidence=sdk_answer.confidence,
        )
    if question.kind is QuestionKind.score:
        if not isinstance(sdk_answer, ScoreAnswer):
            raise JevAnswerError(f"question {question.id!r} expected a score answer")
        return Answer(
            question_id=question.id,
            kind=QuestionKind.score,
            score=sdk_answer.score,
            probabilities=_score_probabilities(question.levels or [], sdk_answer.probabilities),
            confidence=sdk_answer.confidence,
        )
    if not isinstance(sdk_answer, NoulAnswer):
        raise JevAnswerError(f"question {question.id!r} expected a noul answer")
    return Answer(question_id=question.id, kind=QuestionKind.noul, noul=sdk_answer.noul)


def from_sdk_answers(questions: Sequence[Question], response: SystemOneResponse) -> list[Answer]:
    """Map SDK answers back to typed answers in question order."""
    answers: list[Answer] = []
    for question in questions:
        sdk_answer = response.answers.get(question.id)
        if sdk_answer is None:
            raise JevAnswerError(f"response has no answer for question {question.id!r}")
        answers.append(_from_sdk_answer(question, sdk_answer))
    return answers


class JevJudge:
    """A judge that answers typed questions through the TypeSafe decision API."""

    paid = True

    def __init__(
        self,
        model_id: str,
        client: JevClientLike,
        *,
        judge_id: str = "jev",
        prompt_version: str,
        sleep: Callable[[float], None] = time.sleep,
        max_rate_limit_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
    ) -> None:
        self.judge_id = judge_id
        self.model_id = model_id
        self.prompt_version = prompt_version
        self._client = client
        self._sleep = sleep
        self._max_rate_limit_retries = max_rate_limit_retries

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Decide the questions for one state, recording usage and latency in the verdict."""
        if len(state.text) // 4 > MAX_STATE_TOKENS:
            raise StateTooLarge(f"state exceeds {MAX_STATE_TOKENS} tokens")
        with timed() as elapsed:
            try:
                response = self._request(state, questions)
                answers = from_sdk_answers(questions, response)
                usage = Usage(input_tokens=response.usage.input_tokens or 0, output_tokens=0)
            except (JevAnswerError, ValidationError, TypeSafeError) as error:
                return build_verdict(
                    self, state, repeat, [], latency_ms=elapsed(), error=str(error)
                )
            return build_verdict(self, state, repeat, answers, usage=usage, latency_ms=elapsed())

    def _request(self, state: HasStateText, questions: Sequence[Question]) -> SystemOneResponse:
        """Call the client, retrying rate-limit errors with the injected sleep."""
        sdk_questions = to_sdk_questions(questions)
        attempts = 0
        while True:
            try:
                return self._client.system_one(state.text, sdk_questions, model=self.model_id)
            except TypeSafeRateLimitError as error:
                if attempts >= self._max_rate_limit_retries:
                    raise
                attempts += 1
                self._sleep((error.retry_after_ms or 0.0) / 1000)
