"""An LLM judge that answers typed questions under a schema-constrained rubric."""

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from decision_judges.judges.base import HasStateText, build_verdict, timed
from decision_judges.types import Answer, Question, QuestionKind, Usage, Verdict

_VERSION_RE = re.compile(r"^#\s*version:\s*(\S+)\s*$")
_UNIT = {"type": "number", "minimum": 0, "maximum": 1}
_RESPONSE_INSTRUCTIONS = (
    "Answer every question. Return JSON matching the schema exactly: an 'answers' object "
    "keyed by question id and a 'rationale' string. For a choice question give the chosen "
    "option and a confidence in [0, 1]; for a score question give the level and a confidence; "
    "for a noul question give a probability in [0, 1]. Report calibrated confidence."
)


class LlmAnswerError(ValueError):
    """Raised when a model payload cannot map to a typed answer."""


class OpenAILike(Protocol):
    """The single client method the judge depends on."""

    def chat_completions_create(self, **kwargs: Any) -> Any: ...


def read_rubric(path: Path) -> tuple[str, str]:
    """Return the rubric body without its version header and the parsed version."""
    lines = path.read_text(encoding="utf-8").splitlines()
    match = _VERSION_RE.match(lines[0]) if lines else None
    if match is None:
        raise ValueError("rubric missing '# version: N' header on the first line")
    return "\n".join(lines[1:]).strip(), match.group(1)


def prompt_version_for(rubric_version: str, questions: Sequence[Question]) -> str:
    """Derive a prompt version from the rubric version and the question set."""
    material = "".join(f"{q.text}\x1f{q.id}\x1f{q.kind}\x1e" for q in questions)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"{rubric_version}-{digest}"


def _question_block(question: Question) -> str:
    """Render one question as a labelled block for the user prompt."""
    lines = [f"- id: {question.id}", f"  kind: {question.kind}", f"  text: {question.text}"]
    if question.options is not None:
        lines.append(f"  options: {', '.join(question.options)}")
    if question.levels is not None:
        lines.append(f"  levels: {', '.join(question.levels)}")
    return "\n".join(lines)


def build_messages(
    rubric: str, state_text: str, questions: Sequence[Question]
) -> list[dict[str, str]]:
    """Build the system and user messages for one judging call."""
    blocks = "\n".join(_question_block(q) for q in questions)
    user = f"{state_text}\n\nQUESTIONS:\n{blocks}\n\n{_RESPONSE_INSTRUCTIONS}"
    return [{"role": "system", "content": rubric}, {"role": "user", "content": user}]


def _answer_schema(question: Question) -> dict[str, Any]:
    """Return the JSON schema fragment for one question's answer object."""
    props: dict[str, Any]
    if question.kind is QuestionKind.choice:
        props = {"choice": {"enum": list(question.options or [])}, "confidence": dict(_UNIT)}
        required = ["choice", "confidence"]
    elif question.kind is QuestionKind.score:
        props = {"level": {"enum": list(question.levels or [])}, "confidence": dict(_UNIT)}
        required = ["level", "confidence"]
    else:
        props = {"probability": dict(_UNIT)}
        required = ["probability"]
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def response_schema(questions: Sequence[Question]) -> dict[str, Any]:
    """Return a strict JSON schema for the answers-and-rationale payload."""
    answer_props = {q.id: _answer_schema(q) for q in questions}
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "object",
                "properties": answer_props,
                "required": list(answer_props),
                "additionalProperties": False,
            },
            "rationale": {"type": "string"},
        },
        "required": ["answers", "rationale"],
        "additionalProperties": False,
    }


def _require(mapping: dict[str, Any], key: str, field: str) -> Any:
    """Return mapping[key] or raise naming the missing field."""
    if key not in mapping:
        raise LlmAnswerError(f"missing field {field}")
    return mapping[key]


def _unit_number(value: Any, field: str) -> float:
    """Return value as a float in [0, 1] or raise naming the field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlmAnswerError(f"field {field} is not a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise LlmAnswerError(f"field {field} out of range [0, 1]")
    return number


def _distribute(labels: Sequence[str], chosen: str, confidence: float) -> dict[str, float]:
    """Spread one reported confidence over labels as a probability vector.

    An LLM reports a single confidence rather than a full distribution, so the chosen label
    keeps that confidence and the remaining mass is split uniformly across the other labels.
    """
    others = [label for label in labels if label != chosen]
    remainder = (1.0 - confidence) / len(others) if others else 0.0
    return {label: (confidence if label == chosen else remainder) for label in labels}


def _parse_one(question: Question, raw: dict[str, Any], field: str) -> Answer:
    """Map one raw answer object to a typed Answer for its question."""
    if question.kind is QuestionKind.choice:
        options = question.options or []
        choice = _require(raw, "choice", f"{field}.choice")
        if choice not in options:
            raise LlmAnswerError(f"field {field}.choice is not an option")
        confidence = _unit_number(
            _require(raw, "confidence", f"{field}.confidence"), f"{field}.confidence"
        )
        return Answer(
            question_id=question.id,
            kind=QuestionKind.choice,
            choice=choice,
            probabilities=_distribute(options, choice, confidence),
            confidence=confidence,
        )
    if question.kind is QuestionKind.score:
        levels = question.levels or []
        level = _require(raw, "level", f"{field}.level")
        if level not in levels:
            raise LlmAnswerError(f"field {field}.level is not a level")
        confidence = _unit_number(
            _require(raw, "confidence", f"{field}.confidence"), f"{field}.confidence"
        )
        return Answer(
            question_id=question.id,
            kind=QuestionKind.score,
            score=float(levels.index(level)),
            probabilities=_distribute(levels, level, confidence),
            confidence=confidence,
        )
    probability = _unit_number(
        _require(raw, "probability", f"{field}.probability"), f"{field}.probability"
    )
    return Answer(question_id=question.id, kind=QuestionKind.noul, noul=probability)


def parse_answers(questions: Sequence[Question], payload: dict[str, Any]) -> list[Answer]:
    """Map a payload to typed answers, naming any missing or invalid field."""
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, dict):
        raise LlmAnswerError("missing field answers")
    result: list[Answer] = []
    for question in questions:
        field = f"answers.{question.id}"
        raw = _require(answers_obj, question.id, field)
        if not isinstance(raw, dict):
            raise LlmAnswerError(f"field {field} is not an object")
        result.append(_parse_one(question, raw, field))
    return result


def _default_backoff(seconds: int) -> float:
    """Sleep for the given seconds, capped, and return the delay slept."""
    delay = float(min(seconds, 30))
    time.sleep(delay)
    return delay


def _create_with_retries(
    client: OpenAILike,
    kwargs: dict[str, Any],
    max_retries: int,
    backoff: Callable[[int], float],
) -> Any:
    """Call the client, retrying transport errors up to max_retries with injected backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.chat_completions_create(**kwargs)
        except Exception as exc:  # transport boundary; parsing happens after a response
            last_exc = exc
            if attempt + 1 >= max_retries:
                break
            retry_after = getattr(exc, "retry_after", None)
            backoff(int(retry_after) if retry_after is not None else min(2**attempt, 30))
    assert last_exc is not None
    raise last_exc


def _usage_from(response: Any) -> Usage:
    """Read token usage from a chat-completion response, tolerating absence."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return Usage(input_tokens=int(prompt), output_tokens=int(completion))


def _content_from(response: Any) -> str:
    """Return the first choice's message content or raise on a malformed response."""
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise LlmAnswerError("response content is not a string")
    return content


class OpenAiClientAdapter:
    """Adapt the openai client to OpenAILike, reading the API key lazily from the env."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        *,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Build the openai client on first use, using the explicit key or the env."""
        if self._client is None:
            import openai

            key = self._api_key if self._api_key is not None else os.environ[self._api_key_env]
            self._client = openai.OpenAI(base_url=self._base_url, api_key=key)
        return self._client

    def chat_completions_create(self, **kwargs: Any) -> Any:
        """Forward a chat-completions request to the underlying openai client."""
        return self._ensure_client().chat.completions.create(**kwargs)


class LlmJudge:
    """A judge that asks an LLM the typed questions under a versioned rubric."""

    judge_id: str
    model_id: str
    prompt_version: str
    paid = True

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        rubric_path: Path,
        client: OpenAILike,
        *,
        max_retries: int = 5,
        backoff: Callable[[int], float] = _default_backoff,
    ) -> None:
        self.judge_id = judge_id
        self.model_id = model_id
        self._client = client
        self._max_retries = max_retries
        self._backoff = backoff
        self._rubric, self._rubric_version = read_rubric(rubric_path)
        self.prompt_version = self._rubric_version

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Ask the model the questions and map the reply to a verdict, never raising."""
        self.prompt_version = prompt_version_for(self._rubric_version, questions)
        with timed() as elapsed:
            kwargs: dict[str, Any] = {
                "model": self.model_id,
                "messages": build_messages(self._rubric, state.text, questions),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verdict",
                        "strict": True,
                        "schema": response_schema(questions),
                    },
                },
            }
            try:
                response = _create_with_retries(
                    self._client, kwargs, self._max_retries, self._backoff
                )
            except Exception as exc:
                return build_verdict(
                    self,
                    state,
                    repeat,
                    [],
                    latency_ms=elapsed(),
                    error=f"transport error: {type(exc).__name__}",
                )
            usage = _usage_from(response)
            try:
                payload = json.loads(_content_from(response))
                if not isinstance(payload, dict):
                    raise LlmAnswerError("payload is not an object")
                answers = parse_answers(questions, payload)
                raw_rationale = payload.get("rationale")
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                return build_verdict(
                    self,
                    state,
                    repeat,
                    [],
                    usage=usage,
                    latency_ms=elapsed(),
                    error=f"parse error: {exc}",
                )
            rationale = raw_rationale if isinstance(raw_rationale, str) else None
            return build_verdict(
                self,
                state,
                repeat,
                answers,
                rationale=rationale,
                usage=usage,
                latency_ms=elapsed(),
            )
