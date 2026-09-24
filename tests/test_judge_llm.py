"""Tests for the LLM judge, its helpers, and the openai adapter."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from decision_judges.judges.llm import (
    LlmAnswerError,
    LlmJudge,
    OpenAiClientAdapter,
    build_messages,
    parse_answers,
    prompt_version_for,
    read_rubric,
    response_schema,
)
from decision_judges.types import Question, QuestionKind

FIXTURES = Path(__file__).parent / "fixtures" / "llm_responses"

CHOICE_Q = Question(
    id="verdict", kind=QuestionKind.choice, text="pass or fail?", options=["pass", "fail"]
)
NOUL_Q = Question(id="completed", kind=QuestionKind.noul, text="completed?")
SCORE_Q = Question(id="quality", kind=QuestionKind.score, text="quality?", levels=["low", "high"])
QUESTIONS = [CHOICE_Q, NOUL_Q]


class StateStub(BaseModel):
    """A minimal state satisfying HasStateText for judge tests."""

    text: str = "conversation text"
    task_id: str = "retail-0"
    state_hash: str = "hash-0"


def _ns(value: Any) -> Any:
    """Wrap dicts/lists recursively so attribute access mirrors the openai client."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


def _response(fixture: str) -> Any:
    return _ns(json.loads((FIXTURES / fixture).read_text()))


class FakeClient:
    """A client that returns a canned response and records call kwargs."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def chat_completions_create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class BoomClient:
    """A client that always raises, counting the attempts."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.count = 0

    def chat_completions_create(self, **kwargs: Any) -> Any:
        self.count += 1
        raise self._error


@pytest.fixture
def rubric_path(tmp_path: Path) -> Path:
    path = tmp_path / "rubric.md"
    path.write_text("# version: 1\nBe fair.\n")
    return path


def _judge(client: Any, rubric_path: Path, **kwargs: Any) -> LlmJudge:
    return LlmJudge("g3", "model-x", rubric_path, client, **kwargs)


def test_read_rubric_parses_version_and_strips_header(tmp_path: Path) -> None:
    path = tmp_path / "r.md"
    path.write_text("# version: 3\nline one\nline two\n")
    body, version = read_rubric(path)
    assert version == "3"
    assert "version" not in body
    assert body.startswith("line one")


def test_read_rubric_without_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "r.md"
    path.write_text("no header here\nmore\n")
    with pytest.raises(ValueError):
        read_rubric(path)


def test_prompt_version_changes_with_question_text() -> None:
    changed = Question(
        id="verdict", kind=QuestionKind.choice, text="different?", options=["pass", "fail"]
    )
    assert prompt_version_for("1", [CHOICE_Q]) != prompt_version_for("1", [changed])


def test_prompt_version_changes_with_rubric_version() -> None:
    assert prompt_version_for("1", [CHOICE_Q]) != prompt_version_for("2", [CHOICE_Q])
    assert prompt_version_for("1", [CHOICE_Q]).startswith("1-")


def test_build_messages_has_rubric_system_and_all_ids() -> None:
    messages = build_messages("RUBRIC TEXT", "state here", QUESTIONS)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "RUBRIC TEXT"
    assert messages[1]["role"] == "user"
    assert "state here" in messages[1]["content"]
    for question in QUESTIONS:
        assert question.id in messages[1]["content"]


def test_response_schema_forbids_additional_and_enumerates_options() -> None:
    schema = response_schema(QUESTIONS)
    assert schema["additionalProperties"] is False
    answers = schema["properties"]["answers"]
    assert answers["additionalProperties"] is False
    assert set(answers["required"]) == {"verdict", "completed"}
    choice = answers["properties"]["verdict"]
    assert choice["additionalProperties"] is False
    assert choice["properties"]["choice"]["enum"] == ["pass", "fail"]


def test_parse_answers_score_maps_level_index() -> None:
    payload = {"answers": {"quality": {"level": "high", "confidence": 0.7}}, "rationale": "r"}
    [answer] = parse_answers([SCORE_Q], payload)
    assert answer.score == 1.0
    assert answer.confidence == 0.7
    assert answer.probabilities == {"low": pytest.approx(0.3), "high": 0.7}


def test_parse_answers_missing_id_raises_named() -> None:
    payload = {"answers": {"verdict": {"choice": "pass", "confidence": 0.5}}, "rationale": ""}
    with pytest.raises(LlmAnswerError, match="completed"):
        parse_answers(QUESTIONS, payload)


def test_parse_answers_out_of_range_raises() -> None:
    payload = {"answers": {"completed": {"probability": 1.5}}, "rationale": ""}
    with pytest.raises(LlmAnswerError):
        parse_answers([NOUL_Q], payload)


def test_judge_parses_ok_fixture(rubric_path: Path) -> None:
    client = FakeClient(_response("ok.json"))
    verdict = _judge(client, rubric_path).judge(StateStub(), QUESTIONS, repeat=2)

    assert verdict.error is None
    choice = next(a for a in verdict.answers if a.question_id == "verdict")
    noul = next(a for a in verdict.answers if a.question_id == "completed")
    assert choice.choice == "pass"
    assert choice.confidence == 0.8
    assert choice.probabilities == {"pass": 0.8, "fail": pytest.approx(0.2)}
    assert noul.noul == 0.6
    assert verdict.usage.input_tokens == 100
    assert verdict.usage.output_tokens == 20
    assert verdict.rationale == "looks good"
    assert verdict.repeat == 2

    response_format = client.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format


def test_judge_missing_field_sets_error(rubric_path: Path) -> None:
    verdict = _judge(FakeClient(_response("missing_field.json")), rubric_path).judge(
        StateStub(), QUESTIONS, repeat=0
    )
    assert verdict.error is not None
    assert verdict.answers == []
    assert verdict.usage.input_tokens == 50


def test_judge_bad_enum_sets_error(rubric_path: Path) -> None:
    verdict = _judge(FakeClient(_response("bad_enum.json")), rubric_path).judge(
        StateStub(), QUESTIONS, repeat=0
    )
    assert verdict.error is not None
    assert verdict.answers == []


def test_transport_error_retries_max_then_errors(rubric_path: Path) -> None:
    backoff_calls: list[int] = []
    client = BoomClient(RuntimeError("down"))
    judge = _judge(
        client,
        rubric_path,
        max_retries=3,
        backoff=lambda seconds: backoff_calls.append(seconds) or 0.0,
    )
    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert client.count == 3
    assert verdict.error is not None
    assert verdict.answers == []
    assert len(backoff_calls) == 2


def test_retry_after_passed_to_backoff(rubric_path: Path) -> None:
    error = RuntimeError("rate limited")
    error.retry_after = 7  # type: ignore[attr-defined]
    received: list[int] = []
    judge = _judge(
        BoomClient(error),
        rubric_path,
        max_retries=2,
        backoff=lambda seconds: received.append(seconds) or 0.0,
    )
    judge.judge(StateStub(), QUESTIONS, repeat=0)
    assert received == [7]


def test_adapter_construction_needs_no_key() -> None:
    OpenAiClientAdapter("https://example.test/v1", api_key_env="MISSING_KEY_ENV")


def test_adapter_builds_openai_client_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, base_url: str, api_key: str) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("DUMMY_KEY", "secret")
    adapter = OpenAiClientAdapter("https://example.test/v1", api_key_env="DUMMY_KEY")
    adapter._ensure_client()

    assert captured["base_url"] == "https://example.test/v1"
    assert captured["api_key"] == "secret"
