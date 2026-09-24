"""Tests for typed question, answer, and verdict models."""

import pytest
from pydantic import ValidationError

from decision_judges.types import (
    Answer,
    Question,
    QuestionKind,
    Usage,
    Verdict,
)


def test_choice_question_valid() -> None:
    q = Question(
        id="q1",
        kind=QuestionKind.choice,
        text="Pick one",
        options=["a", "b"],
    )
    assert q.options == ["a", "b"]
    assert q.levels is None


def test_score_question_valid() -> None:
    q = Question(
        id="q2",
        kind=QuestionKind.score,
        text="Rate it",
        levels=["low", "high"],
    )
    assert q.levels == ["low", "high"]
    assert q.options is None


def test_noul_question_valid() -> None:
    q = Question(id="q3", kind=QuestionKind.noul, text="Is it true?")
    assert q.options is None
    assert q.levels is None


def test_choice_question_requires_two_options() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", kind=QuestionKind.choice, text="t", options=["only"])


def test_choice_question_rejects_levels() -> None:
    with pytest.raises(ValidationError):
        Question(
            id="q",
            kind=QuestionKind.choice,
            text="t",
            options=["a", "b"],
            levels=["x", "y"],
        )


def test_score_question_requires_two_levels() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", kind=QuestionKind.score, text="t", levels=["one"])


def test_score_question_rejects_options() -> None:
    with pytest.raises(ValidationError):
        Question(
            id="q",
            kind=QuestionKind.score,
            text="t",
            levels=["a", "b"],
            options=["x", "y"],
        )


def test_noul_question_rejects_options_and_levels() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", kind=QuestionKind.noul, text="t", options=["a", "b"])
    with pytest.raises(ValidationError):
        Question(id="q", kind=QuestionKind.noul, text="t", levels=["a", "b"])


def test_choice_answer_valid() -> None:
    a = Answer(
        question_id="q1",
        kind=QuestionKind.choice,
        choice="a",
        probabilities={"a": 0.7, "b": 0.3},
        confidence=0.8,
    )
    assert a.choice == "a"


def test_score_answer_valid() -> None:
    a = Answer(
        question_id="q2",
        kind=QuestionKind.score,
        score=1.0,
        probabilities={"low": 0.4, "high": 0.6},
        confidence=0.5,
    )
    assert a.score == 1.0


def test_noul_answer_valid() -> None:
    a = Answer(question_id="q3", kind=QuestionKind.noul, noul=0.42)
    assert a.noul == 0.42
    assert a.confidence is None


def test_choice_answer_requires_choice_fields() -> None:
    with pytest.raises(ValidationError):
        Answer(question_id="q", kind=QuestionKind.choice, probabilities={"a": 1.0})


def test_choice_answer_rejects_foreign_fields() -> None:
    with pytest.raises(ValidationError):
        Answer(
            question_id="q",
            kind=QuestionKind.choice,
            choice="a",
            probabilities={"a": 0.5, "b": 0.5},
            confidence=0.5,
            noul=0.5,
        )


def test_score_answer_requires_score_fields() -> None:
    with pytest.raises(ValidationError):
        Answer(question_id="q", kind=QuestionKind.score, score=1.0)


def test_noul_answer_rejects_foreign_fields() -> None:
    with pytest.raises(ValidationError):
        Answer(
            question_id="q",
            kind=QuestionKind.noul,
            noul=0.5,
            confidence=0.5,
        )


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        Answer(
            question_id="q",
            kind=QuestionKind.choice,
            choice="a",
            probabilities={"a": 0.7, "b": 0.7},
            confidence=0.5,
        )


def test_probabilities_within_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Answer(
            question_id="q",
            kind=QuestionKind.choice,
            choice="a",
            probabilities={"a": 1.5, "b": -0.5},
            confidence=0.5,
        )


def test_noul_within_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Answer(question_id="q", kind=QuestionKind.noul, noul=1.2)


def test_confidence_within_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Answer(
            question_id="q",
            kind=QuestionKind.choice,
            choice="a",
            probabilities={"a": 0.5, "b": 0.5},
            confidence=1.5,
        )


def test_verdict_json_round_trip() -> None:
    verdict = Verdict(
        judge_id="jev",
        model_id="jev-1.13.0",
        prompt_version="v1",
        state_hash="abc123",
        repeat=0,
        answers=[
            Answer(
                question_id="q1",
                kind=QuestionKind.choice,
                choice="a",
                probabilities={"a": 0.6, "b": 0.4},
                confidence=0.9,
            ),
            Answer(
                question_id="q2",
                kind=QuestionKind.score,
                score=2.0,
                probabilities={"low": 0.2, "high": 0.8},
                confidence=0.7,
            ),
            Answer(question_id="q3", kind=QuestionKind.noul, noul=0.33),
        ],
        usage=Usage(input_tokens=100, output_tokens=20),
        latency_ms=1234,
    )
    dumped = verdict.model_dump_json()
    restored = Verdict.model_validate_json(dumped)
    assert restored == verdict
    assert len(restored.answers) == 3


def test_verdict_defaults() -> None:
    verdict = Verdict(
        judge_id="llm",
        model_id="openai/gpt-4o-mini",
        prompt_version="v1",
        state_hash="hash",
        repeat=1,
        answers=[],
        latency_ms=10,
    )
    assert verdict.usage == Usage()
    assert verdict.extra == {}
    assert verdict.rationale is None
