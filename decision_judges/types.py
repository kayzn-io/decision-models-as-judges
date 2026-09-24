"""Typed question, answer, and verdict models shared across judges."""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class QuestionKind(StrEnum):
    """Kind of typed question a judge answers."""

    choice = "choice"
    score = "score"
    noul = "noul"


def _in_unit_interval(value: float) -> bool:
    """Return whether a value lies within the closed unit interval."""
    return 0.0 <= value <= 1.0


class Question(BaseModel):
    """A typed question with rubric text shared verbatim across judges."""

    id: str
    kind: QuestionKind
    text: str
    options: list[str] | None = None
    levels: list[str] | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "Question":
        """Enforce the fields each question kind requires and forbids."""
        if self.kind is QuestionKind.choice:
            if self.levels is not None:
                raise ValueError("choice question must not define levels")
            if self.options is None or len(self.options) < 2:
                raise ValueError("choice question needs at least two options")
        elif self.kind is QuestionKind.score:
            if self.options is not None:
                raise ValueError("score question must not define options")
            if self.levels is None or len(self.levels) < 2:
                raise ValueError("score question needs at least two levels")
        else:
            if self.options is not None or self.levels is not None:
                raise ValueError("noul question defines neither options nor levels")
        return self


class Answer(BaseModel):
    """A judge's answer to a single typed question."""

    question_id: str
    kind: QuestionKind
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "Answer":
        """Enforce the fields each answer kind requires and forbids."""
        if self.kind is QuestionKind.choice:
            if self.choice is None or self.probabilities is None or self.confidence is None:
                raise ValueError("choice answer needs choice, probabilities, and confidence")
            if self.score is not None or self.noul is not None:
                raise ValueError("choice answer must not carry score or noul")
        elif self.kind is QuestionKind.score:
            if self.score is None or self.probabilities is None or self.confidence is None:
                raise ValueError("score answer needs score, probabilities, and confidence")
            if self.choice is not None or self.noul is not None:
                raise ValueError("score answer must not carry choice or noul")
        else:
            if self.noul is None:
                raise ValueError("noul answer needs noul")
            if (
                self.choice is not None
                or self.probabilities is not None
                or self.score is not None
                or self.confidence is not None
            ):
                raise ValueError("noul answer carries noul only")

        if self.noul is not None and not _in_unit_interval(self.noul):
            raise ValueError("noul must lie in [0, 1]")
        if self.confidence is not None and not _in_unit_interval(self.confidence):
            raise ValueError("confidence must lie in [0, 1]")
        if self.probabilities is not None:
            for value in self.probabilities.values():
                if not _in_unit_interval(value):
                    raise ValueError("each probability must lie in [0, 1]")
            if abs(sum(self.probabilities.values()) - 1.0) > 0.01:
                raise ValueError("probabilities must sum to 1 within 0.01")
        return self


class Usage(BaseModel):
    """Token usage recorded for a judge call."""

    input_tokens: int = 0
    output_tokens: int = 0


class Verdict(BaseModel):
    """A single judge's response over a set of typed questions."""

    judge_id: str
    model_id: str
    prompt_version: str
    state_hash: str
    repeat: int
    answers: list[Answer]
    rationale: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int
    error: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)
