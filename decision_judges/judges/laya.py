"""A local judge backed by the Laya typed-decision model.

The judge is deliberately thin: it maps typed questions to option lists, calls
the injected model for raw option logits, and turns those logits into calibrated
answers. All model-specific loading and tensor work lives in
:mod:`decision_judges.judges.laya_model`.
"""

import math
from collections.abc import Sequence
from typing import Protocol

from decision_judges.judges.base import HasStateText, build_verdict, timed
from decision_judges.types import Answer, Question, QuestionKind, Usage, Verdict

# The model reads ``input_limit`` (512) tokens total, of which the option prompt
# reserves ``head_limit`` (192). The state therefore gets the remainder.
INPUT_LIMIT = 512
HEAD_LIMIT = 192
MAX_STATE_TOKENS = INPUT_LIMIT - HEAD_LIMIT

NOUL_OPTIONS = ("false", "true")


class StateTooLarge(ValueError):
    """The state text is too large to fit the model's input budget."""


class LayaModelLike(Protocol):
    """The model surface the judge depends on."""

    checkpoint_hash: str

    def option_logits(
        self,
        state: str,
        question_type: str,
        instructions: str,
        options: Sequence[str],
    ) -> list[float]: ...

    def tokenizer_len(self, text: str) -> int: ...


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Divide logits by a positive temperature before the softmax."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return [value / temperature for value in logits]


def softmax(logits: Sequence[float]) -> list[float]:
    """Return a numerically stable softmax over the logits."""
    if not logits:
        return []
    highest = max(logits)
    exponentials = [math.exp(value - highest) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def confidence_from_probs(probs: Sequence[float]) -> float:
    """Return ``1 - normalized entropy``; 1.0 for a single option."""
    count = len(probs)
    if count <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    normalized = entropy / math.log(count)
    return max(0.0, min(1.0, 1.0 - normalized))


def expected_level(probs: Sequence[float]) -> float:
    """Return the expected value over ordered levels, ``sum(i * p_i)``."""
    return sum(index * probability for index, probability in enumerate(probs))


def question_options(question: Question) -> tuple[str, list[str]]:
    """Return the model question type and option list for a typed question.

    Choice questions expose their options, score questions their levels, and
    noul questions the fixed false/true pair the model scores.
    """
    if question.kind is QuestionKind.choice:
        return "choice", list(question.options or [])
    if question.kind is QuestionKind.score:
        return "score", list(question.levels or [])
    return "noul", list(NOUL_OPTIONS)


def answer_for(question: Question, probs: Sequence[float]) -> Answer:
    """Build the typed answer for a question from its option probabilities.

    Choice answers take the argmax option, score answers the probability-weighted
    level index, and noul answers the true-slot probability.
    """
    if question.kind is QuestionKind.choice:
        options = question.options or []
        best = max(range(len(options)), key=lambda index: probs[index])
        return Answer(
            question_id=question.id,
            kind=QuestionKind.choice,
            choice=options[best],
            probabilities=dict(zip(options, probs, strict=True)),
            confidence=confidence_from_probs(probs),
        )
    if question.kind is QuestionKind.score:
        levels = question.levels or []
        return Answer(
            question_id=question.id,
            kind=QuestionKind.score,
            score=expected_level(probs),
            probabilities=dict(zip(levels, probs, strict=True)),
            confidence=confidence_from_probs(probs),
        )
    return Answer(question_id=question.id, kind=QuestionKind.noul, noul=probs[1])


class LayaJudge:
    """Answers typed questions with a local Laya decision model."""

    def __init__(
        self,
        model: LayaModelLike,
        *,
        judge_id: str = "laya",
        model_label: str,
        prompt_version: str,
        temperature: float = 1.0,
        device: str,
    ) -> None:
        self.judge_id = judge_id
        self.model_label = model_label
        self.prompt_version = prompt_version
        self.model_id = f"laya:{model.checkpoint_hash[:12]}"
        self._model = model
        self._temperature = temperature
        self._device = device

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Decide the questions for one state, recording usage and latency."""
        input_tokens = self._model.tokenizer_len(state.text)
        if input_tokens > MAX_STATE_TOKENS:
            raise StateTooLarge(f"state exceeds {MAX_STATE_TOKENS} tokens")
        with timed() as elapsed:
            try:
                answers = [self._answer(state, question) for question in questions]
            except Exception as error:  # model failures become verdict errors, never raise
                return build_verdict(
                    self,
                    state,
                    repeat,
                    [],
                    latency_ms=elapsed(),
                    error=str(error),
                    extra=self._extra(),
                )
            usage = Usage(input_tokens=input_tokens, output_tokens=0)
            return build_verdict(
                self,
                state,
                repeat,
                answers,
                usage=usage,
                latency_ms=elapsed(),
                extra=self._extra(),
            )

    def _extra(self) -> dict[str, str]:
        """Return the provenance recorded on every verdict."""
        return {
            "checkpoint_hash": self._model.checkpoint_hash,
            "device": self._device,
            "temperature": str(self._temperature),
            "model_label": self.model_label,
        }

    def _probabilities(
        self, state: str, question_type: str, instructions: str, options: Sequence[str]
    ) -> list[float]:
        """Return softmax probabilities over options after temperature scaling."""
        logits = self._model.option_logits(state, question_type, instructions, options)
        return softmax(apply_temperature(logits, self._temperature))

    def _answer(self, state: HasStateText, question: Question) -> Answer:
        """Answer one typed question, dispatching on its kind."""
        question_type, options = question_options(question)
        probs = self._probabilities(state.text, question_type, question.text, options)
        return answer_for(question, probs)
