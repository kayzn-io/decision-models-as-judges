"""Judge protocol and shared helpers for building verdicts."""

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

from decision_judges.types import Answer, Question, Usage, Verdict


class HasStateText(Protocol):
    """The subset of a serialized state a judge reads."""

    text: str
    task_id: str
    state_hash: str


class Judge(Protocol):
    """A judge that answers typed questions about one state."""

    judge_id: str
    model_id: str
    prompt_version: str

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict: ...


@contextmanager
def timed() -> Iterator[Callable[[], int]]:
    """Yield a callable returning elapsed milliseconds since entry."""
    start = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    yield elapsed


def build_verdict(
    judge: Judge,
    state: HasStateText,
    repeat: int,
    answers: list[Answer],
    *,
    rationale: str | None = None,
    usage: Usage | None = None,
    latency_ms: int,
    error: str | None = None,
    extra: dict[str, str] | None = None,
) -> Verdict:
    """Build a verdict, stamping identity and state fields from the judge."""
    return Verdict(
        judge_id=judge.judge_id,
        model_id=judge.model_id,
        prompt_version=judge.prompt_version,
        state_hash=state.state_hash,
        repeat=repeat,
        answers=answers,
        rationale=rationale,
        usage=usage if usage is not None else Usage(),
        latency_ms=latency_ms,
        error=error,
        extra=extra if extra is not None else {},
    )
