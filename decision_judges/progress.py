"""Progress snapshots and cooperative cancellation for long-running steps.

A :class:`Progress` is a serializable snapshot of one step's advancement that a
callback receives after each unit of work. A :class:`CancelToken` lets a caller
ask a running step to stop between units without raising.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class Progress(BaseModel):
    """A snapshot of a long-running step's advancement."""

    step_id: str
    done: int
    total: int
    spent_usd: float = 0.0
    cap_usd: float | None = None
    last_item: str | None = None
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    cancelled: bool = False
    stopped_reason: str | None = None


ProgressCallback = Callable[[Progress], None]


class CancelToken:
    """A thread-safe, one-shot cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the associated step."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()
