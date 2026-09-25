"""Background step execution with progress persisted atomically to disk.

A :class:`StepRunner` runs a callable on a daemon thread, passing it an
``on_progress`` writer and a :class:`CancelToken`. Every :class:`Progress` the
callable reports is written atomically to ``<state_dir>/<step_id>.json``,
throttled to at most four writes per second, with the final state always
flushed. The status file lets a fresh process read where a step got to and
detect a run the process abandoned mid-flight.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from decision_judges.progress import CancelToken, Progress, utc_now_iso

T = TypeVar("T")

# At most four writes per second per step; the final write bypasses the throttle.
_MIN_WRITE_INTERVAL_S = 0.25


class StepAlreadyRunning(RuntimeError):
    """Raised when starting a step whose thread is still alive."""


class _ProgressWriter:
    """Persists Progress for one step atomically, throttled, forcing the last write."""

    def __init__(self, path: Path, step_id: str) -> None:
        self._path = path
        self._step_id = step_id
        self._lock = threading.Lock()
        self._last_write = 0.0

    def write(self, progress: Progress, *, force: bool = False) -> None:
        """Persist a snapshot, skipping throttled writes unless forced."""
        stamped = progress.model_copy(update={"step_id": self._step_id})
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_write < _MIN_WRITE_INTERVAL_S:
                return
            self._last_write = now
            self._persist(stamped)

    def _persist(self, progress: Progress) -> None:
        """Write the snapshot to a temp file and atomically replace the target."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(progress.model_dump_json(indent=2))
        os.replace(tmp, self._path)


class StepRunner:
    """Runs steps on daemon threads and persists their progress under a directory."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = Path(state_dir)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._tokens: dict[str, CancelToken] = {}

    def _path(self, step_id: str) -> Path:
        """Return the progress file path for a step."""
        return self._state_dir / f"{step_id}.json"

    def start(self, step_id: str, fn: Callable[..., T], *args: object, **kwargs: object) -> None:
        """Run ``fn`` on a daemon thread, persisting every progress it reports.

        The callable is invoked as ``fn(*args, on_progress=<writer>,
        cancel=<token>, **kwargs)``. Raise :class:`StepAlreadyRunning` when a
        thread for the step is still alive.

        A step callable returns None on a clean finish, or a short reason string
        when it stopped early without raising; that reason is recorded on the
        final :class:`Progress` as ``stopped_reason`` with ``finished_at`` set
        and ``error`` left None.
        """
        with self._lock:
            existing = self._threads.get(step_id)
            if existing is not None and existing.is_alive():
                raise StepAlreadyRunning(step_id)

        token = CancelToken()
        writer = _ProgressWriter(self._path(step_id), step_id)
        latest: dict[str, Progress] = {}

        def on_progress(progress: Progress) -> None:
            stamped = progress.model_copy(update={"step_id": step_id})
            latest["value"] = stamped
            writer.write(stamped)

        def run() -> None:
            started_at = utc_now_iso()
            error: str | None = None
            stopped_reason: str | None = None
            try:
                result = fn(*args, on_progress=on_progress, cancel=token, **kwargs)
                if isinstance(result, str) and result:
                    stopped_reason = result
            except BaseException as exc:  # noqa: BLE001 - surfaced via the progress file
                error = f"{type(exc).__name__}: {exc}"
            final = latest.get(
                "value", Progress(step_id=step_id, done=0, total=0, started_at=started_at)
            )
            final = final.model_copy(
                update={
                    "finished_at": utc_now_iso(),
                    "error": error,
                    "cancelled": token.is_cancelled,
                    "stopped_reason": stopped_reason,
                }
            )
            writer.write(final, force=True)
            with self._lock:
                self._threads.pop(step_id, None)

        thread = threading.Thread(target=run, name=f"step-{step_id}", daemon=True)
        with self._lock:
            self._threads[step_id] = thread
            self._tokens[step_id] = token
        thread.start()

    def cancel(self, step_id: str) -> None:
        """Request cancellation of a step if it has a known token."""
        with self._lock:
            token = self._tokens.get(step_id)
        if token is not None:
            token.cancel()

    def status(self, step_id: str) -> Progress | None:
        """Return the last persisted progress for a step, or None when absent."""
        path = self._path(step_id)
        if not path.is_file():
            return None
        return Progress.model_validate_json(path.read_text())

    def is_running(self, step_id: str) -> bool:
        """Return whether the step's thread is still alive."""
        with self._lock:
            thread = self._threads.get(step_id)
        return thread is not None and thread.is_alive()

    def is_cancelling(self, step_id: str) -> bool:
        """Return whether a step was asked to stop but its thread is still alive.

        True while a cancel has been requested and the daemon thread is still
        letting its in-flight work finish; False once the thread ends or when
        no cancel was requested.
        """
        with self._lock:
            token = self._tokens.get(step_id)
            thread = self._threads.get(step_id)
        if token is None or thread is None:
            return False
        return token.is_cancelled and thread.is_alive()

    def running_steps(self) -> list[str]:
        """Return the ids of steps whose threads are still alive."""
        with self._lock:
            items = list(self._threads.items())
        return sorted(step_id for step_id, thread in items if thread.is_alive())

    def stale(self, step_id: str) -> bool:
        """Return whether a step has an unfinished file but no live thread.

        A stale step is one whose process exited mid-run: its progress file was
        never marked finished and no thread is carrying it forward.
        """
        if self.is_running(step_id):
            return False
        status = self.status(step_id)
        return status is not None and status.finished_at is None
