"""Tests for the background StepRunner and its on-disk progress state."""

import time
from pathlib import Path

import pytest

from decision_judges import runner as runner_mod
from decision_judges.progress import CancelToken, Progress, utc_now_iso
from decision_judges.runner import StepAlreadyRunning, StepRunner


def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Poll a predicate until it holds or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def _one_shot(*, on_progress, cancel) -> None:
    """A step that reports a single snapshot and returns."""
    on_progress(Progress(step_id="x", done=1, total=1, started_at=utc_now_iso()))


def _boom(*, on_progress, cancel) -> None:
    """A step that reports then raises."""
    on_progress(Progress(step_id="x", done=1, total=2, started_at=utc_now_iso()))
    raise RuntimeError("kaboom")


def _stops_early(*, on_progress, cancel) -> str:
    """A step that reports once then returns a short stop reason."""
    on_progress(Progress(step_id="x", done=1, total=3, started_at=utc_now_iso()))
    return "aborted: x"


def _loop_until_cancel(*, on_progress, cancel: CancelToken) -> None:
    """A step that reports until it is asked to stop."""
    index = 0
    while not cancel.is_cancelled:
        index += 1
        on_progress(Progress(step_id="x", done=index, total=0, started_at=utc_now_iso()))
        time.sleep(0.01)


def _slow_after_cancel(*, on_progress, cancel: CancelToken) -> None:
    """A step that keeps running for a moment after cancel, like in-flight work."""
    on_progress(Progress(step_id="x", done=1, total=1, started_at=utc_now_iso()))
    while not cancel.is_cancelled:
        time.sleep(0.01)
    time.sleep(0.3)


def test_start_writes_progress_file_ending_with_finished_at(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _one_shot)
    _wait_until(lambda: not runner.is_running("step"))

    status = runner.status("step")
    assert status is not None
    assert status.step_id == "step"
    assert status.finished_at is not None
    assert status.error is None
    assert status.cancelled is False


def test_returned_string_becomes_stopped_reason(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _stops_early)
    _wait_until(lambda: not runner.is_running("step"))

    status = runner.status("step")
    assert status is not None
    assert status.stopped_reason == "aborted: x"
    assert status.finished_at is not None
    assert status.error is None
    assert status.cancelled is False


def test_returned_none_leaves_stopped_reason_none(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _one_shot)
    _wait_until(lambda: not runner.is_running("step"))

    status = runner.status("step")
    assert status is not None
    assert status.stopped_reason is None


def test_exception_surfaces_in_error_and_finishes(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _boom)
    _wait_until(lambda: not runner.is_running("step"))

    status = runner.status("step")
    assert status is not None
    assert status.error == "RuntimeError: kaboom"
    assert status.finished_at is not None


def test_cancel_flips_cancelled_and_stops(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _loop_until_cancel)
    _wait_until(lambda: runner.status("step") is not None)

    runner.cancel("step")
    _wait_until(lambda: not runner.is_running("step"))

    status = runner.status("step")
    assert status is not None
    assert status.cancelled is True
    assert status.finished_at is not None


def test_is_cancelling_is_true_until_the_thread_ends(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _slow_after_cancel)
    _wait_until(lambda: runner.status("step") is not None)
    assert runner.is_cancelling("step") is False

    runner.cancel("step")
    _wait_until(lambda: runner.is_cancelling("step"))
    assert runner.is_running("step") is True

    _wait_until(lambda: not runner.is_running("step"))
    assert runner.is_cancelling("step") is False


def test_is_cancelling_is_false_for_an_unknown_step(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    assert runner.is_cancelling("missing") is False


def test_starting_a_running_step_raises(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    runner.start("step", _loop_until_cancel)
    _wait_until(lambda: runner.is_running("step"))

    with pytest.raises(StepAlreadyRunning):
        runner.start("step", _loop_until_cancel)

    assert runner.running_steps() == ["step"]
    runner.cancel("step")
    _wait_until(lambda: not runner.is_running("step"))


def test_status_reads_back_after_completion(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    assert runner.status("missing") is None

    runner.start("step", _one_shot)
    _wait_until(lambda: not runner.is_running("step"))
    assert runner.status("step") is not None


def test_stale_detection_for_hand_written_unfinished_file(tmp_path: Path) -> None:
    runner = StepRunner(tmp_path)
    path = tmp_path / "orphan.json"
    unfinished = Progress(step_id="orphan", done=3, total=10, started_at=utc_now_iso())
    path.write_text(unfinished.model_dump_json(indent=2))

    assert runner.is_running("orphan") is False
    assert runner.stale("orphan") is True

    finished = unfinished.model_copy(update={"finished_at": utc_now_iso()})
    path.write_text(finished.model_dump_json(indent=2))
    assert runner.stale("orphan") is False


def test_writer_throttles_writes_under_a_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[int] = []
    real_replace = runner_mod.os.replace

    def counting_replace(src: object, dst: object) -> None:
        replaced.append(1)
        real_replace(src, dst)

    monkeypatch.setattr(runner_mod.os, "replace", counting_replace)

    def fast_reporter(*, on_progress, cancel) -> None:
        for index in range(100):
            on_progress(Progress(step_id="x", done=index + 1, total=100, started_at=utc_now_iso()))

    runner = StepRunner(tmp_path)
    runner.start("step", fast_reporter)
    _wait_until(lambda: not runner.is_running("step"))

    # A fast burst of 100 reports collapses to a couple of throttled writes plus
    # the forced final flush, far below one write per report.
    assert len(replaced) <= 6
    status = runner.status("step")
    assert status is not None
    assert status.finished_at is not None
