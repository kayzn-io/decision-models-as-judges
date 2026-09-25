"""Run page tests driven through AppTest over a thin script."""

import shutil
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.progress import Progress, utc_now_iso
from decision_judges.runner import StepRunner
from decision_judges.ui import keys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.run import render\n\nrender()\n"
_FAKE_KEY = "sk-fake-run-key-XYZ"
_PAID_IDS = ("run-agent", "judge-outcome", "gates", "label")


def _script(tmp_path: Path) -> str:
    """Write the thin page script and return its path."""
    script = tmp_path / "run_page.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return str(script)


def _local_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, AppTest]:
    """Copy the fixture tree to a writable root and build a local-mode AppTest."""
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    return root, AppTest.from_file(_script(tmp_path), default_timeout=60)


def _run_buttons(at: AppTest) -> list[str]:
    """Return the keys of every step Run button."""
    return [button.key for button in at.button if button.key and button.key.startswith("run_")]


def _texts(at: AppTest) -> list[str]:
    """Return every rendered markdown and caption value for substring assertions."""
    return [block.value for block in [*at.markdown, *at.caption]]


def _await_finished(root: Path, step_id: str, timeout: float = 30.0) -> Progress:
    """Poll the on-disk progress until the step is finished, then return it."""
    runner = StepRunner(root / "results" / ".progress")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.status(step_id)
        if status is not None and status.finished_at is not None:
            return status
        time.sleep(0.2)
    raise AssertionError(f"step {step_id} did not finish within {timeout}s")


def _await_running(root: Path, step_id: str, timeout: float = 10.0) -> None:
    """Poll the on-disk progress until the step has reported at least once."""
    runner = StepRunner(root / "results" / ".progress")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runner.status(step_id) is not None:
            return
        time.sleep(0.1)
    raise AssertionError(f"step {step_id} never reported progress within {timeout}s")


def _state_files(root: Path) -> int:
    """Count serialized state files under the cache."""
    state_dir = root / "cache" / "state"
    return sum(1 for path in state_dir.rglob("*.json") if path.is_file())


def test_run_shared_mode_shows_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    at = AppTest.from_file(_script(tmp_path), default_timeout=30).run()

    assert not at.exception
    assert any("JUDGES_LOCAL=1" in info.value for info in at.info)


def test_run_renders_eight_cards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert len(_run_buttons(at)) == 8


def test_paid_buttons_disabled_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    for step_id in _PAID_IDS:
        assert at.button(key=f"run_{step_id}").disabled is True, step_id
    assert any("OpenRouter key" in text for text in _texts(at))


def test_entering_a_key_enables_paid_buttons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    assert not at.exception
    assert at.button(key="run_judge-outcome").disabled is False


def test_running_serialize_writes_states_and_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    before = _state_files(root)

    at.button(key="run_serialize").click().run()
    status = _await_finished(root, "serialize")

    assert status.error is None
    assert _state_files(root) > before
    at.run()
    assert any("done" in text.lower() for text in _texts(at))


def test_running_analyze_writes_g6_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    at.button(key="run_analyze").click().run()
    status = _await_finished(root, "analyze")

    assert status.error is None
    assert (root / "results" / "g6_summary.md").is_file()


def test_slow_step_shows_stop_and_cancels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def slow_fake(paths: object, ctx: object) -> None:
        started = utc_now_iso()
        for index in range(200):
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return
            if ctx.on_progress is not None:
                ctx.on_progress(
                    Progress(
                        step_id="judge-outcome",
                        done=index,
                        total=200,
                        started_at=started,
                        last_item=f"item {index}",
                    )
                )
            time.sleep(0.1)

    monkeypatch.setattr("decision_judges.ui.steps._run_judge_outcome", slow_fake)
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    at.button(key="run_judge-outcome").click().run()
    _await_running(root, "judge-outcome")
    at.run()

    assert not at.exception
    at.button(key="stop_judge-outcome").click().run()
    status = _await_finished(root, "judge-outcome")
    assert status.cancelled is True


def test_resume_banner_shows_for_a_stale_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    progress_dir = root / "results" / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    stale = Progress(step_id="serialize", done=1, total=2, started_at=utc_now_iso())
    (progress_dir / "serialize.json").write_text(stale.model_dump_json(indent=2), encoding="utf-8")

    at.run()

    assert not at.exception
    assert any("stopped while the app was closed" in warning.value for warning in at.warning)


def test_run_page_never_renders_the_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    assert not at.exception
    assert all(_FAKE_KEY not in str(text) for text in _texts(at))
