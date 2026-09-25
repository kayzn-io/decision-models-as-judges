"""Trajectories page tests driven through AppTest over a thin script."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.trajectories import render\n\nrender()\n"


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    """Run the trajectories page against the fixture tree in shared mode.

    AppTest.from_function re-executes only the function body, dropping the
    page module's imports, so the page runs from a thin script instead.
    """
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    script = tmp_path / "run_trajectories.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def _rewards(at: AppTest) -> dict[str, str]:
    """Return the rendered metrics as a label-to-value mapping."""
    return {metric.label: metric.value for metric in at.metric}


def _captions(at: AppTest) -> str:
    """Return the concatenated caption text for substring assertions."""
    return "\n".join(caption.value for caption in at.caption)


def test_trajectories_default_shows_pass_and_both_judges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    assert not at.exception

    assert _rewards(at)["Reward"] == "PASS"

    verdict_frame = next(df.value for df in at.dataframe if "judge_id" in df.value.columns)
    assert set(verdict_frame["judge_id"]) == {"code", "fake"}


def test_trajectories_retail_1_shows_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path)
    task_box = next(box for box in at.selectbox if box.key == "task")
    task_box.set_value("retail-1").run()

    assert not at.exception
    assert _rewards(at)["Reward"] == "FAIL"


def test_trajectories_shows_inline_step_scores_for_retail_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    assert not at.exception

    captions = _captions(at)
    assert "necessary" in captions
    assert "%" in captions


def test_trajectories_no_step_block_for_retail_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    task_box = next(box for box in at.selectbox if box.key == "task")
    task_box.set_value("retail-1").run()

    assert not at.exception
    assert "necessary" not in _captions(at)
