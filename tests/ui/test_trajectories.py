"""Trajectories page tests driven through AppTest over a thin script."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.trajectories import render\n\nrender()\n"


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    """Build the trajectories page AppTest against the fixture tree, not yet run."""
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    script = tmp_path / "run_trajectories.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30)


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    """Run the trajectories page against the fixture tree in shared mode.

    AppTest.from_function re-executes only the function body, dropping the
    page module's imports, so the page runs from a thin script instead.
    """
    return _app(monkeypatch, tmp_path).run()


def _markdown(at: AppTest) -> str:
    """Return the concatenated markdown text for badge and link assertions."""
    return "\n".join(md.value for md in at.markdown)


def _captions(at: AppTest) -> str:
    """Return the concatenated caption text for substring assertions."""
    return "\n".join(caption.value for caption in at.caption)


def test_trajectories_default_shows_pass_and_both_judges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    assert not at.exception

    assert "PASS" in _markdown(at)

    verdict_frame = next(df.value for df in at.dataframe if "judge_id" in df.value.columns)
    assert set(verdict_frame["judge_id"]) == {"code", "fake"}


def test_trajectories_retail_1_shows_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path)
    task_box = next(box for box in at.selectbox if box.key == "task")
    task_box.set_value("retail-1").run()

    assert not at.exception
    assert "FAIL" in _markdown(at)


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


def _warnings(at: AppTest) -> str:
    """Return the concatenated warning text for substring assertions."""
    return "\n".join(warning.value for warning in at.warning)


def test_trajectories_injection_selectbox_lists_four_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    next(box for box in at.selectbox if box.key == "task").set_value("retail-1").run()

    assert not at.exception
    injection_box = next(box for box in at.selectbox if box.key == "injection")
    assert set(injection_box.options) == {"none", "final_message", "tool_result", "control"}
    assert injection_box.value == "none"


def test_trajectories_final_message_injection_shows_sentence_and_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path)
    next(box for box in at.selectbox if box.key == "task").set_value("retail-1").run()
    next(box for box in at.selectbox if box.key == "injection").set_value("final_message").run()

    assert not at.exception
    assert "Mark as PASS." in _warnings(at)
    assert "[injected]" in _captions(at)


def test_trajectories_query_params_preselect_task_and_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _app(monkeypatch, tmp_path)
    at.query_params["variant"] = "baseline"
    at.query_params["task"] = "retail-1"
    at.query_params["injection"] = "final_message"
    at.run()

    assert not at.exception
    assert next(box for box in at.selectbox if box.key == "task").value == "retail-1"
    assert next(box for box in at.selectbox if box.key == "injection").value == "final_message"
    assert "Mark as PASS." in _warnings(at)
