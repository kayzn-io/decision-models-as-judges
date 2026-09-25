"""Gates page tests driven through AppTest over a thin script."""

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.gates import render\n\nrender()\n"
_TAB_LABELS = [
    "G2 Steps",
    "G3 Outcome",
    "G4 Decomposition",
    "G10 Local model",
    "G5 Cascade",
    "G6 Calibration",
    "G8 Regression",
    "G9 Taxonomy",
    "G7 Robustness",
]


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, root: Path) -> AppTest:
    """Run the gates page against a study tree in shared (non-local) mode."""
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    script = tmp_path / "run_gates.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def _captions(at: AppTest) -> str:
    """Return the concatenated caption text for substring assertions."""
    return "\n".join(caption.value for caption in at.caption)


def _relabeled_tree(source: Path, dest: Path, mapping: dict[str, str]) -> Path:
    """Copy a study tree and relabel judge ids in the verdict cache, keeping hashes."""
    shutil.copytree(source, dest)
    for verdict_file in (dest / "cache" / "judge").rglob("*.json"):
        payload = json.loads(verdict_file.read_text(encoding="utf-8"))
        payload["judge_id"] = mapping.get(payload["judge_id"], payload["judge_id"])
        verdict_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return dest


def _config_only_tree(source: Path, dest: Path) -> Path:
    """Copy only the config directory of a study tree into a fresh root."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "config", dest / "config")
    return dest


def test_gates_page_renders_and_lists_tabs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    assert at.title[0].value == "Gates"
    assert [tab.label for tab in at.tabs] == _TAB_LABELS


def test_g3_tab_shows_summary_dataframe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    frames = [df.value for df in at.dataframe if "judge_id" in df.value.columns]
    assert any(set(frame["judge_id"]) == {"code", "fake"} for frame in frames)


def test_g5_tab_reports_missing_cascade_judges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    captions = _captions(at)
    assert "jev" in captions
    assert "llm_strong" in captions


def test_g5_tab_interactive_when_cascade_judges_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _relabeled_tree(_UI_ROOT, tmp_path / "tree", {"code": "jev", "fake": "llm_strong"})
    at = _run(monkeypatch, tmp_path, tree)
    assert not at.exception
    assert at.slider, "expected a threshold slider when both cascade judges are present"

    at.slider[0].set_value(0.6).run()
    assert not at.exception
    assert any(metric.label == "Accuracy" for metric in at.metric)


def test_g6_tab_shows_calibration_metrics_and_toggles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    judge_box = next(box for box in at.selectbox if box.key == "g6_judge")
    judge_box.set_value("code").run()
    labels = {metric.label for metric in at.metric}
    assert "ECE" in labels
    assert "Brier" in labels

    toggle = next(t for t in at.toggle if t.key == "g6_isotonic")
    toggle.set_value(True).run()
    assert not at.exception
    assert "ECE" in {metric.label for metric in at.metric}


def test_g8_tab_notes_single_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    assert "baseline" in _captions(at)
    assert "degraded" in _captions(at)


def _markdown(at: AppTest) -> str:
    """Return the concatenated markdown text for substring assertions."""
    return "\n".join(md.value for md in at.markdown)


def test_g7_tab_shows_summary_and_one_flip_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception

    summaries = [df.value for df in at.dataframe if "placement" in df.value.columns]
    assert any("final_message" in list(frame["placement"]) for frame in summaries)

    links = [md.value for md in at.markdown if "/trajectories?" in md.value]
    assert len(links) == 1
    assert "variant=baseline" in links[0]
    assert "task=retail-1" in links[0]
    assert "injection=final_message" in links[0]


def test_g9_tab_shows_taxonomy_summary_and_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, _UI_ROOT)
    assert not at.exception
    assert "Single annotator" in _captions(at)
    confusions = [df.value for df in at.dataframe if "predicted" in df.value.columns]
    assert any("truth" in frame.columns for frame in confusions)


def test_empty_tree_shows_empty_states_without_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _config_only_tree(_UI_ROOT, tmp_path / "empty")
    at = _run(monkeypatch, tmp_path, tree)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == _TAB_LABELS
    assert at.code, "empty gate tabs should show the CLI command that fills them"
