"""Label page tests driven through AppTest over a thin script."""

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.label import render\n\nrender()\n"


def _script(tmp_path: Path) -> str:
    """Write the thin page script and return its path."""
    script = tmp_path / "run_label.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return str(script)


def _texts(at: AppTest) -> str:
    """Return the concatenated markdown text for substring assertions."""
    return "\n".join(block.value for block in at.markdown)


def test_label_shared_mode_shows_info_and_no_radio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    at = AppTest.from_file(_script(tmp_path), default_timeout=30).run()

    assert not at.exception
    assert len(at.radio) == 0
    assert any("JUDGES_LOCAL=1" in info.value for info in at.info)


def _local_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, AppTest]:
    """Copy the fixture tree to a writable root and build a local-mode AppTest."""
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    return root, AppTest.from_file(_script(tmp_path), default_timeout=30)


def test_label_local_shows_radio_and_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert len(at.radio) == 1
    assert list(at.radio[0].options) == [
        "Wrong or missing action",
        "Unrequested write",
        "Skipped confirmation",
        "Identity not verified",
        "Wrong arguments",
        "Premature end",
        "Policy misapplied",
        "Other",
    ]
    assert "0 of 50 labeled" in _texts(at)


def test_label_defaults_to_no_selection_and_disables_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert at.radio[0].value is None
    save = next(button for button in at.button if button.label == "Save label")
    assert save.disabled is True


def test_label_shows_description_of_selected_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from decision_judges.labels import TAXONOMY_DESCRIPTIONS

    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.radio[0].set_value("skipped_confirmation").run()

    assert not at.exception
    captions = "\n".join(caption.value for caption in at.caption)
    assert TAXONOMY_DESCRIPTIONS["skipped_confirmation"] in captions
    save = next(button for button in at.button if button.label == "Save label")
    assert save.disabled is False


def test_label_save_appends_line_and_updates_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception

    at.radio[0].set_value("skipped_confirmation").run()
    at.button[0].click().run()
    assert not at.exception

    jsonl = root / "data" / "labels" / "taxonomy.jsonl"
    assert jsonl.is_file()
    lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["variant"] == "baseline"
    assert record["task_id"] == "retail-1"
    assert record["label"] == "skipped_confirmation"
    assert record["labeler"] == "owner"

    assert "1 of 50 labeled" in _texts(at)


def _rendered(at: AppTest) -> str:
    """Return concatenated markdown and html text for substring assertions."""
    return "\n".join(node.value for node in [*at.get("markdown"), *at.get("html")])


def test_label_has_flow_strip_and_next_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception
    assert 'class="flow-strip"' in _rendered(at)
    assert '<a href="/live" class="in-app-link">Next: Watch the judges work</a>' in _rendered(at)


def test_label_intro_visible_before_first_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception
    assert any(expander.label == "Before you label" for expander in at.expander)
    assert "first cause in the conversation" in _texts(at)


def test_label_tally_appears_after_saving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    assert not at.get("vega_lite_chart")

    at.radio[0].set_value("skipped_confirmation").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.get("vega_lite_chart")
