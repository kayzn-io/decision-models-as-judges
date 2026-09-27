"""Tests for the Technical notes panel: the one place provenance is told in full.

The panel lives two levels deep, inside About these conversations on the Run
page, and appears nowhere else. These tests pin its copy and that isolation.
"""

import re
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.ui.notes import TECHNICAL_NOTES_TITLE, technical_notes
from decision_judges.ui.steps import Provenance

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"

_PROV = Provenance(
    agent_model="openai/gpt-4.1",
    user_model="openai/gpt-4o-mini",
    tau_bench_ref="59a200c6d575d595120f1cb70fea53cef0632f6b",
)


def test_title_is_the_neutral_label() -> None:
    """The label signals depth without emphasis."""
    assert TECHNICAL_NOTES_TITLE == "Technical notes"


def test_notes_tell_the_full_provenance_when_prov_is_given() -> None:
    """With a record the notes name the source, the tasks, the grading, and the rule."""
    text = technical_notes(_PROV)
    assert "Sierra Research" in text
    assert "115" in text
    assert "grading code" in text
    assert "without confirming" in text
    assert "59a200c6" in text


def test_notes_include_the_model_names_when_present() -> None:
    """The two models appear so a reader can reproduce the setup."""
    text = technical_notes(_PROV)
    assert "openai/gpt-4.1" in text
    assert "openai/gpt-4o-mini" in text


def test_notes_omit_model_names_when_prov_is_none() -> None:
    """Without a record the core story stays, but the model names are dropped."""
    text = technical_notes(None)
    assert "openai/gpt-4.1" not in text
    assert "openai/gpt-4o-mini" not in text
    assert "Sierra Research" in text
    assert "115" in text
    assert "grading code" in text


def test_notes_close_on_the_judges_never_seeing_the_ground_truth() -> None:
    """The closing line ties the ground truth to the judges."""
    text = technical_notes(_PROV).strip().lower()
    assert "ground truth" in text
    assert text.endswith("never shown it.")


def test_notes_stay_under_the_word_budget() -> None:
    """The panel is a few short paragraphs, not a wall of text."""
    assert len(technical_notes(_PROV).split()) < 220


def test_notes_prose_carries_no_stray_numbers() -> None:
    """Once the record's own values are removed, only 115 and 230 remain."""
    text = technical_notes(_PROV)
    for token in (
        _PROV.tau_bench_ref,
        _PROV.tau_bench_ref[:12],
        _PROV.agent_model,
        _PROV.user_model,
    ):
        text = text.replace(token, "")
    numbers = set(re.findall(r"\d+", text))
    assert numbers <= {"115", "230"}, numbers


_OTHER_PAGES = (
    ("overview", False),
    ("trajectories", False),
    ("gates", False),
    ("label", True),
    ("live", True),
)


def _page_text(monkeypatch: pytest.MonkeyPatch, base: Path, module: str, *, local: bool) -> str:
    """Render one page module against a fresh fixture copy and return all its text."""
    root = base / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    if local:
        monkeypatch.setenv("JUDGES_LOCAL", "1")
    else:
        monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    if module == "live":
        monkeypatch.setattr("decision_judges.ui.live.laya_available", lambda _study: False)
    script = base / f"{module}_page.py"
    script.write_text(
        f"from decision_judges.ui.pages.{module} import render\n\nrender()\n", encoding="utf-8"
    )
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert not at.exception, module
    nodes = [*at.markdown, *at.caption, *at.get("html"), *at.info]
    return "\n".join(str(node.value) for node in nodes)


def test_technical_notes_text_appears_on_no_other_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Overview, Trajectories, Experiments, Label, and Live never show the notes."""
    for module, local in _OTHER_PAGES:
        text = _page_text(monkeypatch, tmp_path / module, module, local=local)
        assert "Sierra Research" not in text, module
        assert TECHNICAL_NOTES_TITLE not in text, module
