"""Tests for the shared guided-thread components."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _rendered(at: AppTest) -> str:
    """Return concatenated markdown and html text for substring assertions."""
    return "\n".join(node.value for node in [*at.get("markdown"), *at.get("html")])


def _captions(at: AppTest) -> str:
    """Return concatenated caption text for substring assertions."""
    return "\n".join(caption.value for caption in at.caption)


def test_flow_context_renders_active_station(
    monkeypatch: pytest.MonkeyPatch, ui_root: Path
) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(ui_root))
    script = (
        "from decision_judges.ui import components, data\n"
        "from decision_judges.ui.flow import Station\n"
        "components.flow_context(data.Paths.from_env(), Station.verdicts)\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    rendered = _rendered(at)
    assert 'class="flow-strip"' in rendered
    assert 'class="station active"' in rendered


def test_empty_state_with_run_step_points_at_run_page() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.empty_state('The frontier trades cost for accuracy.', "
        "'judges analyze --gate g5', run_step=4)\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    markdown = "\n".join(md.value for md in at.markdown)
    assert "Run step 4 on the Run page" in markdown
    assert "/run" in markdown
    assert "to produce this" in markdown
    assert any(expander.label == "Command line" for expander in at.expander)
    assert any("judges analyze --gate g5" in code.value for code in at.code)


def test_empty_state_without_run_step_shows_command_inline() -> None:
    script = (
        "from decision_judges.ui import components\ncomponents.empty_state('what', 'the-command')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    assert any("the-command" in code.value for code in at.code)
    assert not any(expander.label == "Command line" for expander in at.expander)


def test_how_to_read_renders_rows() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.how_to_read([('Accuracy', 'share matching ground truth')])\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    assert any(expander.label == "How to read this" for expander in at.expander)
    markdown = "\n".join(md.value for md in at.markdown)
    assert "Accuracy" in markdown
    assert "share matching ground truth" in markdown


def test_next_link_renders_link_and_hint() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.next_link('Gates', '/gates', 'See each gate in turn.')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    markdown = "\n".join(md.value for md in at.markdown)
    assert "Next:" in markdown
    assert "[Gates](/gates)" in markdown
    assert "See each gate in turn." in _captions(at)


def test_why_line_renders_muted_note() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.why_line('It orients you before the details.')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    captions = _captions(at)
    assert "Why this page exists" in captions
    assert "It orients you before the details." in captions


def test_keyboard_hint_renders_caption() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.keyboard_hint('Press 1 to 8 to pick a label.')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    assert "Press 1 to 8 to pick a label." in _captions(at)
