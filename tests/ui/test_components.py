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
    anchors = [node.value for node in at.get("html") if "Run step 4 on the Run page" in node.value]
    assert len(anchors) == 1
    assert 'href="/run"' in anchors[0]
    assert "target" not in anchors[0]
    assert "to produce this" in _captions(at)
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


def test_next_link_renders_same_tab_link_and_hint() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.next_link('Gates', '/gates', 'See each gate in turn.')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    anchors = [node.value for node in at.get("html") if "Next: Gates" in node.value]
    assert len(anchors) == 1
    assert 'href="/gates"' in anchors[0]
    assert "target" not in anchors[0]
    assert "See each gate in turn." in _captions(at)


def test_page_link_with_query_renders_anchor_without_target() -> None:
    script = (
        "from decision_judges.ui import components\n"
        "components.page_link('/trajectories', 'See one example', "
        "query={'variant': 'baseline', 'task': 'retail-0'})\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    anchors = [node.value for node in at.get("html") if "See one example" in node.value]
    assert len(anchors) == 1
    assert "target" not in anchors[0]
    assert 'href="/trajectories?' in anchors[0]
    assert "variant=baseline" in anchors[0]
    assert "task=retail-0" in anchors[0]
    assert not at.get("page_link")


def test_page_link_unregistered_falls_back_to_anchor_without_raising() -> None:
    script = (
        "from decision_judges.ui import components\ncomponents.page_link('/run', 'Go to Run')\n"
    )
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception
    anchors = [node.value for node in at.get("html") if "Go to Run" in node.value]
    assert len(anchors) == 1
    assert 'href="/run"' in anchors[0]
    assert "target" not in anchors[0]
    assert not at.get("page_link")


def test_page_link_switches_pages_for_registered_page(
    monkeypatch: pytest.MonkeyPatch, ui_root: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    app = str(repo_root / "decision_judges" / "ui" / "app.py")
    monkeypatch.setenv("JUDGES_ROOT", str(ui_root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception
    labels = [link.proto.label for link in at.get("page_link")]
    assert "Next: Run" in labels


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
