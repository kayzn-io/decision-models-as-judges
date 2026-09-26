"""Every page renders on a config-only study tree with empty states, no exception."""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_PAGES = ("overview", "trajectories", "gates", "label", "live")


def _config_only_tree(dest: Path) -> Path:
    """Copy only the config directory of the fixture tree into a fresh root."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_UI_ROOT / "config", dest / "config")
    return dest


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str) -> AppTest:
    """Run one page against a config-only tree in shared (non-local) mode."""
    root = _config_only_tree(tmp_path / "empty")
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    script = tmp_path / f"run_{page}.py"
    script.write_text(
        f"from decision_judges.ui.pages.{page} import render\n\nrender()\n", encoding="utf-8"
    )
    return AppTest.from_file(str(script), default_timeout=30).run()


def _text(at: AppTest) -> str:
    """Return concatenated caption, code, and info text for substring assertions."""
    parts = [c.value for c in at.caption]
    parts += [c.value for c in at.code]
    parts += [info.value for info in at.info]
    return "\n".join(parts)


@pytest.mark.parametrize("page", _PAGES)
def test_page_renders_without_exception_on_empty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str
) -> None:
    at = _run(monkeypatch, tmp_path, page)
    assert not at.exception


def test_overview_shows_frontier_empty_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, "overview")
    assert any("judges analyze --gate g5" in code.value for code in at.code)


def _html(at: AppTest) -> str:
    """Return concatenated html and markdown text for substring assertions."""
    return "\n".join(node.value for node in [*at.get("html"), *at.get("markdown")])


def test_overview_hero_and_start_cta_on_empty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, "overview")
    assert not at.exception
    rendered = _html(at)
    assert 'class="hero-title"' in rendered
    assert "shows every number's source" in rendered
    starts = [node.value for node in at.get("html") if "Get the conversations first" in node.value]
    assert len(starts) == 1
    assert 'href="/run"' in starts[0]
    assert "Step 1 of 7 next" in _text(at)


def test_overview_tiles_render_on_empty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _run(monkeypatch, tmp_path, "overview")
    assert not at.exception
    rendered = _html(at)
    # Every tile is hollow on an empty tree, and each links to its experiment.
    assert '<div class="tile todo" data-gate="g1">' in rendered
    assert rendered.count('class="tile ') == 10
    assert 'href="/gates?gate=g1"' in rendered


def test_gates_show_command_empty_states(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, "gates")
    assert at.code, "empty gate tabs should show the CLI command that fills them"
    assert any("judges" in code.value for code in at.code)


def test_trajectories_notes_no_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, "trajectories")
    assert "No agent runs are available yet." in _text(at)


def test_label_notes_local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _run(monkeypatch, tmp_path, "label")
    assert "JUDGES_LOCAL=1" in _text(at)
