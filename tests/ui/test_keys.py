"""Tests for the session key store and its survival across page navigation."""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.ui import keys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_APP = str(_REPO_ROOT / "decision_judges" / "ui" / "app.py")
_FAKE_KEY = "sk-fake-persist-key-XYZ"

_UNIT_SCRIPT = (
    "import streamlit as st\n"
    "from decision_judges.ui import keys\n"
    "step = st.session_state.get('_step')\n"
    "if step == 'set':\n"
    "    st.session_state[keys.WIDGET_KEY] = 'sk-unit-key'\n"
    "    keys.store_widget_value()\n"
    "st.write('KEY=' + (keys.get_key() or 'NONE'))\n"
)

_PAGE_SCRIPT = (
    "import streamlit as st\n"
    "from decision_judges.ui import keys\n"
    "from decision_judges.ui.pages import overview, run, trajectories\n"
    "keys.render_key_field()\n"
    "pages = {'overview': overview.render, 'run': run.render, "
    "'trajectories': trajectories.render}\n"
    "pages[st.session_state.get('_page', 'overview')]()\n"
)


def _write(tmp_path: Path, name: str, body: str) -> str:
    """Write a harness script and return its path."""
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return str(script)


def _local_page_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, AppTest]:
    """Copy the fixture tree to a writable root and build a local-mode page harness."""
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    return root, AppTest.from_file(_write(tmp_path, "pages.py", _PAGE_SCRIPT), default_timeout=60)


def _texts(at: AppTest) -> list[str]:
    """Return every rendered markdown and caption value for substring assertions."""
    return [block.value for block in [*at.markdown, *at.caption]]


def test_get_key_is_none_before_the_callback_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    at = AppTest.from_file(_write(tmp_path, "unit.py", _UNIT_SCRIPT), default_timeout=30).run()

    assert not at.exception
    assert any(node.value == "KEY=NONE" for node in at.markdown)


def test_get_key_returns_the_value_after_the_callback_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    at = AppTest.from_file(_write(tmp_path, "unit.py", _UNIT_SCRIPT), default_timeout=30).run()
    at.session_state["_step"] = "set"
    at.run()

    assert not at.exception
    assert at.session_state[keys.SESSION_KEY] == "sk-unit-key"
    assert any(node.value == "KEY=sk-unit-key" for node in at.markdown)


def test_no_key_badge_and_hint_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_page_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert any("No key yet" in text for text in _texts(at))
    assert any("Paid steps need it" in text for text in _texts(at))


def test_run_paid_buttons_disabled_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_page_app(monkeypatch, tmp_path)
    at.session_state["_page"] = "run"
    at.run()

    assert not at.exception
    assert at.button(key="run_judge-outcome").disabled is True
    assert any("sidebar (top left)" in text for text in _texts(at))


def test_key_survives_navigation_and_enables_paid_buttons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_page_app(monkeypatch, tmp_path)
    at.run()

    at.text_input(key=keys.WIDGET_KEY).set_value(_FAKE_KEY).run()
    assert at.session_state[keys.SESSION_KEY] == _FAKE_KEY
    assert any("Key set for this session" in text for text in _texts(at))

    at.session_state["_page"] = "trajectories"
    at.run()
    assert not at.exception
    assert at.session_state[keys.SESSION_KEY] == _FAKE_KEY
    assert any("Key set for this session" in text for text in _texts(at))

    at.session_state["_page"] = "run"
    at.run()
    assert not at.exception
    assert any("Key set for this session" in text for text in _texts(at))
    assert at.button(key="run_judge-outcome").disabled is False


def test_key_text_never_appears_in_any_element(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_page_app(monkeypatch, tmp_path)
    at.run()
    at.text_input(key=keys.WIDGET_KEY).set_value(_FAKE_KEY).run()
    at.session_state["_page"] = "run"
    at.run()

    assert not at.exception
    assert all(_FAKE_KEY not in str(text) for text in _texts(at))


def test_clear_empties_both_the_store_and_the_widget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_page_app(monkeypatch, tmp_path)
    at.run()
    at.text_input(key=keys.WIDGET_KEY).set_value(_FAKE_KEY).run()
    assert at.session_state[keys.SESSION_KEY] == _FAKE_KEY

    at.button(key="openrouter_key_clear").click().run()

    assert not at.exception
    assert at.session_state[keys.SESSION_KEY] == ""
    assert any("No key yet" in text for text in _texts(at))


def test_full_app_renders_the_key_field_in_local_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    at = AppTest.from_file(_APP, default_timeout=60).run()

    assert not at.exception
    assert any("No key yet" in block.value for block in at.sidebar.markdown)


def test_require_key_hint_names_the_sidebar() -> None:
    hint = keys.require_key_hint()

    assert "sidebar" in hint
    assert hint.endswith("to run this step.")
