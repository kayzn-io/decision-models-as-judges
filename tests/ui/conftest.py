"""Fixtures pointing the app at the committed cache fixture tree."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = str(_REPO_ROOT / "decision_judges" / "ui" / "app.py")
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"


@pytest.fixture
def ui_root() -> Path:
    """Return the path to the committed UI cache fixture tree."""
    return _UI_ROOT


@pytest.fixture
def app_test(monkeypatch: pytest.MonkeyPatch, ui_root: Path) -> AppTest:
    """Return an AppTest bound to the fixture tree in shared (non-local) mode."""
    monkeypatch.setenv("JUDGES_ROOT", str(ui_root))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    return AppTest.from_file(_APP, default_timeout=30)
