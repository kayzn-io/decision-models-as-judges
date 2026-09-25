"""Guard test ensuring the UI uses the width parameter Streamlit now expects."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_DIR = _REPO_ROOT / "decision_judges" / "ui"


def test_ui_source_has_no_use_container_width() -> None:
    """No UI module passes the deprecated use_container_width argument."""
    hits = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _UI_DIR.rglob("*.py")
        if "use_container_width" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"use_container_width still present in: {hits}"
