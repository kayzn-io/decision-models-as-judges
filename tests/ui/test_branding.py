"""Branding and theme tests: assets, config, and shared UI helpers."""

import shutil
import tomllib
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_ASSETS = _REPO_ROOT / "decision_judges" / "ui" / "assets"
_CONFIG = _REPO_ROOT / ".streamlit" / "config.toml"
_LOGO = _ASSETS / "kayzn-logo.png"
_FAVICON = _ASSETS / "kayzn-favicon.png"

_PAGES = ("overview", "trajectories", "gates", "label", "live")
_MAX_BYTES = 100 * 1024


def _plain(value: str) -> str:
    """Strip markdown link brackets so linked words read as plain text."""
    return value.replace("[", "").replace("]", "")


def _script(tmp_path: Path, name: str, body: str) -> str:
    """Write a thin runner script and return its path."""
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return str(script)


def test_logo_and_favicon_exist_sized_and_small() -> None:
    for path, size in ((_LOGO, 256), (_FAVICON, 64)):
        assert path.is_file(), f"missing asset {path}"
        assert path.stat().st_size < _MAX_BYTES
        with Image.open(path) as image:
            assert image.size == (size, size)
            assert image.mode == "RGB"


def test_assets_package_marker_exists() -> None:
    assert (_ASSETS / "__init__.py").is_file()


def test_config_theme_is_dark_with_brand_colors() -> None:
    config = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))
    theme = config["theme"]
    assert theme["base"] == "dark"
    assert theme["primaryColor"] == "#E0651F"
    assert theme["textColor"] == "#F1EDE4"
    assert config["server"]["headless"] is True
    assert config["browser"]["gatherUsageStats"] is False


def test_page_header_renders_title_and_purpose(tmp_path: Path) -> None:
    body = (
        "from decision_judges.ui.components import page_header\n\n"
        "page_header('Gates', 'Results for each evaluation gate.')\n"
    )
    at = AppTest.from_file(_script(tmp_path, "run_header.py", body), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Gates"
    assert any("Results for each evaluation gate." in c.value for c in at.caption)


def test_page_header_renders_breadcrumb(tmp_path: Path) -> None:
    body = (
        "from decision_judges.ui.components import page_header\n\n"
        "page_header('Trajectories', 'purpose', ['baseline', 'task_042', 'none'])\n"
    )
    at = AppTest.from_file(_script(tmp_path, "run_crumb.py", body), default_timeout=30).run()
    assert not at.exception
    assert any("baseline / task_042 / none" in c.value for c in at.caption)


def test_empty_state_renders_command_in_code(tmp_path: Path) -> None:
    body = (
        "from decision_judges.ui.components import empty_state\n\n"
        "empty_state('What will appear here.', 'judges results')\n"
    )
    at = AppTest.from_file(_script(tmp_path, "run_empty.py", body), default_timeout=30).run()
    assert not at.exception
    assert any(code.value == "judges results" for code in at.code)
    assert any("What will appear here." in c.value for c in at.caption)


def test_metric_row_renders_labeled_metrics(tmp_path: Path) -> None:
    body = (
        "from decision_judges.ui.components import metric_row\n\n"
        "metric_row([('A', '1'), ('B', '2')])\n"
    )
    at = AppTest.from_file(_script(tmp_path, "run_metric.py", body), default_timeout=30).run()
    assert not at.exception
    assert {metric.label for metric in at.metric} == {"A", "B"}


def _run_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str, local: bool) -> AppTest:
    """Run a page through a thin script in shared or local mode."""
    root = _UI_ROOT
    if local:
        root = tmp_path / "root"
        shutil.copytree(_UI_ROOT, root)
        monkeypatch.setenv("JUDGES_LOCAL", "1")
    else:
        monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    body = f"from decision_judges.ui.pages.{page} import render\n\nrender()\n"
    return AppTest.from_file(_script(tmp_path, f"run_{page}.py", body), default_timeout=30).run()


def _footer_markdowns(at: AppTest) -> list[str]:
    """Return markdown values whose plain text carries the footer sentence."""
    return [md.value for md in at.markdown if "Built by Kayzn" in _plain(md.value)]


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("page", _PAGES)
def test_every_page_renders_footer_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str, local: bool
) -> None:
    at = _run_page(monkeypatch, tmp_path, page, local)
    assert not at.exception
    footers = _footer_markdowns(at)
    assert len(footers) == 1
    assert "Apache License 2.0" in _plain(footers[0])
    assert "kayzn.io" in footers[0]
