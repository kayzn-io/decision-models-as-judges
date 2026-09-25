"""AppTest-backed tests for the flow strip counts and rendering."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.ui import data
from decision_judges.ui.flow import Station, counts

_STRIP_SCRIPT = """
from decision_judges.ui import data
from decision_judges.ui.flow import Station, strip

strip(
    data.Paths.from_env(),
    active=Station.verdicts,
    running="judge",
    paid=True,
    compact=True,
)
"""


def test_counts_match_the_fixture_tree(monkeypatch: pytest.MonkeyPatch, ui_root: Path) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(ui_root))
    result = counts(data.Paths.from_env())
    assert result[Station.tasks] == 3
    assert result[Station.conversations] == 2
    assert result[Station.judge_text] == 7
    assert result[Station.verdicts] == 11
    expected_findings = len(list(ui_root.glob("results/*_findings.md")))
    assert result[Station.findings] == expected_findings


def test_strip_renders_the_svg_without_exception(
    monkeypatch: pytest.MonkeyPatch, ui_root: Path
) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(ui_root))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    at = AppTest.from_string(_STRIP_SCRIPT, default_timeout=30).run()
    assert not at.exception
    rendered = "\n".join(node.value for node in [*at.get("markdown"), *at.get("html")])
    assert "<title>" in rendered
    assert 'role="img"' in rendered
