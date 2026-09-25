"""Overview page and navigation shell tests."""

from streamlit.testing.v1 import AppTest

from decision_judges.ui.app import mode_caption, page_specs


def _metric_values(at: AppTest) -> dict[str, str]:
    """Return the rendered metrics as a label-to-value mapping."""
    return {metric.label: metric.value for metric in at.metric}


def test_overview_renders_counts_roster_and_threats(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception

    metrics = _metric_values(at)
    assert metrics["Agent records"] == "2"
    assert metrics["States (whole + step)"] == "7"
    assert metrics["Verdicts"] == "11"

    roster = at.dataframe[0].value
    assert list(roster["judge"]) == ["code", "llm_cheap", "llm_strong", "jev", "laya"]

    assert any(expander.label == "Threats to validity" for expander in at.expander)


def test_overview_renders_frontier_chart_from_results(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    assert at.get("vega_lite_chart"), "expected the cascade frontier chart on the overview"


def test_navigation_lists_shared_pages_only_when_not_local() -> None:
    assert page_specs(False) == ["Overview", "Trajectories", "Gates"]
    assert "Label" not in page_specs(False)
    assert "Live" not in page_specs(False)


def test_navigation_adds_local_pages_when_local() -> None:
    assert page_specs(True) == ["Overview", "Trajectories", "Gates", "Label", "Live"]


def test_mode_caption_by_mode() -> None:
    assert mode_caption(True) == "Local mode: labeling and live judging enabled"
    assert mode_caption(False) is None


def test_overview_shows_frontier_headline(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    captions = "\n".join(caption.value for caption in at.caption)
    assert "further right costs more, higher is more accurate" in captions
