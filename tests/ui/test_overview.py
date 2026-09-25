"""Overview page and navigation shell tests."""

from streamlit.testing.v1 import AppTest

from decision_judges.ui.app import page_specs


def _metric_values(at: AppTest) -> dict[str, str]:
    """Return the rendered metrics as a label-to-value mapping."""
    return {metric.label: metric.value for metric in at.metric}


def test_overview_renders_counts_roster_and_threats(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception

    metrics = _metric_values(at)
    assert metrics["Agent records"] == "2"
    assert metrics["States (whole + step)"] == "4"
    assert metrics["Verdicts"] == "8"

    roster = at.dataframe[0].value
    assert list(roster["judge"]) == ["code", "llm_cheap", "llm_strong", "jev", "laya"]

    subheaders = [element.value for element in at.subheader]
    assert "Threats to validity" in subheaders


def test_navigation_lists_shared_pages_only_when_not_local() -> None:
    assert page_specs(False) == ["Overview", "Trajectories", "Gates"]
    assert "Label" not in page_specs(False)
    assert "Live" not in page_specs(False)


def test_navigation_adds_local_pages_when_local() -> None:
    assert page_specs(True) == ["Overview", "Trajectories", "Gates", "Label", "Live"]
