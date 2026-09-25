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
    assert page_specs(True) == ["Overview", "Run", "Trajectories", "Gates", "Label", "Live"]


def test_mode_caption_by_mode() -> None:
    assert mode_caption(True) == "Local mode: run the study, label failures, and judge live"
    assert mode_caption(False) is None


def test_overview_shows_frontier_headline(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    captions = "\n".join(caption.value for caption in at.caption)
    assert "further right costs more, higher is more accurate" in captions


def _rendered(at: AppTest) -> str:
    """Return concatenated markdown and html text for substring assertions."""
    return "\n".join(node.value for node in [*at.get("markdown"), *at.get("html")])


def test_overview_has_flow_strip_and_next_link(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    rendered = _rendered(at)
    assert 'class="flow-strip"' in rendered
    markdown = "\n".join(md.value for md in at.markdown)
    assert "Next: [Run](/run)" in markdown


def test_overview_shows_gate_progress_and_continue_button(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    captions = "\n".join(caption.value for caption in at.caption)
    assert "4 of 10 gates have results." in captions
    buttons = at.get("link_button")
    assert any(button.label == "Continue the study" for button in buttons)
    assert all(button.url == "/run" for button in buttons)
