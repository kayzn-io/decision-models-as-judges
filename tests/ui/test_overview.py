"""Overview page and navigation shell tests.

The overview is a front door: a hero, the pipeline strip, ten experiment tiles,
one headline chart, and a single Details expander that holds everything else.
These tests pin that order and the two states the page can open in.
"""

from streamlit.testing.v1 import AppTest

from decision_judges.ui.app import mode_caption, page_specs

_HERO_TITLE = "Decision models as judges"
_HERO_SENTENCE = "shows every number's source"
_HEADLINE_TITLE = "Headline result: when to trust the cheap judge"
_TILE_NAMES = [
    "Difficulty guess",
    "Every action",
    "Pass or fail",
    "Small questions",
    "Cheap first",
    "Confidence",
    "Tricked?",
    "Spotting a drop",
    "Why it failed",
    "Free local model",
]


def _rendered(at: AppTest) -> str:
    """Return concatenated markdown, html, and subheader text for assertions."""
    nodes = [*at.get("markdown"), *at.get("html"), *at.subheader]
    return "\n".join(node.value for node in nodes)


def _captions(at: AppTest) -> str:
    """Return concatenated caption text for substring assertions."""
    return "\n".join(caption.value for caption in at.caption)


def _roster_frame(at: AppTest):
    """Return the judge roster dataframe, found by its judge column."""
    for frame in at.dataframe:
        if "judge" in frame.value.columns:
            return frame.value
    raise AssertionError("no roster dataframe with a judge column was rendered")


def test_overview_renders_without_exception(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception


def test_overview_hero_shows_title_and_sentence(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    rendered = _rendered(at)
    assert 'class="hero-title"' in rendered
    assert _HERO_TITLE in rendered
    assert _HERO_SENTENCE in rendered


def test_overview_has_full_size_flow_strip(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    rendered = _rendered(at)
    assert 'class="flow-strip"' in rendered
    # The full strip carries per-station captions; the compact one does not.
    assert "115 scripted customer requests" in rendered


def test_overview_continue_cta_and_step_hint_on_fixture(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    anchors = [node.value for node in at.get("html") if "Continue the study" in node.value]
    assert len(anchors) == 1
    assert 'href="/run"' in anchors[0]
    assert "in-app-link primary" in anchors[0]
    assert "target" not in anchors[0]
    assert "Step 4 of 8 next" in _captions(at)


def test_overview_shows_ten_tiles_in_gate_order(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    rendered = _rendered(at)
    positions = [rendered.find(name) for name in _TILE_NAMES]
    assert all(position >= 0 for position in positions), positions
    assert positions == sorted(positions), "tiles must render in g1..g10 order"
    for gate_id in (f"g{index}" for index in range(1, 11)):
        assert f'data-gate="{gate_id}"' in rendered


def test_overview_marks_a_tile_done_only_when_its_summary_exists(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    rendered = _rendered(at)
    # The fixture has g3_summary.csv but no g1_summary.csv.
    assert '<div class="tile done" data-gate="g3">' in rendered
    assert '<div class="tile todo" data-gate="g1">' in rendered


def test_overview_headline_chart_from_results(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    assert _HEADLINE_TITLE in _rendered(at)
    assert at.get("vega_lite_chart"), "expected the cascade frontier chart on the overview"


def test_overview_keeps_the_frontier_headline_caption(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    assert "further right costs more, higher is more accurate" in _captions(at)


def test_overview_hides_counts_no_metric_row_above_the_fold(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    # The counts row is gone; nothing about tokens or file counts sits above the fold.
    assert not at.metric


def test_overview_details_expander_holds_the_roster(app_test: AppTest) -> None:
    at = app_test
    at.run()
    assert not at.exception
    assert any(expander.label == "Details" for expander in at.expander)
    roster = _roster_frame(at)
    assert list(roster["judge"]) == ["code", "llm_cheap", "llm_strong", "jev", "laya"]


def test_navigation_lists_shared_pages_only_when_not_local() -> None:
    assert page_specs(False) == ["Overview", "Trajectories", "Experiments"]
    assert "Label" not in page_specs(False)
    assert "Live" not in page_specs(False)


def test_navigation_adds_local_pages_when_local() -> None:
    assert page_specs(True) == ["Overview", "Run", "Trajectories", "Experiments", "Label", "Live"]


def test_mode_caption_by_mode() -> None:
    assert mode_caption(True) == "Local mode: run the study, label failures, and judge live"
    assert mode_caption(False) is None
