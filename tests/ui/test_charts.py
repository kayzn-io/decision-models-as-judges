"""Unit tests for the Altair chart builders."""

import altair as alt
import pandas as pd

from decision_judges.ui import charts

_JEV = "#E0651F"
_LAYA = "#6B6454"
_CODE = "#9A9384"
_OFF_WHITE = "#F1EDE4"


def _marks(chart: alt.TopLevelMixin) -> set[str]:
    """Return the set of mark types in a chart, descending into layers."""
    spec = chart.to_dict()
    found: set[str] = set()

    def visit(node: dict) -> None:
        mark = node.get("mark")
        if isinstance(mark, str):
            found.add(mark)
        elif isinstance(mark, dict):
            found.add(mark["type"])
        for layer in node.get("layer", []):
            visit(layer)

    visit(spec)
    return found


def _color_scale(chart: alt.TopLevelMixin) -> dict:
    """Return the color-encoding scale of a single-layer chart."""
    return chart.to_dict()["encoding"]["color"]["scale"]


def _encoding(chart: alt.TopLevelMixin) -> dict:
    """Return the top-level encoding block of a single-layer chart."""
    return chart.to_dict()["encoding"]


def test_bar_returns_a_bar_mark() -> None:
    frame = pd.DataFrame({"judge_id": ["code", "fake"], "accuracy": [1.0, 0.5]})
    chart = charts.bar(frame, x="judge_id", y="accuracy", title="Accuracy")
    assert isinstance(chart, alt.TopLevelMixin)
    assert "bar" in _marks(chart)


def test_bar_maps_judge_ids_to_fixed_colors() -> None:
    frame = pd.DataFrame(
        {"judge_id": ["jev", "laya", "code", "llm_cheap"], "accuracy": [1, 1, 1, 1]}
    )
    scale = _color_scale(charts.bar(frame, x="judge_id", y="accuracy", color="judge_id"))
    mapping = dict(zip(scale["domain"], scale["range"], strict=True))
    assert mapping["jev"] == _JEV
    assert mapping["laya"] == _LAYA
    assert mapping["code"] == _CODE
    assert mapping["llm_cheap"] == _OFF_WHITE


def test_grouped_bar_returns_bar_mark() -> None:
    frame = pd.DataFrame(
        {
            "judge_id": ["code", "code", "fake", "fake"],
            "aggregator": ["and", "mean", "and", "mean"],
            "auroc": [0.9, 0.8, 0.7, 0.6],
        }
    )
    chart = charts.grouped_bar(frame, x="judge_id", y="auroc", group="aggregator", title="AUROC")
    assert "bar" in _marks(chart)


def test_frontier_layers_line_point_and_reference() -> None:
    frontier = pd.DataFrame(
        {"t": [0.5, 0.9], "cost_per_item": [0.001, 0.01], "accuracy": [0.8, 0.95]}
    )
    reference = pd.DataFrame({"judge": ["jev"], "cost_per_item": [0.02], "accuracy": [0.9]})
    chart = charts.frontier(
        frontier,
        x_cost="cost_per_item",
        y_accuracy="accuracy",
        label="t",
        reference_df=reference,
        title="Frontier",
    )
    marks = _marks(chart)
    assert {"line", "point", "text"} <= marks


def test_reliability_has_diagonal_line_and_points() -> None:
    bins = pd.DataFrame({"mean_prob": [0.2, 0.8], "frac_positive": [0.1, 0.9], "count": [3, 5]})
    marks = _marks(charts.reliability(bins, title="Reliability"))
    assert "line" in marks
    assert "circle" in marks


def test_intervals_has_rule_and_point_marks() -> None:
    frame = pd.DataFrame(
        {
            "judge": ["code", "jev"],
            "lo": [-0.1, 0.0],
            "hi": [0.2, 0.3],
            "est_delta": [0.05, 0.15],
            "true_delta": [0.1, 0.1],
        }
    )
    marks = _marks(
        charts.intervals(
            frame, judge="judge", lo="lo", hi="hi", point="est_delta", truth_x="true_delta"
        )
    )
    assert "rule" in marks
    assert "point" in marks


def test_bar_axis_titles_are_friendly() -> None:
    frame = pd.DataFrame({"judge_id": ["code", "fake"], "auroc": [0.9, 0.8]})
    encoding = _encoding(charts.bar(frame, x="judge_id", y="auroc", color="judge_id"))
    assert encoding["x"]["title"] == "Judge"
    assert encoding["y"]["title"] == "AUROC"
    assert encoding["color"]["title"] == "Judge"


def test_single_category_bar_sets_a_fixed_bar_size() -> None:
    frame = pd.DataFrame({"judge_id": ["code"], "accuracy": [0.9]})
    spec = charts.bar(frame, x="judge_id", y="accuracy").to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["mark"]["size"] == 48


def test_multi_category_bar_has_no_fixed_size() -> None:
    frame = pd.DataFrame({"judge_id": ["code", "fake"], "accuracy": [0.9, 0.8]})
    mark = charts.bar(frame, x="judge_id", y="accuracy").to_dict()["mark"]
    assert "size" not in mark


def test_every_chart_is_280_tall() -> None:
    frame = pd.DataFrame({"judge_id": ["code", "fake"], "accuracy": [0.9, 0.8]})
    assert charts.bar(frame, x="judge_id", y="accuracy").to_dict()["height"] == 280


def test_frontier_axis_titles_and_formats() -> None:
    frontier = pd.DataFrame(
        {"t": [0.5, 0.9], "cost_per_item": [0.001, 0.01], "accuracy": [0.8, 0.95]}
    )
    line = charts.frontier(
        frontier, x_cost="cost_per_item", y_accuracy="accuracy", label="t"
    ).to_dict()["layer"][0]["encoding"]
    assert line["x"]["title"] == "Cost per item (USD)"
    assert line["x"]["axis"]["format"] == "$.4f"
    assert line["y"]["title"] == "Accuracy"
    assert line["y"]["axis"]["format"].endswith("%")


def test_frontier_labels_read_as_threshold_equals() -> None:
    frontier = pd.DataFrame(
        {"t": [0.5, 0.9], "cost_per_item": [0.001, 0.01], "accuracy": [0.8, 0.95]}
    )
    spec = str(
        charts.frontier(
            frontier, x_cost="cost_per_item", y_accuracy="accuracy", label="t"
        ).to_dict()
    )
    assert "t = " in spec
