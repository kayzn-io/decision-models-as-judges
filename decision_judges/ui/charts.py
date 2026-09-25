"""Altair chart builders for the study app.

Each function maps an already-computed DataFrame to a themed Altair chart. The
module encodes and colors data for display; it computes no metrics.
"""

from collections.abc import Sequence

import altair as alt
import pandas as pd

from decision_judges.ui import formatting

_JEV = "#E0651F"
_LAYA = "#6B6454"
_CODE = "#9A9384"
_OFF_WHITE = "#F1EDE4"
_JUDGE_COLUMNS = ("judge", "judge_id")
_HEIGHT = 280
_SINGLE_BAR_SIZE = 48


def _judge_color(judge: str) -> str:
    """Return the fixed color for a judge id by its family prefix."""
    if judge.startswith("jev"):
        return _JEV
    if judge.startswith("laya"):
        return _LAYA
    if judge == "code":
        return _CODE
    return _OFF_WHITE


def _judge_scale(values: Sequence[object]) -> alt.Scale:
    """Return a color scale binding each judge id to its fixed color."""
    domain = sorted({str(value) for value in values})
    return alt.Scale(domain=domain, range=[_judge_color(judge) for judge in domain])


def _tooltips(df: pd.DataFrame) -> list[alt.Tooltip]:
    """Return one tooltip per column so every mark is inspectable."""
    return [alt.Tooltip(str(column)) for column in df.columns]


def _color(df: pd.DataFrame, column: str) -> alt.Color:
    """Return a color encoding, applying the fixed judge scale for judge columns."""
    title = formatting.axis_title(column)
    if column in _JUDGE_COLUMNS:
        return alt.Color(f"{column}:N", scale=_judge_scale(df[column].tolist()), title=title)
    return alt.Color(f"{column}:N", title=title)


def bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> alt.Chart:
    """Return a vertical bar chart of ``y`` by ``x``, colored by ``color`` when given.

    A chart with a single ``x`` category draws a fixed-width bar so one value
    does not stretch across the whole plot.
    """
    single = x in df.columns and df[x].nunique() <= 1
    mark = alt.Chart(df, title=title)
    base = mark.mark_bar(size=_SINGLE_BAR_SIZE) if single else mark.mark_bar()
    encodings: dict[str, object] = {
        "x": alt.X(f"{x}:N", title=formatting.axis_title(x)),
        "y": alt.Y(f"{y}:Q", title=formatting.axis_title(y)),
        "tooltip": _tooltips(df),
    }
    if color is not None:
        encodings["color"] = _color(df, color)
    return base.encode(**encodings).properties(height=_HEIGHT)


def grouped_bar(df: pd.DataFrame, x: str, y: str, group: str, title: str = "") -> alt.Chart:
    """Return grouped bars of ``y`` by ``x``, one bar per ``group`` value."""
    return (
        alt.Chart(df, title=title)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=formatting.axis_title(x)),
            y=alt.Y(f"{y}:Q", title=formatting.axis_title(y)),
            xOffset=alt.XOffset(f"{group}:N"),
            color=_color(df, group),
            tooltip=_tooltips(df),
        )
        .properties(height=_HEIGHT)
    )


def frontier(
    df: pd.DataFrame,
    x_cost: str,
    y_accuracy: str,
    label: str,
    reference_df: pd.DataFrame | None = None,
    title: str = "",
) -> alt.LayerChart | alt.FacetChart:
    """Return the cascade frontier: a labeled cost-accuracy line with reference points.

    Threshold labels read ``t = 0.90`` and alternate above and below the line so
    neighboring labels do not collide; right padding keeps the last label from
    clipping at the plot edge.
    """
    x_title = formatting.axis_title(x_cost)
    y_title = formatting.axis_title(y_accuracy)
    base = alt.Chart(df, title=title)
    line = base.mark_line(point=True, color=_JEV).encode(
        x=alt.X(f"{x_cost}:Q", title=x_title, axis=alt.Axis(format="$.4f")),
        y=alt.Y(
            f"{y_accuracy}:Q",
            title=y_title,
            scale=alt.Scale(domain=[0.0, 1.08]),
            axis=alt.Axis(format=".0%", values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        tooltip=_tooltips(df),
    )
    labeled = base.transform_window(
        _order="row_number()", sort=[alt.SortField(x_cost, order="ascending")]
    ).transform_calculate(_label=f"'t = ' + format(datum['{label}'], '.2f')")
    text_encode: dict[str, object] = {
        "x": alt.X(f"{x_cost}:Q"),
        "y": alt.Y(f"{y_accuracy}:Q"),
        "text": alt.Text("_label:N"),
    }
    above = (
        labeled.transform_filter("datum._order % 2 == 1")
        .mark_text(dy=-12, color=_OFF_WHITE)
        .encode(**text_encode)
    )
    below = (
        labeled.transform_filter("datum._order % 2 == 0")
        .mark_text(dy=16, color=_OFF_WHITE)
        .encode(**text_encode)
    )
    layers: list[alt.Chart] = [line, above, below]
    if reference_df is not None and not reference_df.empty:
        reference = (
            alt.Chart(reference_df)
            .mark_point(size=140, filled=True)
            .encode(
                x=alt.X(f"{x_cost}:Q"),
                y=alt.Y(f"{y_accuracy}:Q"),
                color=_color(reference_df, "judge"),
                tooltip=_tooltips(reference_df),
            )
        )
        layers.append(reference)
    return (
        alt.layer(*layers)
        .properties(height=_HEIGHT, padding={"left": 5, "top": 10, "right": 48, "bottom": 5})
        .interactive()
    )


def reliability(df_bins: pd.DataFrame, title: str = "") -> alt.LayerChart | alt.FacetChart:
    """Return a reliability curve: the diagonal plus bin points sized by count."""
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(color=_CODE, strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title="mean predicted"), y=alt.Y("y:Q", title="fraction positive"))
    )
    points = (
        alt.Chart(df_bins, title=title)
        .mark_circle(color=_JEV)
        .encode(
            x=alt.X("mean_prob:Q", title="mean predicted"),
            y=alt.Y("frac_positive:Q", title="fraction positive"),
            size=alt.Size("count:Q", title="count"),
            tooltip=_tooltips(df_bins),
        )
    )
    return alt.layer(diagonal, points).properties(height=_HEIGHT)


def intervals(
    df: pd.DataFrame,
    judge: str,
    lo: str,
    hi: str,
    point: str,
    truth_x: str,
    title: str = "",
) -> alt.LayerChart | alt.FacetChart:
    """Return per-judge horizontal error bars with a vertical rule at the truth."""
    base = alt.Chart(df, title=title)
    bars = base.mark_rule(size=3).encode(
        y=alt.Y(f"{judge}:N", title=formatting.axis_title(judge)),
        x=alt.X(f"{lo}:Q", title="delta"),
        x2=f"{hi}:Q",
        color=_color(df, judge),
        tooltip=_tooltips(df),
    )
    dots = base.mark_point(filled=True, size=90).encode(
        y=alt.Y(f"{judge}:N"),
        x=alt.X(f"{point}:Q"),
        color=_color(df, judge),
        tooltip=_tooltips(df),
    )
    truth = base.mark_rule(color=_OFF_WHITE, strokeDash=[6, 4]).encode(
        x=alt.X(f"mean({truth_x}):Q", title=formatting.axis_title(truth_x))
    )
    return alt.layer(bars, dots, truth).properties(height=_HEIGHT)
