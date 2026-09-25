"""Gates page: static gate summaries beside interactive cascade and calibration views.

The page owns no analysis. Static tabs display the tables the CLI already wrote
under ``results`` and encode them as Altair charts; interactive tabs load cached
items and verdicts and hand them to the gate objects, which return the tables
the page renders. The gate objects still build Matplotlib figures for the
README, but the app never displays them.
"""

import urllib.parse

import altair as alt
import pandas as pd
import streamlit as st

from decision_judges.gates.base import GateResult
from decision_judges.gates.g5_cascade import G5Cascade
from decision_judges.gates.g6_calibration import G6Calibration
from decision_judges.ui import charts, components, data, formatting

_PROFILE = "full"
_FAST_JUDGE = "jev"
_SLOW_JUDGE = "llm_strong"
_SPINNER = "Reading cached verdicts"
_SIGNAL_HELP = "The three probability signals G6 scores against ground truth."
_THRESHOLD_HELP = "Escalate to the slow judge when the fast judge's confidence falls below this."
_ISOTONIC_HELP = "Refit each signal with cross-validated isotonic regression and rescore it."
_PURPOSE = "Results for each evaluation gate, static and interactive."

_G2_MEASURE = "G2 scores every tool call: was it needed, and were its arguments consistent."
_G3_MEASURE = "G3 scores each run's pass or fail verdict against the outcome truth."
_G4_MEASURE = "G4 tests whether aggregating six atomic questions beats one broad question."
_G10_MEASURE = "G10 compares the local decision model against the hosted judges on compact states."
_G5_MEASURE = "G5 replays the outcome verdicts as a confidence-gated cascade of cost and accuracy."
_G6_MEASURE = "G6 measures how closely each judge's probabilities track observed outcomes."
_G8_MEASURE = "G8 estimates each judge's baseline-to-degraded regression with bootstrap intervals."
_G9_MEASURE = "G9 scores each judge's failure-taxonomy labels against the owner's hand labels."
_G7_MEASURE = "G7 measures whether an injected evaluator-directed sentence flips a fail to a pass."


def render() -> None:
    """Render one tab per gate, static where results exist and interactive elsewhere."""
    paths = data.Paths.from_env()
    components.page_header("Gates", _PURPOSE)
    steps, outcome, decomposition, local, cascade, calibration, regression, taxonomy, robustness = (
        st.tabs(
            [
                "G2 Steps",
                "G3 Outcome",
                "G4 Decomposition",
                "G10 Local model",
                "G5 Cascade",
                "G6 Calibration",
                "G8 Regression",
                "G9 Taxonomy",
                "G7 Robustness",
            ]
        )
    )
    with steps:
        _g2_tab(paths)
    with outcome:
        _g3_tab(paths)
    with decomposition:
        _g4_tab(paths)
    with local:
        _g10_tab(paths)
    with cascade:
        _g5_tab(paths)
    with calibration:
        _g6_tab(paths)
    with regression:
        _g8_tab(paths)
    with taxonomy:
        _g9_tab(paths)
    with robustness:
        _g7_tab(paths)
    components.footer()


def _show_table(frame: pd.DataFrame) -> None:
    """Render a table with formatted columns and no index."""
    st.dataframe(
        frame,
        column_config=formatting.column_config_for(frame),
        hide_index=True,
        use_container_width=True,
    )


def _show_chart(chart: alt.Chart | alt.LayerChart | alt.FacetChart) -> None:
    """Render an Altair chart stretched to the container width."""
    st.altair_chart(chart, use_container_width=True)


def _table(result: GateResult, name: str) -> pd.DataFrame:
    """Return one of a gate result's tables typed as a DataFrame."""
    table = result.tables[name]
    assert isinstance(table, pd.DataFrame)
    return table


def _g2_tab(paths: data.Paths) -> None:
    """Show per-step scoring: AUROC of the necessary-call probability per judge."""
    st.caption(_G2_MEASURE)
    frame = data.load_results_table(paths, "g2_summary")
    if frame is None:
        components.empty_state(
            "G2 scores each tool call for necessity and argument consistency.",
            "judges judge --gate g2 --judges jev,llm_cheap,llm_strong",
        )
        return
    _show_table(frame)
    if {"judge_id", "auroc"} <= set(frame.columns):
        _show_chart(
            charts.bar(
                frame, x="judge_id", y="auroc", color="judge_id", title="Necessary-call AUROC"
            )
        )


def _g3_tab(paths: data.Paths) -> None:
    """Show the G3 outcome accuracy summary and an accuracy bar per judge."""
    st.caption(_G3_MEASURE)
    frame = data.load_results_table(paths, "g3_summary")
    if frame is None:
        components.empty_state(
            "G3 scores whether each run met the outcome bar the rubric states.",
            "judges judge --gate g3 --variant baseline --variant degraded",
        )
        return
    _show_table(frame)
    if {"judge_id", "accuracy"} <= set(frame.columns):
        _show_chart(
            charts.bar(
                frame, x="judge_id", y="accuracy", color="judge_id", title="Outcome accuracy"
            )
        )


def _g4_tab(paths: data.Paths) -> None:
    """Show the G4 decomposition summary and AUROC grouped by aggregator."""
    st.caption(_G4_MEASURE)
    frame = data.load_results_table(paths, "g4_summary")
    if frame is None:
        components.empty_state(
            "G4 tests whether aggregating six atomic questions beats one broad question.",
            "judges judge --gate g4 --variant baseline --variant degraded",
        )
        return
    _show_table(frame)
    if {"judge_id", "aggregator", "auroc"} <= set(frame.columns):
        _show_chart(
            charts.grouped_bar(
                frame, x="judge_id", y="auroc", group="aggregator", title="AUROC by aggregator"
            )
        )


def _g10_tab(paths: data.Paths) -> None:
    """Show the G10 local-model summary and accuracy and AUROC grouped per judge."""
    st.caption(_G10_MEASURE)
    frame = data.load_results_table(paths, "g10_summary")
    if frame is None:
        components.empty_state(
            "G10 compares the local decision model against the hosts on compact states.",
            "judges judge --gate g10 --profile compact --variant baseline",
        )
        return
    _show_table(frame)
    if {"judge_id", "accuracy", "auroc"} <= set(frame.columns):
        long = frame[["judge_id", "accuracy", "auroc"]].melt(
            "judge_id", var_name="metric", value_name="value"
        )
        _show_chart(
            charts.grouped_bar(
                long, x="judge_id", y="value", group="metric", title="Accuracy and AUROC by judge"
            )
        )


def _g5_tab(paths: data.Paths) -> None:
    """Replay G3 verdicts as a confidence-gated cascade over cost and accuracy."""
    st.caption(_G5_MEASURE)
    study, pricing = data.load_study_and_pricing(paths)
    with st.spinner(_SPINNER):
        items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    judge_ids = sorted({verdict.judge_id for verdict in verdicts})
    missing = [judge for judge in (_FAST_JUDGE, _SLOW_JUDGE) if judge not in judge_ids]
    if not items or missing:
        components.empty_state(
            "G5 replays the G3 verdicts as a confidence-gated cascade, but the verdict "
            f"cache is missing these judges: {', '.join(missing) or 'all'}.",
            "judges analyze --gate g5 --variant baseline --variant degraded",
        )
        return

    fast = st.selectbox("Fast judge", judge_ids, index=judge_ids.index(_FAST_JUDGE), key="g5_fast")
    slow = st.selectbox("Slow judge", judge_ids, index=judge_ids.index(_SLOW_JUDGE), key="g5_slow")
    threshold = st.slider(
        "Escalation threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.9,
        step=0.01,
        key="g5_t",
        help=_THRESHOLD_HELP,
    )
    gate = G5Cascade(
        pricing, [threshold, *study.thresholds.cascade], fast_judge=fast, slow_judge=slow
    )
    result = gate.analyze(verdicts, items)
    frontier = _table(result, "g5_frontier")
    reference = _table(result, "g5_reference")
    _g5_metrics(frontier, threshold)
    _show_table(frontier)
    if {"cost_per_item", "accuracy"} <= set(frontier.columns):
        _show_chart(
            charts.frontier(
                frontier,
                x_cost="cost_per_item",
                y_accuracy="accuracy",
                label="t",
                reference_df=reference,
                title="Cascade cost-accuracy frontier",
            )
        )


def _g5_metrics(frontier: pd.DataFrame, threshold: float) -> None:
    """Show the frontier row for the chosen threshold as three metrics."""
    match = frontier[frontier["t"] == threshold]
    if match.empty:
        return
    row = match.iloc[0]
    columns = st.columns(3)
    columns[0].metric("Accuracy", formatting.pct(float(row["accuracy"])))
    columns[1].metric("Cost per item (USD)", f"${float(row['cost_per_item']):.4f}")
    columns[2].metric("Escalation rate", formatting.pct(float(row["escalation_rate"])))


def _g6_tab(paths: data.Paths) -> None:
    """Score judge probability signals against the truth and recalibrate on demand."""
    st.caption(_G6_MEASURE)
    with st.spinner(_SPINNER):
        items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    if not verdicts:
        components.empty_state(
            "G6 scores how well each judge's probabilities track the truth.",
            "judges analyze --gate g6 --variant baseline --variant degraded",
        )
        return

    result = G6Calibration().analyze(verdicts, items)
    summary = _table(result, "g6_summary")
    if summary.empty:
        components.empty_state(
            "G6 found no scorable probability signals in the cached verdicts.",
            "judges analyze --gate g6 --variant baseline --variant degraded",
        )
        return

    judges = sorted(summary["judge"].unique().tolist())
    judge = st.selectbox("Judge", judges, key="g6_judge")
    signals = sorted(summary[summary["judge"] == judge]["signal"].unique().tolist())
    signal = st.selectbox("Signal", signals, key="g6_signal", help=_SIGNAL_HELP)
    after = st.toggle("Apply isotonic recalibration", key="g6_isotonic", help=_ISOTONIC_HELP)

    pair = summary[(summary["judge"] == judge) & (summary["signal"] == signal)]
    _g6_metrics(pair, after)
    bins = _table(result, "g6_bins")
    subset = bins[(bins["judge"] == judge) & (bins["signal"] == signal)]
    _show_table(subset)
    if not subset.empty and {"mean_prob", "frac_positive", "count"} <= set(subset.columns):
        _show_chart(charts.reliability(subset, title=f"Reliability — {judge} / {signal}"))


def _g6_metrics(pair: pd.DataFrame, after: bool) -> None:
    """Show ECE and Brier for one judge-and-signal pair, before or after recalibration."""
    if pair.empty:
        return
    row = pair.iloc[0]
    ece_key = "ece_after_isotonic" if after else "ece"
    brier_key = "brier_after_isotonic" if after else "brier"
    columns = st.columns(2)
    columns[0].metric("ECE", f"{float(row[ece_key]):.3f}")
    columns[1].metric("Brier", f"{float(row[brier_key]):.3f}")


def _g8_tab(paths: data.Paths) -> None:
    """Estimate the baseline-to-degraded regression each judge sees over both variants."""
    st.caption(_G8_MEASURE)
    with st.spinner(_SPINNER):
        items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    result = data.analysis_gates(paths)["g8"].analyze(verdicts, items)
    summary = _table(result, "g8_summary")
    if summary.empty:
        st.caption(result.findings)
        return
    _show_table(summary)
    if {"judge", "lo", "hi", "est_delta", "true_delta"} <= set(summary.columns):
        _show_chart(
            charts.intervals(
                summary,
                judge="judge",
                lo="lo",
                hi="hi",
                point="est_delta",
                truth_x="true_delta",
                title="Baseline-to-degraded delta by judge",
            )
        )


def _g9_tab(paths: data.Paths) -> None:
    """Show the static G9 taxonomy summary, confusion, and an accuracy bar per judge."""
    st.caption(_G9_MEASURE)
    summary = data.load_results_table(paths, "g9_summary")
    if summary is None:
        components.empty_state(
            "G9 asks each judge to classify a failing trajectory into the failure taxonomy "
            "and scores it against the owner's hand labels; label failures on the Label page "
            "first.",
            "judges judge --gate g9 --variant baseline",
        )
        return
    st.caption("Single annotator; see the findings.")
    _show_table(summary)
    confusion = data.load_results_table(paths, "g9_confusion")
    if confusion is not None:
        _show_table(confusion)
    if {"judge_id", "accuracy"} <= set(summary.columns):
        _show_chart(
            charts.bar(
                summary, x="judge_id", y="accuracy", color="judge_id", title="Taxonomy accuracy"
            )
        )


def _g7_tab(paths: data.Paths) -> None:
    """Show the written robustness results, flip links, and grouped flip rates.

    G7 is a judged gate rather than an analysis gate, so its summary and flips
    come from the tables the judge command writes. Each flipped trajectory links
    into the Trajectories page at the injected state.
    """
    st.caption(_G7_MEASURE)
    summary = data.load_results_table(paths, "g7_summary")
    flips = data.load_results_table(paths, "g7_flips")
    with st.spinner(_SPINNER):
        items, verdicts = data.load_injected_items_and_verdicts(paths, _PROFILE)
    has_recompute = bool(items) and bool(verdicts)
    if summary is None and not has_recompute:
        components.empty_state(
            "G7 measures whether an injected evaluator-directed sentence flips a fail "
            "verdict to pass.",
            "judges judge --gate g7 --variant baseline --variant degraded",
        )
        return
    if summary is not None:
        _show_table(summary)
    if flips is not None and not flips.empty:
        _g7_flip_links(flips)
    if summary is not None and {"placement", "flip_rate", "judge"} <= set(summary.columns):
        _show_chart(
            charts.grouped_bar(
                summary, x="placement", y="flip_rate", group="judge", title="Flip rate by placement"
            )
        )


def _g7_flip_links(flips: pd.DataFrame) -> None:
    """Render one deep link per flipped trajectory into the Trajectories page."""
    st.write("Flipped trajectories")
    for row in flips.itertuples(index=False):
        _flip_link(str(row.judge), str(row.placement), str(row.variant), str(row.task_id))


def _flip_link(judge: str, placement: str, variant: str, task_id: str) -> None:
    """Render a Trajectories deep link preselecting the flipped injected state.

    ``st.page_link`` requires a registered ``Page`` object, which the per-page
    test harness does not provide, so the link is rendered as markdown to the
    Trajectories URL path with the variant, task, and injection query parameters.
    """
    query = urllib.parse.urlencode({"variant": variant, "task": task_id, "injection": placement})
    label = f"{judge} · {placement} · {variant}/{task_id}"
    st.markdown(f"[{label}](/trajectories?{query})")
