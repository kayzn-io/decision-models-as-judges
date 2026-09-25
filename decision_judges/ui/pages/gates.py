"""Gates page: static gate summaries beside interactive cascade and calibration views.

The page owns no analysis. Static tabs display the tables and charts the CLI
already wrote under ``results``; interactive tabs load cached items and verdicts
and hand them to the gate objects, which return the tables and figures shown.
"""

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from decision_judges.gates.base import GateResult
from decision_judges.gates.g5_cascade import G5Cascade
from decision_judges.gates.g6_calibration import G6Calibration
from decision_judges.ui import data

_PROFILE = "full"
_FAST_JUDGE = "jev"
_SLOW_JUDGE = "llm_strong"
_SIGNAL_HELP = "The three probability signals G6 scores against ground truth."


def render() -> None:
    """Render one tab per gate, static where results exist and interactive elsewhere."""
    paths = data.Paths.from_env()
    st.title("Gates")
    st.write("Static gate summaries beside interactive cascade, calibration, and regression views.")
    outcome, decomposition, local, cascade, calibration, regression = st.tabs(
        [
            "G3 Outcome",
            "G4 Decomposition",
            "G10 Local model",
            "G5 Cascade",
            "G6 Calibration",
            "G8 Regression",
        ]
    )
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


def _empty_state(what: str, command: str) -> None:
    """Render one sentence on what a gate measures and the command that fills it."""
    st.caption(what)
    st.code(command, language="bash")


def _table(result: GateResult, name: str) -> pd.DataFrame:
    """Return one of a gate result's tables typed as a DataFrame."""
    table = result.tables[name]
    assert isinstance(table, pd.DataFrame)
    return table


def _figure(result: GateResult, name: str) -> Figure:
    """Return one of a gate result's charts typed as a Figure."""
    figure = result.charts[name]
    assert isinstance(figure, Figure)
    return figure


def _static_gate(paths: data.Paths, table: str, chart: str, what: str, command: str) -> None:
    """Show a written results table and chart, or an empty state when absent."""
    frame = data.load_results_table(paths, table)
    if frame is None:
        _empty_state(what, command)
        return
    st.dataframe(frame, hide_index=True)
    image = data.load_chart_path(paths, chart)
    if image is not None:
        st.image(str(image))


def _g3_tab(paths: data.Paths) -> None:
    """Show the G3 outcome accuracy summary the judge command writes."""
    _static_gate(
        paths,
        "g3_summary",
        "g3_accuracy",
        "G3 scores whether each run met the outcome bar the rubric states.",
        "judges judge --gate g3 --variant baseline --variant degraded",
    )


def _g4_tab(paths: data.Paths) -> None:
    """Show the G4 decomposition summary the judge command writes."""
    _static_gate(
        paths,
        "g4_summary",
        "g4_auroc_by_aggregator",
        "G4 tests whether aggregating six atomic questions beats one broad question.",
        "judges judge --gate g4 --variant baseline --variant degraded",
    )


def _g10_tab(paths: data.Paths) -> None:
    """Show the G10 local-model summary the judge command writes."""
    _static_gate(
        paths,
        "g10_summary",
        "g10_zero_shot_vs_finetuned",
        "G10 compares the local decision model against the hosts on compact states.",
        "judges judge --gate g10 --profile compact --variant baseline",
    )


def _g5_tab(paths: data.Paths) -> None:
    """Replay G3 verdicts as a confidence-gated cascade over cost and accuracy."""
    study, pricing = data.load_study_and_pricing(paths)
    items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    judge_ids = sorted({verdict.judge_id for verdict in verdicts})
    missing = [judge for judge in (_FAST_JUDGE, _SLOW_JUDGE) if judge not in judge_ids]
    if not items or missing:
        _empty_state(
            "G5 replays the G3 verdicts as a confidence-gated cascade, but the verdict "
            f"cache is missing these judges: {', '.join(missing) or 'all'}.",
            "judges analyze --gate g5 --variant baseline --variant degraded",
        )
        return

    fast = st.selectbox("Fast judge", judge_ids, index=judge_ids.index(_FAST_JUDGE), key="g5_fast")
    slow = st.selectbox("Slow judge", judge_ids, index=judge_ids.index(_SLOW_JUDGE), key="g5_slow")
    threshold = st.slider(
        "Escalation threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01, key="g5_t"
    )
    gate = G5Cascade(
        pricing, [threshold, *study.thresholds.cascade], fast_judge=fast, slow_judge=slow
    )
    result = gate.analyze(verdicts, items)
    frontier = _table(result, "g5_frontier")
    _g5_metrics(frontier, threshold)
    st.dataframe(frontier, hide_index=True)
    st.pyplot(_figure(result, "g5_frontier"))


def _g5_metrics(frontier: pd.DataFrame, threshold: float) -> None:
    """Show the frontier row for the chosen threshold as three metrics."""
    match = frontier[frontier["t"] == threshold]
    if match.empty:
        return
    row = match.iloc[0]
    columns = st.columns(3)
    columns[0].metric("Accuracy", f"{float(row['accuracy']):.2f}")
    columns[1].metric("Cost per item (USD)", f"{float(row['cost_per_item']):.5f}")
    columns[2].metric("Escalation rate", f"{float(row['escalation_rate']):.2f}")


def _g6_tab(paths: data.Paths) -> None:
    """Score judge probability signals against the truth and recalibrate on demand."""
    items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    if not verdicts:
        _empty_state(
            "G6 scores how well each judge's probabilities track the truth.",
            "judges analyze --gate g6 --variant baseline --variant degraded",
        )
        return

    result = G6Calibration().analyze(verdicts, items)
    summary = _table(result, "g6_summary")
    if summary.empty:
        _empty_state(
            "G6 found no scorable probability signals in the cached verdicts.",
            "judges analyze --gate g6 --variant baseline --variant degraded",
        )
        return

    judges = sorted(summary["judge"].unique().tolist())
    judge = st.selectbox("Judge", judges, key="g6_judge")
    signals = sorted(summary[summary["judge"] == judge]["signal"].unique().tolist())
    signal = st.selectbox("Signal", signals, key="g6_signal", help=_SIGNAL_HELP)
    after = st.toggle("Apply isotonic recalibration", key="g6_isotonic")

    pair = summary[(summary["judge"] == judge) & (summary["signal"] == signal)]
    _g6_metrics(pair, after)
    bins = _table(result, "g6_bins")
    st.dataframe(bins[(bins["judge"] == judge) & (bins["signal"] == signal)], hide_index=True)
    st.pyplot(_figure(result, "g6_reliability"))


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
    items, verdicts = data.load_items_and_verdicts(paths, _PROFILE)
    result = data.analysis_gates(paths)["g8"].analyze(verdicts, items)
    summary = _table(result, "g8_summary")
    if summary.empty:
        st.caption(result.findings)
        return
    st.dataframe(summary, hide_index=True)
    st.pyplot(_figure(result, "g8_intervals"))
