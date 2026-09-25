"""Overview page: study summary, corpus counts, judge roster, and threats."""

import streamlit as st

from decision_judges.ui import charts, components, data, formatting

_PURPOSE = "What the study measures, who the judges are, and what it has cost so far."
_STUDY_SUMMARY = (
    "This study evaluates the typed decision models Jev and Laya as evaluation "
    "judges over tau-bench retail agent trajectories, comparing them with a "
    "deterministic code judge and general LLM judges against the benchmark's "
    "deterministic ground truth."
)
_FRONTIER_WHAT = (
    "The cascade frontier: accuracy against cost per verdict for each confidence threshold."
)
_FRONTIER_COMMAND = "judges analyze --gate g5"


def render() -> None:
    """Render the overview from cached agent runs, states, verdicts, and spend."""
    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    agent_records = data.load_agent_records(paths)
    states = data.load_states(paths)
    verdicts = data.load_verdicts(paths)
    ledger = data.load_ledger(paths)

    components.page_header("Decision models as judges", _PURPOSE)
    st.write(_STUDY_SUMMARY)

    components.metric_row(
        [
            ("Agent records", str(len(agent_records))),
            ("States (whole + step)", str(len(states))),
            ("Verdicts", str(len(verdicts))),
            ("Total spend (USD)", f"${data.total_spend(ledger):.4f}"),
        ]
    )

    st.subheader("Cascade frontier")
    _frontier(paths)

    st.subheader("Judge roster")
    roster = data.judge_roster(study, pricing)
    st.dataframe(
        roster,
        column_config=formatting.column_config_for(roster),
        hide_index=True,
        use_container_width=True,
    )

    if ledger is not None:
        st.subheader("Spend by stage")
        spend = data.spend_by_stage(ledger)
        st.dataframe(
            spend,
            column_config=formatting.column_config_for(spend),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Threats to validity", expanded=False):
        st.markdown(data.threats_text(paths))
    components.footer()


def _frontier(paths: data.Paths) -> None:
    """Render the cascade frontier from results, or an empty state when absent."""
    frame = data.load_results_table(paths, "g5_frontier")
    if frame is None or {"cost_per_item", "accuracy"} - set(frame.columns):
        components.empty_state(_FRONTIER_WHAT, _FRONTIER_COMMAND)
        return
    st.altair_chart(
        charts.frontier(
            frame,
            x_cost="cost_per_item",
            y_accuracy="accuracy",
            label="t",
            title="Cascade cost-accuracy frontier",
        ),
        use_container_width=True,
    )
