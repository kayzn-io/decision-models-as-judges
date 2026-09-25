"""Overview page: study summary, corpus counts, judge roster, and threats."""

import streamlit as st

from decision_judges.ui import components, data

_PURPOSE = "What the study measures, who the judges are, and what it has cost so far."
_STUDY_SUMMARY = (
    "This study evaluates the typed decision models Jev and Laya as evaluation "
    "judges over tau-bench retail agent trajectories, comparing them with a "
    "deterministic code judge and general LLM judges against the benchmark's "
    "deterministic ground truth."
)


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

    columns = st.columns(4)
    columns[0].metric("Agent records", len(agent_records))
    columns[1].metric("States (whole + step)", len(states))
    columns[2].metric("Verdicts", len(verdicts))
    columns[3].metric("Total spend (USD)", f"{data.total_spend(ledger):.4f}")

    st.subheader("Judge roster")
    st.dataframe(data.judge_roster(study, pricing), hide_index=True)

    if ledger is not None:
        st.subheader("Spend by stage")
        st.dataframe(data.spend_by_stage(ledger), hide_index=True)

    st.subheader("Cascade frontier")
    chart = data.frontier_chart(paths)
    if chart is not None:
        st.image(str(chart))
    else:
        st.caption("The cascade frontier chart appears once the cascade gate runs.")

    st.subheader("Threats to validity")
    st.markdown(data.threats_text(paths))
    components.footer()
