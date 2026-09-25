"""Overview page: study summary, corpus counts, judge roster, and threats."""

import streamlit as st

from decision_judges.ui import charts, components, data, formatting

_PURPOSE = "What the study measures, who the judges are, and what it has cost so far."
_WHY = "It orients you before you open any single gate or trajectory."
_NEXT_HINT = "Run the pipeline to fill in any gates that are still empty."
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
_FRONTIER_HEADLINE = (
    "Each point takes the decision model's verdict when its confidence clears the threshold "
    "and the strong LLM's verdict otherwise; further right costs more, higher is more accurate."
)


def render() -> None:
    """Render the overview from cached agent runs, states, verdicts, and spend."""
    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    agent_records = data.load_agent_records(paths)
    states = data.load_states(paths)
    verdicts = data.load_verdicts(paths)
    ledger = data.load_ledger(paths)

    components.page_header("Decision models as judges", _PURPOSE, why=_WHY)
    components.flow_context(paths, None)
    st.write(_STUDY_SUMMARY)

    components.metric_row(
        [
            ("Agent records", str(len(agent_records))),
            ("States (whole + step)", str(len(states))),
            ("Verdicts", str(len(verdicts))),
            ("Total spend (USD)", f"${data.total_spend(ledger):.4f}"),
        ]
    )

    _gate_progress(paths)
    _start_button(bool(agent_records))

    st.subheader("Headline result: the cascade frontier")
    st.caption(_FRONTIER_HEADLINE)
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
    components.next_link("Run", "/run", _NEXT_HINT)
    components.footer()


def _gate_progress(paths: data.Paths) -> None:
    """Render one badge per gate G1..G10 and how many gates have results."""
    present = data.gate_results_present(paths)
    done = sum(1 for _, ok in present if ok)
    badges = "  ".join(f"{':orange[●]' if ok else ':gray[○]'} {gate}" for gate, ok in present)
    st.markdown(badges)
    st.caption(f"{done} of {len(present)} gates have results.")


def _start_button(has_records: bool) -> None:
    """Link to the Run page, worded by whether any agent records exist yet."""
    label = "Continue the study" if has_records else "Start the study"
    st.link_button(label, "/run", type="primary")


def _frontier(paths: data.Paths) -> None:
    """Render the cascade frontier from results, or an empty state when absent."""
    frame = data.load_results_table(paths, "g5_frontier")
    if frame is None or {"cost_per_item", "accuracy"} - set(frame.columns):
        components.empty_state(_FRONTIER_WHAT, _FRONTIER_COMMAND, run_step=4)
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
