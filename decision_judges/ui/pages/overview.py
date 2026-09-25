"""Overview page: the study's front door.

A hero states what the study asks and the next thing to do, the pipeline strip
is the one diagram, ten tiles link to each experiment, one chart carries the
headline result, and a single Details expander holds the roster, spend, counts,
and threats so nothing about tokens or file counts sits above the fold.
"""

import streamlit as st

from decision_judges.ui import charts, components, data, flow, formatting
from decision_judges.ui.flow import Station

_HERO_TITLE = "Decision models as judges"
_HERO_TAGLINE = (
    "Can an AI that only picks answers judge other AIs better than one that writes essays? "
    "This app runs the study and shows every number's source."
)
_HEADLINE_TITLE = "Headline result: when to trust the cheap judge"
_FRONTIER_HEADLINE = (
    "Each point takes the decision model's verdict when its confidence clears the threshold "
    "and the strong LLM's verdict otherwise; further right costs more, higher is more accurate."
)
_FRONTIER_WHAT = (
    "The cascade frontier: accuracy against cost per verdict for each confidence threshold."
)
_FRONTIER_COMMAND = "judges analyze --gate g5"

# The ten experiments in g1..g10 order, each with its plain, front-door name.
_TILE_NAMES: tuple[tuple[str, str], ...] = (
    ("g1", "Difficulty guess"),
    ("g2", "Every action"),
    ("g3", "Pass or fail"),
    ("g4", "Small questions"),
    ("g5", "Cheap first"),
    ("g6", "Confidence"),
    ("g7", "Tricked?"),
    ("g8", "Spotting a drop"),
    ("g9", "Why it failed"),
    ("g10", "Free local model"),
)


def render() -> None:
    """Render the overview from cached agent runs, verdicts, spend, and results."""
    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    agent_records = data.load_agent_records(paths)
    ledger = data.load_ledger(paths)

    components.hero(_HERO_TITLE, _HERO_TAGLINE)
    _call_to_action(paths, bool(agent_records))
    st.divider()

    flow.strip(paths, active=None, compact=False)

    st.subheader("The experiments")
    components.gate_tiles(_tiles(paths))

    st.subheader(_HEADLINE_TITLE)
    st.caption(_FRONTIER_HEADLINE)
    _frontier(paths)

    _details(paths, study, pricing, ledger)
    components.next_link(
        "Run", "/run", "Run the study step by step to fill in what is still empty."
    )
    components.footer()


def _call_to_action(paths: data.Paths, has_records: bool) -> None:
    """Link to the Run page and caption which step comes next."""
    label = "Continue the study" if has_records else "Start the study"
    components.page_link("/run", label, primary=True)
    st.caption(f"Step {_next_step(paths)} of 8 next")


def _next_step(paths: data.Paths) -> int:
    """Return the next study step (1..5) from the folders the strip counts.

    The pipeline fills in order, so the first empty station names the next step:
    conversations, then the reading copy, then verdicts, then the findings the
    analysis writes. Once findings exist the reader is past the core four, so the
    hint points at the fifth step and stops guessing.
    """
    counts = flow.counts(paths)
    if counts[Station.conversations] == 0:
        return 1
    if counts[Station.judge_text] == 0:
        return 2
    if counts[Station.verdicts] == 0:
        return 3
    if counts[Station.findings] == 0:
        return 4
    return 5


def _tiles(paths: data.Paths) -> list[tuple[str, str, bool]]:
    """Pair each experiment id and name with whether its summary results exist."""
    done = {gate.lower(): ok for gate, ok in data.gate_results_present(paths)}
    return [(gate_id, name, done.get(gate_id, False)) for gate_id, name in _TILE_NAMES]


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


def _details(paths: data.Paths, study: object, pricing: object, ledger: object) -> None:
    """Fold the roster, spend, counts, and threats into one Details expander."""
    with st.expander("Details", expanded=False):
        st.markdown("**Judge roster**")
        roster = data.judge_roster(study, pricing)  # type: ignore[arg-type]
        st.dataframe(
            roster,
            column_config=formatting.column_config_for(roster),
            hide_index=True,
            use_container_width=True,
        )

        if ledger is not None:
            st.markdown("**Spend by stage**")
            spend = data.spend_by_stage(ledger)  # type: ignore[arg-type]
            st.dataframe(
                spend,
                column_config=formatting.column_config_for(spend),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("**Counts**")
        st.markdown(_counts_table(paths, ledger))

        st.markdown("**Threats to validity**")
        st.markdown(data.threats_text(paths))


def _counts_table(paths: data.Paths, ledger: object) -> str:
    """Return a small markdown table of the corpus counts and total spend."""
    counts = flow.counts(paths)
    total = data.total_spend(ledger)  # type: ignore[arg-type]
    rows = [
        ("Conversations", str(counts[Station.conversations])),
        ("What judges read", str(counts[Station.judge_text])),
        ("Verdicts", str(counts[Station.verdicts])),
        ("Total spend", f"${total:.4f}"),
    ]
    lines = ["| What | Count |", "| --- | --- |"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines)
