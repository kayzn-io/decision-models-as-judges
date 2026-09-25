"""Live page: run judges on one trajectory with the visitor's own keys (local mode)."""

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import PricingTable, StudyConfig
from decision_judges.serialize import StateRecord
from decision_judges.types import Verdict
from decision_judges.ui import charts, components, data, live
from decision_judges.ui.flow import Station

_PURPOSE = "Run Jev, Laya, and an LLM judge on one run, side by side, on your own keys."
_WHY = "It lets you watch each judge decide on a run of your choosing, live."
_NEXT_HINT = "Back to the start."
_FRAMING = (
    "The text judge explains itself in prose; the decision models return only "
    "probabilities. This page shows that difference side by side."
)
_OPENROUTER_KEY = "live_openrouter_key"
_TYPESAFE_KEY = "live_typesafe_key"
_COUNT_KEY = "live_call_count"
_RESULTS_KEY = "live_results"
_INJECTION = "none"
_LAYA_PROFILE = "compact"

TrajectoryStates = Mapping[tuple[str, str, str, str], StateRecord]


def render() -> None:
    """Render the live judging workflow in local mode, an info banner otherwise."""
    if not data.is_local():
        st.info("Live judging runs only locally with JUDGES_LOCAL=1.")
        components.footer()
        return

    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    records = data.load_agent_records(paths)
    tasks = data.load_tasks_for_ui(paths)
    trajectory_states = data.trajectory_states(data.load_states(paths))

    components.page_header("Live", _PURPOSE, why=_WHY)
    components.flow_context(paths, Station.verdicts)
    keys = _sidebar_keys()

    if not records:
        st.caption("No agent runs are available to judge.")
        components.next_link("Overview", "/overview", _NEXT_HINT)
        components.footer()
        return

    variant, task_id, profile = _controls(records, trajectory_states)
    state = trajectory_states.get((variant, profile, _INJECTION, task_id))

    _judge_controls(
        study, pricing, tasks, records, keys, trajectory_states, variant, task_id, state
    )
    st.caption(_FRAMING)
    _render_results(pricing, state)
    components.next_link("Overview", "/overview", _NEXT_HINT)
    components.footer()


def _sidebar_keys() -> live.LiveKeys:
    """Render the session-key inputs and return the keys held in session state."""
    st.sidebar.subheader("Keys for this session")
    openrouter = st.sidebar.text_input("OpenRouter API key", type="password", key=_OPENROUTER_KEY)
    typesafe = st.sidebar.text_input(
        "TypeSafe API key (optional; only for the direct route)",
        type="password",
        key=_TYPESAFE_KEY,
    )
    st.sidebar.caption("Keys stay in this session only; they are never saved or logged.")
    return live.LiveKeys(openrouter=openrouter or None, typesafe=typesafe or None)


def _controls(
    records: Mapping[tuple[str, str], AgentRecord],
    trajectory_states: TrajectoryStates,
) -> tuple[str, str, str]:
    """Render the variant, task, and profile selectors and return the selection."""
    with st.container(border=True):
        columns = st.columns(3)
        variants = sorted({variant for variant, _ in records})
        with columns[0]:
            variant = str(st.selectbox("Variant", variants, key="live_variant"))
        task_ids = sorted({tid for candidate, tid in records if candidate == variant})
        with columns[1]:
            task_id = str(st.selectbox("Task", task_ids, key="live_task"))
        profiles = sorted(
            {
                prof
                for v, prof, inj, tid in trajectory_states
                if v == variant and tid == task_id and inj == _INJECTION
            }
        )
        with columns[2]:
            profile = str(st.selectbox("Profile", profiles or ["full"], key="live_profile"))
    st.caption(
        f"Jev and the LLM judge read the {profile!r} state; Laya always uses the "
        f"{_LAYA_PROFILE!r} state it was trained on."
    )
    return variant, task_id, profile


def _judge_controls(
    study: StudyConfig,
    pricing: PricingTable,
    tasks: Mapping[str, Task],
    records: Mapping[tuple[str, str], AgentRecord],
    keys: live.LiveKeys,
    trajectory_states: TrajectoryStates,
    variant: str,
    task_id: str,
    state: StateRecord | None,
) -> None:
    """Render the judge button and cap notices, and run judging on a click."""
    count = int(st.session_state.get(_COUNT_KEY, 0))
    within_cap = live.can_call(count)
    key_present = keys.openrouter is not None
    allowed = within_cap and key_present and state is not None
    clicked = st.button("Judge live", type="primary", disabled=not allowed, key="live_judge")

    if not key_present:
        st.caption("Add your OpenRouter key in the sidebar to judge.")
    if state is None:
        st.caption("No serialized state for this selection.")
    if not within_cap:
        st.caption(f"Session limit of {live.LIVE_CALL_CAP} live judgements reached.")

    if clicked and state is not None:
        _run_judging(
            study, pricing, tasks, records, keys, trajectory_states, variant, task_id, state
        )
        st.session_state[_COUNT_KEY] = count + 1


def _run_judging(
    study: StudyConfig,
    pricing: PricingTable,
    tasks: Mapping[str, Task],
    records: Mapping[tuple[str, str], AgentRecord],
    keys: live.LiveKeys,
    trajectory_states: TrajectoryStates,
    variant: str,
    task_id: str,
    state: StateRecord,
) -> None:
    """Judge the selected state with each live judge, timing each in its own block."""
    include_laya = live.laya_available(study)
    if not include_laya:
        st.caption(
            "Laya needs its checkpoint downloaded first; run the finetune or judge commands."
        )
    judges = live.build_live_judges(study, pricing, tasks, records, keys, include_laya=include_laya)
    questions = live.outcome_questions(tasks)
    results: dict[tuple[str, str], Verdict] = st.session_state.setdefault(_RESULTS_KEY, {})
    for judge in judges:
        judged = _state_for(judge.judge_id, state, trajectory_states, variant, task_id)
        with st.status(f"Judging with {judge.judge_id}") as status:
            verdict = judge.judge(judged, questions, 0)
            status.update(label=f"{judge.judge_id} · {verdict.latency_ms} ms", state="complete")
        results[(judge.judge_id, judged.state_hash)] = verdict


def _state_for(
    judge_id: str,
    state: StateRecord,
    trajectory_states: TrajectoryStates,
    variant: str,
    task_id: str,
) -> StateRecord:
    """Return the state a judge reads: Laya's compact state when present, else the selection."""
    if judge_id == "laya":
        compact = trajectory_states.get((variant, _LAYA_PROFILE, _INJECTION, task_id))
        if compact is not None:
            return compact
    return state


def _render_results(pricing: PricingTable, state: StateRecord | None) -> None:
    """Render one column per judge for the selected state's recorded verdicts."""
    results = st.session_state.get(_RESULTS_KEY, {})
    entries = live.order_results(results, state.state_hash) if state is not None else []
    if not entries:
        return
    st.subheader("Live verdicts")
    columns = st.columns(len(entries))
    for column, (judge_id, verdict) in zip(columns, entries, strict=True):
        with column:
            _render_verdict(pricing, judge_id, verdict)
    _latency_race(entries)


def _latency_race(entries: list[tuple[str, Verdict]]) -> None:
    """Render a per-judge latency bar so the speed gap between judges is visible."""
    rows = [
        {"judge": judge_id, "latency_ms": verdict.latency_ms}
        for judge_id, verdict in entries
        if verdict.error is None
    ]
    if not rows:
        return
    st.caption("Response time by judge")
    frame = pd.DataFrame(rows, columns=["judge", "latency_ms"])
    st.altair_chart(charts.bar(frame, "judge", "latency_ms"), use_container_width=True)


def _render_verdict(pricing: PricingTable, judge_id: str, verdict: Verdict) -> None:
    """Render one judge's verdict: metrics, its distribution, and its rationale."""
    st.markdown(f"**{judge_id}**")
    if verdict.error is not None:
        st.error(verdict.error)
        return
    components.metric_row(live.result_metrics(pricing, verdict))
    probabilities = live.verdict_probabilities(verdict)
    if not probabilities.empty:
        st.altair_chart(
            charts.bar(probabilities, "outcome", "probability"), use_container_width=True
        )
    if verdict.rationale:
        with st.expander("Rationale"):
            st.write(verdict.rationale)
