"""Trajectories page: one run's conversation beside its ground truth and verdicts."""

import json

import pandas as pd
import streamlit as st

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.serialize import Injection, StateRecord
from decision_judges.types import Verdict
from decision_judges.ui import data, views

_DEFAULT_PROFILE = "full"
_DEFAULT_INJECTION = "none"


def render() -> None:
    """Render one run's conversation, ground truth, and side-by-side verdicts."""
    paths = data.Paths.from_env()
    records = data.load_agent_records(paths)
    states = data.load_states(paths)
    grouped = data.verdicts_by_state(data.load_verdicts(paths))
    tasks = data.load_tasks_for_ui(paths)

    st.title("Trajectories")
    if not records:
        st.caption("No agent runs are available yet.")
        return

    variant, task_id, profile, injection = _select(records, states)
    record = records[(variant, task_id)]
    state = states.get((variant, profile, injection, task_id))
    task = tasks.get(task_id)

    if state is not None and state.injection is not Injection.none:
        st.warning(
            f"Evaluator-directed '{state.injection.value}' injection is present in this state."
        )

    conversation, ground_truth, verdicts = st.columns(3)
    with conversation:
        _conversation(record)
    with ground_truth:
        _ground_truth(record, task)
    with verdicts:
        _verdicts(state, grouped)


def _select_default(label: str, options: list[str], default: str, key: str) -> str:
    """Render a selectbox defaulting to ``default`` when it is among options."""
    index = options.index(default) if default in options else 0
    return str(st.selectbox(label, options, index=index, key=key))


def _select(
    records: dict[tuple[str, str], AgentRecord],
    states: dict[tuple[str, str, str, str], StateRecord],
) -> tuple[str, str, str, str]:
    """Render the variant, task, profile, and injection selectors and return them."""
    variants = sorted({variant for variant, _ in records})
    variant = str(st.selectbox("Variant", variants, key="variant"))

    task_ids = sorted({tid for candidate, tid in records if candidate == variant})
    task_id = str(st.selectbox("Task", task_ids, key="task"))

    profiles = sorted({prof for v, prof, _, tid in states if v == variant and tid == task_id})
    profile = (
        _select_default("Profile", profiles, _DEFAULT_PROFILE, "profile")
        if profiles
        else _DEFAULT_PROFILE
    )

    injections = sorted(
        {inj for v, prof, inj, tid in states if v == variant and prof == profile and tid == task_id}
    )
    if len(injections) > 1:
        injection = _select_default("Injection", injections, _DEFAULT_INJECTION, "injection")
    else:
        injection = injections[0] if injections else _DEFAULT_INJECTION

    return variant, task_id, profile, injection


def _conversation(record: AgentRecord) -> None:
    """Render the trajectory as chat messages with tool results in expanders."""
    st.subheader("Conversation")
    for turn in views.turns(record):
        if turn.role in ("user", "assistant"):
            with st.chat_message(turn.role):
                if turn.content:
                    st.write(turn.content)
                for call in turn.tool_calls:
                    st.code(call)
        elif turn.role == "tool":
            with st.expander("tool result"):
                st.write(turn.content)
        else:
            st.write(turn.content)


def _ground_truth(record: AgentRecord, task: Task | None) -> None:
    """Render the reward metric, instruction, expected actions, and outputs."""
    st.subheader("Ground truth")
    st.metric("Reward", "PASS" if record.reward >= 1.0 else "FAIL")
    if task is None:
        st.caption("Task definitions are not loaded.")
        return
    st.write(task.instruction)
    action_rows = [
        {
            "action": f"{action.name}({json.dumps(action.kwargs, sort_keys=True)})",
            "matched": matched,
        }
        for action, matched in views.matched_expected_actions(task, record)
    ]
    st.dataframe(pd.DataFrame(action_rows, columns=["action", "matched"]), hide_index=True)
    if task.outputs:
        st.write("Required outputs:")
        for output in task.outputs:
            st.write(f"- {output}")
    else:
        st.caption("No required outputs.")


def _verdicts(state: StateRecord | None, grouped: dict[str, list[Verdict]]) -> None:
    """Render the judge verdicts for the selected state's hash."""
    st.subheader("Verdicts")
    if state is None:
        st.caption("No serialized state for this selection.")
        return
    matching = grouped.get(state.state_hash, [])
    if not matching:
        st.caption("No verdicts for this state.")
        return
    st.dataframe(views.verdict_rows(matching), hide_index=True)
