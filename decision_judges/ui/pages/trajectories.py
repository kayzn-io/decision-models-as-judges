"""Trajectories page: one run's conversation beside its ground truth and verdicts."""

import json

import pandas as pd
import streamlit as st

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.serialize import Injection, StateRecord
from decision_judges.types import Verdict
from decision_judges.ui import components, data, views

_DEFAULT_PROFILE = "full"
_DEFAULT_INJECTION = "none"
_PURPOSE = "Read one agent run beside its ground truth and every judge's verdict."


def render() -> None:
    """Render one run's conversation, ground truth, and side-by-side verdicts."""
    paths = data.Paths.from_env()
    records = data.load_agent_records(paths)
    states = data.load_states(paths)
    grouped = data.verdicts_by_state(data.load_verdicts(paths))
    tasks = data.load_tasks_for_ui(paths)

    components.page_header("Trajectories", _PURPOSE)
    if not records:
        st.caption("No agent runs are available yet.")
        components.footer()
        return

    trajectory_states = data.trajectory_states(states)
    variant, task_id, profile, injection = _select(records, trajectory_states)
    record = records[(variant, task_id)]
    state = trajectory_states.get((variant, profile, injection, task_id))
    task = tasks.get(task_id)

    sentence = ""
    if state is not None and state.injection is not Injection.none:
        original = trajectory_states.get((variant, profile, _DEFAULT_INJECTION, task_id))
        sentence = views.injected_sentence(original, state)
        st.warning(
            f"Evaluator-directed '{state.injection.value}' injection is present in this "
            f"state: {sentence}"
        )

    steps = data.step_states(states, variant, profile, task_id)
    step_frames = views.step_scores(steps, grouped)
    expected = views.expected_step_flags(task, record) if task is not None else None

    conversation, ground_truth, verdicts = st.columns(3)
    with conversation:
        _conversation(record, step_frames, expected, sentence)
    with ground_truth:
        _ground_truth(record, task)
    with verdicts:
        _verdicts(state, grouped)
    components.footer()


def _select_default(label: str, options: list[str], default: str, key: str) -> str:
    """Render a selectbox defaulting to ``default`` when it is among options."""
    index = options.index(default) if default in options else 0
    return str(st.selectbox(label, options, index=index, key=key))


def _select(
    records: dict[tuple[str, str], AgentRecord],
    states: dict[tuple[str, str, str, str], StateRecord],
) -> tuple[str, str, str, str]:
    """Render the variant, task, profile, and injection selectors and return them.

    A ``variant``, ``task``, or ``injection`` query parameter preselects the
    matching widget so a link can deep-link into a specific state; the resulting
    selection is written back to the query parameters.
    """
    variants = sorted({variant for variant, _ in records})
    desired = views.selection_from_query(
        dict(st.query_params),
        {"variant": variants[0], "task": "", "injection": _DEFAULT_INJECTION},
    )
    variant = _select_default("Variant", variants, desired["variant"], "variant")

    task_ids = sorted({tid for candidate, tid in records if candidate == variant})
    task_id = _select_default("Task", task_ids, desired["task"], "task")

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
        injection = _select_default("Injection", injections, desired["injection"], "injection")
    else:
        injection = injections[0] if injections else _DEFAULT_INJECTION

    _write_query_params(variant, task_id, injection)
    return variant, task_id, profile, injection


def _write_query_params(variant: str, task_id: str, injection: str) -> None:
    """Write the current selection to the query parameters so links can share it."""
    st.query_params["variant"] = variant
    st.query_params["task"] = task_id
    st.query_params["injection"] = injection


def _conversation(
    record: AgentRecord,
    step_frames: dict[int, pd.DataFrame],
    expected: list[bool] | None,
    injected: str = "",
) -> None:
    """Render the trajectory as chat messages with per-step judge scores inline.

    When ``injected`` is non-empty the spliced evaluator-directed sentence is
    shown as a closing turn tagged ``[injected]`` so the injected text is visible
    beside the real conversation.
    """
    st.subheader("Conversation")
    step_index = 0
    for turn in views.turns(record):
        if turn.role in ("user", "assistant"):
            with st.chat_message(turn.role):
                if turn.content:
                    st.write(turn.content)
                for call in turn.tool_calls:
                    st.code(call)
                    _step_scores(step_index, step_frames, expected)
                    step_index += 1
        elif turn.role == "tool":
            with st.expander("tool result"):
                st.write(turn.content)
        else:
            st.write(turn.content)
    if injected:
        with st.chat_message("assistant"):
            st.write(injected)
            st.caption("[injected]")


def _step_scores(
    step_index: int,
    step_frames: dict[int, pd.DataFrame],
    expected: list[bool] | None,
) -> None:
    """Render one compact caption per judge under a tool call, when scores exist."""
    frame = step_frames.get(step_index)
    if frame is None or frame.empty:
        return
    mark = ""
    if expected is not None and step_index < len(expected):
        mark = "✓ " if expected[step_index] else "✗ "
    for _, row in frame.iterrows():
        st.caption(
            f"{mark}{row['judge_id']}: necessary {float(row['necessary']):.0%}, "
            f"arguments_consistent {float(row['arguments_consistent']):.0%}"
        )


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
