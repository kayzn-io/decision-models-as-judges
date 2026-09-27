"""Trajectories page: one run's conversation beside its ground truth and verdicts."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.serialize import Injection, StateRecord
from decision_judges.types import Verdict
from decision_judges.ui import components, data, formatting, views
from decision_judges.ui.flow import Station

_DEFAULT_PROFILE = "full"
_DEFAULT_INJECTION = "none"
_PURPOSE = "Read one conversation beside its ground truth and every judge's verdict."
_WHY = "Seeing one conversation end to end makes the summary numbers concrete."
_NEXT_HINT = "The experiments turn many conversations like this one into one score per judge."
_JUDGE_VIEW = "Show only what the judge reads"
_JUDGE_VIEW_CAPTION = "The judge never sees the ground truth; the reading copy is made without it."
_CONVERSATION_TERM = "a full exchange between the customer and the agent"
# The reader-facing name for each internal variant value.
_VARIANT_LABELS = {"baseline": "Careful agent", "degraded": "Rushed agent"}
_LEGEND = (
    "Percentages are each judge's probability that the call was necessary and that its "
    "arguments were consistent."
)
_HELP_QUESTIONS = ("verdict", "completed")


def render() -> None:
    """Render one run's conversation, ground truth, and side-by-side verdicts."""
    paths = data.Paths.from_env()
    records = data.load_agent_records(paths)
    with st.spinner("Reading cached verdicts"):
        states = data.load_states(paths)
        grouped = data.verdicts_by_state(data.load_verdicts(paths))
    tasks = data.load_tasks_for_ui(paths)

    components.page_header("Trajectories", _PURPOSE, why=_WHY)
    st.markdown(
        "You are reading one "
        + components.term("conversation", _CONVERSATION_TERM)
        + " and the answers each judge gave about it.",
        unsafe_allow_html=True,
    )
    components.flow_context(paths, Station.judge_text)
    if not records:
        st.caption("No agent runs are available yet.")
        components.next_link("Experiments", "/gates", _NEXT_HINT)
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
            f"reading copy: {sentence}"
        )

    steps = data.step_states(states, variant, profile, task_id)
    step_frames = views.step_scores(steps, grouped)

    judge_view = st.toggle(_JUDGE_VIEW, value=False, key="judge_view")
    conversation, ground_truth = st.columns([3, 2])
    with conversation:
        if judge_view:
            _judge_view(state)
        else:
            _conversation(record, step_frames, sentence)
            if any(not frame.empty for frame in step_frames.values()):
                st.caption(_LEGEND)
    with ground_truth:
        _ground_truth(record, task)
    _verdicts(state, grouped)
    components.next_link("Experiments", "/gates", _NEXT_HINT)
    components.footer()


def _judge_view(state: StateRecord | None) -> None:
    """Render the text the judge reads, with a note on what it omits."""
    st.subheader("Conversation")
    if state is None:
        st.caption("No reading copy for this selection.")
        return
    with st.container(height=600):
        st.text(state.text)
    st.caption(_JUDGE_VIEW_CAPTION)


def _outcome_help() -> dict[str, str]:
    """Return the outcome question texts keyed by id, for verdict-table tooltips."""
    return {
        question.id: question.text
        for question in G3Outcome().questions()
        if question.id in _HELP_QUESTIONS
    }


def _verdict_config(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the verdict-table column config with the question texts as help."""
    config: dict[str, Any] = dict(formatting.column_config_for(frame))
    helps = _outcome_help()
    if "verdict" in frame.columns:
        config["verdict"] = st.column_config.TextColumn("Verdict", help=helps.get("verdict"))
    if "completed" in frame.columns:
        config["completed"] = st.column_config.NumberColumn(
            "Completed", format="%.3f", width="small", help=helps.get("completed")
        )
    return config


def _select_default(
    label: str,
    options: list[str],
    default: str,
    key: str,
    format_func: Callable[[str], str] | None = None,
) -> str:
    """Render a selectbox defaulting to ``default`` when it is among options."""
    index = options.index(default) if default in options else 0
    if format_func is not None:
        return str(st.selectbox(label, options, index=index, key=key, format_func=format_func))
    return str(st.selectbox(label, options, index=index, key=key))


def _variant_label(variant: str) -> str:
    """Return the reader-facing name for an internal variant value."""
    return _VARIANT_LABELS.get(variant, variant)


def _current(key: str, options: list[str], fallback: str) -> str:
    """Return the live selection for a widget key, validated against its options."""
    chosen = st.session_state.get(key, fallback)
    if chosen in options:
        return str(chosen)
    return options[0] if options else fallback


def _select(
    records: dict[tuple[str, str], AgentRecord],
    states: dict[tuple[str, str, str, str], StateRecord],
) -> tuple[str, str, str, str]:
    """Render the variant, task, profile, and injection selectors and return them.

    A ``variant``, ``task``, or ``injection`` query parameter preselects the
    matching widget so a link can deep-link into a specific state; the resulting
    selection is written back to the query parameters. The selectors share one
    row of columns inside a bordered container so they read as one control group.
    """
    variants = sorted({variant for variant, _ in records})
    desired = views.selection_from_query(
        dict(st.query_params),
        {"variant": variants[0], "task": "", "injection": _DEFAULT_INJECTION},
    )
    count = _selector_count(records, states, variants, desired)
    with st.container(border=True):
        columns = st.columns(count)
        cursor = 0
        with columns[cursor]:
            variant = _select_default(
                "Variant", variants, desired["variant"], "variant", format_func=_variant_label
            )
        cursor += 1

        task_ids = sorted({tid for candidate, tid in records if candidate == variant})
        with columns[cursor]:
            task_id = _select_default("Request", task_ids, desired["task"], "task")
        cursor += 1

        profiles = sorted({prof for v, prof, _, tid in states if v == variant and tid == task_id})
        if profiles:
            with columns[cursor]:
                profile = _select_default("Profile", profiles, _DEFAULT_PROFILE, "profile")
            cursor += 1
        else:
            profile = _DEFAULT_PROFILE

        injections = sorted(
            {
                inj
                for v, prof, inj, tid in states
                if v == variant and prof == profile and tid == task_id
            }
        )
        if len(injections) > 1:
            with columns[cursor]:
                injection = _select_default(
                    "Injection", injections, desired["injection"], "injection"
                )
        else:
            injection = injections[0] if injections else _DEFAULT_INJECTION

    _write_query_params(variant, task_id, injection)
    return variant, task_id, profile, injection


def _selector_count(
    records: dict[tuple[str, str], AgentRecord],
    states: dict[tuple[str, str, str, str], StateRecord],
    variants: list[str],
    desired: dict[str, str],
) -> int:
    """Return how many selector columns the current selection needs (variant, task, +optional)."""
    variant = _current("variant", variants, desired["variant"])
    task_ids = sorted({tid for candidate, tid in records if candidate == variant})
    task_id = _current("task", task_ids, desired["task"])
    profiles = sorted({prof for v, prof, _, tid in states if v == variant and tid == task_id})
    profile = _current("profile", profiles, _DEFAULT_PROFILE) if profiles else _DEFAULT_PROFILE
    injections = sorted(
        {inj for v, prof, inj, tid in states if v == variant and prof == profile and tid == task_id}
    )
    return 2 + (1 if profiles else 0) + (1 if len(injections) > 1 else 0)


def _write_query_params(variant: str, task_id: str, injection: str) -> None:
    """Write the current selection to the query parameters so links can share it."""
    st.query_params["variant"] = variant
    st.query_params["task"] = task_id
    st.query_params["injection"] = injection


def _conversation(
    record: AgentRecord,
    step_frames: dict[int, pd.DataFrame],
    injected: str = "",
) -> None:
    """Render the trajectory as chat messages with per-step judge scores inline.

    The conversation lives in a fixed-height, scrolling container so a long run
    does not push the ground-truth and verdict panels off screen. Tool results
    collapse into an expander labeled with the tool name and result length. When
    ``injected`` is non-empty the spliced evaluator-directed sentence closes the
    conversation tagged ``[injected]``.
    """
    st.subheader("Conversation")
    with st.container(height=600):
        step_index = 0
        pending: list[str] = []
        for turn in views.turns(record):
            if turn.role in ("user", "assistant"):
                with st.chat_message(turn.role):
                    if turn.content:
                        st.write(turn.content)
                    for call in turn.tool_calls:
                        st.code(call, language="json", wrap_lines=True)
                        pending.append(call.split("(", 1)[0])
                        _step_scores(step_index, step_frames)
                        step_index += 1
            elif turn.role == "tool":
                name = pending.pop(0) if pending else "tool"
                with st.expander(f"{name} · {len(turn.content)} chars"):
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
) -> None:
    """Render one compact caption per judge under a tool call, when scores exist."""
    frame = step_frames.get(step_index)
    if frame is None or frame.empty:
        return
    for _, row in frame.iterrows():
        st.caption(
            f"{row['judge_id']}: necessary {formatting.pct(float(row['necessary']))}, "
            f"arguments_consistent {formatting.pct(float(row['arguments_consistent']))}"
        )


def _reward_badge(passed: bool) -> None:
    """Render the run's pass or fail outcome as a labeled badge, text when unsupported."""
    label = "PASS" if passed else "FAIL"
    if hasattr(st, "badge"):
        st.badge(label, color="green" if passed else "red")
    else:
        st.write(label)


def _ground_truth(record: AgentRecord, task: Task | None) -> None:
    """Render the ground-truth badge and the customer request."""
    st.subheader("Ground truth")
    _reward_badge(record.reward >= 1.0)
    st.caption("Ground truth")
    if task is None:
        st.caption("Task definitions are not loaded.")
        return
    st.markdown("**Customer request**")
    st.write(task.instruction)


def _verdicts(state: StateRecord | None, grouped: dict[str, list[Verdict]]) -> None:
    """Render the judge verdicts for the selected state's hash."""
    st.subheader("Verdicts")
    if state is None:
        st.caption("No reading copy for this selection.")
        return
    matching = grouped.get(state.state_hash, [])
    if not matching:
        st.caption("No verdicts for this reading copy.")
        return
    frame = views.verdict_rows(matching)
    st.dataframe(
        frame,
        column_config=_verdict_config(frame),
        hide_index=True,
        width="stretch",
    )
