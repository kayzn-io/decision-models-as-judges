"""Label page: attach a taxonomy label to a failing trajectory (local mode)."""

import pandas as pd
import streamlit as st

from decision_judges.labels import (
    TAXONOMY,
    TAXONOMY_DESCRIPTIONS,
    Label,
    label_counts,
    label_title,
)
from decision_judges.ui import charts, components, data, views
from decision_judges.ui.flow import Station

_TARGET = 50
_WIDGET_KEYS = ("trajectory", "label", "note")
_PURPOSE = "Pick why each failed conversation failed; your labels feed the failure-type experiment."
_WHY = "Your labels are the truth the failure-type experiment scores every judge against."
_NEXT_HINT = "Watching the judges shows how they decide on a conversation you choose."
_INTRO_FLAG = "label_intro_dismissed"
_INTRO = (
    "Consistent labels are hard: two people often read the same failure differently, "
    "so the failure types only hold if one rule is applied every time.\n\n"
    "The rule: choose the first cause in the conversation that made the outcome wrong, "
    "not the last symptom.\n\n"
    "These labels are the truth the failure-type experiment scores every judge against."
)
_CONVERSATION_TERM = "a full exchange between the simulated customer and the agent"
_KEY_HINT = "Press 1 to 8 to pick a label; the buttons work without it too."
# Progressive enhancement: the number keys click the matching radio input when the
# browser runs the script, and the radio stays fully usable when it does not.
_KEY_SCRIPT = """
<script>
document.addEventListener('keydown', function (event) {
  if (event.key >= '1' && event.key <= '8') {
    const doc = window.parent ? window.parent.document : document;
    const radios = doc.querySelectorAll('input[type="radio"]');
    const target = radios[Number(event.key) - 1];
    if (target) { target.click(); }
  }
});
</script>
"""


def render() -> None:
    """Render the labeling workflow for failing trajectories in local mode."""
    if not data.is_local():
        st.info("Labeling runs only locally with JUDGES_LOCAL=1.")
        components.footer()
        return

    paths = data.Paths.from_env()
    store = data.labels_store(paths)
    failing = data.failing_trajectories(paths)

    components.page_header("Judge it yourself", _PURPOSE, why=_WHY)
    st.markdown(
        "Read one failed "
        + components.term("conversation", _CONVERSATION_TERM)
        + " and pick the failure type you see.",
        unsafe_allow_html=True,
    )
    components.flow_context(paths, Station.findings)
    if not failing:
        st.caption("No failed conversations to label.")
        components.next_link("Watch the judges work", "/live", _NEXT_HINT)
        components.footer()
        return

    _intro()
    labeled = store.latest()
    done, target = store.progress(_TARGET)
    progress_column, tally_column = st.columns([2, 1])
    with progress_column:
        st.progress(min(done / target, 1.0) if target else 0.0)
        st.write(f"{done} of {target} labeled")
    with tally_column:
        _tally(store)

    variant, task_id = _select(failing, labeled)
    record = data.load_agent_records(paths)[(variant, task_id)]
    components.render_conversation(views.turns(record))

    label = st.radio("Label", TAXONOMY, index=None, format_func=label_title, key="label")
    st.html(_KEY_SCRIPT)
    components.keyboard_hint(_KEY_HINT)
    if label is not None:
        st.caption(TAXONOMY_DESCRIPTIONS[str(label)])
    note = st.text_area("Note", key="note")
    if st.button("Save label", key="save", type="primary", disabled=label is None):
        store.append(Label(variant=variant, task_id=task_id, label=str(label), note=note))
        st.session_state[_INTRO_FLAG] = True
        _advance()
        st.toast("Label saved.")
        st.rerun()
    components.next_link("Watch the judges work", "/live", _NEXT_HINT)
    components.footer()


def _intro() -> None:
    """Show the labeling guidance, open until the first label is saved this session."""
    dismissed = bool(st.session_state.get(_INTRO_FLAG, False))
    with st.expander("Before you label", expanded=not dismissed):
        st.markdown(_INTRO)


def _tally(store: data.LabelStore) -> None:
    """Render a small bar of the running label counts, when any labels exist."""
    counts = label_counts(store.load())
    if not counts:
        return
    frame = pd.DataFrame(counts, columns=["category", "count"])
    st.altair_chart(charts.bar(frame, "category", "count"), use_container_width=True)


def _select(
    failing: list[tuple[str, str]], labeled: dict[tuple[str, str], Label]
) -> tuple[str, str]:
    """Render the trajectory selector, defaulting to the first unlabeled pair."""
    unlabeled = [pair for pair in failing if pair not in labeled]
    default = unlabeled[0] if unlabeled else failing[0]
    options = [f"{variant} · {task_id}" for variant, task_id in failing]
    choice = st.selectbox("Conversation", options, index=failing.index(default), key="trajectory")
    return failing[options.index(str(choice))]


def _advance() -> None:
    """Clear the widget state so the next run defaults to the next unlabeled pair."""
    for key in _WIDGET_KEYS:
        st.session_state.pop(key, None)
