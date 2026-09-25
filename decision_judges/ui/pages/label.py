"""Label page: attach a taxonomy label to a failing trajectory (local mode)."""

import streamlit as st

from decision_judges.labels import TAXONOMY, TAXONOMY_DESCRIPTIONS, Label, label_title
from decision_judges.ui import components, data, views

_TARGET = 50
_WIDGET_KEYS = ("trajectory", "label", "note")
_PURPOSE = "Assign a failure type to each failed run; labels feed the taxonomy gate."


def render() -> None:
    """Render the labeling workflow for failing trajectories in local mode."""
    if not data.is_local():
        st.info("Labeling runs only locally with JUDGES_LOCAL=1.")
        components.footer()
        return

    paths = data.Paths.from_env()
    store = data.labels_store(paths)
    failing = data.failing_trajectories(paths)

    components.page_header("Label", _PURPOSE)
    if not failing:
        st.caption("No failing trajectories to label.")
        components.footer()
        return

    labeled = store.latest()
    done, target = store.progress(_TARGET)
    st.progress(min(done / target, 1.0) if target else 0.0)
    st.write(f"{done} of {target} labeled")

    variant, task_id = _select(failing, labeled)
    record = data.load_agent_records(paths)[(variant, task_id)]
    components.render_conversation(views.turns(record))

    label = st.radio("Label", TAXONOMY, index=None, format_func=label_title, key="label")
    if label is not None:
        st.caption(TAXONOMY_DESCRIPTIONS[str(label)])
    note = st.text_area("Note", key="note")
    if st.button("Save label", key="save", type="primary", disabled=label is None):
        store.append(Label(variant=variant, task_id=task_id, label=str(label), note=note))
        _advance()
        st.toast("Label saved.")
        st.rerun()
    components.footer()


def _select(
    failing: list[tuple[str, str]], labeled: dict[tuple[str, str], Label]
) -> tuple[str, str]:
    """Render the trajectory selector, defaulting to the first unlabeled pair."""
    unlabeled = [pair for pair in failing if pair not in labeled]
    default = unlabeled[0] if unlabeled else failing[0]
    options = [f"{variant} · {task_id}" for variant, task_id in failing]
    choice = st.selectbox("Trajectory", options, index=failing.index(default), key="trajectory")
    return failing[options.index(str(choice))]


def _advance() -> None:
    """Clear the widget state so the next run defaults to the next unlabeled pair."""
    for key in _WIDGET_KEYS:
        st.session_state.pop(key, None)
