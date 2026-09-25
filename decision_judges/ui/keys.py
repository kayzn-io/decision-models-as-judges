"""Session-scoped OpenRouter key held in a non-widget slot that survives navigation.

Streamlit drops a widget's session-state entry on any run that does not render
that widget, so a key field shown only on some pages loses its value when the
visitor moves to another page. The value therefore lives under a plain
session-state key that no widget owns; the sidebar field copies into it on
change, and every page reads it through ``get_key``. ``app.main`` renders the
field on every page so the store is refreshed on each run.
"""

from typing import Literal

import streamlit as st

SESSION_KEY = "openrouter_key"
WIDGET_KEY = "openrouter_key_input"

_CLEAR_KEY = "openrouter_key_clear"
_FIELD_CAPTION = "Kept in this browser session only; never saved or logged."
_STATUS_CAPTION = "Paid steps need it; free steps run without one."
_HINT = "Add your OpenRouter key in the sidebar (top left) to run this step."


def get_key() -> str | None:
    """Return the session key, or None when none is set."""
    return st.session_state.get(SESSION_KEY) or None


def require_key_hint() -> str:
    """Return the sentence a card shows when a paid step needs the key."""
    return _HINT


def store_widget_value() -> None:
    """Copy the widget's current value into the non-widget session slot."""
    st.session_state[SESSION_KEY] = st.session_state.get(WIDGET_KEY, "")


def clear_key() -> None:
    """Empty both the stored key and the widget."""
    st.session_state[SESSION_KEY] = ""
    st.session_state[WIDGET_KEY] = ""


def render_key_field() -> None:
    """Render the sidebar key field and an always-visible status badge."""
    has_key = get_key() is not None
    with st.sidebar.expander("Key for this session", expanded=not has_key):
        st.text_input(
            "OpenRouter API key",
            type="password",
            key=WIDGET_KEY,
            value=st.session_state.get(SESSION_KEY, ""),
            on_change=store_widget_value,
        )
        st.caption(_FIELD_CAPTION)
        st.button("Clear", key=_CLEAR_KEY, on_click=clear_key)
    _status(has_key)


def _status(has_key: bool) -> None:
    """Render the key-status badge, with a prompt caption when no key is set."""
    if has_key:
        _badge("Key set for this session", icon=":material/key:", color="green")
        return
    _badge("No key yet", color="gray")
    st.sidebar.caption(_STATUS_CAPTION)


def _badge(text: str, *, color: Literal["green", "gray"], icon: str | None = None) -> None:
    """Render a sidebar badge, falling back to markdown on older Streamlit."""
    if hasattr(st.sidebar, "badge"):
        st.sidebar.badge(text, icon=icon, color=color)
        return
    label = f"{icon} {text}" if icon else text
    st.sidebar.markdown(f":{color}-badge[{label}]")
