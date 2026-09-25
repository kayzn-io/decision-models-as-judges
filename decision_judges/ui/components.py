"""Shared Streamlit rendering helpers reused across pages."""

from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from decision_judges import __version__
from decision_judges.ui.views import Turn

_ASSETS = Path(__file__).parent / "assets"
LOGO = _ASSETS / "kayzn-logo.png"
FAVICON = _ASSETS / "kayzn-favicon.png"

_REPO = "https://github.com/kayzn-io/decision-models-as-judges"
_FOOTER = (
    "Built by [Kayzn](https://kayzn.io). Open source under the "
    f"[Apache License 2.0]({_REPO}/blob/main/LICENSE). [Source]({_REPO})"
)


def page_header(title: str, purpose: str, breadcrumb: Sequence[str] = ()) -> None:
    """Render a page title, a one-line muted purpose, and an optional breadcrumb."""
    st.title(title)
    st.caption(purpose)
    if breadcrumb:
        st.caption(" / ".join(breadcrumb))


def footer() -> None:
    """Render the divider and the Kayzn attribution and license links."""
    st.divider()
    st.markdown(_FOOTER)


def empty_state(what: str, command: str) -> None:
    """State what a view will show and the command that produces it."""
    st.caption(what)
    st.code(command, language="bash")


def metric_row(items: Sequence[tuple[str, str]]) -> None:
    """Lay out labeled metrics as bordered cards in equal columns."""
    if not items:
        return
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items, strict=True):
        column.metric(label, value, border=True)


def sidebar_brand() -> None:
    """Show the logo, the app name, and the version in the sidebar."""
    if hasattr(st, "logo"):
        st.logo(str(LOGO))
    else:
        st.sidebar.image(str(LOGO))
    st.sidebar.caption("Decision models as judges")
    st.sidebar.caption(f"v{__version__}")


def render_conversation(turns: list[Turn]) -> None:
    """Render trajectory turns as compact chat messages.

    User and assistant turns become chat messages with their tool calls shown
    as code, tool results collapse into an expander, and any other role renders
    as plain text.
    """
    for turn in turns:
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
