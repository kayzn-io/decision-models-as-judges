"""Shared Streamlit rendering helpers reused across pages."""

import streamlit as st

from decision_judges.ui.views import Turn


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
