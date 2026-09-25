"""Live page (placeholder; enabled only in local mode)."""

import streamlit as st

from decision_judges.ui import components

_PURPOSE = "Stream a judge's verdicts as an agent run unfolds."


def render() -> None:
    """Render the live placeholder."""
    components.page_header("Live", _PURPOSE)
    st.write("Live judging is coming soon.")
    components.footer()
