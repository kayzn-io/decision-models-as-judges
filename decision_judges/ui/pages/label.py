"""Label page (placeholder; enabled only in local mode)."""

import streamlit as st


def render() -> None:
    """Render the label placeholder."""
    st.title("Label")
    st.write("Human labeling is coming soon.")
