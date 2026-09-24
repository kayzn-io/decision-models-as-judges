"""Streamlit entrypoint: a navigation shell over the cached data layer.

The runnable body is guarded by a Streamlit script-run context so importing
this module for its helpers has no side effects.
"""

import streamlit as st
from streamlit.navigation.page import StreamlitPage
from streamlit.runtime.scriptrunner import get_script_run_ctx

from decision_judges import __version__
from decision_judges.ui import data
from decision_judges.ui.pages import gates, label, live, overview, trajectories

_SHARED_PAGES = [
    ("Overview", overview.render),
    ("Trajectories", trajectories.render),
    ("Gates", gates.render),
]
_LOCAL_PAGES = [
    ("Label", label.render),
    ("Live", live.render),
]


def page_specs(local: bool) -> list[str]:
    """Return the ordered page titles, adding the local-only pages when local."""
    titles = [title for title, _ in _SHARED_PAGES]
    if local:
        titles += [title for title, _ in _LOCAL_PAGES]
    return titles


def build_pages(local: bool) -> list[StreamlitPage]:
    """Build the navigation pages, including local-only pages when local."""
    entries = list(_SHARED_PAGES)
    if local:
        entries += _LOCAL_PAGES
    return [
        st.Page(render, title=title, url_path=title.lower(), default=index == 0)
        for index, (title, render) in enumerate(entries)
    ]


def main() -> None:
    """Configure the page, build navigation, and run the selected page."""
    st.set_page_config(page_title="Decision models as judges", layout="wide")
    local = data.is_local()
    mode = "local mode" if local else "shared mode"
    pages = build_pages(local)
    navigation = st.navigation(pages)
    st.sidebar.caption(f"v{__version__} · {mode}")
    navigation.run()


if get_script_run_ctx() is not None:
    main()
