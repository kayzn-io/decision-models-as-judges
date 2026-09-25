"""Streamlit entrypoint: a navigation shell over the cached data layer.

The runnable body is guarded by a Streamlit script-run context so importing
this module for its helpers has no side effects.
"""

from collections.abc import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage
from streamlit.runtime.scriptrunner import get_script_run_ctx

from decision_judges.ui import components, data
from decision_judges.ui.pages import gates, label, live, overview, run, trajectories

_Render = Callable[[], None]

_SHARED_PAGES: list[tuple[str, _Render]] = [
    ("Overview", overview.render),
    ("Trajectories", trajectories.render),
    ("Gates", gates.render),
]
_LOCAL_PAGES: list[tuple[str, _Render]] = [
    ("Label", label.render),
    ("Live", live.render),
]
_RUN_PAGE: tuple[str, _Render] = ("Run", run.render)


def page_specs(local: bool) -> list[str]:
    """Return the shared page titles, adding the local-only pages when local."""
    titles = [title for title, _ in _SHARED_PAGES]
    if local:
        titles += [title for title, _ in _LOCAL_PAGES]
    return titles


def mode_caption(local: bool) -> str | None:
    """Return the sidebar mode caption, or None when there is nothing to say."""
    if local:
        return "Local mode: labeling and live judging enabled"
    return None


def _ordered_pages(local: bool) -> list[tuple[str, _Render]]:
    """Return the navigation pages, inserting Run second in local mode."""
    if not local:
        return list(_SHARED_PAGES)
    overview_page, *rest = _SHARED_PAGES
    return [overview_page, _RUN_PAGE, *rest, *_LOCAL_PAGES]


def build_pages(local: bool) -> list[StreamlitPage]:
    """Build the navigation pages, with the Run page second in local mode."""
    return [
        st.Page(render, title=title, url_path=title.lower(), default=index == 0)
        for index, (title, render) in enumerate(_ordered_pages(local))
    ]


def main() -> None:
    """Configure the page, build navigation, and run the selected page."""
    st.set_page_config(
        page_title="Decision models as judges",
        page_icon=str(components.FAVICON),
        layout="wide",
    )
    local = data.is_local()
    components.sidebar_brand()
    pages = build_pages(local)
    navigation = st.navigation(pages)
    caption = mode_caption(local)
    if caption is not None:
        st.sidebar.caption(caption)
    navigation.run()


if get_script_run_ctx() is not None:
    main()
