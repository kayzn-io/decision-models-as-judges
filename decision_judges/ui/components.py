"""Shared Streamlit rendering helpers reused across pages."""

import urllib.parse
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from streamlit.navigation.page import StreamlitPage
from streamlit.runtime.scriptrunner import get_script_run_ctx

from decision_judges import __version__
from decision_judges.ui.views import Turn

if TYPE_CHECKING:
    from decision_judges.ui import data
    from decision_judges.ui.flow import Station

_ASSETS = Path(__file__).parent / "assets"
LOGO = _ASSETS / "kayzn-logo.png"
FAVICON = _ASSETS / "kayzn-favicon.png"
MOTION_CSS = _ASSETS / "motion.css"

_MOTION_MARKER = "_flow_motion_css_injected"


@lru_cache(maxsize=1)
def _motion_css() -> str:
    """Return the motion stylesheet text, read once per process."""
    return MOTION_CSS.read_text(encoding="utf-8")


def motion_styles() -> None:
    """Inject the flow-strip stylesheet once per script run."""
    ctx = get_script_run_ctx()
    if ctx is not None:
        if getattr(ctx, _MOTION_MARKER, False):
            return
        try:
            setattr(ctx, _MOTION_MARKER, True)
        except (AttributeError, TypeError):
            pass
    st.html(f"<style>{_motion_css()}</style>")


_PAGE_REGISTRY: dict[str, StreamlitPage] = {}


def register_pages(pages: Mapping[str, StreamlitPage]) -> None:
    """Record the navigation pages by URL path so links can switch pages in place."""
    _PAGE_REGISTRY.clear()
    _PAGE_REGISTRY.update(pages)


def _live_page(url_path: str) -> StreamlitPage | None:
    """Return the registered page for a URL path when this run registered it.

    Liveness is scoped to the current script run: a page counts only when
    ``st.navigation`` registered its URL path this run, so a per-page test
    harness that never built the navigation falls back to a plain anchor.
    """
    page = _PAGE_REGISTRY.get(url_path)
    ctx = get_script_run_ctx()
    if page is None or ctx is None:
        return None
    live = {info.get("url_pathname") for info in ctx.pages_manager.get_pages().values()}
    return page if url_path in live else None


def _in_app_anchor(href: str, label: str, *, primary: bool) -> None:
    """Render a same-tab anchor styled as a link, or as a primary button.

    The anchor is emitted through ``st.html`` rather than ``st.markdown``:
    Streamlit's markdown renderer forces ``target="_blank"`` on every link,
    which would open in-app navigation in a new tab. ``st.html`` leaves the
    markup untouched, so with no target the browser navigates in the same tab.
    """
    motion_styles()
    css_class = "in-app-link primary" if primary else "in-app-link"
    st.html(f'<a href="{href}" class="{css_class}">{label}</a>')


def page_link(
    path: str,
    label: str,
    *,
    query: Mapping[str, str] | None = None,
    icon: str | None = None,
    primary: bool = False,
) -> None:
    """Link to another app page in the same tab.

    When the destination page is registered this run, the link switches pages
    through Streamlit so the sidebar highlights it, carrying any query
    parameters. Otherwise it falls back to a same-tab anchor, which the per-page
    test harness relies on since it registers no pages.
    """
    page = _live_page(path.lstrip("/"))
    if page is not None:
        st.page_link(
            page,
            label=label,
            icon=icon,
            query_params=dict(query) if query else None,
        )
        return
    href = f"{path}?{urllib.parse.urlencode(dict(query))}" if query else path
    text = f"{icon} {label}" if icon else label
    _in_app_anchor(href, text, primary=primary)


_REPO = "https://github.com/kayzn-io/decision-models-as-judges"
_FOOTER = (
    "Built by [Kayzn](https://kayzn.io). Open source under the "
    f"[Apache License 2.0]({_REPO}/blob/main/LICENSE). [Source]({_REPO})"
)


def page_header(
    title: str, purpose: str, breadcrumb: Sequence[str] = (), why: str | None = None
) -> None:
    """Render a page title, a one-line muted purpose, an optional why line and breadcrumb."""
    st.title(title)
    st.caption(purpose)
    if why:
        why_line(why)
    if breadcrumb:
        st.caption(" / ".join(breadcrumb))


def why_line(text: str) -> None:
    """Render a muted one-line note on why the current page exists."""
    st.caption(f"Why this page exists: {text}")


def flow_context(paths: "data.Paths", station: "Station | None") -> None:
    """Render the compact pipeline strip with one station marked active."""
    from decision_judges.ui import flow

    flow.strip(paths, active=station, compact=True)


def next_link(label: str, path: str, hint: str) -> None:
    """Link to the next page in the same tab, above the footer, with a muted hint."""
    page_link(path, f"Next: {label}")
    st.caption(hint)


def footer() -> None:
    """Render the divider and the Kayzn attribution and license links."""
    st.divider()
    st.markdown(_FOOTER)


def empty_state(what: str, command: str, run_step: int | None = None) -> None:
    """State what a view will show and how to produce it.

    Without a run step the CLI command sits inline. With one, the view points to
    the matching step on the Run page and folds the command into an expander so
    it stays available without competing with the primary call to action.
    """
    st.caption(what)
    if run_step is None:
        st.code(command, language="bash")
        return
    page_link("/run", f"Run step {run_step} on the Run page")
    st.caption("Run it there to produce this.")
    with st.expander("Command line"):
        st.code(command, language="bash")


def how_to_read(rows: Sequence[tuple[str, str]]) -> None:
    """Show a plain-word key to a table's columns in a collapsible panel."""
    with st.expander("How to read this"):
        lines = ["| Column | Meaning |", "| --- | --- |"]
        lines += [f"| {name} | {meaning} |" for name, meaning in rows]
        st.markdown("\n".join(lines))


def keyboard_hint(text: str) -> None:
    """Render a small caption describing an optional keyboard shortcut."""
    st.caption(text)


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
                    st.code(call, language="json", wrap_lines=True)
        elif turn.role == "tool":
            with st.expander("tool result"):
                st.write(turn.content)
        else:
            st.write(turn.content)
