"""Pipeline flow strip: five stations joined by pipes, drawn as one SVG.

The strip visualizes the study as a left-to-right pipeline. Each station is a
folder of records whose count comes from the read-only data loaders, and each
pipe is a study step that moves data from one station to the next. ``render_svg``
is a pure function so it is unit-testable without a Streamlit runtime; ``strip``
injects the shared motion stylesheet and renders the SVG on a page.
"""

from enum import StrEnum

import streamlit as st

from decision_judges.ui import components, data


class Station(StrEnum):
    """A stage of the pipeline, each backed by one folder of records."""

    tasks = "tasks"
    conversations = "conversations"
    judge_text = "judge_text"
    verdicts = "verdicts"
    findings = "findings"


PIPES: dict[str, tuple[Station, Station]] = {
    "run-agent": (Station.tasks, Station.conversations),
    "serialize": (Station.conversations, Station.judge_text),
    "judge": (Station.judge_text, Station.verdicts),
    "analyze": (Station.verdicts, Station.findings),
}

KEYFRAMES: tuple[str, ...] = ("flow-dots", "station-glow")

_ORDER: tuple[Station, ...] = (
    Station.tasks,
    Station.conversations,
    Station.judge_text,
    Station.verdicts,
    Station.findings,
)
_LABELS: dict[Station, str] = {
    Station.tasks: "Tasks",
    Station.conversations: "Conversations",
    Station.judge_text: "Judge text",
    Station.verdicts: "Verdicts",
    Station.findings: "Findings",
}
_TITLE = "Study pipeline: tasks to findings"

_VIEW_W = 1040
_MARGIN = 20
_BOX_W = 160
_GAP = 50
_DOTS = 4
_DEFAULT_DURATION = 1.8


def _findings_count(paths: data.Paths) -> int:
    """Count the published findings files under the results directory."""
    results = paths.results_dir
    if not results.is_dir():
        return 0
    return sum(1 for _ in results.glob("*_findings.md"))


def counts(paths: data.Paths) -> dict[Station, int]:
    """Return the record count backing each station, read through the loaders."""
    try:
        tasks = len(data.load_tasks_for_ui(paths))
    except Exception:
        tasks = 0
    return {
        Station.tasks: tasks,
        Station.conversations: len(data.load_agent_records(paths)),
        Station.judge_text: len(data.load_states(paths)),
        Station.verdicts: len(data.load_verdicts(paths)),
        Station.findings: _findings_count(paths),
    }


def _box_x(index: int) -> int:
    """Return the left edge x of the station box at an index."""
    return _MARGIN + index * (_BOX_W + _GAP)


def _dims(compact: bool) -> tuple[int, int, int]:
    """Return the (height, box top, box height) for the chosen density."""
    if compact:
        return 48, 6, 24
    return 120, 18, 84


def _station_svg(station: Station, count: int, active: bool, compact: bool) -> str:
    """Return one station group: a rounded rect with its count and label."""
    height, box_y, box_h = _dims(compact)
    center_x = _box_x(_ORDER.index(station)) + _BOX_W // 2
    label = _LABELS[station]
    cls = "station active" if active else "station"
    if compact:
        count_y, count_size, label_y, label_size = box_y + 17, 14, height - 3, 11
    else:
        count_y, count_size, label_y, label_size = box_y + 42, 30, box_y + 68, 15
    return (
        f'<g class="{cls}" aria-label="{label}: {count}">'
        f'<rect x="{_box_x(_ORDER.index(station))}" y="{box_y}" '
        f'width="{_BOX_W}" height="{box_h}" rx="6" />'
        f'<text class="count" x="{center_x}" y="{count_y}" '
        f'text-anchor="middle" font-size="{count_size}">{count}</text>'
        f'<text class="label" x="{center_x}" y="{label_y}" '
        f'text-anchor="middle" font-size="{label_size}">{label}</text>'
        f"</g>"
    )


def _pipe_path(index: int, center_y: int) -> str:
    """Return the path data joining the box at an index to the next box."""
    x1 = _box_x(index) + _BOX_W
    x2 = _box_x(index + 1)
    return f"M {x1} {center_y} L {x2} {center_y}"


def _dots_svg(path: str, paid: bool) -> str:
    """Return the animated flow dots spread along a pipe path."""
    kind = "paid" if paid else "free"
    step = _DEFAULT_DURATION / _DOTS
    dots = []
    for index in range(_DOTS):
        delay = round(-index * step, 3)
        dots.append(
            f'<circle class="dot {kind}" r="4" '
            f"style=\"offset-path: path('{path}'); animation-delay: {delay}s\" />"
        )
    return "".join(dots)


def _arrowhead_svg(path_index: int, center_y: int) -> str:
    """Return a static arrowhead at a pipe midpoint for reduced-motion users."""
    x1 = _box_x(path_index) + _BOX_W
    x2 = _box_x(path_index + 1)
    mid = (x1 + x2) // 2
    return (
        f'<path class="arrowhead" d="M {mid - 5} {center_y - 5} '
        f'L {mid + 5} {center_y} L {mid - 5} {center_y + 5} Z" />'
    )


def _pipes_svg(running: str | None, paid: bool, center_y: int) -> str:
    """Return every pipe, marking the running step with dots and an arrowhead."""
    parts: list[str] = []
    for index, (step, _stations) in enumerate(PIPES.items()):
        path = _pipe_path(index, center_y)
        is_running = step == running
        cls = "pipe flow" if is_running else "pipe"
        parts.append(f'<path class="{cls}" data-step="{step}" d="{path}" />')
        if is_running:
            parts.append(_dots_svg(path, paid))
            parts.append(_arrowhead_svg(index, center_y))
    return "".join(parts)


def render_svg(
    counts: dict[Station, int],
    *,
    active: Station | None,
    running: str | None,
    paid: bool,
    compact: bool,
) -> str:
    """Return the pipeline strip as a standalone SVG string.

    The active station is stroked and glows via the ``active`` class, the
    running pipe carries a ``flow`` class with dots colored ``paid`` or ``free``,
    and compact mode renders a short strip with labels beneath the counts while
    full mode renders a taller strip with labels inside each station.
    """
    height, box_y, box_h = _dims(compact)
    center_y = box_y + box_h // 2
    stations = "".join(
        _station_svg(station, counts.get(station, 0), station == active, compact)
        for station in _ORDER
    )
    return (
        f'<svg class="flow-strip" role="img" viewBox="0 0 {_VIEW_W} {height}" '
        f'width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid meet">'
        f"<title>{_TITLE}</title>"
        f'<g class="pipes">{_pipes_svg(running, paid, center_y)}</g>'
        f'<g class="stations">{stations}</g>'
        f"</svg>"
    )


def strip(
    paths: data.Paths,
    *,
    active: Station | None = None,
    running: str | None = None,
    paid: bool = False,
    compact: bool = True,
) -> None:
    """Inject the motion styles once and render the strip for the current counts.

    The SVG is written through ``st.markdown`` with HTML enabled because
    ``st.html`` sanitizes SVG elements away, which would leave the strip blank.
    """
    components.motion_styles()
    svg = render_svg(counts(paths), active=active, running=running, paid=paid, compact=compact)
    st.markdown(svg, unsafe_allow_html=True)
