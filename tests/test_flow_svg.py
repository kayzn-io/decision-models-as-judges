"""Pure-rendering tests for the pipeline flow strip SVG and its stylesheet."""

from decision_judges.ui.components import MOTION_CSS
from decision_judges.ui.flow import KEYFRAMES, PIPES, Station, render_svg

_COUNTS = {
    Station.tasks: 3,
    Station.conversations: 2,
    Station.judge_text: 7,
    Station.verdicts: 11,
    Station.findings: 1,
}

_LABELS = {
    Station.tasks: "Tasks",
    Station.conversations: "Conversations",
    Station.judge_text: "Judge text",
    Station.verdicts: "Verdicts",
    Station.findings: "Findings",
}


def _svg(**overrides: object) -> str:
    """Render the strip with sensible defaults, overriding named keywords."""
    params: dict[str, object] = {
        "active": None,
        "running": None,
        "paid": False,
        "compact": True,
    }
    params.update(overrides)
    return render_svg(_COUNTS, **params)  # type: ignore[arg-type]


def test_renders_five_stations_with_labels_and_counts() -> None:
    svg = _svg()
    for label in _LABELS.values():
        assert f">{label}</text>" in svg
    for count in _COUNTS.values():
        assert f">{count}</text>" in svg


def test_pipes_cover_every_step() -> None:
    svg = _svg()
    for step in PIPES:
        assert f'data-step="{step}"' in svg


def test_active_station_carries_active_class() -> None:
    svg = _svg(active=Station.verdicts)
    assert svg.count("station active") == 1
    assert 'aria-label="Verdicts: 11"' in svg


def test_no_active_station_has_no_active_class() -> None:
    assert "station active" not in _svg(active=None)


def test_running_marks_the_named_pipe_with_flow_class() -> None:
    svg = _svg(running="judge")
    assert '<path class="pipe flow" data-step="judge"' in svg
    assert '<path class="pipe" data-step="run-agent"' in svg


def test_no_running_has_no_flow_class() -> None:
    assert "pipe flow" not in _svg(running=None)


def test_running_dots_are_paid_or_free_per_flag() -> None:
    paid = _svg(running="judge", paid=True)
    free = _svg(running="judge", paid=False)
    assert "dot paid" in paid and "dot free" not in paid
    assert "dot free" in free and "dot paid" not in free
    dots = paid.count('class="dot paid"')
    assert 3 <= dots <= 5


def test_compact_and_full_differ_in_height() -> None:
    compact = _svg(compact=True)
    full = _svg(compact=False)
    assert 'height="48"' in compact
    assert 'height="48"' not in full
    assert 'height="120"' in full


def test_each_station_has_an_aria_label() -> None:
    svg = _svg()
    for station, label in _LABELS.items():
        assert f'aria-label="{label}: {_COUNTS[station]}"' in svg


def test_svg_declares_role_img_and_a_title() -> None:
    svg = _svg()
    assert 'role="img"' in svg
    assert "<title>" in svg and "</title>" in svg


def test_reduced_motion_pipe_shows_a_static_arrowhead() -> None:
    assert 'class="arrowhead"' in _svg(running="judge")


def test_motion_css_defines_used_keyframes_and_reduced_motion() -> None:
    css = MOTION_CSS.read_text(encoding="utf-8")
    assert "@media (prefers-reduced-motion: reduce)" in css
    for name in KEYFRAMES:
        assert f"@keyframes {name}" in css
