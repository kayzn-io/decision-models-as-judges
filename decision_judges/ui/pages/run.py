"""Run page: guide the learner through the eight study steps and execute each one.

The page is a thin view. Each step's readiness, examples, and work live in
``steps``; this module renders the cards, launches steps on a background
``StepRunner`` held in session state, and reflects their progress. Paid steps
read the OpenRouter key from the same session field the Live page uses, so one
sidebar entry serves both pages.
"""

from datetime import datetime

import streamlit as st

from decision_judges.runner import StepAlreadyRunning, StepRunner
from decision_judges.ui import components, data, flow, steps
from decision_judges.ui.steps import RunContext, RunStep, StepStatus

_PURPOSE = "Eight steps, in order. Each shows what goes in, what comes out, and what it costs."
_OPENROUTER_KEY = "live_openrouter_key"
_RUNNER_KEY = "run_step_runner"
_BALLOONS_KEY = "run_first_paid_finished"
_SETTLE_MARKER = "run_settle_css"

_STATION_LABELS = {
    "tasks": "Tasks",
    "conversations": "Conversations",
    "judge_text": "Judge text",
    "verdicts": "Verdicts",
    "findings": "Findings",
}
_STATUS_COLORS = {"locked": "gray", "ready": "blue", "partial": "orange", "done": "green"}

_SETTLE_CSS = (
    "<style>.settle{border-left:3px solid #16a34a;padding:0.4rem 0.75rem;"
    "border-radius:4px;animation:settle-in 0.4s ease-out}"
    "@keyframes settle-in{from{opacity:0;transform:translateY(4px)}"
    "to{opacity:1;transform:none}}</style>"
)


def render() -> None:
    """Render the eight study steps as runnable cards in local mode."""
    if not data.is_local():
        st.info("Running the study runs only locally with JUDGES_LOCAL=1.")
        components.footer()
        return

    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    ledger = data.load_ledger(paths)
    key = _sidebar_key()
    runner = _runner(paths)

    components.page_header("Run the study", _PURPOSE)
    _resume_banner(runner)
    _settle_style()

    running = _running_step(runner)
    flow.strip(
        paths,
        active=None,
        running=running.pipe if running is not None else None,
        paid=running.paid if running is not None else False,
        compact=False,
    )

    for index, step in enumerate(steps.STEPS, start=1):
        _card(index, step, paths, study, pricing, ledger, key, runner, running)
    components.footer()


def _sidebar_key() -> str | None:
    """Render the shared OpenRouter key field and return the session value."""
    st.sidebar.subheader("Key for this session")
    st.sidebar.text_input("OpenRouter API key", type="password", key=_OPENROUTER_KEY)
    st.sidebar.caption("Keys stay in this session only; they are never saved or logged.")
    return st.session_state.get(_OPENROUTER_KEY) or None


def _runner(paths: data.Paths) -> StepRunner:
    """Return the step runner held in session state, creating it once."""
    if _RUNNER_KEY not in st.session_state:
        st.session_state[_RUNNER_KEY] = StepRunner(paths.results_dir / ".progress")
    return st.session_state[_RUNNER_KEY]


def _running_step(runner: StepRunner) -> RunStep | None:
    """Return the step whose thread is currently alive, or None."""
    for step in steps.STEPS:
        if runner.is_running(step.id):
            return step
    return None


def _resume_banner(runner: StepRunner) -> None:
    """Warn about the first step whose work was interrupted while the app was closed."""
    for index, step in enumerate(steps.STEPS, start=1):
        if runner.stale(step.id):
            st.warning(
                f"Step {index} stopped while the app was closed; its work is saved. "
                "Run it again to continue."
            )
            return


def _settle_style() -> None:
    """Inject the settle animation stylesheet once per session."""
    if not st.session_state.get(_SETTLE_MARKER):
        st.session_state[_SETTLE_MARKER] = True
        st.html(_SETTLE_CSS)


def _card(
    index: int,
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    ledger: object,
    key: str | None,
    runner: StepRunner,
    running: RunStep | None,
) -> None:
    """Render one bordered step card: header, pills, examples, controls, and state."""
    status = step.status(paths)
    with st.container(border=True):
        _header(index, step)
        _pills(step, study, ledger, status)
        _show_me(step, paths)
        _controls(step, paths, study, pricing, key, runner, status, running)
        _run_state(step, paths, runner)


def _header(index: int, step: RunStep) -> None:
    """Render the numbered title and one-line purpose."""
    st.markdown(f"### {index}. {step.title}")
    st.caption(step.purpose)


def _label(station: str) -> str:
    """Return the display label for a station key."""
    return _STATION_LABELS.get(station, station)


def _cost_text(step: RunStep, study: object, ledger: object) -> tuple[str, str]:
    """Return the cost badge text and its color."""
    cap = step.cap_usd(study)  # type: ignore[arg-type]
    if cap is None:
        return "free", "green"
    text = f"cap \\${cap:.0f}"
    spent = step.spent_usd(ledger)  # type: ignore[arg-type]
    if spent > 0:
        text += f" · spent \\${spent:.2f}"
    return text, "orange"


def _pills(step: RunStep, study: object, ledger: object, status: StepStatus) -> None:
    """Render the input, output, cost, and status badges on one line."""
    cost, cost_color = _cost_text(step, study, ledger)
    status_color = _STATUS_COLORS.get(status.state, "gray")
    line = (
        f":gray-badge[{_label(step.input_station)}] → "
        f":blue-badge[{_label(step.output_station)}] &nbsp;&nbsp; "
        f":{cost_color}-badge[{cost}] &nbsp;&nbsp; "
        f":{status_color}-badge[{status.state}]"
    )
    st.markdown(line)
    if status.detail:
        st.caption(status.detail)


def _show_me(step: RunStep, paths: data.Paths) -> None:
    """Render the input and output examples side by side with the learning note."""
    with st.expander("Show me"):
        left, right = st.columns(2)
        with left:
            st.caption(f"Input · {_label(step.input_station)}")
            st.code(step.example_input(paths), wrap_lines=True)
        with right:
            st.caption(f"Output · {_label(step.output_station)}")
            st.code(step.example_output(paths), wrap_lines=True)
        st.markdown(step.learn)


def _controls(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    key: str | None,
    runner: StepRunner,
    status: StepStatus,
    running: RunStep | None,
) -> None:
    """Render the Run button and the caption explaining any disabled state."""
    locked = status.state == "locked"
    needs_key = step.paid and not key
    busy = running is not None and running.id != step.id
    disabled = locked or needs_key or busy
    clicked = st.button("Run", key=f"run_{step.id}", type="primary", disabled=disabled)

    caption = _disabled_reason(status, locked, needs_key, busy)
    if caption:
        st.caption(caption)
    if clicked:
        _launch(step, paths, study, pricing, key, runner)


def _disabled_reason(status: StepStatus, locked: bool, needs_key: bool, busy: bool) -> str | None:
    """Return the caption explaining why the Run button is disabled, if it is."""
    if locked:
        return status.detail
    if needs_key:
        return "Add your OpenRouter key in the sidebar to run this step."
    if busy:
        return "Another step is running; wait for it to finish."
    return None


def _launch(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    key: str | None,
    runner: StepRunner,
) -> None:
    """Start the step on a background thread with a fresh run context."""

    def fn(*, on_progress: object, cancel: object) -> object:
        ctx = RunContext(
            study=study,  # type: ignore[arg-type]
            pricing=pricing,  # type: ignore[arg-type]
            key=key,
            on_progress=on_progress,  # type: ignore[arg-type]
            cancel=cancel,  # type: ignore[arg-type]
        )
        return step.run(paths, ctx)

    try:
        runner.start(step.id, fn)
    except StepAlreadyRunning:
        pass


def _run_state(step: RunStep, paths: data.Paths, runner: StepRunner) -> None:
    """Render the running panel while alive, or the finished panel once done."""
    if runner.is_running(step.id):
        _running_panel(step, runner)
        return
    status = runner.status(step.id)
    if status is not None and status.finished_at is not None:
        _finished_panel(step, paths, status)


@st.fragment(run_every="1s")
def _running_panel(step: RunStep, runner: StepRunner) -> None:
    """Show live progress, a dollar meter, elapsed time, and a Stop button.

    When the step's thread has ended, rerun the whole page once so the status
    badge, the flow strip counts, and the finished panel all reflect the new
    files on disk instead of the snapshot this fragment started from.
    """
    if not runner.is_running(step.id):
        st.rerun()
    status = runner.status(step.id)
    if status is None:
        st.caption("Starting…")
        _stop_button(step, runner)
        return
    total = status.total or 1
    st.progress(min(status.done / total, 1.0), text=f"{status.done} of {status.total}")
    _dollar_meter(status)
    st.caption(f"Elapsed {_elapsed(status.started_at, None)} · {status.last_item or ''}")
    _stop_button(step, runner)


def _dollar_meter(status: object) -> None:
    """Render the spend against the cap, in amber once it nears the cap."""
    cap = getattr(status, "cap_usd", None)
    if not cap:
        return
    spent = getattr(status, "spent_usd", 0.0)
    ratio = min(spent / cap, 1.0)
    st.progress(ratio, text=f"${spent:.2f} of ${cap:.0f}")
    if ratio >= 0.8:
        st.caption(":orange[Approaching the spend cap.]")


def _stop_button(step: RunStep, runner: StepRunner) -> None:
    """Render the Stop button that cancels the running step."""
    if st.button("Stop", key=f"stop_{step.id}"):
        runner.cancel(step.id)


def _finished_panel(step: RunStep, paths: data.Paths, status: object) -> None:
    """Render the error, or the settled success line, findings, and Gates link."""
    error = getattr(status, "error", None)
    if error:
        st.error(error)
        return
    cancelled = getattr(status, "cancelled", False)
    word = "cancelled" if cancelled else "done"
    st.markdown(f'<div class="settle">Step {step.title}: {word}.</div>', unsafe_allow_html=True)
    findings = _step_findings(step, paths)
    if findings:
        st.caption(findings)
    st.markdown("[See the gates](/gates)")
    if not cancelled:
        _maybe_balloons(step)


def _step_findings(step: RunStep, paths: data.Paths) -> str:
    """Return the findings paragraphs this step produced, joined together."""
    texts: list[str] = []
    for stage in step.stages:
        path = paths.results_dir / f"{stage}_findings.md"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def _maybe_balloons(step: RunStep) -> None:
    """Celebrate the first paid step to finish in this session, once."""
    if step.paid and not st.session_state.get(_BALLOONS_KEY):
        st.session_state[_BALLOONS_KEY] = True
        st.balloons()


def _elapsed(started_at: str, finished_at: str | None) -> str:
    """Return a short elapsed-time string between two ISO timestamps."""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(start.tzinfo)
    except ValueError:
        return "0s"
    seconds = max(int((end - start).total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"
