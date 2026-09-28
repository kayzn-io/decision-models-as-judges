"""Run page: guide the learner through the study steps and execute each one.

The page is a thin view. Each step's readiness, examples, and work live in
``steps``; this module renders the cards, launches steps on a background
``StepRunner`` held in session state, and reflects their progress. Paid steps
read the OpenRouter key from the same session field the Live page uses, so one
sidebar entry serves both pages.
"""

from datetime import datetime

import streamlit as st

from decision_judges.bench.run_agent import MissingCredentials
from decision_judges.runner import StepAlreadyRunning, StepRunner
from decision_judges.ui import components, data, flow, keys, notes, steps
from decision_judges.ui.steps import RunContext, RunStep, StepStatus

_PURPOSE = (
    "Seven steps that test the judges on the shipped conversations. Each shows what goes "
    "in, what comes out, and what it costs."
)
_INTRO = "Paid steps use your OpenRouter key only to run the judges."
_RUNNER_KEY = "run_step_runner"
_LAST_RUNNING_KEY = "run_last_running_id"

# The material card's status badge color per state.
_MATERIAL_BADGE_COLORS = {"shipped": "green", "regenerating": "orange", "missing": "red"}
_MATERIAL_MISSING_LINE = (
    "The study needs conversations first. Open Regenerate below to create them with your own agent."
)
_REGENERATE_WARNING = (
    "This replaces the shipped conversations and costs about $10 through your OpenRouter key."
)
_REGENERATE_EXPLAINER = "Replace the shipped conversations with new ones from your own agent runs."
_VARIANT_EXPLAINER = (
    "The rushed agent skips confirming details with the customer before acting; "
    "the careful agent does not."
)
# The human label for each agent whose conversations the app ships with.
_VARIANT_LABELS = {"baseline": "Careful agent", "degraded": "Rushed agent"}

_STATION_LABELS = {
    "tasks": "Requests",
    "conversations": "Conversations",
    "judge_text": "What judges read",
    "verdicts": "Verdicts",
    "findings": "Findings",
}
_STATUS_COLORS = {"locked": "gray", "ready": "blue", "partial": "orange", "done": "green"}

# The plain noun each step's output station adds to the produced line.
_PRODUCED_NOUN = {
    "conversations": "conversations",
    "judge_text": "reading copies",
    "verdicts": "verdicts",
    "findings": "findings",
}

# One "Notice:" line per station, telling the learner what to look for in an example.
_NOTICE = {
    "tasks": "Notice: one customer request, as the customer stated it.",
    "conversations": "Notice: the ground truth is the last line, and the judges never see it.",
    "judge_text": "Notice: the ground truth is gone from what the judge reads.",
    "verdicts": "Notice: this is what one judge answered, with how sure it was.",
    "findings": "Notice: these are written results, ready to publish.",
}


def render() -> None:
    """Render the material panel and the seven runnable step cards in local mode."""
    if not data.is_local():
        st.info("Running the study runs only locally with JUDGES_LOCAL=1.")
        components.footer()
        return

    paths = data.Paths.from_env()
    study, pricing = data.load_study_and_pricing(paths)
    ledger = data.load_ledger(paths)
    key = keys.get_key()
    runner = _runner(paths)

    components.page_header("Run the study", _PURPOSE)
    st.markdown(_INTRO)
    _resume_banner(runner)

    running = _running_step(runner)
    flow.strip(
        paths,
        active=None,
        running=running.pipe if running is not None else None,
        paid=running.paid if running is not None else False,
        compact=False,
        flash=_flash_station(runner, running),
    )

    for step in steps.STEPS:
        _card(step, paths, study, pricing, ledger, key, runner, running)
    components.next_link(
        "Trajectories",
        "/trajectories",
        "Read one of the conversations a step produced, beside its ground truth and every verdict.",
    )
    components.footer()


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
    for step in steps.STEPS:
        if runner.stale(step.id):
            st.warning(_stale_message(step))
            return


def _stale_message(step: RunStep) -> str:
    """Return the resume warning for a stalled step, numbered or material."""
    index = steps.display_index(step)
    if index is None:
        return (
            "Regenerating the conversations stopped while the app was closed; its work is "
            "saved. Run it again to continue."
        )
    return (
        f"Step {index} stopped while the app was closed; its work is saved. "
        "Run it again to continue."
    )


def _flash_station(runner: StepRunner, running: RunStep | None) -> flow.Station | None:
    """Return the destination station to flash for the one render after a finish.

    The just-finished step is the one that was running on the previous render
    and is no longer running now, having finished without error or cancellation.
    The previous running id is remembered in session state so the flash fires
    exactly once.
    """
    previous = st.session_state.get(_LAST_RUNNING_KEY)
    current = running.id if running is not None else None
    st.session_state[_LAST_RUNNING_KEY] = current
    if current is not None or previous is None:
        return None
    status = runner.status(previous)
    if status is None or status.finished_at is None:
        return None
    if status.error or status.cancelled or status.stopped_reason:
        return None
    step = _step_by_id(previous)
    if step is None:
        return None
    return flow.Station(step.output_station)


def _step_by_id(step_id: str) -> RunStep | None:
    """Return the step with a given id, or None when no step matches."""
    for step in steps.STEPS:
        if step.id == step_id:
            return step
    return None


def _card(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    ledger: object,
    key: str | None,
    runner: StepRunner,
    running: RunStep | None,
) -> None:
    """Render one bordered step card, material or numbered."""
    with st.container(border=True):
        if step.material:
            _material_card(step, paths, study, pricing, key, runner, running)
        else:
            _numbered_card(step, paths, study, pricing, ledger, key, runner, running)


def _numbered_card(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    ledger: object,
    key: str | None,
    runner: StepRunner,
    running: RunStep | None,
) -> None:
    """Render a numbered step card: header, pills, examples, controls, and state."""
    index = steps.display_index(step)
    status = step.status(paths)
    _header(index, step)
    _pills(step, study, ledger, status)
    _plan_section(step, paths, study)
    _show_me(step, paths)
    _controls(step, paths, study, pricing, key, runner, status, running)
    _run_state(step, index, paths, runner)


def _material_card(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    key: str | None,
    runner: StepRunner,
    running: RunStep | None,
) -> None:
    """Render the unnumbered material panel describing the shipped conversations."""
    badge = steps.material_status(paths, regenerating=runner.is_running(step.id))
    st.markdown(f"### {step.title}")
    st.caption(step.purpose)
    _material_badge(badge)
    if badge == "missing":
        st.caption(_MATERIAL_MISSING_LINE)
    _variant_counts(paths)
    _provenance(paths)
    _show_me(step, paths)
    _regenerate(step, paths, study, pricing, key, runner, running, badge == "missing")


def _material_badge(badge: str) -> None:
    """Render the shipped / regenerating / missing status badge."""
    color = _MATERIAL_BADGE_COLORS.get(badge, "gray")
    st.markdown(f":{color}-badge[{badge}]")


def _variant_counts(paths: data.Paths) -> None:
    """Show each variant's conversation count and pass rate, side by side."""
    counts = steps.variant_counts(paths)
    if not counts:
        return
    columns = st.columns(len(counts))
    for column, item in zip(columns, counts, strict=True):
        with column:
            st.markdown(f"**{_VARIANT_LABELS.get(item.variant, item.variant)}**")
            st.caption(
                f"{item.conversations} conversations · {item.pass_rate * 100:.1f}% pass "
                "in ground truth"
            )


def _provenance(paths: data.Paths) -> None:
    """Fold the models and the ground-truth note behind an 'About these conversations' expander."""
    prov = steps.provenance(paths)
    with st.expander("About these conversations"):
        lines = []
        if prov is not None:
            lines.append(f"- Support agent: `{prov.agent_model}`")
            lines.append(f"- Customer: `{prov.user_model}`")
        lines.append(
            "- Ground truth was fixed before any judge read a conversation; judges never see it."
        )
        st.markdown("\n".join(lines))
        with st.expander(notes.TECHNICAL_NOTES_TITLE, expanded=False):
            st.markdown(notes.technical_notes(prov))


def _regenerate(
    step: RunStep,
    paths: data.Paths,
    study: object,
    pricing: object,
    key: str | None,
    runner: StepRunner,
    running: RunStep | None,
    expanded: bool,
) -> None:
    """Hold the regenerate explainer, cost, warning, Run button, and live panel."""
    with st.expander("Regenerate with your own agent", expanded=expanded):
        st.markdown(_REGENERATE_EXPLAINER)
        st.caption(_VARIANT_EXPLAINER)
        cap = step.cap_usd(study)  # type: ignore[arg-type]
        if cap is not None:
            st.caption(f"Spend cap: \\${cap:.0f}")
        st.warning(_REGENERATE_WARNING)
        status = step.status(paths)
        _controls(step, paths, study, pricing, key, runner, status, running)
        _run_state(step, None, paths, runner)


def _header(index: int | None, step: RunStep) -> None:
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


def _plan_section(step: RunStep, paths: data.Paths, study: object) -> None:
    """Render the plan-of-work panel for a step that declares one."""
    if step.plan is None:
        return
    plan = step.plan(paths, study)  # type: ignore[arg-type]
    components.plan_panel(plan, step.cap_usd(study))  # type: ignore[arg-type]


def _show_me(step: RunStep, paths: data.Paths) -> None:
    """Render the input and output examples side by side with the learning note."""
    with st.expander("Show me"):
        left, right = st.columns(2)
        with left:
            st.caption("What goes in")
            notice = _NOTICE.get(step.input_station)
            if notice:
                st.caption(notice)
            st.code(step.example_input(paths), wrap_lines=True)
        with right:
            st.caption("What comes out")
            notice = _NOTICE.get(step.output_station)
            if notice:
                st.caption(notice)
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
        return keys.require_key_hint()
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


def _run_state(step: RunStep, index: int | None, paths: data.Paths, runner: StepRunner) -> None:
    """Render the running panel while alive, or the finished panel once done."""
    if runner.is_running(step.id):
        _running_panel(step, runner)
        return
    st.session_state.pop(f"stopping_{step.id}", None)
    status = runner.status(step.id)
    if status is not None and status.finished_at is not None:
        _finished_panel(step, index, paths, status)


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
    _progress_bars(step, status)
    _dollar_meter(status)
    st.caption(
        f"Elapsed {_elapsed(status.started_at, None)} · {_format_last_item(status.last_item, step)}"
    )
    _stop_button(step, runner)


def _progress_bars(step: RunStep, status: object) -> None:
    """Render the batch bar, naming the running batch and the overall progress.

    An unphased step keeps its single bar. A phased step gets a bold line naming
    the running batch, the batch bar with its count beneath, and, when the totals
    across all batches are known, a second bar for the progress across them.
    """
    done = getattr(status, "done", 0) or 0
    total = getattr(status, "total", 0) or 0
    phase = getattr(status, "phase", None)
    if phase is None:
        st.progress(min(done / (total or 1), 1.0), text=f"{done} of {total}")
        return
    index = getattr(status, "phase_index", None) or 1
    count = getattr(status, "phase_count", None) or 1
    st.markdown(f"**Batch {index} of {count}: {phase}**")
    st.progress(min(done / (total or 1), 1.0))
    noun = "judge calls" if step.pipe == "judge" else "items"
    st.caption(f"{done:,} of {total:,} {noun} in this batch")
    _overall_bar(status)


def _overall_bar(status: object) -> None:
    """Render the progress across every batch when their totals are known."""
    overall_total = getattr(status, "overall_total", None)
    if not overall_total:
        return
    overall_done = getattr(status, "overall_done", 0) or 0
    st.progress(min(overall_done / overall_total, 1.0))
    st.caption(f"Overall: {overall_done:,} of {overall_total:,}")


def _format_last_item(last_item: str | None, step: RunStep) -> str:
    """Return the last judged item as a sentence, or the raw line when it is not one.

    A judge step reports ``task · judge · verdict``; this reads it back as a
    sentence naming the conversation, the judge, and its verdict. Any other line
    is left as it is.
    """
    if not last_item:
        return ""
    if step.pipe != "judge":
        return last_item
    parts = last_item.split(" · ")
    if len(parts) != 3:
        return last_item
    task_id, judge_id, label = parts
    return f"Conversation {task_id}, judged by {steps.judge_display_name(judge_id)}: {label}"


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
    """Render the Stop button, or a disabled Stopping state once cancel is asked."""
    if runner.is_cancelling(step.id) or st.session_state.get(f"stopping_{step.id}"):
        st.button("Stopping", key=f"stop_{step.id}", disabled=True)
        st.caption(
            "Letting the conversations already in progress finish; nothing started so far is lost."
        )
        return
    if st.button("Stop", key=f"stop_{step.id}"):
        runner.cancel(step.id)
        st.session_state[f"stopping_{step.id}"] = True
        st.rerun()


def _finished_panel(step: RunStep, index: int | None, paths: data.Paths, status: object) -> None:
    """Render the error, the stopped-early warning, the stopped line, or success."""
    error = getattr(status, "error", None)
    if error:
        st.error(_plain_error(error))
        return
    stopped_reason = getattr(status, "stopped_reason", None)
    if stopped_reason and stopped_reason != "cancelled":
        _stopped_early_panel(index, stopped_reason)
        return
    if getattr(status, "cancelled", False):
        done = getattr(status, "done", 0) or 0
        st.caption(
            f"Stopped after {done} conversations. "
            "Everything finished so far is saved; run again to continue."
        )
        return
    components.success_moment(
        _done_title(index),
        _produced_text(step, status),
        "See the experiments",
        "/gates",
    )
    findings = _step_findings(step, paths)
    if findings:
        st.caption(findings)


def _done_title(index: int | None) -> str:
    """Return the success-card title for a finished step."""
    return "Conversations regenerated" if index is None else f"Step {index} done"


def _stopped_early_panel(index: int | None, reason: str) -> None:
    """Warn that a step stopped early without an error, and how to proceed."""
    header = "Regenerating stopped early" if index is None else f"Step {index} stopped early"
    st.warning(f"{header}\n\n{reason}\n\n{_stop_guidance(reason)}")


def _stop_guidance(reason: str) -> str:
    """Return the one-line next step chosen by what the stop reason mentions."""
    lowered = reason.lower()
    if "credentials" in lowered or "api key" in lowered:
        return "Add a valid OpenRouter key in the sidebar and run again."
    if "401" in lowered or "not found" in lowered or "unauthorized" in lowered:
        return (
            "Your key is not enabled for this model. Check the model page on "
            "OpenRouter for your account, then run again; finished work is kept."
        )
    if "spend cap" in lowered:
        return (
            "The spend cap for this step is reached; raise it in config/study.toml "
            "if you intend to spend more."
        )
    return "Run again to retry; finished items are kept."


def _produced_text(step: RunStep, status: object) -> str:
    """Name what the step wrote, with its count when the status carries a total."""
    noun = _PRODUCED_NOUN.get(step.output_station, "results")
    done = getattr(status, "done", 0) or 0
    total = getattr(status, "total", 0) or 0
    if total > 1:
        return f"Wrote {done:,} of {total:,} {noun}."
    return f"Wrote the {noun}."


def _plain_error(error: str) -> str:
    """Return a missing-credentials error without its exception type prefix."""
    prefix = f"{MissingCredentials.__name__}: "
    if error.startswith(prefix):
        return error[len(prefix) :]
    return error


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
