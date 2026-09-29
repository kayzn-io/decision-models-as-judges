"""Run page tests driven through AppTest over a thin script."""

import shutil
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from decision_judges.progress import Progress, utc_now_iso
from decision_judges.runner import StepRunner
from decision_judges.ui import keys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "tests" / "fixtures" / "ui_cache"
_SCRIPT = "from decision_judges.ui.pages.run import render\n\nrender()\n"
_FAKE_KEY = "sk-fake-run-key-XYZ"
_PAID_IDS = ("run-agent", "judge-outcome", "gates", "label")


def _script(tmp_path: Path) -> str:
    """Write the thin page script and return its path."""
    script = tmp_path / "run_page.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return str(script)


def _local_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, AppTest]:
    """Copy the fixture tree to a writable root and build a local-mode AppTest."""
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    return root, AppTest.from_file(_script(tmp_path), default_timeout=60)


def _run_buttons(at: AppTest) -> list[str]:
    """Return the keys of every step Run button."""
    return [button.key for button in at.button if button.key and button.key.startswith("run_")]


def _texts(at: AppTest) -> list[str]:
    """Return every rendered markdown, caption, and progress-bar label for substring assertions."""
    texts = [block.value for block in [*at.markdown, *at.caption]]
    texts += [getattr(bar, "text", "") or "" for bar in at.get("progress")]
    return texts


def _await_finished(root: Path, step_id: str, timeout: float = 30.0) -> Progress:
    """Poll the on-disk progress until the step is finished, then return it."""
    runner = StepRunner(root / "results" / ".progress")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.status(step_id)
        if status is not None and status.finished_at is not None:
            return status
        time.sleep(0.2)
    raise AssertionError(f"step {step_id} did not finish within {timeout}s")


def _await_running(root: Path, step_id: str, timeout: float = 10.0) -> None:
    """Poll the on-disk progress until the step has reported at least once."""
    runner = StepRunner(root / "results" / ".progress")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runner.status(step_id) is not None:
            return
        time.sleep(0.1)
    raise AssertionError(f"step {step_id} never reported progress within {timeout}s")


def _state_files(root: Path) -> int:
    """Count serialized state files under the cache."""
    state_dir = root / "cache" / "state"
    return sum(1 for path in state_dir.rglob("*.json") if path.is_file())


def test_run_shared_mode_shows_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JUDGES_ROOT", str(_UI_ROOT))
    monkeypatch.delenv("JUDGES_LOCAL", raising=False)
    at = AppTest.from_file(_script(tmp_path), default_timeout=30).run()

    assert not at.exception
    assert any("JUDGES_LOCAL=1" in info.value for info in at.info)


def test_run_renders_material_and_seven_numbered_cards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    # One Run button per numbered step plus the regenerate button on the material panel.
    assert len(_run_buttons(at)) == 8
    headers = [block.value for block in at.markdown if block.value.startswith("### ")]
    assert any(header.startswith("### The conversations") for header in headers)
    assert sum(1 for header in headers if header[4:5].isdigit()) == 7


def test_material_card_shows_counts_provenance_and_run_only_in_regenerate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    texts = _texts(at)
    assert any("Careful agent" in text for text in texts)
    assert any("Ground truth was fixed before any judge" in text for text in texts)
    labels = [expander.label for expander in at.expander]
    assert "About these conversations" in labels
    regen = next(e for e in at.expander if e.label == "Regenerate with your own agent")
    assert any(button.key == "run_run-agent" for button in regen.button)
    assert any("$10" in warning.value for warning in at.warning)


def test_material_card_has_collapsed_technical_notes_with_full_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    labels = [expander.label for expander in at.expander]
    assert "About these conversations" in labels
    assert "Technical notes" in labels
    notes = next(e for e in at.expander if e.label == "Technical notes")
    assert notes.proto.expanded is False
    joined = "\n".join(block.value for block in notes.markdown)
    assert "Sierra Research" in joined


def test_material_card_missing_asks_for_conversations_and_opens_regenerate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    shutil.copytree(_UI_ROOT, root)
    shutil.rmtree(root / "cache" / "agent")
    (root / "cache" / "agent").mkdir(parents=True)
    monkeypatch.setenv("JUDGES_ROOT", str(root))
    monkeypatch.setenv("JUDGES_LOCAL", "1")
    at = AppTest.from_file(_script(tmp_path), default_timeout=60).run()

    assert not at.exception
    assert any("needs conversations" in text.lower() for text in _texts(at))
    regen = next(e for e in at.expander if e.label == "Regenerate with your own agent")
    assert regen.proto.expanded is True


def test_paid_buttons_disabled_without_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    for step_id in _PAID_IDS:
        assert at.button(key=f"run_{step_id}").disabled is True, step_id
    assert any("OpenRouter key" in text for text in _texts(at))


def test_entering_a_key_enables_paid_buttons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    assert not at.exception
    assert at.button(key="run_judge-outcome").disabled is False


def test_running_serialize_writes_states_and_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    before = _state_files(root)

    at.button(key="run_serialize").click().run()
    status = _await_finished(root, "serialize")

    assert status.error is None
    assert _state_files(root) > before
    at.run()
    assert any("done" in text.lower() for text in _texts(at))


def test_running_analyze_writes_g6_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    at.button(key="run_analyze").click().run()
    status = _await_finished(root, "analyze")

    assert status.error is None
    assert (root / "results" / "g6_summary.md").is_file()


def test_running_analyze_rebuilds_g3_summary_from_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The g3 table is rewritten from every cached verdict, not left as the last batch wrote it."""
    root, at = _local_app(monkeypatch, tmp_path)
    stale = root / "results" / "g3_summary.md"
    stale.write_text("| judge_id |\n|---|\n| laya_base |\n", encoding="utf-8")
    at.run()

    at.button(key="run_analyze").click().run()
    status = _await_finished(root, "analyze")

    assert status.error is None
    rebuilt = stale.read_text(encoding="utf-8")
    assert "laya_base" not in rebuilt
    assert "code" in rebuilt and "fake" in rebuilt
    assert (root / "results" / "g3_findings.md").is_file()


def test_slow_step_shows_stop_and_cancels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def slow_fake(paths: object, ctx: object) -> None:
        started = utc_now_iso()
        for index in range(200):
            if ctx.cancel is not None and ctx.cancel.is_cancelled:
                return
            if ctx.on_progress is not None:
                ctx.on_progress(
                    Progress(
                        step_id="judge-outcome",
                        done=index,
                        total=200,
                        started_at=started,
                        last_item=f"item {index}",
                    )
                )
            time.sleep(0.1)

    monkeypatch.setattr("decision_judges.ui.steps._run_judge_outcome", slow_fake)
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    at.button(key="run_judge-outcome").click().run()
    _await_running(root, "judge-outcome")
    at.run()

    assert not at.exception
    at.button(key="stop_judge-outcome").click().run()
    status = _await_finished(root, "judge-outcome")
    assert status.cancelled is True


def test_slow_step_shows_a_stopping_state_then_the_stopped_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def slow_fake(paths: object, ctx: object) -> None:
        started = utc_now_iso()
        if ctx.on_progress is not None:
            ctx.on_progress(
                Progress(
                    step_id="judge-outcome",
                    done=1,
                    total=4,
                    started_at=started,
                    last_item="item 1",
                )
            )
        while not (ctx.cancel is not None and ctx.cancel.is_cancelled):
            time.sleep(0.02)
        time.sleep(0.5)

    monkeypatch.setattr("decision_judges.ui.steps._run_judge_outcome", slow_fake)
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    at.button(key="run_judge-outcome").click().run()
    _await_running(root, "judge-outcome")
    at.run()

    at.button(key="stop_judge-outcome").click().run()

    assert not at.exception
    stopping = at.button(key="stop_judge-outcome")
    assert stopping.label == "Stopping"
    assert stopping.disabled is True
    assert any(
        "Letting the conversations already in progress finish" in text for text in _texts(at)
    )

    status = _await_finished(root, "judge-outcome")
    assert status.cancelled is True
    at.run()
    assert not at.exception
    assert any("Stopped after" in text for text in _texts(at))


def test_resume_banner_shows_for_a_stale_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    progress_dir = root / "results" / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    stale = Progress(step_id="serialize", done=1, total=2, started_at=utc_now_iso())
    (progress_dir / "serialize.json").write_text(stale.model_dump_json(indent=2), encoding="utf-8")

    at.run()

    assert not at.exception
    assert any("stopped while the app was closed" in warning.value for warning in at.warning)


def test_run_page_never_renders_the_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    assert not at.exception
    assert all(_FAKE_KEY not in str(text) for text in _texts(at))


def test_ui_source_has_no_balloons() -> None:
    ui_dir = _REPO_ROOT / "decision_judges" / "ui"
    hits = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in ui_dir.rglob("*.py")
        if "balloons" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"balloons still present in: {hits}"


def test_finished_serialize_shows_done_and_the_produced_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    at.button(key="run_serialize").click().run()
    _await_finished(root, "serialize")
    at.run()

    assert not at.exception
    texts = _texts(at)
    assert any("done" in text.lower() for text in texts)
    assert any("reading copies" in text for text in texts)


def test_cancelled_step_shows_the_stopped_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    progress_dir = root / "results" / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    stopped = Progress(
        step_id="serialize",
        done=1,
        total=2,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        cancelled=True,
    )
    (progress_dir / "serialize.json").write_text(
        stopped.model_dump_json(indent=2), encoding="utf-8"
    )

    at.run()

    assert not at.exception
    assert any("Stopped after 1 conversations" in text for text in _texts(at))


def _plant_stopped(root: Path, step_id: str, reason: str) -> None:
    """Write a finished progress file that stopped early with a reason."""
    progress_dir = root / "results" / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    stopped = Progress(
        step_id=step_id,
        done=0,
        total=115,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        stopped_reason=reason,
    )
    (progress_dir / f"{step_id}.json").write_text(
        stopped.model_dump_json(indent=2), encoding="utf-8"
    )


def test_stopped_step_shows_warning_and_key_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    _plant_stopped(
        root,
        "run-agent",
        "aborted: the first 3 tasks failed with Missing credentials",
    )

    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("Regenerating stopped early" in w for w in warnings)
    assert any("Missing credentials" in w for w in warnings)
    assert any("OpenRouter key" in w for w in warnings)
    assert all("Conversations regenerated" not in text for text in _texts(at))


def test_stopped_step_with_spend_cap_shows_cap_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    _plant_stopped(root, "run-agent", "the spend cap for this step is reached")

    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("Regenerating stopped early" in w for w in warnings)
    assert any("config/study.toml" in w for w in warnings)


def test_stopped_step_with_unauthorized_model_shows_model_key_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, at = _local_app(monkeypatch, tmp_path)
    _plant_stopped(
        root,
        "judge-outcome",
        "jev stopped after 5 failed calls: 401 User not found. Other judges continued.",
    )

    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("Step 2 stopped early" in w for w in warnings)
    assert any("not enabled for this model" in w for w in warnings)


def test_ask_the_judges_shows_the_plan_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    texts = _texts(at)
    assert any("What happens when you press Run" in text for text in texts)
    assert any("Rule-based check (free)" in text for text in texts)
    assert any("Fast text model" in text for text in texts)
    assert any("Strong text model" in text for text in texts)
    assert any("Jev (decision model)" in text for text in texts)
    assert any("openai/gpt-4o-mini" in text for text in texts)
    assert any("openai/gpt-5" in text for text in texts)
    assert any("jev-1.13.0" in text for text in texts)
    assert any("2 questions" in text and "5 times" in text for text in texts)


def test_draw_conclusions_panel_says_no_new_judge_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _root, at = _local_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert any("No new judge calls" in text for text in _texts(at))


def test_format_last_item_reads_like_a_sentence_for_judge_steps() -> None:
    from decision_judges.config import load_study
    from decision_judges.ui import steps
    from decision_judges.ui.pages.run import _format_last_item

    study = load_study(_UI_ROOT / "config" / "study.toml")
    judge_step = next(s for s in steps.STEPS if s.id == "judge-outcome")
    line = _format_last_item(
        "retail-42 · jev · pass", judge_step, study, "Careful agent, short text"
    )
    assert line == (
        "Conversation retail-42, short text, judged by "
        f"Jev (decision model) ({study.models.jev}): pass"
    )
    # The free rule-based judge has no model to name, and an unphased step names no version.
    assert (
        _format_last_item("retail-1 · code · fail", judge_step, study, None)
        == "Conversation retail-1, judged by Rule-based check (free): fail"
    )


def test_format_last_item_leaves_non_judge_lines_alone() -> None:
    from decision_judges.ui import steps
    from decision_judges.ui.pages.run import _format_last_item

    serialize_step = next(s for s in steps.STEPS if s.id == "serialize")
    assert _format_last_item("baseline · 2 runs", serialize_step, None, None) == "baseline · 2 runs"
    assert _format_last_item(None, serialize_step, None, None) == ""


def test_running_panel_names_the_batch_and_shows_overall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def slow_fake(paths: object, ctx: object) -> None:
        started = utc_now_iso()
        if ctx.on_progress is not None:  # type: ignore[attr-defined]
            ctx.on_progress(  # type: ignore[attr-defined]
                Progress(
                    step_id="judge-outcome",
                    done=1204,
                    total=2300,
                    started_at=started,
                    last_item="retail-42 · jev · pass",
                    phase="Careful agent, short text",
                    phase_index=2,
                    phase_count=4,
                    overall_done=3504,
                    overall_total=9200,
                    spent_usd=12.5,
                    cap_usd=75.0,
                )
            )
        while not (ctx.cancel is not None and ctx.cancel.is_cancelled):  # type: ignore[attr-defined]
            time.sleep(0.02)

    monkeypatch.setattr("decision_judges.ui.steps._run_judge_outcome", slow_fake)
    root, at = _local_app(monkeypatch, tmp_path)
    at.run()
    at.session_state[keys.SESSION_KEY] = _FAKE_KEY
    at.run()

    at.button(key="run_judge-outcome").click().run()
    _await_running(root, "judge-outcome")
    at.run()

    assert not at.exception
    texts = _texts(at)
    assert any("Batch 2 of 4: Careful agent, short text" in text for text in texts)
    assert any("1,204 of 2,300 judge calls in this batch" in text for text in texts)
    assert any("Overall: 3,504 of 9,200" in text for text in texts)
    assert any("short text, judged by Jev (decision model) (" in text for text in texts)
    assert any("across all batches" in text for text in texts)
    assert any("Spent $" in text and "cap for this step" in text for text in texts)

    at.button(key="stop_judge-outcome").click().run()
    _await_finished(root, "judge-outcome")
