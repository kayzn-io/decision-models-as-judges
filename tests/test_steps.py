"""Pure tests for the eight run steps: order, unlock rules, status, and examples."""

from pathlib import Path

from decision_judges.config import load_study
from decision_judges.ui import steps
from decision_judges.ui.data import Paths

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ui_cache"

_EXPECTED_IDS = [
    "run-agent",
    "serialize",
    "judge-outcome",
    "analyze",
    "laya",
    "gates",
    "label",
    "results",
]
_EXPECTED_PIPES = ["run-agent", "serialize", "judge", "analyze", "judge", "judge", "judge", None]


def _paths(root: Path) -> Paths:
    """Build a Paths rooted at a directory without touching the environment."""
    return Paths(
        repo_root=root,
        cache_dir=root / "cache",
        results_dir=root / "results",
        config_dir=root / "config",
        data_dir=root / "data",
    )


def test_there_are_eight_steps() -> None:
    assert len(steps.STEPS) == 8


def test_step_ids_are_unique_and_ordered() -> None:
    ids = [step.id for step in steps.STEPS]
    assert ids == _EXPECTED_IDS
    assert len(set(ids)) == 8


def test_step_pipes_match_the_pipeline() -> None:
    assert [step.pipe for step in steps.STEPS] == _EXPECTED_PIPES


def test_empty_tree_leaves_only_the_first_step_ready(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    reasons = [step.unlock(paths) for step in steps.STEPS]
    assert reasons[0] is None
    assert all(reason is not None for reason in reasons[1:])


def test_fixture_tree_unlocks_every_step() -> None:
    paths = _paths(_FIXTURE)
    assert all(step.unlock(paths) is None for step in steps.STEPS)


def test_first_step_is_partial_on_the_fixture() -> None:
    status = steps.STEPS[0].status(_paths(_FIXTURE))
    assert status.state == "partial"
    assert status.done == 2
    assert status.total == 115


def test_locked_step_reports_locked_status_on_an_empty_tree(tmp_path: Path) -> None:
    status = steps.STEPS[1].status(_paths(tmp_path))
    assert status.state == "locked"
    assert status.detail


def test_examples_are_non_empty_on_the_fixture() -> None:
    paths = _paths(_FIXTURE)
    for step in steps.STEPS:
        assert step.example_input(paths).strip(), step.id
        assert step.example_output(paths).strip(), step.id


def test_examples_are_non_empty_on_an_empty_tree(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for step in steps.STEPS:
        assert step.example_input(paths).strip(), step.id
        assert step.example_output(paths).strip(), step.id


def test_paid_steps_carry_a_cap_and_free_steps_do_not() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    caps = {step.id: step.cap_usd(study) for step in steps.STEPS}
    assert caps["run-agent"] == 25.0
    assert caps["judge-outcome"] == 50.0
    assert caps["gates"] == 55.0
    assert caps["label"] == 5.0
    assert caps["serialize"] is None
    assert caps["analyze"] is None
    assert caps["laya"] is None
    assert caps["results"] is None


def test_paid_flags_match_the_priced_stages() -> None:
    paid = {step.id: step.paid for step in steps.STEPS}
    assert paid == {
        "run-agent": True,
        "serialize": False,
        "judge-outcome": True,
        "analyze": False,
        "laya": False,
        "gates": True,
        "label": True,
        "results": False,
    }


def test_step_one_returns_first_stop_reason_and_stops_after_one_variant(
    monkeypatch, tmp_path: Path
) -> None:
    from decision_judges.bench import load as bench_load
    from decision_judges.bench import run_agent as run_agent_mod
    from decision_judges.bench.run_agent import RunSummary
    from decision_judges.config import load_pricing
    from decision_judges.ui.steps import RunContext

    calls: list[str] = []

    monkeypatch.setattr(bench_load, "load_tasks", lambda: [])
    monkeypatch.setattr(run_agent_mod, "build_tau_runner", lambda study, variant: object())

    def fake_run_variant(variant, *args, **kwargs):
        calls.append(variant)
        return RunSummary(
            variant=variant,
            completed=0,
            excluded=0,
            skipped_existing=0,
            pass_rate=0.0,
            stopped_reason="aborted: the first 3 tasks failed with Missing credentials",
        )

    monkeypatch.setattr(run_agent_mod, "run_variant", fake_run_variant)

    study = load_study(_FIXTURE / "config" / "study.toml")
    pricing = load_pricing(_FIXTURE / "config" / "pricing.toml")
    ctx = RunContext(study=study, pricing=pricing)

    result = steps.STEPS[0].run(_paths(tmp_path), ctx)

    assert result == "aborted: the first 3 tasks failed with Missing credentials"
    assert calls == ["baseline"]


def test_exactly_one_material_step_and_it_is_first() -> None:
    material = [step for step in steps.STEPS if step.material]
    assert len(material) == 1
    assert steps.STEPS[0].material is True
    assert material[0].id == "run-agent"
    assert material[0].title == "The conversations"


def test_display_indices_number_only_the_seven_runnable_steps() -> None:
    indices = [steps.display_index(step) for step in steps.STEPS]
    assert indices[0] is None
    assert indices[1:] == [1, 2, 3, 4, 5, 6, 7]


def test_material_status_shipped_on_the_fixture_and_missing_on_empty(tmp_path: Path) -> None:
    assert steps.material_status(_paths(_FIXTURE)) == "shipped"
    assert steps.material_status(_paths(tmp_path)) == "missing"


def test_material_status_regenerating_when_flagged() -> None:
    assert steps.material_status(_paths(_FIXTURE), regenerating=True) == "regenerating"


def test_variant_counts_and_provenance_read_from_the_fixture() -> None:
    counts = steps.variant_counts(_paths(_FIXTURE))
    assert [count.variant for count in counts] == ["baseline"]
    assert counts[0].conversations == 2
    assert counts[0].pass_rate == 0.5
    prov = steps.provenance(_paths(_FIXTURE))
    assert prov is not None
    assert prov.agent_model == "openai/gpt-4.1"
    assert prov.user_model == "openai/gpt-4o-mini"
    assert prov.tau_bench_ref


def test_variant_counts_and_provenance_empty_on_an_empty_tree(tmp_path: Path) -> None:
    assert steps.variant_counts(_paths(tmp_path)) == []
    assert steps.provenance(_paths(tmp_path)) is None


def test_conversation_excerpt_ends_with_the_ground_truth() -> None:
    """The example exists to show the ground truth, so a real excerpt must end with it."""
    from decision_judges.ui.steps import _conversation_excerpt

    class Record:
        reward = 0.0
        trajectory = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "I want a refund."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_order_details"}}],
            },
            {"role": "tool", "content": "{...}"},
            {"role": "assistant", "content": "Refunded."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "Anything else?"},
        ]

    text = _conversation_excerpt(Record(), head=2, tail=2)
    lines = text.split("\n")
    assert lines[0].startswith("user:")  # the system turn is skipped
    assert "[... 2 turns omitted ...]" in lines
    assert lines[-1] == "ground truth: fail"


def test_short_conversation_excerpt_has_no_elision() -> None:
    from decision_judges.ui.steps import _conversation_excerpt

    class Record:
        reward = 1.0
        trajectory = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    assert _conversation_excerpt(Record()) == "user: hi\nassistant: hello\nground truth: pass"


def _step(step_id: str) -> steps.RunStep:
    """Return the step with a given id."""
    return next(step for step in steps.STEPS if step.id == step_id)


def test_run_plan_calls_formula_scales_to_the_full_study() -> None:
    """calls is conversations times text versions times the summed per-judge repeats."""
    from decision_judges.ui.steps import JudgeLine, RunPlan

    judges = [
        JudgeLine(name=name, model_id="m", paid=True, repeats=5) for name in ("a", "b", "c", "d")
    ]
    plan = RunPlan.build(
        judges=judges,
        conversations=230,
        questions=2,
        text_versions=["full text", "short text"],
    )
    assert plan.calls == 230 * len(["full text", "short text"]) * (5 + 5 + 5 + 5)
    assert plan.calls == 9200


def test_ask_the_judges_plan_reads_names_and_models_from_the_study() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    plan = _step("judge-outcome").plan(_paths(_FIXTURE), study)  # type: ignore[misc]

    assert plan.conversations == 2
    assert plan.questions == 2
    assert len(plan.text_versions) == 2
    assert [line.repeats for line in plan.judges] == [5, 5, 5, 5]
    assert plan.calls == 2 * 2 * (5 + 5 + 5 + 5)

    names = [line.name for line in plan.judges]
    assert "Rule-based check (free)" in names
    assert "Fast text model" in names
    assert "Strong text model" in names
    assert "Jev (decision model)" in names

    model_ids = [line.model_id for line in plan.judges]
    assert study.models.llm_cheap in model_ids
    assert study.models.llm_strong in model_ids
    assert study.models.jev in model_ids


def test_ask_the_judges_sentence_names_the_questions_and_repeats() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    sentence = _step("judge-outcome").plan(_paths(_FIXTURE), study).sentence()  # type: ignore[misc]
    assert "2 questions" in sentence
    assert "5 times" in sentence
    assert "80 judge calls" in sentence


def test_full_study_sentence_reads_as_the_owner_asked() -> None:
    """At the study's own size the sentence names 230 conversations and 9,200 calls."""
    from decision_judges.ui.steps import JudgeLine, RunPlan

    judges = [
        JudgeLine(name="Rule-based check (free)", model_id="", paid=False, repeats=5),
        JudgeLine(name="Fast text model", model_id="m", paid=True, repeats=5),
        JudgeLine(name="Strong text model", model_id="m", paid=True, repeats=5),
        JudgeLine(name="Jev (decision model)", model_id="m", paid=True, repeats=5),
    ]
    plan = RunPlan.build(
        judges=judges,
        conversations=230,
        questions=2,
        text_versions=["full text", "short text"],
    )
    assert plan.sentence() == (
        "Asks 4 judges 2 questions about 230 conversations, 5 times each, "
        "on 2 versions of the text: 9,200 judge calls."
    )


def test_draw_conclusions_plan_makes_no_new_judge_calls() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    plan = _step("analyze").plan(_paths(_FIXTURE), study)  # type: ignore[misc]
    assert plan.calls == 0
    assert plan.judges == []
    assert plan.free_note is not None
    assert "no new judge calls" in plan.free_note.lower()
    assert plan.sentence() == plan.free_note


def test_steps_without_judge_calls_report_zero_calls_and_a_note() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    for step_id in ("serialize", "analyze", "results"):
        plan = _step(step_id).plan(_paths(_FIXTURE), study)  # type: ignore[misc]
        assert plan.calls == 0, step_id
        assert plan.free_note, step_id


def test_every_numbered_step_carries_a_plan_and_the_material_panel_has_none() -> None:
    for step in steps.STEPS:
        if step.material:
            assert step.plan is None, step.id
        else:
            assert step.plan is not None, step.id


def test_label_plan_covers_the_labeled_failures() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    plan = _step("label").plan(_paths(_FIXTURE), study)  # type: ignore[misc]
    assert plan.conversations == 1
    names = [line.name for line in plan.judges]
    assert "Jev (decision model)" in names
    assert "Strong text model" in names
    assert "Laya, trained on these conversations (free, local)" in names


def test_laya_plan_is_free_and_local() -> None:
    study = load_study(_FIXTURE / "config" / "study.toml")
    plan = _step("laya").plan(_paths(_FIXTURE), study)  # type: ignore[misc]
    assert all(line.paid is False for line in plan.judges)
    assert plan.free_note is not None
    names = [line.name for line in plan.judges]
    assert "Laya, as published (free, local)" in names
    assert "Laya, trained on these conversations (free, local)" in names


def test_stopped_reason_from_formats_a_single_judge() -> None:
    from decision_judges.ui.steps import _stopped_reason_from

    assert _stopped_reason_from({"jev": "401 User not found."}) == (
        "jev stopped after 5 failed calls: 401 User not found. Other judges continued."
    )


def test_stopped_reason_from_joins_multiple_judges() -> None:
    from decision_judges.ui.steps import _stopped_reason_from

    message = _stopped_reason_from({"jev": "401 User not found.", "laya_base": "boom error"})
    assert message is not None
    assert "jev stopped after 5 failed calls: 401 User not found." in message
    assert "laya_base stopped after 5 failed calls: boom error" in message
    assert "; " in message
    assert message.endswith("Other judges continued.")


def test_stopped_reason_from_empty_is_none() -> None:
    from decision_judges.ui.steps import _stopped_reason_from

    assert _stopped_reason_from({}) is None
