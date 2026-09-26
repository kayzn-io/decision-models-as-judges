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
