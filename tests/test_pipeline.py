"""Tests for the injectable orchestration pipeline."""

import builtins
import json
from pathlib import Path

import pytest

from decision_judges import pipeline
from decision_judges.bench.load import Task, load_tasks
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import StudyConfig, load_study
from decision_judges.serialize import StateProfile, StateRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tasks_sample.json"
STUDY_FILE = REPO_ROOT / "config" / "study.toml"
RUBRIC_FILE = REPO_ROOT / "config" / "rubrics" / "g3_outcome.md"


def _tasks() -> dict[str, Task]:
    """Return the sample tasks keyed by id."""
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {task.task_id: task for task in load_tasks(source=source)}


def _record(task_id: str, reward: float, *, excluded: bool = False) -> AgentRecord:
    """Build an agent record with a tiny valid trajectory."""
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=[
            {"role": "user", "content": "please help"},
            {"role": "assistant", "content": "done"},
        ],
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
        excluded=excluded,
    )


def _study() -> StudyConfig:
    """Load the repository study configuration."""
    return load_study(STUDY_FILE)


def _state(task_id: str) -> StateRecord:
    """Build a minimal serialized state for a task."""
    return StateRecord(
        variant="baseline",
        task_id=task_id,
        profile=StateProfile.full,
        text="transcript",
        token_estimate=10,
        state_hash=f"hash-{task_id}",
    )


# --- serialize_all ---------------------------------------------------------


def test_serialize_all_writes_both_profiles_for_both_variants(tmp_path: Path) -> None:
    tasks = _tasks()
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
        "retail-2": _record("retail-2", 1.0, excluded=True),
    }
    out_dir = tmp_path / "state"

    count = pipeline.serialize_all(
        records, tasks, out_dir, [StateProfile.full, StateProfile.compact]
    )

    assert count == 4
    for task_id in ("retail-0", "retail-1"):
        for profile in ("full", "compact"):
            path = out_dir / "baseline" / profile / f"{task_id}.json"
            assert path.is_file()
            loaded = StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
            assert loaded.task_id == task_id
            assert loaded.profile.value == profile
    # Excluded record is skipped entirely.
    assert not (out_dir / "baseline" / "full" / "retail-2.json").exists()


def test_serialize_all_is_idempotent(tmp_path: Path) -> None:
    tasks = _tasks()
    records = {"retail-0": _record("retail-0", 1.0)}
    out_dir = tmp_path / "state"

    first = pipeline.serialize_all(records, tasks, out_dir, [StateProfile.full])
    second = pipeline.serialize_all(records, tasks, out_dir, [StateProfile.full])

    assert first == second == 1


# --- load_agent_records ----------------------------------------------------


def test_load_agent_records_skips_invalid_with_warning(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    variant_dir = agent_dir / "baseline"
    variant_dir.mkdir(parents=True)
    good = _record("retail-0", 1.0)
    (variant_dir / "retail-0.json").write_text(good.model_dump_json(), encoding="utf-8")
    (variant_dir / "retail-1.json").write_text("{ not valid json", encoding="utf-8")

    records, warnings = pipeline.load_agent_records(agent_dir, "baseline")

    assert set(records) == {"retail-0"}
    assert len(warnings) == 1
    assert "retail-1.json" in warnings[0]


def test_load_agent_records_missing_variant_dir(tmp_path: Path) -> None:
    records, warnings = pipeline.load_agent_records(tmp_path / "agent", "degraded")

    assert records == {}
    assert warnings == []


# --- items_from_states -----------------------------------------------------


def test_items_from_states_pairs_truth() -> None:
    states = [_state("retail-0"), _state("retail-1")]
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }

    items = pipeline.items_from_states(states, records)

    by_task = {item.state.task_id: item for item in items}
    assert by_task["retail-0"].truth_label == "pass"
    assert by_task["retail-0"].truth_value == 1.0
    assert by_task["retail-1"].truth_label == "fail"
    assert by_task["retail-1"].truth_value == 0.0


def test_items_from_states_skips_states_without_record() -> None:
    states = [_state("retail-0"), _state("retail-9")]
    records = {"retail-0": _record("retail-0", 1.0)}

    items = pipeline.items_from_states(states, records)

    assert [item.state.task_id for item in items] == ["retail-0"]


# --- judge specs and construction ------------------------------------------


def test_judge_specs_from_maps_known_names() -> None:
    study = _study()
    specs = {
        spec.name: spec
        for spec in pipeline.judge_specs_from(
            ["code", "llm_cheap", "llm_strong", "jev", "fake"], study
        )
    }

    assert specs["code"].kind == "code"
    assert specs["code"].model_id == "none"
    assert specs["llm_cheap"].kind == "llm"
    assert specs["llm_cheap"].model_id == study.models.llm_cheap
    assert specs["llm_strong"].model_id == study.models.llm_strong
    assert specs["jev"].kind == "jev"
    assert specs["jev"].model_id == study.models.jev
    assert specs["fake"].kind == "fake"


def test_judge_specs_from_unknown_name_lists_options() -> None:
    with pytest.raises(ValueError) as excinfo:
        pipeline.judge_specs_from(["bogus"], _study())

    message = str(excinfo.value)
    assert "bogus" in message
    assert "code" in message
    assert "fake" in message


def test_build_judges_constructs_code_and_fake() -> None:
    study = _study()
    specs = pipeline.judge_specs_from(["code", "fake"], study)

    judges = pipeline.build_judges(
        specs, study=study, tasks={}, records={}, rubric_path=RUBRIC_FILE, prompt_version="pv"
    )

    by_id = {judge.judge_id: judge for judge in judges}
    assert set(by_id) == {"code", "fake"}
    assert by_id["fake"].model_id == "none"
    assert by_id["code"].model_id == "none"


def test_build_judges_llm_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    study = _study()
    specs = pipeline.judge_specs_from(["llm_cheap"], study)

    judges = pipeline.build_judges(
        specs, study=study, tasks={}, records={}, rubric_path=RUBRIC_FILE, prompt_version="pv"
    )

    assert judges[0].model_id == study.models.llm_cheap


def test_build_judges_code_fake_does_not_import_typesafe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    seen: list[str] = []

    def guard(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] == "typesafe_sdk":
            seen.append(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", guard)
    specs = pipeline.judge_specs_from(["code", "fake"], _study())
    pipeline.build_judges(
        specs, study=_study(), tasks={}, records={}, rubric_path=RUBRIC_FILE, prompt_version="pv"
    )

    assert seen == []


# --- repeats parsing -------------------------------------------------------


def test_parse_repeats_shared_int() -> None:
    assert pipeline.parse_repeats(["3"]) == 3


def test_parse_repeats_per_judge_mapping() -> None:
    assert pipeline.parse_repeats(["jev=2", "code=1"]) == {"jev": 2, "code": 1}


def test_parse_repeats_empty_uses_default() -> None:
    assert pipeline.parse_repeats([]) == 1


def test_parse_repeats_rejects_mixed() -> None:
    with pytest.raises(ValueError):
        pipeline.parse_repeats(["3", "jev=2"])


# --- gate registry ---------------------------------------------------------


def test_gate_registry_contains_g3() -> None:
    registry = pipeline.gate_registry()
    assert set(registry) == {"g3", "g10"}
    assert registry["g3"]().gate_id == "g3"
    assert registry["g10"]().gate_id == "g10"


# --- run_gate and analyze_gate end to end (offline, fake judge) ------------


def _make_states_and_records(tmp_path: Path) -> tuple[list[StateRecord], dict[str, AgentRecord]]:
    """Serialize two records and load the resulting states back."""
    tasks = _tasks()
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    state_dir = tmp_path / "state"
    pipeline.serialize_all(records, tasks, state_dir, [StateProfile.full])
    states = [
        StateRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((state_dir / "baseline" / "full").glob("*.json"))
    ]
    return states, records


def test_run_and_analyze_gate_with_fake_judge(tmp_path: Path) -> None:
    from decision_judges.cache import Cache
    from decision_judges.config import load_pricing
    from decision_judges.spend import Spend

    states, records = _make_states_and_records(tmp_path)
    items = pipeline.items_from_states(states, records)
    gate = pipeline.gate_registry()["g3"]()
    judges: list[object] = [pipeline.FakeJudge(judge_id="fake", prompt_version="pv")]
    cache = Cache(tmp_path / "judge")
    pricing = load_pricing(REPO_ROOT / "config" / "pricing.toml")
    spend = Spend(pricing, {"g3": 0.0}, tmp_path / "ledger.json")

    verdicts = pipeline.run_gate(gate, items, judges, cache, spend, 2)
    findings = pipeline.analyze_gate(gate, verdicts, items, tmp_path / "results")

    assert len(verdicts) == 4
    assert (tmp_path / "results" / "g3_summary.md").is_file()
    assert (tmp_path / "results" / "g3_accuracy.png").is_file()
    assert findings.strip() != ""


def test_fake_judge_answers_every_question() -> None:
    gate = pipeline.gate_registry()["g3"]()
    judge = pipeline.FakeJudge()
    state = _state("retail-0")

    verdict = judge.judge(state, gate.questions(), 0)

    answers = {answer.question_id: answer for answer in verdict.answers}
    assert answers["verdict"].choice == "pass"
    assert answers["completed"].noul == 0.5
    assert verdict.usage.input_tokens == 0
    assert verdict.error is None
