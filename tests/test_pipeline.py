"""Tests for the injectable orchestration pipeline."""

import builtins
import json
from pathlib import Path

import pytest

from decision_judges import pipeline
from decision_judges.bench.load import Task, load_tasks
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import StudyConfig, load_pricing, load_study
from decision_judges.gates.g5_cascade import G5Cascade
from decision_judges.gates.g6_calibration import G6Calibration
from decision_judges.gates.g8_regression import G8Regression
from decision_judges.serialize import Injection, StateProfile, StateRecord
from decision_judges.types import Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tasks_sample.json"
STUDY_FILE = REPO_ROOT / "config" / "study.toml"
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"
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


# --- step states -----------------------------------------------------------


def _record_with_calls(task_id: str = "retail-0") -> AgentRecord:
    """Build a record with one expected and one unexpected tool call."""
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=[
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c0",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "#W2378156"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": '{"order_id": "#W9"}',
                        },
                    }
                ],
            },
        ],
        reward=1.0,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
    )


def test_serialize_all_writes_step_files_with_padded_names(tmp_path: Path) -> None:
    out_dir = tmp_path / "state"
    pipeline.serialize_all(
        {"retail-0": _record_with_calls()}, _tasks(), out_dir, [StateProfile.full]
    )

    steps_dir = out_dir / "baseline" / "full" / "steps"
    assert (steps_dir / "retail-0.000.json").is_file()
    assert (steps_dir / "retail-0.001.json").is_file()
    first = StateRecord.model_validate_json((steps_dir / "retail-0.000.json").read_text())
    assert first.step_index == 0


def test_read_step_states_keys_by_task_and_index(tmp_path: Path) -> None:
    out_dir = tmp_path / "state"
    pipeline.serialize_all(
        {"retail-0": _record_with_calls()}, _tasks(), out_dir, [StateProfile.full]
    )

    step_states = pipeline.read_step_states(out_dir, "baseline", StateProfile.full)

    assert set(step_states) == {("retail-0", 0), ("retail-0", 1)}
    assert step_states[("retail-0", 1)].step_index == 1


def test_read_step_states_missing_dir_is_empty(tmp_path: Path) -> None:
    assert pipeline.read_step_states(tmp_path / "state", "baseline", StateProfile.full) == {}


def test_items_for_gate_g2_labels_each_step(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    record = _record_with_calls()
    (agent_dir / "baseline").mkdir(parents=True)
    (agent_dir / "baseline" / "retail-0.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    pipeline.serialize_all({"retail-0": record}, _tasks(), state_dir, [StateProfile.full])

    items = pipeline.items_for_gate(
        "g2", state_dir, agent_dir, StateProfile.full, "baseline", _tasks()
    )

    by_step = {item.state.step_index: item for item in items}
    assert set(by_step) == {0, 1}
    assert by_step[0].truth_label == "necessary"  # get_order_details is expected
    assert by_step[0].truth_value == 1.0
    assert by_step[1].truth_label == "unnecessary"  # cancel is not expected
    assert by_step[1].truth_value == 0.0


def test_items_for_gate_g3_uses_whole_states(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    _seed_variant(agent_dir, state_dir, "baseline")

    items = pipeline.items_for_gate(
        "g3", state_dir, agent_dir, StateProfile.full, "baseline", _tasks()
    )

    by_task = {item.state.task_id: item for item in items}
    assert set(by_task) == {"retail-0", "retail-1"}
    assert all(item.state.step_index is None for item in items)


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


# --- jev client kwargs -----------------------------------------------------


def test_jev_client_kwargs_openrouter_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from decision_judges.config import JevRoute

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-open")
    kwargs = pipeline.jev_client_kwargs(JevRoute())

    assert kwargs == {"api_key": "sk-open", "base_url": "https://openrouter.ai/api"}


def test_jev_client_kwargs_typesafe_uses_typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from decision_judges.config import JevRoute

    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-typesafe")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kwargs = pipeline.jev_client_kwargs(JevRoute(provider="typesafe"))

    assert kwargs == {"api_key": "sk-typesafe"}


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


def test_gate_registry_contains_judging_gates() -> None:
    registry = pipeline.gate_registry()
    assert set(registry) == {"g2", "g3", "g4", "g7", "g10"}
    assert registry["g2"]().gate_id == "g2"
    assert registry["g3"]().gate_id == "g3"
    assert registry["g4"]().gate_id == "g4"
    assert registry["g7"]().gate_id == "g7"
    assert registry["g10"]().gate_id == "g10"


def test_default_repeats_g2_returns_study_mapping() -> None:
    plan = pipeline.default_repeats("g2", _study(), ["jev", "code", "unknown"])
    assert plan == {"jev": 5, "code": 1, "unknown": 1}


def test_default_repeats_other_gate_is_one() -> None:
    assert pipeline.default_repeats("g3", _study(), ["jev", "code"]) == 1


# --- analysis registry -----------------------------------------------------


def test_analysis_registry_keys_and_types() -> None:
    study = _study()
    pricing = load_pricing(PRICING_FILE)

    registry = pipeline.analysis_registry(study, pricing)

    assert set(registry) == {"g5", "g6", "g8"}
    assert isinstance(registry["g5"], G5Cascade)
    assert isinstance(registry["g6"], G6Calibration)
    assert isinstance(registry["g8"], G8Regression)
    assert registry["g5"].gate_id == "g5"
    assert registry["g6"].gate_id == "g6"
    assert registry["g8"].gate_id == "g8"


# --- load_verdicts ---------------------------------------------------------


def _verdict(state_hash: str, *, judge_id: str = "code") -> Verdict:
    """Build a minimal valid verdict for a state hash."""
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=0,
        answers=[],
        latency_ms=1,
    )


def test_load_verdicts_skips_quarantine_and_invalid(tmp_path: Path) -> None:
    cache_dir = tmp_path / "judge"
    shard = cache_dir / "ab"
    shard.mkdir(parents=True)
    (shard / "good.json").write_text(_verdict("hash-a").model_dump_json(), encoding="utf-8")
    (shard / "broken.json").write_text("{ not valid", encoding="utf-8")
    quarantine = cache_dir / "_quarantine"
    quarantine.mkdir(parents=True)
    (quarantine / "bad.json").write_text(_verdict("hash-b").model_dump_json(), encoding="utf-8")

    verdicts, warnings = pipeline.load_verdicts(cache_dir)

    assert [verdict.state_hash for verdict in verdicts] == ["hash-a"]
    assert len(warnings) == 1
    assert "broken.json" in warnings[0]


def test_load_verdicts_filters_by_judge_id(tmp_path: Path) -> None:
    cache_dir = tmp_path / "judge"
    shard = cache_dir / "ab"
    shard.mkdir(parents=True)
    (shard / "one.json").write_text(
        _verdict("hash-a", judge_id="code").model_dump_json(), encoding="utf-8"
    )
    (shard / "two.json").write_text(
        _verdict("hash-b", judge_id="fake").model_dump_json(), encoding="utf-8"
    )

    verdicts, warnings = pipeline.load_verdicts(cache_dir, judge_ids={"code"})

    assert [verdict.judge_id for verdict in verdicts] == ["code"]
    assert warnings == []


def test_load_verdicts_missing_dir_is_empty(tmp_path: Path) -> None:
    verdicts, warnings = pipeline.load_verdicts(tmp_path / "absent")

    assert verdicts == []
    assert warnings == []


# --- read_states and items_for_variants ------------------------------------


def _record_v(task_id: str, reward: float, variant: str) -> AgentRecord:
    """Build an agent record under a chosen variant."""
    return _record(task_id, reward).model_copy(update={"variant": variant})


def _seed_variant(agent_dir: Path, state_dir: Path, variant: str) -> None:
    """Write two agent records and their full states for one variant."""
    records = {
        "retail-0": _record_v("retail-0", 1.0, variant),
        "retail-1": _record_v("retail-1", 0.0, variant),
    }
    variant_dir = agent_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    for task_id, record in records.items():
        (variant_dir / f"{task_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
    pipeline.serialize_all(records, _tasks(), state_dir, [StateProfile.full])


def test_read_states_keys_by_task_id(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    _seed_variant(agent_dir, state_dir, "baseline")

    states = pipeline.read_states(state_dir, "baseline", StateProfile.full)

    assert set(states) == {"retail-0", "retail-1"}
    assert states["retail-0"].task_id == "retail-0"


def test_read_states_missing_dir_is_empty(tmp_path: Path) -> None:
    assert pipeline.read_states(tmp_path / "state", "degraded", StateProfile.full) == {}


def test_items_for_variants_concatenates_both(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    _seed_variant(agent_dir, state_dir, "baseline")
    _seed_variant(agent_dir, state_dir, "degraded")

    items = pipeline.items_for_variants(
        state_dir, agent_dir, StateProfile.full, ["baseline", "degraded"]
    )

    assert len(items) == 4
    assert {item.state.variant for item in items} == {"baseline", "degraded"}


def test_items_for_variants_skips_absent_variant(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    _seed_variant(agent_dir, state_dir, "baseline")

    items = pipeline.items_for_variants(
        state_dir, agent_dir, StateProfile.full, ["baseline", "degraded"]
    )

    assert {item.state.variant for item in items} == {"baseline"}


# --- filter_verdicts_to_items ----------------------------------------------


def test_filter_verdicts_to_items_keeps_matching() -> None:
    states = [_state("retail-0"), _state("retail-1")]
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    items = pipeline.items_from_states(states, records)
    verdicts = [_verdict("hash-retail-0"), _verdict("unmatched")]

    kept = pipeline.filter_verdicts_to_items(verdicts, items)

    assert [verdict.state_hash for verdict in kept] == ["hash-retail-0"]


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


# --- injected states (G7) --------------------------------------------------


def test_serialize_all_writes_injected_only_for_failing(tmp_path: Path) -> None:
    tasks = _tasks()
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    out_dir = tmp_path / "state"

    pipeline.serialize_all(records, tasks, out_dir, [StateProfile.full])

    injected_dir = out_dir / "baseline" / "full" / "injected"
    for kind in ("final_message", "tool_result", "control"):
        assert (injected_dir / f"retail-1.{kind}.json").is_file()
    assert not (injected_dir / "retail-0.final_message.json").exists()


def test_read_injected_states_keys_by_task_and_injection(tmp_path: Path) -> None:
    out_dir = tmp_path / "state"
    pipeline.serialize_all(
        {"retail-1": _record("retail-1", 0.0)}, _tasks(), out_dir, [StateProfile.full]
    )

    injected = pipeline.read_injected_states(out_dir, "baseline", StateProfile.full)

    assert set(injected) == {
        ("retail-1", Injection.final_message),
        ("retail-1", Injection.tool_result),
        ("retail-1", Injection.control),
    }
    assert injected[("retail-1", Injection.control)].injection is Injection.control


def test_read_injected_states_missing_dir_is_empty(tmp_path: Path) -> None:
    assert pipeline.read_injected_states(tmp_path / "state", "baseline", StateProfile.full) == {}


def test_items_for_gate_g7_uses_injected_states(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    record = _record("retail-1", 0.0)
    (agent_dir / "baseline").mkdir(parents=True)
    (agent_dir / "baseline" / "retail-1.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    pipeline.serialize_all({"retail-1": record}, _tasks(), state_dir, [StateProfile.full])

    items = pipeline.items_for_gate(
        "g7", state_dir, agent_dir, StateProfile.full, "baseline", _tasks()
    )

    assert len(items) == 3
    assert all(item.truth_label == "fail" for item in items)
    assert {item.state.injection for item in items} == {
        Injection.final_message,
        Injection.tool_result,
        Injection.control,
    }


def test_verdicts_for_analysis_non_g7_passthrough(tmp_path: Path) -> None:
    states = [_state("retail-0")]
    records = {"retail-0": _record("retail-0", 1.0)}
    items = pipeline.items_from_states(states, records)
    run_verdicts = [_verdict("hash-retail-0")]

    verdicts, out_items = pipeline.verdicts_for_analysis(
        "g3", tmp_path, run_verdicts, items, tmp_path, tmp_path, StateProfile.full, "baseline"
    )

    assert verdicts == run_verdicts
    assert out_items == items


def test_verdicts_for_analysis_g7_unions_originals_and_injected(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    state_dir = tmp_path / "state"
    cache_dir = tmp_path / "judge"
    record = _record("retail-1", 0.0)
    (agent_dir / "baseline").mkdir(parents=True)
    (agent_dir / "baseline" / "retail-1.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    pipeline.serialize_all({"retail-1": record}, _tasks(), state_dir, [StateProfile.full])

    originals = pipeline.read_states(state_dir, "baseline", StateProfile.full)
    injected = pipeline.read_injected_states(state_dir, "baseline", StateProfile.full)
    shard = cache_dir / "aa"
    shard.mkdir(parents=True)
    all_states = [*originals.values(), *injected.values()]
    for index, state in enumerate(all_states):
        (shard / f"v{index}.json").write_text(
            _verdict(state.state_hash).model_dump_json(), encoding="utf-8"
        )

    injected_items = pipeline.items_for_gate(
        "g7", state_dir, agent_dir, StateProfile.full, "baseline", _tasks()
    )
    verdicts, items = pipeline.verdicts_for_analysis(
        "g7", cache_dir, [], injected_items, state_dir, agent_dir, StateProfile.full, "baseline"
    )

    assert len(items) == 4
    injections = {item.state.injection for item in items}
    assert Injection.none in injections
    assert {Injection.final_message, Injection.tool_result, Injection.control} <= injections
    assert len(verdicts) == 4
