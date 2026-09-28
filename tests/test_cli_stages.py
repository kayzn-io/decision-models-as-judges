"""Tests for the serialize, judge, and results CLI stages."""

import json
from pathlib import Path

from typer.testing import CliRunner

from decision_judges import pipeline
from decision_judges.bench.load import Task, load_tasks
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cli import app
from decision_judges.labels import Label, LabelStore
from decision_judges.serialize import StateProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tasks_sample.json"
STUDY_FILE = REPO_ROOT / "config" / "study.toml"
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"

MARKERS = "<!-- results:start -->\n{body}\n<!-- results:end -->"
EMPTY_SUMMARY = "No cached results yet. Tables and charts appear here as evaluation stages run."


def _tasks() -> dict[str, Task]:
    """Return the sample tasks keyed by id."""
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {task.task_id: task for task in load_tasks(source=source)}


def _record(task_id: str, reward: float) -> AgentRecord:
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
    )


def _write_agent_records(agent_dir: Path) -> dict[str, AgentRecord]:
    """Write two baseline agent records to the agent cache and return them."""
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    variant_dir = agent_dir / "baseline"
    variant_dir.mkdir(parents=True, exist_ok=True)
    for task_id, record in records.items():
        (variant_dir / f"{task_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
    return records


# --- export-tasks ----------------------------------------------------------


def test_cli_export_tasks_writes_file(monkeypatch, tmp_path: Path) -> None:
    from decision_judges.bench import load as bench_load

    fixture_tasks = load_tasks(source=json.loads(FIXTURE.read_text(encoding="utf-8")))
    monkeypatch.setattr(bench_load, "load_tasks", lambda: fixture_tasks)
    out = tmp_path / "data" / "tasks.json"

    runner = CliRunner()
    result = runner.invoke(app, ["export-tasks", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.is_file()
    reloaded = load_tasks(source=json.loads(out.read_text(encoding="utf-8")))
    assert reloaded == fixture_tasks


# --- serialize -------------------------------------------------------------
def test_cli_serialize_writes_states_and_prints_counts(tmp_path: Path) -> None:
    agent_dir = tmp_path / "cache" / "agent"
    state_dir = tmp_path / "cache" / "state"
    _write_agent_records(agent_dir)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "serialize",
            "--agent-dir",
            str(agent_dir),
            "--state-dir",
            str(state_dir),
            "--variant",
            "baseline",
            "--tasks-fixture",
            str(FIXTURE),
        ],
    )

    assert result.exit_code == 0, result.output
    for profile in ("full", "compact"):
        assert (state_dir / "baseline" / profile / "retail-0.json").is_file()
        assert (state_dir / "baseline" / profile / "retail-1.json").is_file()
    assert "baseline: 4 states" in result.output


# --- judge -----------------------------------------------------------------


def _serialize_states(tmp_path: Path) -> tuple[Path, Path]:
    """Set up agent records and full-profile states, returning their dirs."""
    agent_dir = tmp_path / "cache" / "agent"
    state_dir = tmp_path / "cache" / "state"
    records = _write_agent_records(agent_dir)
    pipeline.serialize_all(records, _tasks(), state_dir, [StateProfile.full])
    return agent_dir, state_dir


def _judge_args(tmp_path: Path, agent_dir: Path, state_dir: Path) -> list[str]:
    """Build the judge invocation arguments for a code+fake run."""
    return _judge_args_for(tmp_path, agent_dir, state_dir, "g3")


def _judge_args_for(tmp_path: Path, agent_dir: Path, state_dir: Path, gate: str) -> list[str]:
    """Build code+fake judge invocation arguments for a chosen gate."""
    return [
        "judge",
        "--gate",
        gate,
        "--profile",
        "full",
        "--variant",
        "baseline",
        "--judges",
        "code,fake",
        "--repeats",
        "2",
        "--agent-dir",
        str(agent_dir),
        "--state-dir",
        str(state_dir),
        "--cache-dir",
        str(tmp_path / "cache" / "judge"),
        "--results-dir",
        str(tmp_path / "results"),
        "--study",
        str(STUDY_FILE),
        "--pricing",
        str(PRICING_FILE),
        "--ledger",
        str(tmp_path / "results" / "spend.json"),
        "--tasks-fixture",
        str(FIXTURE),
    ]


def test_cli_judge_runs_gate_and_warms_cache(tmp_path: Path) -> None:
    agent_dir, state_dir = _serialize_states(tmp_path)
    cache_dir = tmp_path / "cache" / "judge"
    results_dir = tmp_path / "results"
    runner = CliRunner()
    args = _judge_args(tmp_path, agent_dir, state_dir)

    first = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    verdict_files = sorted(cache_dir.rglob("*.json"))
    assert verdict_files, "expected cached verdict files"
    assert (results_dir / "g3_summary.md").is_file()
    assert (results_dir / "g3_accuracy.png").is_file()
    assert "accuracy" in first.output.lower()

    before = {path: path.stat().st_mtime_ns for path in verdict_files}
    second = runner.invoke(app, args)

    assert second.exit_code == 0, second.output
    after = {path: path.stat().st_mtime_ns for path in sorted(cache_dir.rglob("*.json"))}
    assert after == before, "warm cache must not recompute any verdict"


def test_cli_judge_g7_after_g3_writes_summary_and_flips(tmp_path: Path) -> None:
    agent_dir, state_dir = _serialize_states(tmp_path)
    results_dir = tmp_path / "results"
    runner = CliRunner()

    g3 = runner.invoke(app, _judge_args_for(tmp_path, agent_dir, state_dir, "g3"))
    assert g3.exit_code == 0, g3.output

    g7 = runner.invoke(app, _judge_args_for(tmp_path, agent_dir, state_dir, "g7"))
    assert g7.exit_code == 0, g7.output
    assert (results_dir / "g7_summary.md").is_file()
    assert (results_dir / "g7_flips.csv").is_file()


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


def test_cli_judge_g2_writes_step_verdicts_and_summary(tmp_path: Path) -> None:
    agent_dir = tmp_path / "cache" / "agent"
    state_dir = tmp_path / "cache" / "state"
    cache_dir = tmp_path / "cache" / "judge"
    results_dir = tmp_path / "results"
    record = _record_with_calls()
    (agent_dir / "baseline").mkdir(parents=True)
    (agent_dir / "baseline" / "retail-0.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    pipeline.serialize_all({"retail-0": record}, _tasks(), state_dir, [StateProfile.full])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "judge",
            "--gate",
            "g2",
            "--profile",
            "full",
            "--variant",
            "baseline",
            "--judges",
            "fake",
            "--repeats",
            "1",
            "--agent-dir",
            str(agent_dir),
            "--state-dir",
            str(state_dir),
            "--cache-dir",
            str(cache_dir),
            "--results-dir",
            str(results_dir),
            "--study",
            str(STUDY_FILE),
            "--pricing",
            str(PRICING_FILE),
            "--ledger",
            str(results_dir / "spend.json"),
            "--tasks-fixture",
            str(FIXTURE),
        ],
    )

    assert result.exit_code == 0, result.output
    verdict_files = sorted(cache_dir.rglob("*.json"))
    assert len(verdict_files) == 2, "expected one verdict per step under cache/judge"
    assert (results_dir / "g2_summary.md").is_file()


def _g9_judge_args(
    tmp_path: Path, agent_dir: Path, state_dir: Path, labels_path: Path
) -> list[str]:
    """Build fake-only judge invocation arguments for the g9 gate with a labels path."""
    return [
        "judge",
        "--gate",
        "g9",
        "--profile",
        "full",
        "--variant",
        "baseline",
        "--judges",
        "fake",
        "--repeats",
        "1",
        "--agent-dir",
        str(agent_dir),
        "--state-dir",
        str(state_dir),
        "--cache-dir",
        str(tmp_path / "cache" / "judge"),
        "--results-dir",
        str(tmp_path / "results"),
        "--study",
        str(STUDY_FILE),
        "--pricing",
        str(PRICING_FILE),
        "--ledger",
        str(tmp_path / "results" / "spend.json"),
        "--tasks-fixture",
        str(FIXTURE),
        "--labels",
        str(labels_path),
    ]


def test_cli_judge_g9_no_labels_exits_zero_with_message(tmp_path: Path) -> None:
    agent_dir, state_dir = _serialize_states(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    runner = CliRunner()

    result = runner.invoke(app, _g9_judge_args(tmp_path, agent_dir, state_dir, labels_path))

    assert result.exit_code == 0, result.output
    assert "No hand labels found at" in result.output
    assert "Label page" in result.output


def test_cli_judge_g9_with_labels_writes_summary(tmp_path: Path) -> None:
    agent_dir, state_dir = _serialize_states(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    LabelStore(labels_path).append(
        Label(variant="baseline", task_id="retail-1", label="premature_end")
    )
    runner = CliRunner()

    result = runner.invoke(app, _g9_judge_args(tmp_path, agent_dir, state_dir, labels_path))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "g9_summary.md").is_file()


def test_cli_judge_unknown_gate_lists_registry(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["judge", "--gate", "nope"])

    assert result.exit_code != 0
    assert "g3" in result.output


def test_cli_judge_analysis_gate_mentions_analyze(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["judge", "--gate", "g5"])

    assert result.exit_code != 0
    assert "analyze" in result.output.lower()


# --- analyze ---------------------------------------------------------------


def _analyze_args(tmp_path: Path, agent_dir: Path, state_dir: Path, gate: str) -> list[str]:
    """Build analyze invocation arguments for one baseline variant."""
    return [
        "analyze",
        "--gate",
        gate,
        "--profile",
        "full",
        "--variant",
        "baseline",
        "--agent-dir",
        str(agent_dir),
        "--state-dir",
        str(state_dir),
        "--cache-dir",
        str(tmp_path / "cache" / "judge"),
        "--results-dir",
        str(tmp_path / "results"),
        "--study",
        str(STUDY_FILE),
        "--pricing",
        str(PRICING_FILE),
    ]


def _warm_g3_cache(tmp_path: Path) -> tuple[Path, Path]:
    """Serialize states and run the g3 judge to fill the verdict cache."""
    agent_dir, state_dir = _serialize_states(tmp_path)
    runner = CliRunner()
    judged = runner.invoke(app, _judge_args(tmp_path, agent_dir, state_dir))
    assert judged.exit_code == 0, judged.output
    return agent_dir, state_dir


def test_cli_analyze_g6_writes_summary(tmp_path: Path) -> None:
    agent_dir, state_dir = _warm_g3_cache(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, _analyze_args(tmp_path, agent_dir, state_dir, "g6"))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "g6_summary.md").is_file()


def test_cli_analyze_g8_single_variant_note(tmp_path: Path) -> None:
    agent_dir, state_dir = _warm_g3_cache(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, _analyze_args(tmp_path, agent_dir, state_dir, "g8"))

    assert result.exit_code == 0, result.output
    assert "careful agent" in result.output and "rushed agent" in result.output


def test_cli_analyze_g5_zero_coverage(tmp_path: Path) -> None:
    agent_dir, state_dir = _warm_g3_cache(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, _analyze_args(tmp_path, agent_dir, state_dir, "g5"))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "g5_coverage.md").is_file()


def test_cli_analyze_rejects_non_analysis_gate(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["analyze", "--gate", "g3", "--study", str(STUDY_FILE), "--pricing", str(PRICING_FILE)],
    )

    assert result.exit_code != 0
    assert "g5" in result.output
    assert "g6" in result.output
    assert "g8" in result.output


def test_cli_analyze_no_verdicts_message(tmp_path: Path) -> None:
    agent_dir, state_dir = _serialize_states(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, _analyze_args(tmp_path, agent_dir, state_dir, "g6"))

    assert result.exit_code == 0, result.output
    assert "judge" in result.output.lower()


# --- results ---------------------------------------------------------------


def test_cli_results_lists_gate_tables(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "g3_summary.md").write_text("| a |\n|---|\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(MARKERS.format(body="placeholder"), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "results",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--results-dir",
            str(results_dir),
            "--readme",
            str(readme),
        ],
    )

    assert result.exit_code == 0, result.output
    body = (results_dir / "summary.md").read_text(encoding="utf-8")
    assert "### Outcome judging" in body
    assert "| a |" in body
    assert "### Outcome judging" in readme.read_text(encoding="utf-8")
