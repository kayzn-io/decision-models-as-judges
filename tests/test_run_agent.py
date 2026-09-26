"""Tests for the resumable agent runner and its degraded policy variant.

These unit tests inject a fake runner and never import tau_bench or touch the
network. The real ``build_tau_runner`` is left to the marked smoke test.
"""

import os
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from decision_judges import cli
from decision_judges.bench import load as bench_load
from decision_judges.bench import run_agent as run_agent_mod
from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import (
    CONFIRMATION_RULE,
    DEGRADED_RULE,
    AgentRecord,
    MissingCredentials,
    RawRunResult,
    RunSummary,
    classify_failure,
    degraded_policy,
    run_variant,
)
from decision_judges.config import PricingTable
from decision_judges.spend import Spend

ROOT = Path(__file__).resolve().parent.parent
EXCERPT = Path(__file__).parent / "fixtures" / "retail_policy_excerpt.txt"


def _task(index: int) -> Task:
    """Build a minimal task with a positional retail id."""
    return Task(task_id=f"retail-{index}", instruction="do a thing", actions=[], outputs=[])


def _pricing() -> PricingTable:
    """Price the fake agent model at 1 USD per million tokens each way."""
    return PricingTable(
        effective_date=date(2026, 9, 24),
        source="test",
        models={"agent-model": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}},
    )


def _spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    return Spend(_pricing(), caps, tmp_path / "ledger.json")


class FakeRunner:
    """A runner that records its calls and returns a fixed result."""

    def __init__(
        self,
        *,
        agent_model: str = "agent-model",
        user_model: str = "user-model",
        policy: str = "POLICY",
        est_input: int = 1_000,
        est_output: int = 100,
        reward: float = 1.0,
    ) -> None:
        self.agent_model = agent_model
        self.user_model = user_model
        self.policy = policy
        self.est_input = est_input
        self.est_output = est_output
        self.reward = reward
        self.calls: list[int] = []

    def __call__(self, task_index: int, policy: str) -> RawRunResult:
        self.calls.append(task_index)
        return RawRunResult(
            reward=self.reward,
            messages=[{"role": "assistant", "content": f"done {task_index}"}],
            info={"ok": True},
            est_input_tokens=self.est_input,
            est_output_tokens=self.est_output,
        )


def test_run_variant_writes_records_and_is_resumable(tmp_path: Path) -> None:
    runner = FakeRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1), _task(2)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=2)

    assert isinstance(summary, RunSummary)
    assert summary.completed == 3
    assert summary.excluded == 0
    assert summary.skipped_existing == 0
    assert summary.pass_rate == pytest.approx(1.0)
    assert summary.stopped_reason is None
    assert len(runner.calls) == 3
    for i in range(3):
        path = tmp_path / "out" / "baseline" / f"retail-{i}.json"
        record = AgentRecord.model_validate_json(path.read_text())
        assert record.variant == "baseline"
        assert record.task_id == f"retail-{i}"
        assert record.excluded is False

    rerun_runner = FakeRunner()
    rerun_spend = _spend(tmp_path, {"agent": 100.0})
    rerun = run_variant(
        "baseline", tasks, rerun_runner, rerun_spend, tmp_path / "out", concurrency=2
    )
    assert rerun.skipped_existing == 3
    assert rerun.completed == 0
    assert rerun_runner.calls == []


def test_run_variant_excludes_failed_task(tmp_path: Path) -> None:
    class RaisingRunner(FakeRunner):
        def __call__(self, task_index: int, policy: str) -> RawRunResult:
            if task_index == 1:
                raise RuntimeError("boom")
            return super().__call__(task_index, policy)

    runner = RaisingRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1), _task(2)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert summary.completed == 2
    assert summary.excluded == 1
    assert summary.pass_rate == pytest.approx(1.0)

    excluded_path = tmp_path / "out" / "baseline" / "retail-1.json"
    excluded = AgentRecord.model_validate_json(excluded_path.read_text())
    assert excluded.excluded is True
    assert excluded.exclusion_reason == "RuntimeError: boom"

    exclusions = (tmp_path / "out" / "exclusions.md").read_text().strip().splitlines()
    assert exclusions == ["- baseline/retail-1: RuntimeError: boom"]

    # The failed task's reservation was cancelled: only the two settled runs count.
    assert spend.reserved("agent") == pytest.approx(0.0)
    assert spend.spent("agent") == pytest.approx(2 * (1_000 + 100) / 1_000_000)


class _AuthenticationError(RuntimeError):
    """A credentials failure whose type name carries 'Authentication'."""


def test_classify_failure_config_messages() -> None:
    assert classify_failure(RuntimeError("Missing credentials")) == "config"
    assert classify_failure(RuntimeError("Invalid API key provided")) == "config"
    assert classify_failure(RuntimeError("401 Unauthorized")) == "config"
    assert classify_failure(RuntimeError("403 Forbidden")) == "config"
    assert classify_failure(_AuthenticationError("nope")) == "config"


def test_classify_failure_transient_messages() -> None:
    assert classify_failure(TimeoutError("request timed out")) == "transient"
    assert classify_failure(ConnectionError("connection reset by peer")) == "transient"
    assert classify_failure(RuntimeError("429 Too Many Requests")) == "transient"
    assert classify_failure(RuntimeError("InternalServerError: 500")) == "transient"
    assert classify_failure(RuntimeError("ServiceUnavailable")) == "transient"
    assert classify_failure(RuntimeError("RateLimitError")) == "transient"


def test_classify_failure_task_messages() -> None:
    assert classify_failure(ValueError("order #W9 not found")) == "task"
    assert classify_failure(KeyError("user_id")) == "task"


class CredsRunner(FakeRunner):
    """A runner that always fails with a credentials error."""

    def __call__(self, task_index: int, policy: str) -> RawRunResult:
        self.calls.append(task_index)
        raise _AuthenticationError("Missing credentials")


def test_config_failure_writes_no_records_and_reruns(tmp_path: Path) -> None:
    runner = CredsRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1), _task(2)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert summary.failed_config == 3
    assert summary.failed_transient == 0
    assert summary.completed == 0
    assert summary.excluded == 0
    assert list((tmp_path / "out" / "baseline").glob("*.json")) == []
    assert not (tmp_path / "out" / "exclusions.md").exists()
    assert spend.reserved("agent") == pytest.approx(0.0)
    assert spend.spent("agent") == pytest.approx(0.0)

    good = FakeRunner()
    rerun = run_variant(
        "baseline", tasks, good, _spend(tmp_path, {"agent": 100.0}), tmp_path / "out", concurrency=1
    )
    assert rerun.completed == 3
    assert rerun.skipped_existing == 0
    assert sorted(good.calls) == [0, 1, 2]


def test_run_aborts_after_three_setup_failures(tmp_path: Path) -> None:
    runner = CredsRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(i) for i in range(10)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert summary.stopped_reason is not None
    assert summary.stopped_reason.startswith("aborted: the first 3 tasks failed with")
    assert len(runner.calls) == 3
    assert summary.failed_config == 3
    assert list((tmp_path / "out" / "baseline").glob("*.json")) == []


def test_task_failure_recorded_and_skipped_on_rerun(tmp_path: Path) -> None:
    class TaskFailRunner(FakeRunner):
        def __call__(self, task_index: int, policy: str) -> RawRunResult:
            self.calls.append(task_index)
            raise ValueError("environment blew up")

    runner = TaskFailRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert summary.excluded == 1
    assert summary.failed_config == 0
    assert summary.failed_transient == 0
    record = AgentRecord.model_validate_json(
        (tmp_path / "out" / "baseline" / "retail-0.json").read_text()
    )
    assert record.excluded is True
    assert record.exclusion_reason == "ValueError: environment blew up"

    rerun_runner = TaskFailRunner()
    rerun = run_variant(
        "baseline",
        tasks,
        rerun_runner,
        _spend(tmp_path, {"agent": 100.0}),
        tmp_path / "out",
        concurrency=1,
    )
    assert rerun.skipped_existing == 1
    assert rerun_runner.calls == []


class PreflightRunner(FakeRunner):
    """A runner whose preflight refuses to start when credentials are missing."""

    def preflight(self) -> None:
        raise MissingCredentials(
            "OPENROUTER_API_KEY is not set. Set it in the app sidebar or run "
            "export OPENROUTER_API_KEY=... before this step."
        )


def test_preflight_missing_credentials_stops_before_any_task(tmp_path: Path) -> None:
    runner = PreflightRunner()
    spend = _spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1)]

    with pytest.raises(MissingCredentials):
        run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert runner.calls == []
    assert spend.reserved("agent") == pytest.approx(0.0)
    assert spend.spent("agent") == pytest.approx(0.0)
    assert list((tmp_path / "out" / "baseline").glob("*.json")) == []


def test_run_agent_cli_missing_credentials_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = PreflightRunner(agent_model="openai/gpt-4.1", user_model="openai/gpt-4o-mini")
    monkeypatch.setattr(run_agent_mod, "build_tau_runner", lambda study, variant: runner)
    monkeypatch.setattr(bench_load, "load_tasks", lambda: [_task(0), _task(1)])

    result = CliRunner().invoke(
        cli.app,
        [
            "run-agent",
            "--variant",
            "baseline",
            "--out-dir",
            str(tmp_path / "agent"),
            "--study",
            str(ROOT / "config" / "study.toml"),
            "--pricing",
            str(ROOT / "config" / "pricing.toml"),
            "--ledger",
            str(tmp_path / "spend.json"),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "OPENROUTER_API_KEY" in result.output


def test_spend_is_reserved_before_runner_runs(tmp_path: Path) -> None:
    class ReservedCheckRunner(FakeRunner):
        def __init__(self, spend: Spend, stage: str) -> None:
            super().__init__()
            self._spend = spend
            self._stage = stage
            self.reserved_seen: list[float] = []

        def __call__(self, task_index: int, policy: str) -> RawRunResult:
            self.reserved_seen.append(self._spend.reserved(self._stage))
            return super().__call__(task_index, policy)

    spend = _spend(tmp_path, {"agent": 100.0})
    runner = ReservedCheckRunner(spend, "agent")
    tasks = [_task(0), _task(1)]

    run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert runner.reserved_seen
    assert all(amount > 0.0 for amount in runner.reserved_seen)


def test_spend_cap_stops_scheduling(tmp_path: Path) -> None:
    # Reserve estimate is 60k input + 4k output = 0.064 USD; a 0.1 cap fits one.
    runner = FakeRunner(est_input=60_000, est_output=4_000)
    spend = _spend(tmp_path, {"agent": 0.1})
    tasks = [_task(0), _task(1), _task(2)]

    summary = run_variant("baseline", tasks, runner, spend, tmp_path / "out", concurrency=1)

    assert summary.stopped_reason is not None
    written = list((tmp_path / "out" / "baseline").glob("*.json"))
    assert len(written) == 1
    assert summary.completed == 1


def test_degraded_policy_replaces_confirmation_rule_with_act_at_once() -> None:
    wiki = EXCERPT.read_text()
    assert CONFIRMATION_RULE in wiki
    degraded = degraded_policy(wiki)
    assert CONFIRMATION_RULE not in degraded
    assert DEGRADED_RULE in degraded
    assert len(degraded) > 0
    assert "You should not make up any information" in degraded


def test_degraded_policy_raises_when_rule_absent() -> None:
    with pytest.raises(ValueError, match="confirmation rule not found in policy text"):
        degraded_policy("a policy with no such rule")


def test_run_agent_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner(agent_model="openai/gpt-4.1", user_model="openai/gpt-4o-mini")
    monkeypatch.setattr(run_agent_mod, "build_tau_runner", lambda study, variant: runner)
    monkeypatch.setattr(bench_load, "load_tasks", lambda: [_task(0), _task(1)])

    out_dir = tmp_path / "agent"
    ledger = tmp_path / "spend.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "run-agent",
            "--variant",
            "baseline",
            "--out-dir",
            str(out_dir),
            "--study",
            str(ROOT / "config" / "study.toml"),
            "--pricing",
            str(ROOT / "config" / "pricing.toml"),
            "--ledger",
            str(ledger),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "baseline" / "retail-0.json").is_file()
    assert (out_dir / "baseline" / "retail-1.json").is_file()


@pytest.mark.tau_bench
def test_build_tau_runner_binds_config_without_running() -> None:
    pytest.importorskip("tau_bench")
    from tau_bench.envs.retail.wiki import WIKI

    from decision_judges.config import load_study

    study = load_study(ROOT / "config" / "study.toml")
    baseline = run_agent_mod.build_tau_runner(study, "baseline")
    degraded = run_agent_mod.build_tau_runner(study, "degraded")

    assert baseline.agent_model == study.models.agent
    assert baseline.user_model == study.models.user_sim
    assert baseline.policy == WIKI
    assert CONFIRMATION_RULE not in degraded.policy


def test_preflight_hands_the_openrouter_key_to_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """litellm reads OPENAI_API_KEY; preflight must populate it from the OpenRouter key."""
    from decision_judges.bench import run_agent as mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = mod._TauRunner.__new__(mod._TauRunner)
    runner._key_env = "OPENROUTER_API_KEY"
    runner.preflight()
    assert os.environ["OPENAI_API_KEY"] == "sk-or-test-key"
