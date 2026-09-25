"""Tests for progress reporting and cooperative cancellation across pipeline steps."""

from collections.abc import Callable, Sequence
from pathlib import Path

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import RawRunResult, run_variant
from decision_judges.cache import Cache
from decision_judges.config import PricingTable, load_pricing
from decision_judges.gates.base import Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.judges.base import HasStateText, build_verdict
from decision_judges.progress import CancelToken, Progress, utc_now_iso
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Spend
from decision_judges.training.finetune_laya import run_cross_validation
from decision_judges.types import Answer, Question, QuestionKind, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"
STUB = Path(__file__).parent / "fixtures" / "laya_stub"


# --- Progress and CancelToken ----------------------------------------------


def test_cancel_token_flips_once_set() -> None:
    token = CancelToken()
    assert token.is_cancelled is False
    token.cancel()
    assert token.is_cancelled is True


def test_progress_round_trips_json() -> None:
    progress = Progress(step_id="g3", done=2, total=5, spent_usd=0.5, started_at=utc_now_iso())
    assert Progress.model_validate_json(progress.model_dump_json()) == progress


# --- Gate.run --------------------------------------------------------------


def _state(state_hash: str) -> StateRecord:
    return StateRecord(
        variant="baseline",
        task_id=f"retail-{state_hash}",
        profile=StateProfile.full,
        text="transcript",
        token_estimate=100,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str) -> Item:
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


class _FakeJudge:
    """A local judge whose verdict choice equals the item truth by state hash."""

    paid = False

    def __init__(self, judge_id: str, truth: dict[str, str]) -> None:
        self.judge_id = judge_id
        self.model_id = "none"
        self.prompt_version = "pv"
        self._truth = truth

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        choice = self._truth[state.state_hash]
        passed = choice == "pass"
        answers = [
            Answer(question_id="completed", kind=QuestionKind.noul, noul=1.0 if passed else 0.0),
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice=choice,
                probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
                confidence=1.0,
            ),
        ]
        return build_verdict(self, state, repeat, answers, latency_ms=1)


def _spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    return Spend(load_pricing(PRICING_FILE), caps, tmp_path / "ledger.json")


def test_gate_run_reports_progress_for_every_call(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = _FakeJudge("code", truth)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    seen: list[Progress] = []
    verdicts = G3Outcome().run(items, [judge], cache, spend, repeats=1, on_progress=seen.append)

    assert len(verdicts) == 3
    assert [p.done for p in seen] == [1, 2, 3]
    assert all(p.total == 3 for p in seen)
    assert all(isinstance(p.spent_usd, float) for p in seen)
    assert all(p.last_item is not None and " · " in p.last_item for p in seen)
    assert seen[-1].last_item is not None and seen[-1].last_item.endswith("pass")


def test_gate_run_stops_when_cancelled_mid_plan(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = _FakeJudge("code", truth)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    cancel = CancelToken()

    def on_progress(progress: Progress) -> None:
        cancel.cancel()  # cancel after the very first call

    verdicts = G3Outcome().run(
        items, [judge], cache, spend, repeats=1, on_progress=on_progress, cancel=cancel
    )

    assert len(verdicts) == 1
    assert len(verdicts) < len(items)


# --- run_variant -----------------------------------------------------------


def _task(index: int) -> Task:
    return Task(task_id=f"retail-{index}", instruction="do a thing", actions=[], outputs=[])


def _agent_pricing() -> PricingTable:
    from datetime import date

    return PricingTable(
        effective_date=date(2026, 9, 24),
        source="test",
        models={"agent-model": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}},
    )


def _agent_spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    return Spend(_agent_pricing(), caps, tmp_path / "ledger.json")


class _FakeRunner:
    agent_model = "agent-model"
    user_model = "user-model"
    policy = "POLICY"

    def __init__(self, on_call: Callable[[int], None] | None = None) -> None:
        self._on_call = on_call

    def __call__(self, task_index: int, policy: str) -> RawRunResult:
        if self._on_call is not None:
            self._on_call(task_index)
        return RawRunResult(
            reward=1.0,
            messages=[{"role": "assistant", "content": f"done {task_index}"}],
            info={"ok": True},
            est_input_tokens=1_000,
            est_output_tokens=100,
        )


def test_run_variant_reports_per_task(tmp_path: Path) -> None:
    runner = _FakeRunner()
    spend = _agent_spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1), _task(2)]

    seen: list[Progress] = []
    summary = run_variant(
        "baseline",
        tasks,
        runner,
        spend,
        tmp_path / "out",
        concurrency=1,
        on_progress=seen.append,
    )

    assert summary.completed == 3
    assert [p.done for p in seen] == [1, 2, 3]
    assert all(p.total == 3 for p in seen)
    assert all(p.last_item is not None and "retail-" in p.last_item for p in seen)


def test_run_variant_honours_cancel_with_stopped_reason(tmp_path: Path) -> None:
    cancel = CancelToken()
    runner = _FakeRunner()
    spend = _agent_spend(tmp_path, {"agent": 100.0})
    tasks = [_task(0), _task(1), _task(2)]

    def on_progress(progress: Progress) -> None:
        cancel.cancel()  # cancel after the first task completes

    summary = run_variant(
        "baseline",
        tasks,
        runner,
        spend,
        tmp_path / "out",
        concurrency=1,
        on_progress=on_progress,
        cancel=cancel,
    )

    assert summary.stopped_reason == "cancelled"
    assert summary.completed == 1
    written = list((tmp_path / "out" / "baseline").glob("*.json"))
    assert len(written) < 3


# --- run_cross_validation --------------------------------------------------

VERDICT_Q = Question(
    id="verdict", kind=QuestionKind.choice, text="Pass or fail?", options=["pass", "fail"]
)
COMPLETED_Q = Question(id="completed", kind=QuestionKind.noul, text="Did the agent complete it?")
FT_QUESTIONS = [VERDICT_Q, COMPLETED_Q]


def _cv_state(task_id: str) -> StateRecord:
    text = f"state for {task_id}"
    return StateRecord(
        variant="baseline",
        task_id=task_id,
        profile=StateProfile.compact,
        text=text,
        token_estimate=len(text) // 4,
        state_hash=f"hash-{task_id}",
    )


def _cv_inputs() -> tuple[dict[str, StateRecord], dict[str, float]]:
    task_ids = [f"retail-{index}" for index in range(4)]
    states = {task_id: _cv_state(task_id) for task_id in task_ids}
    rewards = {task_id: float(index % 2 == 0) for index, task_id in enumerate(task_ids)}
    return states, rewards


def test_run_cross_validation_reports_per_fold_and_per_step(tmp_path: Path) -> None:
    states, rewards = _cv_inputs()
    seen: list[Progress] = []

    manifests = run_cross_validation(
        STUB,
        states,
        rewards,
        FT_QUESTIONS,
        tmp_path / "models",
        k=2,
        seed=7,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4,
        device="cpu",
        max_steps=2,
        on_progress=seen.append,
    )

    assert len(manifests) == 2
    step_reports = [p for p in seen if p.last_item is not None and "step" in p.last_item]
    fold_reports = [p for p in seen if p.last_item is not None and "complete" in p.last_item]
    assert step_reports
    assert len(fold_reports) == 2


def test_run_cross_validation_cancels_cleanly(tmp_path: Path) -> None:
    states, rewards = _cv_inputs()
    cancel = CancelToken()
    seen: list[Progress] = []

    def on_progress(progress: Progress) -> None:
        seen.append(progress)
        if progress.last_item is not None and "step" in progress.last_item:
            cancel.cancel()

    manifests = run_cross_validation(
        STUB,
        states,
        rewards,
        FT_QUESTIONS,
        tmp_path / "models",
        k=2,
        seed=7,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4,
        device="cpu",
        max_steps=2,
        on_progress=on_progress,
        cancel=cancel,
    )

    assert len(manifests) < 2
    assert any(p.last_item is not None and "step" in p.last_item for p in seen)
