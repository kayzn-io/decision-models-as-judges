"""Tests for the gate framework and the G3 outcome gate."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache, cache_key
from decision_judges.config import load_pricing
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.judges.base import HasStateText, build_verdict
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Spend, SpendCapExceeded
from decision_judges.types import Answer, Question, QuestionKind, Usage, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"

_PRICED_MODEL = "openai/gpt-4o-mini"


# --- fixtures / builders ---------------------------------------------------


def _state(
    state_hash: str,
    *,
    token_estimate: int = 100,
    profile: StateProfile = StateProfile.full,
) -> StateRecord:
    """Build a minimal serialized state with a chosen hash and token estimate."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=profile,
        text="transcript",
        token_estimate=token_estimate,
        state_hash=state_hash,
    )


def _item(
    state_hash: str,
    truth_label: str,
    *,
    token_estimate: int = 100,
    profile: StateProfile = StateProfile.full,
) -> Item:
    """Build an item pairing a state with its ground-truth label and value."""
    return Item(
        state=_state(state_hash, token_estimate=token_estimate, profile=profile),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _make_verdict(
    judge: "FakeJudge",
    state: HasStateText,
    repeat: int,
    choice: str | None,
    *,
    error: str | None = None,
    usage: Usage | None = None,
    latency_ms: int = 1,
) -> Verdict:
    """Build a verdict answering the completed and verdict questions, or an error verdict."""
    answers: list[Answer] = []
    if error is None and choice is not None:
        passed = choice == "pass"
        answers = [
            Answer(
                question_id="completed",
                kind=QuestionKind.noul,
                noul=1.0 if passed else 0.0,
            ),
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice=choice,
                probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
                confidence=1.0,
            ),
        ]
    return build_verdict(
        judge, state, repeat, answers, usage=usage, latency_ms=latency_ms, error=error
    )


class FakeJudge:
    """A judge whose choice/error per state and repeat is scripted for tests."""

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        prompt_version: str,
        decide: Callable[[str, int], tuple[str | None, str | None]],
        *,
        usage: Usage | None = None,
        paid: bool = True,
    ) -> None:
        self.judge_id = judge_id
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.paid = paid
        self._decide = decide
        self._usage = usage
        self.calls: list[tuple[str, int]] = []

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        self.calls.append((state.state_hash, repeat))
        choice, error = self._decide(state.state_hash, repeat)
        return _make_verdict(self, state, repeat, choice, error=error, usage=self._usage)


def _perfect(truth: dict[str, str]) -> Callable[[str, int], tuple[str | None, str | None]]:
    """Return a decide function whose choice always equals the item's truth."""
    return lambda state_hash, repeat: (truth[state_hash], None)


def _spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    """Build a real Spend over the repo pricing table and a temp ledger."""
    return Spend(load_pricing(PRICING_FILE), caps, tmp_path / "ledger.json")


# --- run() -----------------------------------------------------------------


def test_run_calls_each_judge_items_times_repeats_then_zero_on_warm(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    ja = FakeJudge("a", "none", "pv-a", _perfect(truth), paid=False)
    jb = FakeJudge("b", "none", "pv-b", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    gate = G3Outcome()

    verdicts = gate.run(items, [ja, jb], cache, spend, repeats=2)

    assert len(verdicts) == 12
    assert len(ja.calls) == 6
    assert len(jb.calls) == 6

    ja2 = FakeJudge("a", "none", "pv-a", _perfect(truth), paid=False)
    jb2 = FakeJudge("b", "none", "pv-b", _perfect(truth), paid=False)
    warm = gate.run(items, [ja2, jb2], cache, spend, repeats=2)

    assert len(warm) == 12
    assert ja2.calls == []
    assert jb2.calls == []


def test_run_verdict_order_is_deterministic(tmp_path: Path) -> None:
    truth = {"h2": "pass", "h0": "fail", "h1": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    ja = FakeJudge("b", "none", "pv", _perfect(truth), paid=False)
    jb = FakeJudge("a", "none", "pv", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    verdicts = G3Outcome().run(items, [ja, jb], cache, spend, repeats=2)

    keys = [(v.judge_id, v.state_hash, v.repeat) for v in verdicts]
    assert keys == sorted(keys)


def test_priced_judge_reserves_and_settles_per_miss(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = FakeJudge(
        "llm",
        _PRICED_MODEL,
        "pv",
        _perfect(truth),
        usage=Usage(input_tokens=1000, output_tokens=100),
    )
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 50.0})

    G3Outcome().run(items, [judge], cache, spend, repeats=1)

    assert spend.total() > 0.0
    assert spend.spent("g3") == pytest.approx(3 * (1000 * 0.15 + 100 * 0.60) / 1_000_000)


def test_code_style_judge_records_no_spend(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = FakeJudge("code", "none", "code-1", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    G3Outcome().run(items, [judge], cache, spend, repeats=2)

    assert spend.total() == 0.0
    assert spend.spent("g3") == 0.0


def test_spend_cap_exceeded_propagates_and_cancels_cleanly(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = FakeJudge(
        "llm",
        _PRICED_MODEL,
        "pv",
        _perfect(truth),
        usage=Usage(input_tokens=1000, output_tokens=100),
    )
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0004})

    with pytest.raises(SpendCapExceeded):
        G3Outcome().run(items, [judge], cache, spend, repeats=1)

    one_call = (1000 * 0.15 + 100 * 0.60) / 1_000_000
    assert spend.total() == pytest.approx(one_call)
    assert spend.spent("g3") == pytest.approx(one_call)


def test_per_judge_repeats_mapping_is_honored(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail", "h2": "pass"}
    items = [_item(h, label) for h, label in truth.items()]
    jev = FakeJudge("jev", "none", "pv-jev", _perfect(truth), paid=False)
    llm = FakeJudge("llm", "none", "pv-llm", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    G3Outcome().run(items, [jev, llm], cache, spend, repeats={"jev": 2, "llm": 1})

    assert len(jev.calls) == 6
    assert len(llm.calls) == 3


# --- errored-verdict retry -------------------------------------------------


def test_errored_cached_verdict_is_retried_and_replaced(tmp_path: Path) -> None:
    truth = {"h0": "pass"}
    items = [_item("h0", "pass")]
    judge = FakeJudge("a", "none", "pv-a", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    key = cache_key("a", "none", "pv-a", "h0", 0)
    cache.put(key, _make_verdict(judge, items[0].state, 0, None, error="401 User not found"))

    verdicts = G3Outcome().run(items, [judge], cache, spend, repeats=1)

    assert judge.calls == [("h0", 0)]
    stored = cache.peek(key, Verdict)
    assert stored is not None
    assert stored.error is None
    assert stored.answers != []
    assert verdicts[0].error is None


def test_successful_cached_verdict_still_short_circuits(tmp_path: Path) -> None:
    truth = {"h0": "pass"}
    items = [_item("h0", "pass")]
    judge = FakeJudge("a", "none", "pv-a", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    key = cache_key("a", "none", "pv-a", "h0", 0)
    cache.put(key, _make_verdict(judge, items[0].state, 0, "pass"))

    verdicts = G3Outcome().run(items, [judge], cache, spend, repeats=1)

    assert judge.calls == []
    assert verdicts[0].error is None


def test_retry_of_errored_cache_reserves_and_settles_spend(tmp_path: Path) -> None:
    items = [_item("h0", "pass")]
    judge = FakeJudge(
        "llm",
        _PRICED_MODEL,
        "pv",
        _perfect({"h0": "pass"}),
        usage=Usage(input_tokens=1000, output_tokens=100),
    )
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 50.0})
    key = cache_key("llm", _PRICED_MODEL, "pv", "h0", 0)
    cache.put(key, _make_verdict(judge, items[0].state, 0, None, error="401 User not found"))

    G3Outcome().run(items, [judge], cache, spend, repeats=1)

    assert judge.calls == [("h0", 0)]
    assert spend.spent("g3") == pytest.approx((1000 * 0.15 + 100 * 0.60) / 1_000_000)


# --- stop a judge after five identical failures ----------------------------


def test_judge_that_fails_first_five_calls_stops_while_others_continue(tmp_path: Path) -> None:
    truth = {f"h{i}": "pass" for i in range(8)}
    items = [_item(h, "pass") for h in truth]
    good = FakeJudge("good", "none", "pv", _perfect(truth), paid=False)
    bad = FakeJudge("jev", "none", "pv", lambda h, r: (None, "401 User not found."), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    gate = G3Outcome()

    verdicts = gate.run(items, [good, bad], cache, spend, repeats=1)

    assert len(good.calls) == 8
    assert len(bad.calls) == 5
    assert len(verdicts) == 13
    assert "jev" in gate.last_stopped
    assert "401 User not found." in gate.last_stopped["jev"]

    ordered = sorted(truth)
    skipped = [cache_key("jev", "none", "pv", h, 0) for h in ordered[5:]]
    assert all(cache.peek(key, Verdict) is None for key in skipped)


def test_varying_errors_do_not_stop_the_judge(tmp_path: Path) -> None:
    truth = {f"h{i}": "pass" for i in range(6)}
    items = [_item(h, "pass") for h in truth]
    bad = FakeJudge("jev", "none", "pv", lambda h, r: (None, f"error at {h}"), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    gate = G3Outcome()

    gate.run(items, [bad], cache, spend, repeats=1)

    assert len(bad.calls) == 6
    assert gate.last_stopped == {}


def test_last_stopped_is_empty_on_a_clean_run(tmp_path: Path) -> None:
    truth = {"h0": "pass", "h1": "fail"}
    items = [_item(h, label) for h, label in truth.items()]
    judge = FakeJudge("a", "none", "pv", _perfect(truth), paid=False)
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})
    gate = G3Outcome()

    gate.run(items, [judge], cache, spend, repeats=1)

    assert gate.last_stopped == {}


# --- questions / prompt version -------------------------------------------


def test_g3_questions_ids_kinds_and_options() -> None:
    questions = G3Outcome().questions()
    by_id = {q.id: q for q in questions}

    assert set(by_id) == {"completed", "verdict"}
    assert by_id["completed"].kind is QuestionKind.noul
    assert (
        by_id["completed"].text == "The agent completed the user request in line with the policy."
    )
    assert by_id["verdict"].kind is QuestionKind.choice
    assert by_id["verdict"].options == ["pass", "fail"]
    assert by_id["verdict"].text.strip() != ""


def test_prompt_version_stable_across_instances_and_changes_with_rubric(tmp_path: Path) -> None:
    assert G3Outcome().prompt_version == G3Outcome().prompt_version

    modified = tmp_path / "rubric.md"
    modified.write_text("# version: 1\n\nA different rubric body entirely.\n", encoding="utf-8")
    changed = G3Outcome(rubric_path=modified)

    assert changed.prompt_version != G3Outcome().prompt_version


# --- build_items -----------------------------------------------------------


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


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, instruction="do the thing", actions=[], outputs=[])


def test_build_items_skips_excluded_and_maps_truth() -> None:
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
        "retail-2": _record("retail-2", 1.0, excluded=True),
    }
    tasks = {tid: _task(tid) for tid in records}

    items = G3Outcome().build_items(records, tasks, StateProfile.compact)

    assert len(items) == 2
    by_task = {item.state.task_id: item for item in items}
    assert by_task["retail-0"].truth_label == "pass"
    assert by_task["retail-0"].truth_value == 1.0
    assert by_task["retail-1"].truth_label == "fail"
    assert by_task["retail-1"].truth_value == 0.0


# --- analyze ---------------------------------------------------------------


def test_analyze_computes_metrics_and_builds_outputs() -> None:
    hashes = ["h0", "h1", "h2", "h3"]
    truths = ["pass", "pass", "fail", "fail"]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]

    perfect = FakeJudge("perfect", "none", "pv", lambda h, r: ("", None))
    noisy = FakeJudge("noisy", "none", "pv", lambda h, r: ("", None))
    noisy_map = {
        ("h0", 0): ("pass", None),
        ("h0", 1): ("pass", None),
        ("h1", 0): ("fail", None),
        ("h1", 1): ("pass", None),
        ("h2", 0): (None, "boom"),
        ("h2", 1): ("fail", None),
        ("h3", 0): ("pass", None),
        ("h3", 1): ("pass", None),
    }

    verdicts: list[Verdict] = []
    for state_hash, truth in zip(hashes, truths, strict=True):
        state = _state(state_hash)
        for repeat in (0, 1):
            verdicts.append(_make_verdict(perfect, state, repeat, truth))
        for repeat in (0, 1):
            choice, error = noisy_map[(state_hash, repeat)]
            verdicts.append(_make_verdict(noisy, state, repeat, choice, error=error))

    result = G3Outcome().analyze(verdicts, items)

    assert isinstance(result, GateResult)
    df = result.tables["g3_summary"]
    assert len(df) == 2

    perfect_row = df[df["judge_id"] == "perfect"].iloc[0]
    assert perfect_row["accuracy"] == pytest.approx(1.0)
    assert perfect_row["kappa"] == pytest.approx(1.0)
    assert perfect_row["f1_fail"] == pytest.approx(1.0)
    assert perfect_row["modal_agreement"] == pytest.approx(1.0)
    assert perfect_row["error_rate"] == pytest.approx(0.0)

    noisy_row = df[df["judge_id"] == "noisy"].iloc[0]
    assert noisy_row["accuracy"] == pytest.approx(0.5)
    assert noisy_row["kappa"] == pytest.approx(0.0)
    assert noisy_row["f1_fail"] == pytest.approx(0.5)
    assert noisy_row["modal_agreement"] == pytest.approx(0.875)
    assert noisy_row["error_rate"] == pytest.approx(0.125)

    assert "perfect" in result.findings
    assert isinstance(result.charts["g3_accuracy"], Figure)
