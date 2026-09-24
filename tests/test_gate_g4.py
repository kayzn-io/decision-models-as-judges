"""Tests for the G4 decomposition gate and its aggregators."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g4_decomposition import (
    BroadAggregator,
    G4Decomposition,
    LogRegCVAggregator,
    MeanAggregator,
    MinAggregator,
)
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Verdict

_ATOMIC = [
    "all_actions",
    "no_unrequested_writes",
    "confirmed_writes",
    "identity_verified",
    "final_states_outcome",
    "no_wrong_refusal",
]
_ALL = ["completed", *_ATOMIC]


# --- builders --------------------------------------------------------------


def _state(state_hash: str) -> StateRecord:
    """Build a minimal full-profile state with a chosen hash."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.full,
        text="transcript",
        token_estimate=100,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str) -> Item:
    """Pair a state with its ground-truth label and value."""
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _verdict(
    judge_id: str,
    state_hash: str,
    repeat: int,
    nouls: dict[str, float],
    *,
    error: str | None = None,
    latency_ms: int = 1,
) -> Verdict:
    """Build a verdict carrying one noul answer per supplied question id."""
    answers: list[Answer] = []
    if error is None:
        answers = [
            Answer(question_id=qid, kind=QuestionKind.noul, noul=value)
            for qid, value in nouls.items()
        ]
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=answers,
        latency_ms=latency_ms,
        error=error,
    )


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


def _informative_fixture() -> tuple[list[Verdict], list[Item]]:
    """Atomic nouls separate pass from fail while the broad noul is noisy."""
    hashes = ["h0", "h1", "h2", "h3"]
    truths = ["pass", "pass", "fail", "fail"]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]
    completed = {"h0": 0.4, "h1": 0.6, "h2": 0.6, "h3": 0.4}
    verdicts: list[Verdict] = []
    for h, t in zip(hashes, truths, strict=True):
        base = 0.9 if t == "pass" else 0.1
        nouls = {qid: base for qid in _ATOMIC}
        nouls["completed"] = completed[h]
        verdicts.append(_verdict("judge", h, 0, nouls))
    return verdicts, items


# --- questions / prompt version -------------------------------------------


def test_questions_ids_texts_and_kinds() -> None:
    questions = G4Decomposition().questions()
    by_id = {q.id: q for q in questions}
    expected = {
        "completed": "The agent completed the user request in line with the policy.",
        "all_actions": "Every action the user requested was performed.",
        "no_unrequested_writes": (
            "The agent made no order-changing action the user did not request."
        ),
        "confirmed_writes": (
            "The agent obtained explicit confirmation before each order-changing action."
        ),
        "identity_verified": (
            "The agent verified the user identity before accessing account details."
        ),
        "final_states_outcome": "The final message states the outcome of the request.",
        "no_wrong_refusal": "The agent did not refuse a request the policy permits.",
    }

    assert [q.id for q in questions] == list(expected)
    assert set(by_id) == set(expected)
    for qid, text in expected.items():
        assert by_id[qid].text == text
        assert by_id[qid].kind is QuestionKind.noul


def test_prompt_version_stable_and_changes_with_rubric(tmp_path: Path) -> None:
    assert G4Decomposition().prompt_version == G4Decomposition().prompt_version

    modified = tmp_path / "rubric.md"
    modified.write_text("# version: 2\n\nA different rubric body.\n", encoding="utf-8")
    changed = G4Decomposition(rubric_path=modified)

    assert changed.rubric_path == modified
    assert changed.prompt_version != G4Decomposition().prompt_version


# --- build_items -----------------------------------------------------------


def test_build_items_equals_g3_items() -> None:
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
    }
    tasks = {tid: _task(tid) for tid in records}

    g4_items = G4Decomposition().build_items(records, tasks, StateProfile.full)
    g3_items = G3Outcome().build_items(records, tasks, StateProfile.full)

    assert [i.state.state_hash for i in g4_items] == [i.state.state_hash for i in g3_items]
    assert [i.truth_label for i in g4_items] == [i.truth_label for i in g3_items]
    assert [i.truth_value for i in g4_items] == [i.truth_value for i in g3_items]


# --- aggregators -----------------------------------------------------------


def test_min_mean_and_broad_aggregators_exact() -> None:
    rows = [
        {
            "completed": 0.5,
            "all_actions": 0.2,
            "no_unrequested_writes": 0.4,
            "confirmed_writes": 0.6,
            "identity_verified": 0.8,
            "final_states_outcome": 1.0,
            "no_wrong_refusal": 0.3,
        },
        {qid: 0.9 for qid in _ALL} | {"completed": 0.1},
    ]
    truth = [1.0, 0.0]

    assert MinAggregator().scores(rows, truth) == pytest.approx([0.2, 0.9])
    assert MeanAggregator().scores(rows, truth) == pytest.approx([3.3 / 6, 0.9])
    assert BroadAggregator().scores(rows, truth) == pytest.approx([0.5, 0.1])


def test_logreg_cv_never_trains_on_the_scored_item() -> None:
    rows: list[dict[str, float]] = []
    truth: list[float] = []
    for i in range(6):
        base = 0.8 if i < 3 else 0.2
        rows.append({qid: base for qid in _ALL})
        truth.append(1.0 if i < 3 else 0.0)

    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class Spy(LogRegCVAggregator):
        def _fit_predict(
            self,
            features: object,
            labels: object,
            train_idx: list[int],
            test_idx: list[int],
        ) -> object:
            seen.append((tuple(train_idx), tuple(test_idx)))
            return super()._fit_predict(features, labels, train_idx, test_idx)

    scores = Spy(k=3, seed=0).scores(rows, truth)

    assert len(scores) == 6
    covered: set[int] = set()
    for train_idx, test_idx in seen:
        assert set(train_idx).isdisjoint(set(test_idx))
        covered |= set(test_idx)
    assert covered == set(range(6))
    assert all(0.0 <= score <= 1.0 for score in scores)


# --- analyze ---------------------------------------------------------------


def test_analyze_atomic_aggregation_beats_broad() -> None:
    verdicts, items = _informative_fixture()

    frame = G4Decomposition().analyze(verdicts, items).tables["g4_summary"]
    by_agg = {row["aggregator"]: row for _, row in frame.iterrows()}

    assert by_agg["broad"]["auroc"] == pytest.approx(0.5)
    assert by_agg["mean"]["auroc"] == pytest.approx(1.0)
    assert by_agg["min"]["auroc"] > by_agg["broad"]["auroc"]
    assert by_agg["mean"]["auroc"] > by_agg["broad"]["auroc"]


def test_analyze_table_shape_and_columns() -> None:
    verdicts, items = _informative_fixture()

    frame = G4Decomposition().analyze(verdicts, items).tables["g4_summary"]

    assert len(frame) == 4
    assert set(frame.columns) == {"judge_id", "aggregator", "auroc", "accuracy", "n", "dropped"}
    assert set(frame["aggregator"]) == {"broad", "min", "mean", "logreg_cv"}
    assert (frame["n"] == 4).all()
    assert (frame["dropped"] == 0).all()


def test_analyze_drops_item_missing_an_atomic_answer() -> None:
    hashes = ["h0", "h1", "h2", "h3"]
    truths = ["pass", "pass", "fail", "fail"]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]
    verdicts: list[Verdict] = []
    for h, t in zip(hashes, truths, strict=True):
        base = 0.9 if t == "pass" else 0.1
        nouls = {qid: base for qid in _ATOMIC}
        nouls["completed"] = 0.5
        if h == "h1":
            del nouls["confirmed_writes"]
        verdicts.append(_verdict("judge", h, 0, nouls))

    frame = G4Decomposition().analyze(verdicts, items).tables["g4_summary"]
    row = frame[frame["aggregator"] == "mean"].iloc[0]

    assert row["dropped"] == 1
    assert row["n"] == 3


def test_analyze_averages_repeats_per_question() -> None:
    items = [_item("h0", "pass")]
    nouls_a = {qid: 0.8 for qid in _ALL}
    nouls_b = {qid: 0.4 for qid in _ALL}
    verdicts = [_verdict("judge", "h0", 0, nouls_a), _verdict("judge", "h0", 1, nouls_b)]

    frame = G4Decomposition().analyze(verdicts, items).tables["g4_summary"]
    row = frame[frame["aggregator"] == "mean"].iloc[0]

    assert row["n"] == 1
    # A single pass item cannot form both AUROC classes.
    assert row["accuracy"] == pytest.approx(1.0)


def test_analyze_chart_present_and_findings_name_winner() -> None:
    verdicts, items = _informative_fixture()

    result = G4Decomposition().analyze(verdicts, items)

    assert isinstance(result, GateResult)
    assert isinstance(result.charts["g4_auroc_by_aggregator"], Figure)
    assert any(name in result.findings for name in ("min", "mean", "logreg_cv"))


def test_analyze_empty_returns_placeholder_findings() -> None:
    result = G4Decomposition().analyze([], [])

    assert result.tables["g4_summary"].empty
    assert isinstance(result.charts["g4_auroc_by_aggregator"], Figure)
    assert result.findings
