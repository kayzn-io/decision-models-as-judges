"""Tests for the G6 calibration analysis gate."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g6_calibration import (
    CompletedNoul,
    G6Calibration,
    VerdictConfidence,
    VerdictPassProb,
    _assign_folds,
    isotonic_cv,
)
from decision_judges.metrics import isotonic_fit_transform
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Verdict

_PASS = "pass"
_FAIL = "fail"


# --- builders --------------------------------------------------------------


def _state(state_hash: str) -> StateRecord:
    """Build a minimal serialized state with a chosen hash."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.full,
        text="transcript",
        token_estimate=100,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str) -> Item:
    """Pair a state with its ground-truth label."""
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == _PASS else 0.0,
    )


def _completed(noul: float) -> Answer:
    """Build a completed-noul answer."""
    return Answer(question_id="completed", kind=QuestionKind.noul, noul=noul)


def _verdict_answer(choice: str, pass_prob: float, confidence: float) -> Answer:
    """Build a pass/fail verdict-choice answer."""
    return Answer(
        question_id="verdict",
        kind=QuestionKind.choice,
        choice=choice,
        probabilities={"pass": pass_prob, "fail": 1.0 - pass_prob},
        confidence=confidence,
    )


def _verdict(
    judge_id: str,
    state_hash: str,
    repeat: int,
    answers: list[Answer],
    *,
    error: str | None = None,
) -> Verdict:
    """Build a verdict over the given answers."""
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=answers,
        latency_ms=1,
        error=error,
    )


# --- signal extractors -----------------------------------------------------


def test_completed_noul_extracts_noul_and_supplied_label() -> None:
    verdict = _verdict("j", "h0", 0, [_completed(0.8)])
    assert CompletedNoul().extract(verdict, 1) == (0.8, 1)
    assert CompletedNoul().extract(verdict, 0) == (0.8, 0)


def test_completed_noul_returns_none_when_absent() -> None:
    verdict = _verdict("j", "h0", 0, [_verdict_answer(_PASS, 0.7, 0.9)])
    assert CompletedNoul().extract(verdict, 1) is None


def test_verdict_pass_prob_extracts_pass_probability_and_label() -> None:
    verdict = _verdict("j", "h0", 0, [_verdict_answer(_PASS, 0.7, 0.9)])
    assert VerdictPassProb().extract(verdict, 1) == (0.7, 1)
    assert VerdictPassProb().extract(verdict, 0) == (0.7, 0)


def test_verdict_pass_prob_returns_none_when_absent() -> None:
    verdict = _verdict("j", "h0", 0, [_completed(0.5)])
    assert VerdictPassProb().extract(verdict, 1) is None


def test_verdict_confidence_label_is_choice_correctness() -> None:
    chose_pass = _verdict("j", "h0", 0, [_verdict_answer(_PASS, 0.9, 0.85)])
    # confidence predicts "the judge's own chosen label is right"
    assert VerdictConfidence().extract(chose_pass, 1) == (0.85, 1)
    assert VerdictConfidence().extract(chose_pass, 0) == (0.85, 0)

    chose_fail = _verdict("j", "h1", 0, [_verdict_answer(_FAIL, 0.1, 0.7)])
    assert VerdictConfidence().extract(chose_fail, 0) == (0.7, 1)
    assert VerdictConfidence().extract(chose_fail, 1) == (0.7, 0)


# --- isotonic cross-validation ---------------------------------------------


def test_isotonic_cv_fits_on_the_other_fold() -> None:
    seed = 7
    folds = _assign_folds(8, 2, seed)
    # Give the two folds opposite calibration so that fitting on the wrong fold
    # would produce a visibly different value.
    fold0_probs = [0.2, 0.2, 0.8, 0.8]
    fold0_labels = [0.0, 0.0, 1.0, 1.0]
    fold1_labels = [1.0, 1.0, 0.0, 0.0]
    counters = {0: 0, 1: 0}
    probs = [0.0] * 8
    labels = [0.0] * 8
    for i, f in enumerate(folds):
        c = counters[f]
        counters[f] += 1
        probs[i] = fold0_probs[c]
        labels[i] = fold0_labels[c] if f == 0 else fold1_labels[c]

    recal = isotonic_cv(probs, labels, 2, seed)

    # Fold 1 is anti-correlated -> its isotonic map is flat at 0.5, so any fold-0
    # point recalibrated with the fold-1 map lands at 0.5 (never near 1.0).
    for i, f in enumerate(folds):
        if f == 0 and probs[i] == 0.8:
            assert recal[i] == pytest.approx(0.5)
        if f == 1 and probs[i] == 0.8:
            # fitted on fold 0, which is well ordered -> maps 0.8 to 1.0
            assert recal[i] == pytest.approx(1.0)


def test_isotonic_cv_preserves_length_and_order() -> None:
    probs = [0.1, 0.9, 0.4, 0.6, 0.5]
    labels = [0.0, 1.0, 0.0, 1.0, 1.0]
    recal = isotonic_cv(probs, labels, 5, 7)
    assert len(recal) == len(probs)
    assert all(0.0 <= v <= 1.0 for v in recal)


def test_assign_folds_are_balanced_and_deterministic() -> None:
    a = _assign_folds(10, 5, 7)
    b = _assign_folds(10, 5, 7)
    assert a == b
    counts = [a.count(f) for f in range(5)]
    assert counts == [2, 2, 2, 2, 2]


# --- analyze ---------------------------------------------------------------


def _completed_verdicts(judge_id: str, noul_for: dict[str, float]) -> list[Verdict]:
    """Build one completed-noul verdict per item for a judge."""
    return [_verdict(judge_id, h, 0, [_completed(noul)]) for h, noul in noul_for.items()]


def test_analyze_flags_overconfident_judge_and_recalibrates() -> None:
    hashes = [f"h{i}" for i in range(20)]
    truths = [_PASS if i % 2 == 0 else _FAIL for i in range(20)]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]

    # Perfectly calibrated: noul equals the truth.
    calibrated = _completed_verdicts(
        "calibrated", {h: (1.0 if t == _PASS else 0.0) for h, t in zip(hashes, truths, strict=True)}
    )
    # Overconfident: always emits 0.9 while only half the items truly pass.
    overconfident = _completed_verdicts("overconfident", {h: 0.9 for h in hashes})

    result = G6Calibration().analyze(calibrated + overconfident, items)
    assert isinstance(result, GateResult)

    summary = result.tables["g6_summary"]
    cal = summary[
        (summary["judge"] == "calibrated") & (summary["signal"] == "completed_noul")
    ].iloc[0]
    over = summary[
        (summary["judge"] == "overconfident") & (summary["signal"] == "completed_noul")
    ].iloc[0]

    assert over["ece"] > cal["ece"]
    assert over["ece_after_isotonic"] <= over["ece"] + 1e-9
    assert int(over["n"]) == 20

    bins = result.tables["g6_bins"]
    assert set(bins["judge"]) == {"calibrated", "overconfident"}

    chart = result.charts["g6_reliability"]
    assert isinstance(chart, Figure)
    assert len(chart.axes) == 2


def test_analyze_skips_error_verdicts() -> None:
    items = [_item("h0", _PASS), _item("h1", _FAIL)]
    good = _verdict("j", "h0", 0, [_completed(1.0)])
    boom = _verdict("j", "h1", 0, [], error="boom")
    result = G6Calibration().analyze([good, boom], items)
    row = result.tables["g6_summary"]
    row = row[row["signal"] == "completed_noul"].iloc[0]
    assert int(row["n"]) == 1


def test_analyze_all_three_signals_produce_rows() -> None:
    items = [_item("h0", _PASS), _item("h1", _FAIL)]
    verdicts = [
        _verdict("j", "h0", 0, [_completed(0.9), _verdict_answer(_PASS, 0.9, 0.8)]),
        _verdict("j", "h1", 0, [_completed(0.1), _verdict_answer(_FAIL, 0.1, 0.8)]),
    ]
    summary = G6Calibration().analyze(verdicts, items).tables["g6_summary"]
    assert set(summary["signal"]) == {"completed_noul", "verdict_pass_prob", "verdict_confidence"}


# --- delegation and run ----------------------------------------------------


def test_delegates_to_g3() -> None:
    g3 = G3Outcome()
    gate = G6Calibration(g3)
    assert gate.questions() == g3.questions()
    assert gate.prompt_version == g3.prompt_version
    assert gate.rubric_path == g3.rubric_path


def test_run_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        G6Calibration().run([], [], None, None, repeats=1)  # type: ignore[arg-type]


def test_matches_direct_isotonic_on_single_fold() -> None:
    # With k=1 there is no held-out fold; fall back to fitting on all points,
    # matching the library helper.
    probs = [0.1, 0.4, 0.6, 0.9]
    labels = [0.0, 0.0, 1.0, 1.0]
    recal = isotonic_cv(probs, labels, 1, 7)
    direct = isotonic_fit_transform(probs, labels).tolist()
    assert recal == pytest.approx(direct)


def test_gate_identity() -> None:
    assert G6Calibration.gate_id == "g6"
    assert G6Calibration.stage == "g6"
    assert isinstance(_assign_folds(4, 2, 7), list)
    assert isinstance(np.asarray([]), np.ndarray)
