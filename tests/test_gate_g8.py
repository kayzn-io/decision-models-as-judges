"""Tests for the G8 regression detection analysis gate."""

from collections.abc import Mapping, Sequence

import pytest
from matplotlib.figure import Figure

from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g8_regression import (
    G8Regression,
    estimate_delta,
    false_alarm,
    pass_labels,
    true_delta,
)
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Verdict

_PASS = "pass"
_FAIL = "fail"


# --- builders --------------------------------------------------------------


def _state(variant: str, task_id: str) -> StateRecord:
    """Build a minimal serialized state keyed by variant and task."""
    return StateRecord(
        variant=variant,
        task_id=task_id,
        profile=StateProfile.compact,
        text=f"{variant}:{task_id}",
        token_estimate=10,
        state_hash=f"{variant}-{task_id}",
    )


def _item(variant: str, task_id: str, truth_label: str) -> Item:
    """Build an item pairing a state with its ground-truth pass/fail label."""
    return Item(
        state=_state(variant, task_id),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == _PASS else 0.0,
    )


def _verdict(
    judge_id: str,
    state_hash: str,
    repeat: int,
    choice: str | None,
    *,
    error: str | None = None,
) -> Verdict:
    """Build a verdict answering the verdict choice question, or an error verdict."""
    answers: list[Answer] = []
    if error is None and choice is not None:
        passed = choice == _PASS
        answers = [
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice=choice,
                probabilities={"pass": 1.0, "fail": 0.0} if passed else {"pass": 0.0, "fail": 1.0},
                confidence=1.0,
            )
        ]
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


# --- pass_labels -----------------------------------------------------------


def test_pass_labels_takes_modal_choice_and_sorts_by_task() -> None:
    by_task: Mapping[str, Sequence[Verdict]] = {
        "b": [
            _verdict("j", "hb", 0, _PASS),
            _verdict("j", "hb", 1, _PASS),
            _verdict("j", "hb", 2, _PASS),
            _verdict("j", "hb", 3, _FAIL),
        ],
        "a": [
            _verdict("j", "ha", 0, _FAIL),
            _verdict("j", "ha", 1, _FAIL),
            _verdict("j", "ha", 2, _PASS),
            _verdict("j", "ha", 3, _PASS),
        ],
    }

    # a ties 2-2 and resolves to fail; b is modally pass. Sorted by task: a, b.
    assert pass_labels(by_task, None) == [0, 1]


def test_pass_labels_honors_repeat_subsets() -> None:
    by_task: Mapping[str, Sequence[Verdict]] = {
        "a": [
            _verdict("j", "ha", 0, _FAIL),
            _verdict("j", "ha", 1, _FAIL),
            _verdict("j", "ha", 2, _PASS),
            _verdict("j", "ha", 3, _PASS),
        ],
        "b": [
            _verdict("j", "hb", 0, _PASS),
            _verdict("j", "hb", 1, _PASS),
            _verdict("j", "hb", 2, _PASS),
            _verdict("j", "hb", 3, _FAIL),
        ],
    }

    assert pass_labels(by_task, {0, 1}) == [0, 1]
    assert pass_labels(by_task, {2, 3}) == [1, 0]


def test_pass_labels_skips_error_verdicts() -> None:
    by_task: Mapping[str, Sequence[Verdict]] = {
        "a": [
            _verdict("j", "ha", 0, None, error="boom"),
            _verdict("j", "ha", 1, _PASS),
            _verdict("j", "ha", 2, _PASS),
        ],
    }

    assert pass_labels(by_task, None) == [1]


# --- true_delta ------------------------------------------------------------


def test_true_delta_is_degraded_minus_baseline_pass_rate() -> None:
    items = [
        _item("baseline", "t0", _PASS),
        _item("baseline", "t1", _PASS),
        _item("degraded", "t0", _FAIL),
        _item("degraded", "t1", _PASS),
    ]

    assert true_delta(items) == pytest.approx(-0.5)


# --- estimate_delta --------------------------------------------------------


def test_estimate_delta_contains_true_difference_and_excludes_zero_on_a_big_gap() -> None:
    base = [1] * 40 + [0] * 60
    degraded = [1] * 90 + [0] * 10

    point, lo, hi = estimate_delta(base, degraded, 2000, 7)

    assert point == pytest.approx(0.5, abs=0.02)
    assert lo <= 0.5 <= hi
    assert lo > 0.0


# --- false_alarm -----------------------------------------------------------


def test_false_alarm_false_for_identical_label_sets() -> None:
    labels = [1, 0, 1, 0, 1]

    assert false_alarm(labels, labels, 2000, 7) is False


def test_false_alarm_true_when_interval_excludes_zero() -> None:
    assert false_alarm([0] * 20, [1] * 20, 2000, 7) is True


# --- analyze ---------------------------------------------------------------

_TASKS = ("t0", "t1", "t2", "t3")
_DEGRADED_TRUTH = {"t0": _PASS, "t1": _FAIL, "t2": _FAIL, "t3": _FAIL}
_NOISY = {"t0": _PASS, "t1": _PASS, "t2": _FAIL, "t3": _FAIL}


def _both_variant_items() -> list[Item]:
    """Baseline passes every task; the degraded variant regresses on three."""
    items = [_item("baseline", task, _PASS) for task in _TASKS]
    items += [_item("degraded", task, _DEGRADED_TRUTH[task]) for task in _TASKS]
    return items


def _analysis_verdicts() -> list[Verdict]:
    """Build four repeats per item for an accurate judge and a noisy judge."""
    verdicts: list[Verdict] = []
    for task in _TASKS:
        base_hash = f"baseline-{task}"
        deg_hash = f"degraded-{task}"
        for repeat in range(4):
            # accurate judge tracks the truth on both variants
            verdicts.append(_verdict("acc", base_hash, repeat, _PASS))
            verdicts.append(_verdict("acc", deg_hash, repeat, _DEGRADED_TRUTH[task]))
            # noisy judge reports the same pass rate on both variants
            verdicts.append(_verdict("noisy", base_hash, repeat, _NOISY[task]))
            verdicts.append(_verdict("noisy", deg_hash, repeat, _NOISY[task]))
    return verdicts


def test_analyze_detects_regression_for_accurate_judge_only() -> None:
    result = G8Regression().analyze(_analysis_verdicts(), _both_variant_items())

    assert isinstance(result, GateResult)
    frame = result.tables["g8_summary"]
    assert len(frame) == 2
    assert set(frame.columns) == {
        "judge",
        "true_delta",
        "est_delta",
        "lo",
        "hi",
        "detected",
        "covers_truth",
        "false_alarm",
    }
    assert frame["true_delta"].iloc[0] == pytest.approx(-0.75)

    acc = frame[frame["judge"] == "acc"].iloc[0]
    assert bool(acc["detected"]) is True
    assert bool(acc["covers_truth"]) is True
    assert bool(acc["false_alarm"]) is False

    noisy = frame[frame["judge"] == "noisy"].iloc[0]
    assert bool(noisy["detected"]) is False

    assert "acc" in result.findings
    assert isinstance(result.charts["g8_intervals"], Figure)


def test_analyze_reports_missing_variant_and_empty_tables() -> None:
    items = [_item("baseline", task, _PASS) for task in _TASKS]

    result = G8Regression().analyze(_analysis_verdicts(), items)

    assert result.tables["g8_summary"].empty
    assert "baseline" in result.findings and "degraded" in result.findings
    assert isinstance(result.charts["g8_intervals"], Figure)


# --- delegation and run ----------------------------------------------------


def test_delegates_metadata_to_g3() -> None:
    g3 = G3Outcome()
    gate = G8Regression(g3)

    assert gate.questions() == g3.questions()
    assert gate.prompt_version == g3.prompt_version
    assert gate.rubric_path == g3.rubric_path
    assert gate.gate_id == "g8"
    assert gate.stage == "g8"


def test_run_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        G8Regression().run([], [], None, None, repeats=1)  # type: ignore[arg-type]
