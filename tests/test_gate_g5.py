"""Tests for the G5 confidence-gated cascade analysis gate."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.config import load_pricing
from decision_judges.gates.base import Item
from decision_judges.gates.g5_cascade import (
    CascadeDecision,
    G5Cascade,
    ItemVerdict,
    cascade,
    cost_of,
    modal_verdict,
    three_tier,
)
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Usage, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"

_FAST_MODEL = "jev-1.13.0"
_SLOW_MODEL = "openai/gpt-5"
_CHEAP_MODEL = "openai/gpt-4o-mini"


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
    """Build an item pairing a state with its ground-truth label."""
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _verdict(
    judge_id: str,
    model_id: str,
    state_hash: str,
    choice: str,
    confidence: float,
    *,
    usage: Usage | None = None,
    repeat: int = 0,
) -> Verdict:
    """Build a choice verdict for one judge, item, and repeat."""
    probabilities = {"pass": 1.0, "fail": 0.0} if choice == "pass" else {"pass": 0.0, "fail": 1.0}
    answer = Answer(
        question_id="verdict",
        kind=QuestionKind.choice,
        choice=choice,
        probabilities=probabilities,
        confidence=confidence,
    )
    return Verdict(
        judge_id=judge_id,
        model_id=model_id,
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=[answer],
        usage=usage if usage is not None else Usage(),
        latency_ms=1,
    )


# --- modal_verdict ---------------------------------------------------------


def test_modal_verdict_returns_mode_and_mean_confidence() -> None:
    verdicts = [
        _verdict("jev", _FAST_MODEL, "h0", "pass", 0.8, repeat=0),
        _verdict("jev", _FAST_MODEL, "h0", "pass", 0.6, repeat=1),
        _verdict("jev", _FAST_MODEL, "h0", "fail", 0.4, repeat=2),
    ]

    label, confidence = modal_verdict(verdicts)

    assert label == "pass"
    assert confidence == pytest.approx((0.8 + 0.6 + 0.4) / 3)


# --- cost_of ---------------------------------------------------------------


def test_cost_of_averages_priced_usage_over_repeats() -> None:
    pricing = load_pricing(PRICING_FILE)
    usage = Usage(input_tokens=1000, output_tokens=100)
    verdicts = [
        _verdict("cheap", _CHEAP_MODEL, "h0", "pass", 1.0, usage=usage, repeat=0),
        _verdict("cheap", _CHEAP_MODEL, "h0", "pass", 1.0, usage=usage, repeat=1),
    ]

    expected = (1000 * 0.15 + 100 * 0.60) / 1_000_000
    assert cost_of(verdicts, pricing) == pytest.approx(expected)


def test_cost_of_treats_unpriced_none_model_as_free() -> None:
    pricing = load_pricing(PRICING_FILE)
    usage = Usage(input_tokens=1000, output_tokens=100)
    verdicts = [_verdict("code", "none", "h0", "pass", 1.0, usage=usage)]

    assert cost_of(verdicts, pricing) == pytest.approx(0.0)


# --- cascade ---------------------------------------------------------------


def test_cascade_keeps_fast_at_or_above_threshold() -> None:
    fast = ItemVerdict(label="pass", confidence=0.9, cost=0.001)
    slow = ItemVerdict(label="fail", confidence=1.0, cost=0.01)

    decision = cascade(fast, slow, 0.8)

    assert decision == CascadeDecision(label="pass", cost=0.001, escalated=False)


def test_cascade_escalates_below_threshold_and_sums_cost() -> None:
    fast = ItemVerdict(label="pass", confidence=0.7, cost=0.001)
    slow = ItemVerdict(label="fail", confidence=1.0, cost=0.01)

    decision = cascade(fast, slow, 0.9)

    assert decision.label == "fail"
    assert decision.escalated is True
    assert decision.cost == pytest.approx(0.011)


# --- three_tier ------------------------------------------------------------


def test_three_tier_keeps_fast_at_top_tier() -> None:
    fast = ItemVerdict(label="pass", confidence=0.95, cost=0.001)
    slow = ItemVerdict(label="fail", confidence=1.0, cost=0.01)

    decision = three_tier(fast, slow, 0.9, 0.6, "fail", 2.0)

    assert decision == CascadeDecision(label="pass", cost=0.001, escalated=False)


def test_three_tier_escalates_middle_tier_to_slow() -> None:
    fast = ItemVerdict(label="pass", confidence=0.7, cost=0.001)
    slow = ItemVerdict(label="fail", confidence=1.0, cost=0.01)

    decision = three_tier(fast, slow, 0.9, 0.6, "fail", 2.0)

    assert decision.label == "fail"
    assert decision.escalated is True
    assert decision.cost == pytest.approx(0.011)


def test_three_tier_routes_to_human_below_low_threshold() -> None:
    fast = ItemVerdict(label="pass", confidence=0.4, cost=0.001)
    slow = ItemVerdict(label="fail", confidence=1.0, cost=0.01)

    decision = three_tier(fast, slow, 0.9, 0.6, "pass", 2.0)

    assert decision.label == "pass"
    assert decision.escalated is True
    assert decision.cost == pytest.approx(2.001)


# --- analyze fixture -------------------------------------------------------

_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

# Per item: (truth, fast choice, fast confidence). Fast is confident when right.
_FIXTURE = {
    "h0": ("pass", "pass", 0.97),
    "h1": ("fail", "fail", 0.92),
    "h2": ("pass", "fail", 0.75),
    "h3": ("fail", "pass", 0.55),
}


def _analyze_inputs() -> tuple[list[Verdict], list[Item]]:
    """Build covered fast/slow verdicts, one uncovered item, and their items."""
    fast_usage = Usage(input_tokens=1000, output_tokens=0)
    slow_usage = Usage(input_tokens=1000, output_tokens=100)
    verdicts: list[Verdict] = []
    items: list[Item] = []
    for state_hash, (truth, fast_choice, confidence) in _FIXTURE.items():
        items.append(_item(state_hash, truth))
        verdicts.append(
            _verdict("jev", _FAST_MODEL, state_hash, fast_choice, confidence, usage=fast_usage)
        )
        verdicts.append(
            _verdict("llm_strong", _SLOW_MODEL, state_hash, truth, 0.99, usage=slow_usage)
        )
    # An uncovered item: fast judge only, no slow verdict.
    items.append(_item("h4", "pass"))
    verdicts.append(_verdict("jev", _FAST_MODEL, "h4", "pass", 0.80, usage=fast_usage))
    return verdicts, items


def _gate() -> G5Cascade:
    return G5Cascade(load_pricing(PRICING_FILE), _THRESHOLDS)


def test_analyze_frontier_has_one_row_per_threshold() -> None:
    verdicts, items = _analyze_inputs()

    result = _gate().analyze(verdicts, items)

    frontier = result.tables["g5_frontier"]
    assert len(frontier) == len(_THRESHOLDS)
    assert (frontier["n"] == 4).all()


def test_analyze_accuracy_rises_and_cost_falls_with_threshold() -> None:
    verdicts, items = _analyze_inputs()

    frontier = _gate().analyze(verdicts, items).tables["g5_frontier"]
    ordered = frontier.sort_values("t").reset_index(drop=True)

    low = ordered.iloc[0]
    high = ordered.iloc[-1]
    assert low["t"] == pytest.approx(0.50)
    assert high["t"] == pytest.approx(0.99)
    assert low["accuracy"] <= high["accuracy"]
    assert high["cost_per_item"] > low["cost_per_item"]
    assert list(ordered["cost_per_item"]) == sorted(ordered["cost_per_item"])


def test_analyze_reference_table_has_fast_and_slow_rows() -> None:
    verdicts, items = _analyze_inputs()

    reference = _gate().analyze(verdicts, items).tables["g5_reference"]
    judges = set(reference["judge"])

    assert {"jev", "llm_strong"} <= judges
    fast_row = reference[reference["judge"] == "jev"].iloc[0]
    slow_row = reference[reference["judge"] == "llm_strong"].iloc[0]
    assert fast_row["accuracy"] == pytest.approx(0.5)
    assert slow_row["accuracy"] == pytest.approx(1.0)
    assert slow_row["cost_per_item"] > fast_row["cost_per_item"]


def test_analyze_counts_uncovered_items() -> None:
    verdicts, items = _analyze_inputs()

    coverage = _gate().analyze(verdicts, items).tables["g5_coverage"]
    row = coverage.iloc[0]

    assert row["n_covered"] == 4
    assert row["n_uncovered"] == 1


def test_analyze_three_tier_table_and_chart_present() -> None:
    verdicts, items = _analyze_inputs()

    result = _gate().analyze(verdicts, items)

    three = result.tables["g5_three_tier"]
    assert len(three) == len(_THRESHOLDS)
    assert (three["t_lo"] >= 0.5).all()
    assert isinstance(result.charts["g5_frontier"], Figure)
    assert result.findings.strip() != ""


# --- delegation and run ----------------------------------------------------


def test_delegates_questions_and_prompt_version_to_g3() -> None:
    gate = _gate()

    assert [q.id for q in gate.questions()] == ["completed", "verdict"]
    assert gate.prompt_version != ""
    assert gate.rubric_path.name == "g3_outcome.md"


def test_run_raises_because_gate_makes_no_calls() -> None:
    gate = _gate()

    with pytest.raises(NotImplementedError):
        gate.run([], [], None, None, repeats=1)  # type: ignore[arg-type]
