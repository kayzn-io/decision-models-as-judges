"""Tests for the spend tracker with per-stage caps and a persisted ledger."""

from datetime import date
from pathlib import Path

import pytest

from decision_judges.config import PricingTable, UnpricedModel
from decision_judges.spend import Reservation, Spend, SpendCapExceeded
from decision_judges.types import Usage


def _pricing() -> PricingTable:
    """Build a small pricing table where 100k input tokens on model m cost 1 USD."""
    return PricingTable(
        effective_date=date(2026, 9, 24),
        source="test",
        models={
            "m": {"input_per_mtok": 10.0, "output_per_mtok": 20.0},
            "n": {"input_per_mtok": 5.0, "output_per_mtok": 0.0},
        },
    )


def _spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    return Spend(_pricing(), caps, tmp_path / "ledger.json")


def test_reserve_within_cap_then_settle_records_actual(tmp_path: Path) -> None:
    spend = _spend(tmp_path, {"agent": 5.0})
    reservation = spend.reserve("agent", "m", est_input_tokens=100_000)
    assert isinstance(reservation, Reservation)
    assert reservation.estimated_usd == pytest.approx(1.0)
    assert reservation.settled is False
    assert spend.reserved("agent") == pytest.approx(1.0)

    actual = spend.settle(reservation, Usage(input_tokens=50_000))
    assert actual == pytest.approx(0.5)
    assert reservation.settled is True
    assert spend.spent("agent") == pytest.approx(0.5)
    assert spend.total() == pytest.approx(0.5)
    assert spend.reserved("agent") == pytest.approx(0.0)


def test_reserve_exceeding_cap_raises_before_any_call(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    spend = Spend(_pricing(), {"agent": 5.0}, ledger_path)
    with pytest.raises(SpendCapExceeded) as exc:
        spend.reserve("agent", "m", est_input_tokens=600_000)
    message = str(exc.value)
    assert "agent" in message
    assert spend.spent("agent") == pytest.approx(0.0)
    assert not ledger_path.exists()


def test_unconfigured_stage_refuses(tmp_path: Path) -> None:
    spend = _spend(tmp_path, {"agent": 5.0})
    with pytest.raises(SpendCapExceeded) as exc:
        spend.reserve("ghost", "m", est_input_tokens=1)
    assert "ghost" in str(exc.value)


def test_outstanding_reservations_count_against_cap(tmp_path: Path) -> None:
    spend = _spend(tmp_path, {"agent": 5.0})
    first = spend.reserve("agent", "m", est_input_tokens=300_000)
    with pytest.raises(SpendCapExceeded):
        spend.reserve("agent", "m", est_input_tokens=300_000)

    spend.cancel(first)
    assert spend.reserved("agent") == pytest.approx(0.0)
    again = spend.reserve("agent", "m", est_input_tokens=300_000)
    assert again.estimated_usd == pytest.approx(3.0)


def test_ledger_persists_and_reloads(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    spend = Spend(_pricing(), {"agent": 5.0, "user": 5.0}, ledger_path)
    r1 = spend.reserve("agent", "m", est_input_tokens=100_000)
    spend.settle(r1, Usage(input_tokens=100_000))
    r2 = spend.reserve("user", "n", est_input_tokens=100_000)
    spend.settle(r2, Usage(input_tokens=100_000))
    assert ledger_path.exists()

    reloaded = Spend(_pricing(), {"agent": 5.0, "user": 5.0}, ledger_path)
    assert reloaded.spent("agent") == pytest.approx(1.0)
    assert reloaded.spent("user") == pytest.approx(0.5)
    assert reloaded.total() == pytest.approx(1.5)


def test_by_model_totals(tmp_path: Path) -> None:
    spend = _spend(tmp_path, {"agent": 100.0})
    r1 = spend.reserve("agent", "m", est_input_tokens=100_000)
    spend.settle(r1, Usage(input_tokens=100_000))
    r2 = spend.reserve("agent", "n", est_input_tokens=100_000)
    spend.settle(r2, Usage(input_tokens=200_000))
    totals = spend.by_model()
    assert totals["m"] == pytest.approx(1.0)
    assert totals["n"] == pytest.approx(1.0)


def test_unpriced_model_raises_from_reserve(tmp_path: Path) -> None:
    spend = _spend(tmp_path, {"agent": 5.0})
    with pytest.raises(UnpricedModel):
        spend.reserve("agent", "no-such-model", est_input_tokens=1)
