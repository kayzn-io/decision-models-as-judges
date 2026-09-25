"""Spend tracking with per-stage caps and a persisted ledger."""

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from decision_judges.config import PricingTable
from decision_judges.types import Usage


class SpendCapExceeded(RuntimeError):
    """Raised when a reservation would push a stage past its spend cap."""

    def __init__(self, stage: str, cap: float | None, spent: float, estimate: float) -> None:
        if cap is None:
            message = (
                f"stage {stage!r} has no configured spend cap; refusing paid call "
                f"(estimate ${estimate:.6f})"
            )
        else:
            message = (
                f"stage {stage!r} would exceed cap ${cap:.6f}: "
                f"spent ${spent:.6f} + estimate ${estimate:.6f}"
            )
        super().__init__(message)
        self.stage = stage
        self.cap = cap
        self.spent = spent
        self.estimate = estimate


class Reservation(BaseModel):
    """A held estimate of spend for a stage awaiting settlement."""

    reservation_id: str = Field(default_factory=lambda: uuid4().hex)
    stage: str
    model_id: str
    estimated_usd: float
    settled: bool = False


class LedgerEntry(BaseModel):
    """A settled spend record for one model call."""

    stage: str
    model_id: str
    input_tokens: int
    output_tokens: int
    usd: float
    timestamp: str


class Ledger(BaseModel):
    """Persisted spend totals and settled entries."""

    per_stage: dict[str, float] = Field(default_factory=dict)
    per_model: dict[str, float] = Field(default_factory=dict)
    entries: list[LedgerEntry] = Field(default_factory=list)

    def record(self, entry: LedgerEntry) -> None:
        """Add a settled entry and fold its cost into the totals."""
        self.entries.append(entry)
        self.per_stage[entry.stage] = self.per_stage.get(entry.stage, 0.0) + entry.usd
        self.per_model[entry.model_id] = self.per_model.get(entry.model_id, 0.0) + entry.usd


class Spend:
    """Enforces per-stage spend caps and persists settled spend to a ledger."""

    def __init__(self, pricing: PricingTable, caps: Mapping[str, float], ledger_path: Path) -> None:
        self._pricing = pricing
        self._caps = dict(caps)
        self._ledger_path = ledger_path
        self._ledger = self._load()
        self._reserved: dict[str, float] = {}
        self._open: set[str] = set()

    def _load(self) -> Ledger:
        """Load the ledger from disk if present, else start empty."""
        if self._ledger_path.exists():
            return Ledger.model_validate_json(self._ledger_path.read_text())
        return Ledger()

    def _persist(self) -> None:
        """Write the ledger atomically via a temp file and os.replace."""
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._ledger_path.with_suffix(self._ledger_path.suffix + ".tmp")
        tmp.write_text(self._ledger.model_dump_json(indent=2))
        os.replace(tmp, self._ledger_path)

    def reserve(
        self, stage: str, model_id: str, est_input_tokens: int, est_output_tokens: int = 0
    ) -> Reservation:
        """Estimate a call's cost and hold it against the stage cap or refuse."""
        estimate = self._pricing.cost(
            model_id, Usage(input_tokens=est_input_tokens, output_tokens=est_output_tokens)
        )
        cap = self._caps.get(stage)
        current = self.spent(stage) + self.reserved(stage)
        if cap is None or current + estimate > cap:
            raise SpendCapExceeded(stage, cap, self.spent(stage), estimate)
        reservation = Reservation(stage=stage, model_id=model_id, estimated_usd=estimate)
        self._reserved[stage] = self.reserved(stage) + estimate
        self._open.add(reservation.reservation_id)
        return reservation

    def _release(self, reservation: Reservation) -> None:
        """Drop a reservation's held estimate from the stage total."""
        if reservation.reservation_id not in self._open:
            raise ValueError("reservation is not open")
        self._open.discard(reservation.reservation_id)
        remaining = self.reserved(reservation.stage) - reservation.estimated_usd
        if remaining > 0.0:
            self._reserved[reservation.stage] = remaining
        else:
            self._reserved.pop(reservation.stage, None)

    def settle(self, reservation: Reservation, usage: Usage) -> float:
        """Replace the held estimate with the actual cost and persist it."""
        self._release(reservation)
        actual = self._pricing.cost(reservation.model_id, usage)
        self._ledger.record(
            LedgerEntry(
                stage=reservation.stage,
                model_id=reservation.model_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                usd=actual,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        reservation.settled = True
        self._persist()
        return actual

    def cancel(self, reservation: Reservation) -> None:
        """Release a reservation without recording any spend."""
        self._release(reservation)

    def cap(self, stage: str) -> float | None:
        """Return the configured spend cap for a stage, or None when uncapped."""
        return self._caps.get(stage)

    def spent(self, stage: str) -> float:
        """Return the settled spend recorded for a stage."""
        return self._ledger.per_stage.get(stage, 0.0)

    def reserved(self, stage: str) -> float:
        """Return the outstanding reserved estimate for a stage."""
        return self._reserved.get(stage, 0.0)

    def total(self) -> float:
        """Return the settled spend across all stages."""
        return sum(self._ledger.per_stage.values())

    def by_model(self) -> dict[str, float]:
        """Return settled spend totals keyed by model."""
        return dict(self._ledger.per_model)
