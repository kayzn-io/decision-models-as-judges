"""The G10 gate: local decision model (Laya) zero-shot and fine-tuned vs hosts.

Every judge answers the G3 outcome questions on the compact state profile, so
the local model is compared head to head against the hosted decision model and
the cheap LLM against the same harness reward.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome, _mode, _verdict_choice
from decision_judges.metrics import (
    accuracy,
    auroc,
    brier,
    cohens_kappa,
    ece,
    latency_summary,
    temperature_fit,
)
from decision_judges.serialize import StateProfile
from decision_judges.types import Question, Verdict

_PASS = "pass"
_LAYA = "laya"
_LAYA_FT = "laya_ft"


def _pass_prob(verdict: Verdict) -> float | None:
    """Return P(pass) from the verdict question's probabilities, or None."""
    for answer in verdict.answers:
        if answer.question_id == "verdict" and answer.probabilities is not None:
            return answer.probabilities.get(_PASS)
    return None


def _first_extra(group: Sequence[Verdict], key: str) -> str:
    """Return the first non-empty value for an extra key across the group."""
    for verdict in group:
        value = verdict.extra.get(key)
        if value:
            return value
    return ""


class _Reduced(NamedTuple):
    """Per-item modal choices, truth labels, mean P(pass), and binary truth."""

    modal: list[str]
    truth: list[str]
    pass_probs: list[float]
    binary: list[float]


def _reduce(group: Sequence[Verdict], by_hash: Mapping[str, Item]) -> _Reduced:
    """Collapse a judge's non-error verdicts into per-item scored sequences."""
    choices: dict[str, list[str]] = defaultdict(list)
    probs: dict[str, list[float]] = defaultdict(list)
    for verdict in group:
        if verdict.error is not None:
            continue
        choice = _verdict_choice(verdict)
        if choice is not None:
            choices[verdict.state_hash].append(choice)
        prob = _pass_prob(verdict)
        if prob is not None:
            probs[verdict.state_hash].append(prob)

    reduced = _Reduced([], [], [], [])
    for state_hash, item_choices in choices.items():
        item = by_hash.get(state_hash)
        if item is None or item.truth_label is None:
            continue
        reduced.modal.append(_mode(item_choices))
        reduced.truth.append(item.truth_label)
        item_probs = probs.get(state_hash)
        if item_probs:
            reduced.pass_probs.append(float(np.mean(item_probs)))
            reduced.binary.append(1.0 if item.truth_label == _PASS else 0.0)
    return reduced


class G10LocalModel(Gate):
    """Compare zero-shot and fine-tuned Laya against the hosts on compact states."""

    gate_id = "g10"
    stage = "g10"

    def __init__(self, outcome: G3Outcome | None = None) -> None:
        self._outcome = outcome if outcome is not None else G3Outcome()

    def questions(self) -> list[Question]:
        """Return the G3 outcome questions every judge answers."""
        return self._outcome.questions()

    @property
    def rubric_path(self) -> Path:
        """Path of the shared G3 outcome rubric."""
        return self._outcome.rubric_path

    @property
    def prompt_version(self) -> str:
        """Return the G3 prompt version judges share."""
        return self._outcome.prompt_version

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Build G3 items, forcing the compact profile this gate is defined on."""
        if profile is not StateProfile.compact:
            raise ValueError("g10 evaluates the compact state profile only")
        return self._outcome.build_items(records, tasks, StateProfile.compact)

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Summarize each judge and, for fine-tuned Laya, each checkpoint fold.

        Error verdicts are excluded from accuracy, AUROC, and calibration but
        counted in the error rate. Local judges (Laya) incur no API cost, so the
        spend ledger records zero for them; this table flags them instead.
        """
        by_hash = {item.state.state_hash: item for item in items}
        groups: dict[str, list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            if verdict.state_hash in by_hash:
                groups[verdict.judge_id].append(verdict)

        rows = [self._summarize(judge_id, groups[judge_id], by_hash) for judge_id in sorted(groups)]
        frame = pd.DataFrame(rows)
        tables: dict[str, object] = {"g10_summary": frame}
        folds = self._folds(groups, by_hash)
        if folds is not None:
            tables["g10_folds"] = folds
        return GateResult(
            tables=tables,
            charts={"g10_zero_shot_vs_finetuned": self._chart(frame)},
            findings=self._findings(frame),
        )

    def _summarize(
        self, judge_id: str, group: Sequence[Verdict], by_hash: Mapping[str, Item]
    ) -> dict[str, object]:
        """Reduce one judge's verdicts into a summary row."""
        reduced = _reduce(group, by_hash)
        local = judge_id.startswith(_LAYA)
        best_t, ece_before, ece_after = self._calibration(reduced)
        total = len(group)
        errors = sum(1 for verdict in group if verdict.error is not None)
        latency = latency_summary([float(verdict.latency_ms) for verdict in group])
        return {
            "judge_id": judge_id,
            "local": local,
            "n_items": len(reduced.modal),
            "accuracy": accuracy(reduced.modal, reduced.truth) if reduced.modal else float("nan"),
            "kappa": (
                cohens_kappa(reduced.modal, reduced.truth) if reduced.modal else float("nan")
            ),
            "auroc": auroc(reduced.pass_probs, reduced.binary) if reduced.binary else float("nan"),
            "ece": ece_before,
            "brier": brier(reduced.pass_probs, reduced.binary) if reduced.binary else float("nan"),
            "best_T": best_t,
            "ece_after": ece_after,
            "latency_p50": latency.p50,
            "error_rate": errors / total if total else 0.0,
            "checkpoint_hash": _first_extra(group, "checkpoint_hash") if local else "",
            "temperature": _first_extra(group, "temperature") if local else "",
        }

    def _calibration(self, reduced: _Reduced) -> tuple[float, float, float]:
        """Return best temperature, ECE, and ECE after temperature fitting."""
        if not reduced.binary:
            return float("nan"), float("nan"), float("nan")
        ece_before = ece(reduced.pass_probs, reduced.binary)
        best_t, best_probs = temperature_fit(reduced.pass_probs, reduced.binary)
        ece_after = ece(best_probs.tolist(), reduced.binary)
        return best_t, ece_before, ece_after

    def _folds(
        self, groups: Mapping[str, Sequence[Verdict]], by_hash: Mapping[str, Item]
    ) -> pd.DataFrame | None:
        """Return one row per fine-tuned checkpoint, or None if none are present."""
        ft_verdicts: list[Verdict] = []
        for judge_id, group in groups.items():
            if judge_id.startswith(_LAYA_FT):
                ft_verdicts.extend(group)
        if not ft_verdicts:
            return None

        by_ckpt: dict[str, list[Verdict]] = defaultdict(list)
        for verdict in ft_verdicts:
            by_ckpt[verdict.extra.get("checkpoint_hash", "")].append(verdict)

        rows: list[dict[str, object]] = []
        for checkpoint_hash in sorted(by_ckpt):
            reduced = _reduce(by_ckpt[checkpoint_hash], by_hash)
            rows.append(
                {
                    "checkpoint_hash": checkpoint_hash,
                    "n_items": len(reduced.modal),
                    "accuracy": (
                        accuracy(reduced.modal, reduced.truth) if reduced.modal else float("nan")
                    ),
                    "ece": ece(reduced.pass_probs, reduced.binary)
                    if reduced.binary
                    else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def _chart(self, frame: pd.DataFrame) -> Figure:
        """Return grouped accuracy and AUROC bars per judge, built without pyplot."""
        figure = Figure()
        axes = figure.subplots()
        if not frame.empty:
            positions = np.arange(len(frame))
            width = 0.4
            axes.bar(positions - width / 2, frame["accuracy"].tolist(), width, label="accuracy")
            axes.bar(positions + width / 2, frame["auroc"].tolist(), width, label="auroc")
            axes.set_xticks(positions)
            axes.set_xticklabels(frame["judge_id"].tolist())
            axes.legend()
        axes.set_ylabel("score")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G10 zero-shot vs fine-tuned")
        return figure

    def _findings(self, frame: pd.DataFrame) -> str:
        """Return two to three factual sentences on the top judge and calibration."""
        if frame.empty:
            return "No verdicts were available to analyze."
        best = frame.sort_values("accuracy", ascending=False).iloc[0]
        parts = [
            f"Judge {str(best['judge_id'])!r} had the highest accuracy on the compact state "
            f"at {float(best['accuracy']):.2f}."
        ]
        judges = set(frame["judge_id"])
        if "laya_base" in judges and "laya_ft" in judges:
            base = float(frame.loc[frame["judge_id"] == "laya_base", "accuracy"].iloc[0])
            fine = float(frame.loc[frame["judge_id"] == "laya_ft", "accuracy"].iloc[0])
            parts.append(f"Fine-tuning moved Laya accuracy from {base:.2f} to {fine:.2f}.")
        lowered = [
            str(judge_id)
            for judge_id, before, after in zip(
                frame["judge_id"], frame["ece"], frame["ece_after"], strict=True
            )
            if float(after) < float(before) - 1e-9
        ]
        if lowered:
            parts.append("Temperature fitting lowered ECE for " + ", ".join(lowered) + ".")
        else:
            parts.append("Temperature fitting did not lower ECE for any judge.")
        return " ".join(parts)
