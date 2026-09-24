"""The G6 calibration gate: how well do judge probabilities track the truth?

G6 is analysis-only. It reuses the G3 outcome verdicts and never calls a judge.
For each judge it scores three probability signals against ground truth, reports
their expected calibration error and Brier score, and recomputes both after a
cross-validated isotonic recalibration to show how much miscalibration a
monotone map can remove.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.isotonic import IsotonicRegression

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.judges.base import Judge
from decision_judges.metrics import brier, ece, reliability_bins
from decision_judges.serialize import StateProfile
from decision_judges.spend import Spend
from decision_judges.types import Answer, Question, Verdict

_PASS = "pass"


def _answer(verdict: Verdict, question_id: str) -> Answer | None:
    """Return the answer to a question by id, or None if the verdict lacks it."""
    for answer in verdict.answers:
        if answer.question_id == question_id:
            return answer
    return None


class Signal(Protocol):
    """A probability the judge emits, paired with the label it should predict."""

    name: str

    def extract(self, verdict: Verdict, truth_pass: int) -> tuple[float, int] | None:
        """Return (probability, label) for the verdict, or None if unavailable.

        ``truth_pass`` is the withheld truth (1 when the item truly passed) that
        each signal turns into the label its probability is trying to predict.
        """


class CompletedNoul:
    """The 'completed' noul read as the probability that the item passed."""

    name = "completed_noul"

    def extract(self, verdict: Verdict, truth_pass: int) -> tuple[float, int] | None:
        """Return the completed noul against the item's pass label."""
        answer = _answer(verdict, "completed")
        if answer is None or answer.noul is None:
            return None
        return answer.noul, truth_pass


class VerdictPassProb:
    """The verdict choice's probability mass on 'pass'."""

    name = "verdict_pass_prob"

    def extract(self, verdict: Verdict, truth_pass: int) -> tuple[float, int] | None:
        """Return the pass probability against the item's pass label."""
        answer = _answer(verdict, "verdict")
        if answer is None or answer.probabilities is None:
            return None
        prob = answer.probabilities.get(_PASS)
        if prob is None:
            return None
        return prob, truth_pass


class VerdictConfidence:
    """The verdict choice's confidence, read as a probability about a different
    question: is the judge right when it says it is confident?

    The label is therefore choice correctness (1 when the chosen label matches
    the truth), not whether the item passed.
    """

    name = "verdict_confidence"

    def extract(self, verdict: Verdict, truth_pass: int) -> tuple[float, int] | None:
        """Return the confidence against whether the judge's choice was correct."""
        answer = _answer(verdict, "verdict")
        if answer is None or answer.choice is None or answer.confidence is None:
            return None
        chose_pass = 1 if answer.choice == _PASS else 0
        label = 1 if chose_pass == truth_pass else 0
        return answer.confidence, label


def _signals() -> list[Signal]:
    """Return the three calibration signals in a stable order."""
    return [CompletedNoul(), VerdictPassProb(), VerdictConfidence()]


def _assign_folds(n: int, k: int, seed: int) -> list[int]:
    """Assign each of n indices to one of k folds, balanced and seed-deterministic."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = [0] * n
    for position, index in enumerate(perm):
        folds[int(index)] = position % k
    return folds


def _isotonic_apply(
    train_probs: Sequence[float],
    train_labels: Sequence[float],
    test_probs: Sequence[float],
) -> list[float]:
    """Fit a monotone map on the training fold and apply it to the test fold."""
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(np.asarray(train_probs, dtype=float), np.asarray(train_labels, dtype=float))
    return [float(value) for value in model.predict(np.asarray(test_probs, dtype=float))]


def isotonic_cv(probs: Sequence[float], labels: Sequence[float], k: int, seed: int) -> list[float]:
    """Return isotonically recalibrated probabilities via k-fold cross-validation.

    Each fold is recalibrated by a map fit only on the other folds, so no point
    is scored by a map trained on itself. With a single fold, or a fold whose
    complement is empty, the map is fit on all available points instead.
    """
    n = len(probs)
    if n == 0:
        return []
    if k <= 1:
        return _isotonic_apply(probs, labels, probs)

    folds = _assign_folds(n, k, seed)
    out = [0.0] * n
    for fold in range(k):
        test_idx = [i for i in range(n) if folds[i] == fold]
        train_idx = [i for i in range(n) if folds[i] != fold]
        if not test_idx:
            continue
        if not train_idx:
            for i in test_idx:
                out[i] = float(probs[i])
            continue
        recalibrated = _isotonic_apply(
            [probs[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [probs[i] for i in test_idx],
        )
        for position, i in enumerate(test_idx):
            out[i] = recalibrated[position]
    return out


class G6Calibration(Gate):
    """Score the calibration of G3 judge probabilities without calling a judge."""

    gate_id = "g6"
    stage = "g6"

    def __init__(
        self,
        g3: G3Outcome | None = None,
        *,
        bins: int = 10,
        k: int = 5,
        seed: int = 7,
    ) -> None:
        self._g3 = g3 if g3 is not None else G3Outcome()
        self.bins = bins
        self.k = k
        self.seed = seed

    def questions(self) -> list[Question]:
        """Delegate to the G3 gate's question set."""
        return self._g3.questions()

    @property
    def prompt_version(self) -> str:
        """Delegate to the G3 gate's prompt version."""
        return self._g3.prompt_version

    @property
    def rubric_path(self) -> Path:
        """Delegate to the G3 gate's rubric path."""
        return self._g3.rubric_path

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Delegate item construction to the G3 gate."""
        return self._g3.build_items(records, tasks, profile)

    def run(
        self,
        items: Sequence[Item],
        judges: Sequence[Judge],
        cache: Cache,
        spend: Spend,
        *,
        repeats: Mapping[str, int] | int,
    ) -> list[Verdict]:
        """Raise, since G6 analyzes existing G3 verdicts rather than judging."""
        raise NotImplementedError("G6 is analysis-only over G3 verdicts")

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Score each judge-and-signal pair and recalibrate, returning tables and a chart.

        Error verdicts are skipped. Repeats of the same item are collapsed to a
        single point by averaging both the probability and the label.
        """
        by_hash = {item.state.state_hash: item for item in items}
        judges = sorted({v.judge_id for v in verdicts if v.error is None})
        signals = _signals()

        points: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        summary_rows: list[dict[str, object]] = []
        bin_rows: list[dict[str, object]] = []

        for judge in judges:
            for signal in signals:
                probs, labels = self._collect(judge, signal, verdicts, by_hash)
                points[(judge, signal.name)] = (probs, labels)
                if not probs:
                    continue
                recal = isotonic_cv(probs, labels, self.k, self.seed)
                summary_rows.append(
                    {
                        "judge": judge,
                        "signal": signal.name,
                        "n": len(probs),
                        "ece": ece(probs, labels, self.bins),
                        "brier": brier(probs, labels),
                        "ece_after_isotonic": ece(recal, labels, self.bins),
                        "brier_after_isotonic": brier(recal, labels),
                    }
                )
                for rbin in reliability_bins(probs, labels, self.bins):
                    bin_rows.append(
                        {
                            "judge": judge,
                            "signal": signal.name,
                            "bin_lower": rbin.lower,
                            "bin_upper": rbin.upper,
                            "mean_prob": rbin.mean_prob,
                            "frac_positive": rbin.frac_positive,
                            "count": rbin.size,
                        }
                    )

        summary = pd.DataFrame(summary_rows)
        return GateResult(
            tables={"g6_summary": summary, "g6_bins": pd.DataFrame(bin_rows)},
            charts={"g6_reliability": self._reliability_chart(judges, signals, points)},
            findings=self._findings(summary),
        )

    def _collect(
        self,
        judge: str,
        signal: Signal,
        verdicts: Sequence[Verdict],
        by_hash: Mapping[str, Item],
    ) -> tuple[list[float], list[float]]:
        """Gather one averaged (probability, label) point per item for a judge and signal."""
        per_item: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for verdict in verdicts:
            if verdict.judge_id != judge or verdict.error is not None:
                continue
            item = by_hash.get(verdict.state_hash)
            if item is None or item.truth_label is None:
                continue
            truth_pass = 1 if item.truth_label == _PASS else 0
            extracted = signal.extract(verdict, truth_pass)
            if extracted is not None:
                per_item[verdict.state_hash].append(extracted)

        probs: list[float] = []
        labels: list[float] = []
        for state_hash in sorted(per_item):
            pairs = per_item[state_hash]
            probs.append(sum(prob for prob, _ in pairs) / len(pairs))
            labels.append(sum(label for _, label in pairs) / len(pairs))
        return probs, labels

    def _reliability_chart(
        self,
        judges: Sequence[str],
        signals: Sequence[Signal],
        points: Mapping[tuple[str, str], tuple[list[float], list[float]]],
    ) -> Figure:
        """Return one reliability subplot per judge, built without pyplot."""
        figure = Figure(figsize=(4 * max(len(judges), 1), 4))
        if not judges:
            return figure
        axes = figure.subplots(1, len(judges), squeeze=False)[0]
        for axis, judge in zip(axes, judges, strict=True):
            axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray")
            for signal in signals:
                probs, labels = points.get((judge, signal.name), ([], []))
                if not probs:
                    continue
                curve = reliability_bins(probs, labels, self.bins)
                axis.plot(
                    [b.mean_prob for b in curve],
                    [b.frac_positive for b in curve],
                    marker="o",
                    label=signal.name,
                )
            axis.set_title(judge)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("mean predicted probability")
            axis.set_ylabel("fraction positive")
            axis.legend(loc="best", fontsize="small")
        return figure

    def _findings(self, summary: pd.DataFrame) -> str:
        """Return two to three factual sentences on best calibration and recalibration."""
        if summary.empty:
            return "No verdicts were available to analyze."
        best = summary.sort_values("ece").iloc[0]
        improved = int((summary["ece_after_isotonic"] < summary["ece"] - 1e-9).sum())
        total = len(summary)
        mean_delta = float((summary["ece"] - summary["ece_after_isotonic"]).mean())
        return (
            f"Signal {str(best['signal'])!r} from judge {str(best['judge'])!r} was the "
            f"best calibrated at ECE {float(best['ece']):.3f}. Isotonic recalibration lowered "
            f"ECE for {improved} of {total} judge-signal pairs, a mean reduction of "
            f"{mean_delta:.3f}. Recalibration is fit with cross-validation, so the gains "
            f"reflect held-out folds rather than a map scored on its own data."
        )
