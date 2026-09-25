"""The G8 gate: detect a pass-rate regression between two agent variants.

Analysis only. Over G3 verdicts for the baseline and degraded variants, each
judge estimates the pass-rate delta from its own verdicts with a bootstrap
interval, which is checked against the true delta from ground truth. A
false-alarm test judges the baseline variant against itself on disjoint repeat
sets, so an interval that excludes zero there is a spurious signal.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome, _verdict_choice
from decision_judges.judges.base import Judge
from decision_judges.metrics import bootstrap_delta
from decision_judges.progress import CancelToken, ProgressCallback
from decision_judges.serialize import StateProfile
from decision_judges.spend import Spend
from decision_judges.types import Question, Verdict

_PASS = "pass"
_BASELINE = "baseline"
_DEGRADED = "degraded"


def _modal_choice(choices: Sequence[str]) -> str:
    """Return the most frequent choice, resolving ties by sort order."""
    return Counter(sorted(choices)).most_common(1)[0][0]


def _mean(values: Sequence[float]) -> float:
    """Return the mean of the values, or NaN when there are none."""
    return float(np.mean(values)) if len(values) else float("nan")


def _excludes_zero(lo: float, hi: float) -> bool:
    """Return whether a closed interval lies entirely above or below zero."""
    return lo > 0.0 or hi < 0.0


def pass_labels(
    verdicts_for_variant_judge: Mapping[str, Sequence[Verdict]],
    repeats: set[int] | None,
) -> list[int]:
    """Return a per-task pass label from the modal verdict over selected repeats.

    Verdicts are grouped by task; tasks are visited in sorted order. For each
    task the modal verdict choice over the chosen repeats (all repeats when
    ``repeats`` is None) yields 1 for pass and 0 otherwise, ignoring error and
    unanswered verdicts.
    """
    labels: list[int] = []
    for task_id in sorted(verdicts_for_variant_judge):
        choices: list[str] = []
        for verdict in verdicts_for_variant_judge[task_id]:
            if verdict.error is not None:
                continue
            if repeats is not None and verdict.repeat not in repeats:
                continue
            choice = _verdict_choice(verdict)
            if choice is not None:
                choices.append(choice)
        labels.append(1 if choices and _modal_choice(choices) == _PASS else 0)
    return labels


def true_delta(items: Sequence[Item]) -> float:
    """Return the ground-truth pass-rate delta, degraded minus baseline, by task."""
    truth: dict[str, dict[str, float]] = defaultdict(dict)
    for item in items:
        if item.truth_label is None:
            continue
        indicator = 1.0 if item.truth_label == _PASS else 0.0
        truth[item.state.variant][item.state.task_id] = indicator
    base = list(truth.get(_BASELINE, {}).values())
    degraded = list(truth.get(_DEGRADED, {}).values())
    return _mean(degraded) - _mean(base)


def estimate_delta(
    base_labels: Sequence[int],
    degraded_labels: Sequence[int],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return the bootstrap point estimate and 95% interval of the delta.

    The delta is the degraded pass rate minus the baseline pass rate.
    """
    result = bootstrap_delta(base_labels, degraded_labels, n_boot, seed)
    return result.delta, result.lo, result.hi


def false_alarm(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    n_boot: int,
    seed: int,
) -> bool:
    """Return whether the bootstrap interval between two label sets excludes zero."""
    result = bootstrap_delta(labels_a, labels_b, n_boot, seed)
    return _excludes_zero(result.lo, result.hi)


class G8Regression(Gate):
    """Estimate the baseline-to-degraded pass-rate regression each judge sees."""

    gate_id = "g8"
    stage = "g8"

    def __init__(
        self,
        g3: G3Outcome | None = None,
        *,
        n_boot: int = 2000,
        seed: int = 7,
        alpha: float = 0.05,
    ) -> None:
        self._g3 = g3 if g3 is not None else G3Outcome()
        self._n_boot = n_boot
        self._seed = seed
        self._alpha = alpha

    def questions(self) -> list[Question]:
        """Return the G3 outcome questions every judge answers."""
        return self._g3.questions()

    @property
    def rubric_path(self) -> Path:
        """Path of the shared G3 outcome rubric."""
        return self._g3.rubric_path

    @property
    def prompt_version(self) -> str:
        """Return the G3 prompt version judges share."""
        return self._g3.prompt_version

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Delegate item construction to the G3 outcome gate."""
        return self._g3.build_items(records, tasks, profile)

    def run(
        self,
        items: Sequence[Item],
        judges: Sequence[Judge],
        cache: Cache,
        spend: Spend,
        *,
        repeats: Mapping[str, int] | int,
        on_progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> list[Verdict]:
        """Reject direct execution; G8 analyzes verdicts the G3 gate produces."""
        raise NotImplementedError("g8 analyzes existing g3 verdicts and does not judge")

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Estimate each judge's regression, test for false alarms, and report.

        The items must cover both the baseline and degraded variants; otherwise
        the summary table is empty and the findings say which variants are
        missing.
        """
        by_hash = {item.state.state_hash: item for item in items}
        variants = {item.state.variant for item in items}
        if not {_BASELINE, _DEGRADED} <= variants:
            return GateResult(
                tables={"g8_summary": pd.DataFrame()},
                charts={"g8_intervals": self._intervals_chart(pd.DataFrame(), float("nan"))},
                findings=self._missing_findings(variants),
            )

        td = true_delta(items)
        grouped = self._group(verdicts, by_hash)
        rows = [self._summarize(judge_id, grouped[judge_id], td) for judge_id in sorted(grouped)]
        frame = pd.DataFrame(rows)
        return GateResult(
            tables={"g8_summary": frame},
            charts={"g8_intervals": self._intervals_chart(frame, td)},
            findings=self._findings(frame, td),
        )

    def _group(
        self, verdicts: Sequence[Verdict], by_hash: Mapping[str, Item]
    ) -> dict[str, dict[str, dict[str, list[Verdict]]]]:
        """Group verdicts by judge, then variant, then task id."""
        grouped: dict[str, dict[str, dict[str, list[Verdict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for verdict in verdicts:
            item = by_hash.get(verdict.state_hash)
            if item is None:
                continue
            grouped[verdict.judge_id][item.state.variant][item.state.task_id].append(verdict)
        return grouped

    def _summarize(
        self,
        judge_id: str,
        by_variant: Mapping[str, Mapping[str, Sequence[Verdict]]],
        td: float,
    ) -> dict[str, object]:
        """Reduce one judge's verdicts into a regression summary row."""
        base_map = by_variant.get(_BASELINE, {})
        degraded_map = by_variant.get(_DEGRADED, {})
        base_labels = pass_labels(base_map, None)
        degraded_labels = pass_labels(degraded_map, None)
        point, lo, hi = estimate_delta(base_labels, degraded_labels, self._n_boot, self._seed)
        return {
            "judge": judge_id,
            "true_delta": td,
            "est_delta": point,
            "lo": lo,
            "hi": hi,
            "detected": _excludes_zero(lo, hi),
            "covers_truth": lo <= td <= hi,
            "false_alarm": self._false_alarm(base_map),
        }

    def _false_alarm(self, base_map: Mapping[str, Sequence[Verdict]]) -> bool | None:
        """Test the baseline against itself on repeats {0,1} vs {2,3}, if available."""
        present = {verdict.repeat for group in base_map.values() for verdict in group}
        if not {0, 1, 2, 3} <= present:
            return None
        labels_a = pass_labels(base_map, {0, 1})
        labels_b = pass_labels(base_map, {2, 3})
        return false_alarm(labels_a, labels_b, self._n_boot, self._seed)

    def _intervals_chart(self, frame: pd.DataFrame, td: float) -> Figure:
        """Return a horizontal interval per judge with the true delta as a line."""
        figure = Figure()
        axes = figure.subplots()
        if not frame.empty:
            positions = np.arange(len(frame))
            for pos, lo, hi, est in zip(
                positions, frame["lo"], frame["hi"], frame["est_delta"], strict=True
            ):
                axes.plot([lo, hi], [pos, pos], marker="|", color="tab:blue")
                axes.plot([est], [pos], marker="o", color="tab:blue")
            axes.set_yticks(positions)
            axes.set_yticklabels(frame["judge"].tolist())
        if not np.isnan(td):
            axes.axvline(td, linestyle="--", color="black", label="true delta")
            axes.legend()
        axes.set_xlabel("degraded - baseline pass-rate delta")
        axes.set_title("G8 estimated regression intervals")
        return figure

    def _missing_findings(self, variants: set[str]) -> str:
        """Return a finding naming which of the two required variants are present."""
        present = ", ".join(sorted(variants)) if variants else "none"
        return (
            "The regression analysis needs both the baseline and degraded variants, "
            f"but the items cover only: {present}."
        )

    def _findings(self, frame: pd.DataFrame, td: float) -> str:
        """Return two to three factual sentences on detection, coverage, and alarms."""
        if frame.empty:
            return "No verdicts were available to analyze the regression."
        detected = [
            str(j) for j, flag in zip(frame["judge"], frame["detected"], strict=True) if flag
        ]
        covered = [
            str(j) for j, flag in zip(frame["judge"], frame["covers_truth"], strict=True) if flag
        ]
        alarms = [
            str(j)
            for j, flag in zip(frame["judge"], frame["false_alarm"], strict=True)
            if flag is True
        ]
        skipped = [
            str(j)
            for j, flag in zip(frame["judge"], frame["false_alarm"], strict=True)
            if flag is None
        ]

        detect_text = ", ".join(detected) if detected else "no judges"
        cover_text = ", ".join(covered) if covered else "no judges"
        sentences = [
            f"The true degraded-minus-baseline pass-rate delta is {td:.2f}.",
            f"Detected the regression: {detect_text}; covered the true delta: {cover_text}.",
        ]
        if alarms:
            sentences.append(
                "Raised a false alarm on the baseline variant: " + ", ".join(alarms) + "."
            )
        elif skipped:
            sentences.append(
                "The false-alarm test was skipped for "
                + ", ".join(skipped)
                + " for lack of four repeats."
            )
        else:
            sentences.append("No judge raised a false alarm on the baseline variant.")
        return " ".join(sentences)
