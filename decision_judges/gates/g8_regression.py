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
from decision_judges.gates.names import as_points, judge_name, variant_name
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


def _join_names(names: Sequence[str]) -> str:
    """Join names as 'a', 'a and b', or 'a, b, and c'."""
    items = list(names)
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _change_phrase(delta: float) -> str:
    """Describe a pass-rate change as a drop, a rise, or no change."""
    if delta < 0:
        return f"a drop of {as_points(delta)}"
    if delta > 0:
        return f"a rise of {as_points(delta)}"
    return "no change"


def _bound_phrase(delta: float) -> str:
    """Describe one interval bound as points worse, points better, or no change."""
    if delta < 0:
        return f"{as_points(delta)} worse"
    if delta > 0:
        return f"{as_points(delta)} better"
    return "no change"


def _capitalize(text: str) -> str:
    """Capitalize the first character of a sentence, leaving the rest unchanged."""
    return text[:1].upper() + text[1:] if text else text


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
        """Return a finding naming which of the two required agents are present."""
        present = (
            ", ".join(variant_name(v) for v in sorted(variants)) if variants else "neither of them"
        )
        return (
            "This experiment needs both the careful agent and the rushed agent, but the "
            f"conversations cover only: {present}."
        )

    def _findings(self, frame: pd.DataFrame, td: float) -> str:
        """Return three plain paragraphs: what was tested, the numbers, the choice."""
        if frame.empty:
            return "No conversations were available to analyze the regression."
        what = (
            "The rushed agent skips confirming with customers; this experiment asks whether each "
            "judge notices that it does worse."
        )
        return "\n\n".join([what, self._numbers(frame, td), self._alarm_and_meaning(frame, td)])

    def _numbers(self, frame: pd.DataFrame, td: float) -> str:
        """Return the true gap, each judge's estimate, and a classification sentence."""
        direction = "less often" if td < 0 else "more often" if td > 0 else "just as often"
        small = abs(td) < 0.05
        tail = ", a gap small enough to be hard to detect." if small else "."
        truth = (
            f"In the ground truth the rushed agent passed {as_points(td)} {direction} than the "
            f"careful agent{tail}"
        )
        clauses: list[str] = []
        right_size: list[str] = []
        exaggerated: list[str] = []
        missed: list[str] = []
        for judge, est, lo, hi, detected, covers in zip(
            frame["judge"],
            frame["est_delta"],
            frame["lo"],
            frame["hi"],
            frame["detected"],
            frame["covers_truth"],
            strict=True,
        ):
            name = judge_name(str(judge))
            interval = (
                f"somewhere between {_bound_phrase(float(lo))} and {_bound_phrase(float(hi))}"
            )
            change = _change_phrase(float(est))
            clauses.append(f"{_capitalize(name)} saw {change}, {interval}.")
            if not bool(detected):
                missed.append(name)
            elif bool(covers):
                right_size.append(name)
            else:
                exaggerated.append(name)
        groups: list[str] = []
        if right_size:
            groups.append(f"{_join_names(right_size)} got the size about right")
        if exaggerated:
            groups.append(f"{_join_names(exaggerated)} exaggerated the drop")
        if missed:
            groups.append(f"{_join_names(missed)} missed it")
        classification = _capitalize("; ".join(groups)) + "." if groups else ""
        return " ".join([truth, *clauses, classification]).strip()

    def _alarm_and_meaning(self, frame: pd.DataFrame, td: float) -> str:
        """Return the false-alarm result and an honest note on exaggeration."""
        alarms = [
            judge_name(str(j))
            for j, flag in zip(frame["judge"], frame["false_alarm"], strict=True)
            if flag is True
        ]
        skipped = [
            judge_name(str(j))
            for j, flag in zip(frame["judge"], frame["false_alarm"], strict=True)
            if flag is None
        ]
        if alarms:
            alarm = (
                f"{_capitalize(_join_names(alarms))} reported a drop when shown two halves of the "
                "careful agent's own conversations, a false alarm."
            )
        elif skipped:
            alarm = (
                "The false-alarm test was skipped for "
                f"{_join_names(skipped)} for lack of four repeats per conversation."
            )
        else:
            alarm = (
                "None of the judges reported a drop when shown two halves of the careful agent's "
                "own conversations."
            )
        exaggerated = [
            judge_name(str(j))
            for j, det, cov in zip(
                frame["judge"], frame["detected"], frame["covers_truth"], strict=True
            )
            if bool(det) and not bool(cov)
        ]
        right_size = [
            judge_name(str(j))
            for j, det, cov in zip(
                frame["judge"], frame["detected"], frame["covers_truth"], strict=True
            )
            if bool(det) and bool(cov)
        ]
        meaning_parts: list[str] = []
        if exaggerated:
            meaning_parts.append(
                f"{_capitalize(_join_names(exaggerated))} treated the rushed agent's behaviour as "
                "failure rather than judging its results, so the drop they report is larger than "
                "the real regression"
            )
        if right_size:
            meaning_parts.append(f"{_join_names(right_size)} is the estimate you could trust here")
        elif abs(td) < 0.05:
            meaning_parts.append("the true gap is so small that no judge pins it down well")
        meaning = _capitalize("; ".join(meaning_parts)) + "." if meaning_parts else ""
        return " ".join(part for part in (alarm, meaning) if part)
