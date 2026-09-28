"""The G5 cascade gate: analyze G3 verdicts as a confidence-gated cascade.

A cascade keeps a fast decision model's verdict when its confidence clears a
threshold and otherwise escalates to a strong LLM, paying the LLM only where it
escalates. A three-tier variant routes the least confident items to a human at
a nominal cost with ground-truth accuracy. This gate makes no judge calls; it
reduces existing G3 verdicts into a cost-accuracy frontier.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache
from decision_judges.config import PricingTable
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.names import as_money, as_percent, as_points, judge_name
from decision_judges.judges.base import Judge
from decision_judges.metrics import accuracy
from decision_judges.progress import CancelToken, ProgressCallback
from decision_judges.serialize import StateProfile
from decision_judges.spend import Spend
from decision_judges.types import Question, Verdict

_UNPRICED_MODEL_ID = "none"
_CHEAP_JUDGE = "llm_cheap"
_THREE_TIER_GAP = 0.2
_THREE_TIER_FLOOR = 0.5


class ItemVerdict(NamedTuple):
    """A judge's aggregate verdict for one item: label, confidence, unit cost."""

    label: str
    confidence: float
    cost: float


class CascadeDecision(NamedTuple):
    """The label, USD cost, and escalation flag a cascade returns for one item."""

    label: str
    cost: float
    escalated: bool


def _verdict_choice(verdict: Verdict) -> tuple[str | None, float]:
    """Return the verdict answer's choice and confidence, or None and zero."""
    for answer in verdict.answers:
        if answer.question_id == "verdict":
            return answer.choice, answer.confidence or 0.0
    return None, 0.0


def _mode(choices: Sequence[str]) -> str:
    """Return the most frequent choice, resolving ties by sort order."""
    return Counter(sorted(choices)).most_common(1)[0][0]


def modal_verdict(verdicts: Sequence[Verdict]) -> tuple[str, float]:
    """Return the modal verdict choice and the mean confidence over repeats."""
    choices: list[str] = []
    confidences: list[float] = []
    for verdict in verdicts:
        choice, confidence = _verdict_choice(verdict)
        if choice is not None:
            choices.append(choice)
            confidences.append(confidence)
    label = _mode(choices)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return label, confidence


def _unit_cost(verdict: Verdict, pricing: PricingTable) -> float:
    """Return the USD cost of one verdict, treating the unpriced model as free."""
    if verdict.model_id == _UNPRICED_MODEL_ID:
        return 0.0
    return pricing.cost(verdict.model_id, verdict.usage)


def cost_of(verdicts: Sequence[Verdict], pricing: PricingTable) -> float:
    """Return the mean USD cost of a single judgment across the repeats."""
    if not verdicts:
        return 0.0
    return sum(_unit_cost(verdict, pricing) for verdict in verdicts) / len(verdicts)


def cascade(fast: ItemVerdict, slow: ItemVerdict, t: float) -> CascadeDecision:
    """Keep the fast verdict at confidence >= t, else escalate to the slow one."""
    if fast.confidence >= t:
        return CascadeDecision(label=fast.label, cost=fast.cost, escalated=False)
    return CascadeDecision(label=slow.label, cost=fast.cost + slow.cost, escalated=True)


def three_tier(
    fast: ItemVerdict,
    slow: ItemVerdict,
    t_hi: float,
    t_lo: float,
    truth: str,
    human_cost: float,
) -> CascadeDecision:
    """Route by fast confidence: keep it, escalate to slow, or send to a human."""
    if fast.confidence >= t_hi:
        return CascadeDecision(label=fast.label, cost=fast.cost, escalated=False)
    if fast.confidence >= t_lo:
        return CascadeDecision(label=slow.label, cost=fast.cost + slow.cost, escalated=True)
    return CascadeDecision(label=truth, cost=fast.cost + human_cost, escalated=True)


class G5Cascade(Gate):
    """Analyze G3 verdicts as a confidence-gated cascade over cost and accuracy."""

    gate_id = "g5"
    stage = "g5"

    def __init__(
        self,
        pricing: PricingTable,
        thresholds: Sequence[float],
        *,
        fast_judge: str = "jev",
        slow_judge: str = "llm_strong",
        human_cost_usd: float = 2.0,
        g3: G3Outcome | None = None,
    ) -> None:
        self._pricing = pricing
        self._thresholds = list(thresholds)
        self._fast_judge = fast_judge
        self._slow_judge = slow_judge
        self._human_cost = human_cost_usd
        self._g3 = g3 if g3 is not None else G3Outcome()

    @property
    def rubric_path(self) -> Path:
        """Path of the G3 rubric this gate's verdicts share."""
        return self._g3.rubric_path

    @property
    def prompt_version(self) -> str:
        """Prompt version of the G3 verdicts this gate consumes."""
        return self._g3.prompt_version

    def questions(self) -> list[Question]:
        """Return the G3 questions, since this gate consumes G3 verdicts."""
        return self._g3.questions()

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Delegate item construction to G3."""
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
        """Refuse to run: G5 analyzes existing G3 verdicts and makes no calls."""
        raise NotImplementedError("G5 analyzes G3 verdicts and makes no judge calls")

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Reduce verdicts into frontier, reference, three-tier, and coverage tables."""
        grouped: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            grouped[(verdict.state_hash, verdict.judge_id)].append(verdict)

        covered: list[tuple[str, ItemVerdict, ItemVerdict, str]] = []
        uncovered = 0
        for item in items:
            state_hash = item.state.state_hash
            fast_verdicts = grouped.get((state_hash, self._fast_judge))
            slow_verdicts = grouped.get((state_hash, self._slow_judge))
            if not fast_verdicts or not slow_verdicts or item.truth_label is None:
                uncovered += 1
                continue
            fast = self._item_verdict(fast_verdicts)
            slow = self._item_verdict(slow_verdicts)
            covered.append((state_hash, fast, slow, item.truth_label))

        frontier = pd.DataFrame([self._frontier_row(t, covered) for t in self._thresholds])
        reference = pd.DataFrame(self._reference_rows(covered, grouped))
        three = pd.DataFrame([self._three_tier_row(t, covered) for t in self._thresholds])
        coverage = pd.DataFrame([{"n_covered": len(covered), "n_uncovered": uncovered}])
        return GateResult(
            tables={
                "g5_frontier": frontier,
                "g5_reference": reference,
                "g5_three_tier": three,
                "g5_coverage": coverage,
            },
            charts={"g5_frontier": self._frontier_chart(frontier, reference)},
            findings=self._findings(frontier, reference),
        )

    def _item_verdict(self, verdicts: Sequence[Verdict]) -> ItemVerdict:
        """Build the aggregate verdict and unit cost for one item and judge."""
        label, confidence = modal_verdict(verdicts)
        cost = cost_of(verdicts, self._pricing)
        return ItemVerdict(label=label, confidence=confidence, cost=cost)

    def _frontier_row(
        self,
        t: float,
        covered: Sequence[tuple[str, ItemVerdict, ItemVerdict, str]],
    ) -> dict[str, object]:
        """Return one cascade frontier row at threshold t over covered items."""
        labels: list[str] = []
        truths: list[str] = []
        costs: list[float] = []
        escalated = 0
        for _, fast, slow, truth in covered:
            decision = cascade(fast, slow, t)
            labels.append(decision.label)
            truths.append(truth)
            costs.append(decision.cost)
            escalated += int(decision.escalated)
        n = len(covered)
        return {
            "t": t,
            "accuracy": accuracy(labels, truths) if labels else float("nan"),
            "cost_per_item": sum(costs) / n if n else float("nan"),
            "escalation_rate": escalated / n if n else float("nan"),
            "n": n,
        }

    def _reference_rows(
        self,
        covered: Sequence[tuple[str, ItemVerdict, ItemVerdict, str]],
        grouped: Mapping[tuple[str, str], list[Verdict]],
    ) -> list[dict[str, object]]:
        """Return accuracy and cost for the fast, slow, and cheap reference judges."""
        rows: list[dict[str, object]] = []
        for judge in (self._fast_judge, self._slow_judge, _CHEAP_JUDGE):
            labels: list[str] = []
            truths: list[str] = []
            costs: list[float] = []
            for state_hash, _, _, truth in covered:
                judge_verdicts = grouped.get((state_hash, judge))
                if not judge_verdicts:
                    continue
                aggregate = self._item_verdict(judge_verdicts)
                labels.append(aggregate.label)
                truths.append(truth)
                costs.append(aggregate.cost)
            if not labels:
                continue
            rows.append(
                {
                    "judge": judge,
                    "accuracy": accuracy(labels, truths),
                    "cost_per_item": sum(costs) / len(costs),
                }
            )
        return rows

    def _three_tier_row(
        self,
        t_hi: float,
        covered: Sequence[tuple[str, ItemVerdict, ItemVerdict, str]],
    ) -> dict[str, object]:
        """Return one three-tier row at t_hi and its floored low threshold."""
        t_lo = max(t_hi - _THREE_TIER_GAP, _THREE_TIER_FLOOR)
        labels: list[str] = []
        truths: list[str] = []
        costs: list[float] = []
        human = 0
        for _, fast, slow, truth in covered:
            decision = three_tier(fast, slow, t_hi, t_lo, truth, self._human_cost)
            labels.append(decision.label)
            truths.append(truth)
            costs.append(decision.cost)
            if fast.confidence < t_lo:
                human += 1
        n = len(covered)
        return {
            "t_hi": t_hi,
            "t_lo": t_lo,
            "accuracy": accuracy(labels, truths) if labels else float("nan"),
            "cost_per_item": sum(costs) / n if n else float("nan"),
            "human_rate": human / n if n else float("nan"),
            "n": n,
        }

    def _frontier_chart(self, frontier: pd.DataFrame, reference: pd.DataFrame) -> Figure:
        """Return a cost-accuracy figure with the cascade frontier and reference markers."""
        figure = Figure()
        axes = figure.subplots()
        if not frontier.empty:
            ordered = frontier.sort_values("cost_per_item")
            axes.plot(
                ordered["cost_per_item"].tolist(),
                ordered["accuracy"].tolist(),
                marker="o",
                label="cascade",
            )
            for _, row in frontier.iterrows():
                axes.annotate(
                    f"{float(row['t']):.2f}",
                    (float(row["cost_per_item"]), float(row["accuracy"])),
                )
        for _, row in reference.iterrows():
            axes.scatter(
                [float(row["cost_per_item"])],
                [float(row["accuracy"])],
                marker="s",
                label=str(row["judge"]),
            )
        axes.set_xlabel("cost per item (USD)")
        axes.set_ylabel("accuracy")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G5 cascade cost-accuracy frontier")
        axes.legend()
        return figure

    def _findings(self, frontier: pd.DataFrame, reference: pd.DataFrame) -> str:
        """Return three plain paragraphs: what a cascade is, the numbers, the choice."""
        if frontier.empty:
            return "No conversations were available to analyze."
        priced = frontier[frontier["cost_per_item"] > 0]
        if priced.empty:
            return "Every cascade point cost nothing, so cost cannot be compared."
        priced = priced.assign(value=priced["accuracy"] / priced["cost_per_item"])
        best = priced.sort_values("value", ascending=False).iloc[0]
        strong = judge_name(self._slow_judge)

        what = (
            "A cascade asks the cheap judge first and only pays for "
            f"{strong} when the cheap judge is unsure."
        )
        best_acc = float(best["accuracy"])
        best_cost = float(best["cost_per_item"])
        share = as_percent(float(best["escalation_rate"]))
        numbers = [
            f"The best cascade sets its confidence bar at {as_percent(float(best['t']))}, "
            f"reaches {as_percent(best_acc)} accuracy at {as_money(best_cost)} per conversation, "
            f"and sends {share} of conversations on to {strong}."
        ]
        numbers.append(self._compare_to_strong(reference, best_acc, best_cost, strong))
        numbers.append(self._compare_to_cheap(reference))
        meaning = self._cascade_meaning(reference, best_acc)
        body = " ".join(part for part in numbers if part)
        return "\n\n".join([what, body, meaning])

    def _compare_to_strong(
        self, reference: pd.DataFrame, best_acc: float, best_cost: float, strong: str
    ) -> str:
        """Return a sentence comparing the cascade with sending everything to the strong model."""
        slow = reference[reference["judge"] == self._slow_judge]
        if slow.empty:
            return ""
        s_acc = float(slow.iloc[0]["accuracy"])
        s_cost = float(slow.iloc[0]["cost_per_item"])
        saving = (s_cost - best_cost) / s_cost if s_cost > 0 else 0.0
        direction = "more" if best_acc >= s_acc else "less"
        return (
            f"Sending every conversation to {strong} reaches {as_percent(s_acc)} accuracy at "
            f"{as_money(s_cost)} per conversation, so the cascade costs {as_percent(saving)} less "
            f"while scoring about {as_points(best_acc - s_acc)} {direction}."
        )

    def _compare_to_cheap(self, reference: pd.DataFrame) -> str:
        """Return a sentence on the cheap judge alone when it is present."""
        cheap = reference[reference["judge"] == _CHEAP_JUDGE]
        if cheap.empty:
            return ""
        c_acc = float(cheap.iloc[0]["accuracy"])
        c_cost = float(cheap.iloc[0]["cost_per_item"])
        return (
            f"{judge_name(_CHEAP_JUDGE).capitalize()} on its own reaches {as_percent(c_acc)} "
            f"accuracy at {as_money(c_cost)} per conversation."
        )

    def _cascade_meaning(self, reference: pd.DataFrame, best_acc: float) -> str:
        """Return an honest sentence on whether the cascade is worth it here."""
        slow = reference[reference["judge"] == self._slow_judge]
        cheap = reference[reference["judge"] == _CHEAP_JUDGE]
        strong = judge_name(self._slow_judge)
        if slow.empty:
            return (
                "For someone choosing a judge, the cascade trades some accuracy for a lower cost "
                "per conversation."
            )
        s_acc = float(slow.iloc[0]["accuracy"])
        if not cheap.empty and float(cheap.iloc[0]["accuracy"]) >= s_acc - 0.02:
            c_cost = float(cheap.iloc[0]["cost_per_item"])
            return (
                f"For someone choosing a judge, paying for {strong} buys little here: "
                f"{judge_name(_CHEAP_JUDGE)} on its own is already about as accurate at "
                f"{as_money(c_cost)} per conversation, so the cheaper judge is the better default."
            )
        if best_acc >= s_acc - 0.02:
            return (
                "For someone choosing a judge, the cascade is worth it here: it stays within "
                f"{as_points(best_acc - s_acc)} of {strong} while costing far less per "
                "conversation."
            )
        return (
            "For someone choosing a judge, the cascade saves money but gives up more than a "
            f"couple of points against {strong}, so it is worth it only when cost matters most."
        )
