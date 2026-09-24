"""The G3 outcome gate: did the agent complete the request in line with policy?"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.judges.llm import prompt_version_for, read_rubric
from decision_judges.metrics import (
    accuracy,
    cohens_kappa,
    latency_summary,
    modal_agreement,
    precision_recall_f1,
)
from decision_judges.serialize import StateProfile, serialize
from decision_judges.types import Question, QuestionKind, Verdict

_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "config" / "rubrics" / "g3_outcome.md"
_COMPLETED_TEXT = "The agent completed the user request in line with the policy."
_PASS = "pass"
_FAIL = "fail"


def _mode(choices: Sequence[str]) -> str:
    """Return the most frequent choice, resolving ties by sort order."""
    return Counter(sorted(choices)).most_common(1)[0][0]


def _verdict_choice(verdict: Verdict) -> str | None:
    """Return the verdict question's chosen option, or None if absent."""
    for answer in verdict.answers:
        if answer.question_id == "verdict":
            return answer.choice
    return None


class G3Outcome(Gate):
    """Judge whether each run met the outcome bar stated in the G3 rubric."""

    gate_id = "g3"
    stage = "g3"

    def __init__(self, rubric_path: Path = _DEFAULT_RUBRIC_PATH) -> None:
        self._rubric_path = rubric_path
        self._questions: list[Question] | None = None
        self._prompt_version: str | None = None

    def questions(self) -> list[Question]:
        """Return the completed-noul and pass/fail-choice questions, loading the rubric once."""
        if self._questions is None:
            body, version = read_rubric(self._rubric_path)
            self._questions = [
                Question(id="completed", kind=QuestionKind.noul, text=_COMPLETED_TEXT),
                Question(
                    id="verdict",
                    kind=QuestionKind.choice,
                    text=body,
                    options=[_PASS, _FAIL],
                ),
            ]
            self._prompt_version = prompt_version_for(version, self._questions)
        return self._questions

    @property
    def prompt_version(self) -> str:
        """Return the prompt version judges share, derived from rubric and questions."""
        if self._prompt_version is None:
            self.questions()
        assert self._prompt_version is not None
        return self._prompt_version

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Serialize each non-excluded run and label it pass at reward >= 1.0 else fail."""
        items: list[Item] = []
        for record in records.values():
            if record.excluded:
                continue
            state = serialize(record, tasks[record.task_id], profile)
            label = _PASS if record.reward >= 1.0 else _FAIL
            items.append(Item(state=state, truth_label=label, truth_value=record.reward))
        return items

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Summarize per judge and profile, returning a table, an accuracy chart, and findings.

        Error verdicts are excluded from accuracy, kappa, F1, and repeatability
        but counted in the error rate. USD cost lives in the spend ledger, so
        this table reports mean input tokens and latency percentiles instead.
        """
        by_hash = {item.state.state_hash: item for item in items}
        groups: dict[tuple[str, StateProfile], list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            item = by_hash.get(verdict.state_hash)
            if item is not None:
                groups[(verdict.judge_id, item.state.profile)].append(verdict)

        rows = [
            self._summarize(judge_id, profile, groups[(judge_id, profile)], by_hash)
            for judge_id, profile in sorted(groups)
        ]
        frame = pd.DataFrame(rows)
        return GateResult(
            tables={"g3_summary": frame},
            charts={"g3_accuracy": self._accuracy_chart(frame)},
            findings=self._findings(frame),
        )

    def _summarize(
        self,
        judge_id: str,
        profile: StateProfile,
        group: Sequence[Verdict],
        by_hash: Mapping[str, Item],
    ) -> dict[str, object]:
        """Reduce one judge-and-profile group of verdicts into a summary row."""
        per_item: dict[str, list[str]] = defaultdict(list)
        for verdict in group:
            if verdict.error is not None:
                continue
            choice = _verdict_choice(verdict)
            if choice is not None:
                per_item[verdict.state_hash].append(choice)

        modal: list[str] = []
        truth: list[str] = []
        repeats: list[list[str]] = []
        for state_hash, choices in per_item.items():
            label = by_hash[state_hash].truth_label
            if not choices or label is None:
                continue
            modal.append(_mode(choices))
            truth.append(label)
            repeats.append(choices)

        total = len(group)
        errors = sum(1 for verdict in group if verdict.error is not None)
        latency = latency_summary([float(verdict.latency_ms) for verdict in group])
        return {
            "judge_id": judge_id,
            "profile": profile.value,
            "n_items": len(modal),
            "accuracy": accuracy(modal, truth) if modal else float("nan"),
            "kappa": cohens_kappa(modal, truth) if modal else float("nan"),
            "f1_fail": precision_recall_f1(modal, truth, _FAIL).f1 if modal else float("nan"),
            "modal_agreement": modal_agreement(repeats).mean if repeats else float("nan"),
            "error_rate": errors / total if total else 0.0,
            "mean_input_tokens": (
                sum(verdict.usage.input_tokens for verdict in group) / total if total else 0.0
            ),
            "latency_p50": latency.p50,
            "latency_p95": latency.p95,
        }

    def _accuracy_chart(self, frame: pd.DataFrame) -> Figure:
        """Return a bar figure of accuracy per judge, built without pyplot."""
        figure = Figure()
        axes = figure.subplots()
        if not frame.empty:
            axes.bar(frame["judge_id"].tolist(), frame["accuracy"].tolist())
        axes.set_ylabel("accuracy")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G3 outcome accuracy by judge")
        return figure

    def _findings(self, frame: pd.DataFrame) -> str:
        """Return two to three factual sentences on the best judge and repeatability."""
        if frame.empty:
            return "No verdicts were available to analyze."
        best = frame.sort_values("accuracy", ascending=False).iloc[0]
        return (
            f"Judge {str(best['judge_id'])!r} had the highest outcome accuracy at "
            f"{float(best['accuracy']):.2f}. Modal agreement across repeats ranged from "
            f"{float(frame['modal_agreement'].min()):.2f} to "
            f"{float(frame['modal_agreement'].max()):.2f}. USD cost per verdict is recorded "
            f"in the spend ledger, not in this table."
        )
