"""The G9 taxonomy gate: classify a failed run and score it against hand labels."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.judges.llm import prompt_version_for, read_rubric
from decision_judges.labels import TAXONOMY, LabelStore
from decision_judges.metrics import accuracy, cohens_kappa
from decision_judges.serialize import StateProfile, serialize
from decision_judges.types import Question, QuestionKind, Verdict

_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "config" / "rubrics" / "g9_taxonomy.md"
_FAILURE_TYPE = "failure_type"
_FAILURE_TYPE_TEXT = (
    "Which failure type best explains why this conversation did not complete the user request?"
)
_SUMMARY_COLUMNS = ["judge_id", "n", "accuracy", "kappa", "error_rate"]
_CONFUSION_COLUMNS = ["judge", "truth", "predicted", "count"]


def labels_from_store(store: LabelStore) -> dict[tuple[str, str], str]:
    """Return the newest hand label per ``(variant, task_id)`` as plain strings."""
    return {key: label.label for key, label in store.latest().items()}


def _failure_choice(verdict: Verdict) -> str | None:
    """Return the failure-type question's chosen option, or None if absent."""
    for answer in verdict.answers:
        if answer.question_id == _FAILURE_TYPE:
            return answer.choice
    return None


def _mode(choices: Sequence[str]) -> str:
    """Return the most frequent choice, resolving ties by sort order."""
    return Counter(sorted(choices)).most_common(1)[0][0]


class G9Taxonomy(Gate):
    """Classify each failing run's failure type and score it against hand labels."""

    gate_id = "g9"
    stage = "g9"

    def __init__(
        self,
        labels: Mapping[tuple[str, str], str] | None = None,
        rubric_path: Path = _DEFAULT_RUBRIC_PATH,
    ) -> None:
        self._labels = dict(labels) if labels else {}
        self._rubric_path = rubric_path
        self._questions: list[Question] | None = None
        self._prompt_version: str | None = None

    def questions(self) -> list[Question]:
        """Return the single failure-type choice question, loading the rubric once."""
        if self._questions is None:
            _, version = read_rubric(self._rubric_path)
            self._questions = [
                Question(
                    id=_FAILURE_TYPE,
                    kind=QuestionKind.choice,
                    text=_FAILURE_TYPE_TEXT,
                    options=list(TAXONOMY),
                )
            ]
            self._prompt_version = prompt_version_for(version, self._questions)
        return self._questions

    @property
    def rubric_path(self) -> Path:
        """Path of the rubric file this gate loads."""
        return self._rubric_path

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
        """Serialize each labeled run and attach its hand label as the truth.

        Only records whose ``(variant, task_id)`` carries a hand label are
        scorable, so unlabeled records are skipped. An empty label set yields no
        items.
        """
        items: list[Item] = []
        if not self._labels:
            return items
        for record in records.values():
            label = self._labels.get((record.variant, record.task_id))
            if label is None:
                continue
            state = serialize(record, tasks[record.task_id], profile)
            items.append(Item(state=state, truth_label=label))
        return items

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Score modal failure-type against the hand labels per judge.

        The labels come from one owner, so the findings mark the scores as
        single-annotator. Error verdicts are excluded from accuracy and kappa
        but counted in the error rate.
        """
        by_hash = {item.state.state_hash: item for item in items}
        if not by_hash:
            return self._empty_result()

        groups: dict[str, list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            if verdict.state_hash in by_hash:
                groups[verdict.judge_id].append(verdict)

        summary_rows: list[dict[str, object]] = []
        confusion_rows: list[dict[str, object]] = []
        for judge_id in sorted(groups):
            row, cells = self._summarize(judge_id, groups[judge_id], by_hash)
            summary_rows.append(row)
            confusion_rows.extend(cells)

        summary = pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)
        confusion = pd.DataFrame(confusion_rows, columns=_CONFUSION_COLUMNS)
        return GateResult(
            tables={"g9_summary": summary, "g9_confusion": confusion},
            charts={"g9_accuracy": self._accuracy_chart(summary)},
            findings=self._findings(summary, confusion),
        )

    def _summarize(
        self,
        judge_id: str,
        group: Sequence[Verdict],
        by_hash: Mapping[str, Item],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Reduce one judge's verdicts into a summary row and its confusion cells."""
        per_item: dict[str, list[str]] = defaultdict(list)
        for verdict in group:
            if verdict.error is not None:
                continue
            choice = _failure_choice(verdict)
            if choice is not None:
                per_item[verdict.state_hash].append(choice)

        modal: list[str] = []
        truth: list[str] = []
        confusion: Counter[tuple[str, str]] = Counter()
        for state_hash, choices in per_item.items():
            label = by_hash[state_hash].truth_label
            if not choices or label is None:
                continue
            predicted = _mode(choices)
            modal.append(predicted)
            truth.append(label)
            confusion[(label, predicted)] += 1

        total = len(group)
        errors = sum(1 for verdict in group if verdict.error is not None)
        row: dict[str, object] = {
            "judge_id": judge_id,
            "n": len(modal),
            "accuracy": accuracy(modal, truth) if modal else float("nan"),
            "kappa": cohens_kappa(modal, truth) if modal else float("nan"),
            "error_rate": errors / total if total else 0.0,
        }
        cells = [
            {"judge": judge_id, "truth": label, "predicted": predicted, "count": count}
            for (label, predicted), count in sorted(confusion.items())
        ]
        return row, cells

    def _empty_result(self) -> GateResult:
        """Return an empty result naming the Label page when no labels exist."""
        summary = pd.DataFrame(columns=_SUMMARY_COLUMNS)
        confusion = pd.DataFrame(columns=_CONFUSION_COLUMNS)
        return GateResult(
            tables={"g9_summary": summary, "g9_confusion": confusion},
            charts={"g9_accuracy": self._accuracy_chart(summary)},
            findings=(
                "The label store has no failure labels yet, so this gate has nothing to "
                "score. Label failing trajectories on the Label page first."
            ),
        )

    def _accuracy_chart(self, summary: pd.DataFrame) -> Figure:
        """Return a bar figure of accuracy per judge, built without pyplot."""
        figure = Figure()
        axes = figure.subplots()
        if not summary.empty:
            axes.bar(summary["judge_id"].tolist(), summary["accuracy"].tolist())
        axes.set_ylabel("accuracy")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G9 taxonomy accuracy by judge")
        return figure

    def _findings(self, summary: pd.DataFrame, confusion: pd.DataFrame) -> str:
        """Return three factual sentences opening with the single-annotator caveat."""
        if summary.empty:
            return "Single annotator: no verdicts were available to score against the hand labels."
        best = summary.sort_values("accuracy", ascending=False).iloc[0]
        return (
            "Single annotator: these labels come from one owner, so treat the scores as "
            f"indicative. Judge {str(best['judge_id'])!r} classified failures most accurately "
            f"at {float(best['accuracy']):.2f}. {self._most_confused(confusion)}"
        )

    def _most_confused(self, confusion: pd.DataFrame) -> str:
        """Name the truth-and-predicted pair judges confused most, or note none."""
        if confusion.empty:
            return "No judge confused one failure type for another."
        errors = confusion[confusion["truth"] != confusion["predicted"]]
        if errors.empty:
            return "No judge confused one failure type for another."
        grouped = errors.groupby(["truth", "predicted"])["count"].sum().reset_index()
        top = grouped.sort_values(
            ["count", "truth", "predicted"], ascending=[False, True, True]
        ).iloc[0]
        return (
            f"The most confused pair was truth {str(top['truth'])!r} predicted as "
            f"{str(top['predicted'])!r}."
        )
