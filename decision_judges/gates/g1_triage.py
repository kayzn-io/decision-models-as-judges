"""The G1 triage gate: how hard is a task, and which category, before any run?

A judge sees only the task instruction and the policy summary, never a
trajectory, and answers a difficulty score and a category. The analysis is
exploratory: it correlates predicted difficulty with the task's expected-action
count and its write-action count, and asks whether difficulty predicts run
failure.
"""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task, write_actions_for
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.judges.llm import prompt_version_for, read_rubric
from decision_judges.metrics import auroc, spearman
from decision_judges.serialize import (
    POLICY_SUMMARY,
    Injection,
    LeakageError,
    StateProfile,
    StateRecord,
)
from decision_judges.types import Question, QuestionKind, Verdict

_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "config" / "rubrics" / "g1_triage.md"

_DIFFICULTY = "difficulty"
_CATEGORY = "category"
_DIFFICULTY_TEXT = (
    "How hard is this task for a support agent to complete correctly under the policy?"
)
_CATEGORY_TEXT = "Which category best describes the task?"
_LEVELS = ["trivial", "routine", "multi_step", "hard"]
_CATEGORIES = [
    "single_lookup",
    "single_write",
    "multiple_writes",
    "cancellation_or_return",
    "ambiguous",
]

_PASS = "pass"
_FAIL = "fail"
_NAN = float("nan")

_SUMMARY_COLUMNS = [
    "judge_id",
    "variant",
    "n",
    "spearman_actions",
    "spearman_writes",
    "auroc_fail",
]


def _guard_task_text(text: str, task: Task) -> None:
    """Raise LeakageError if triage text carries a task's required output string.

    The triage state holds only the instruction and policy, so the only ground
    truth that could slip in is a graded answer; this mirrors the outputs check
    the trajectory leakage guard applies.
    """
    lowered = text.lower()
    for output in task.outputs:
        if output and output.lower() in lowered:
            raise LeakageError("leaked required output string")


class G1Triage(Gate):
    """Triage each task's difficulty and category before any agent runs."""

    gate_id = "g1"
    stage = "g1"

    def __init__(
        self,
        rubric_path: Path = _DEFAULT_RUBRIC_PATH,
        *,
        tasks: Mapping[str, Task] | None = None,
    ) -> None:
        self._rubric_path = rubric_path
        self._tasks = tasks
        self._questions: list[Question] | None = None
        self._prompt_version: str | None = None

    def questions(self) -> list[Question]:
        """Return the difficulty-score and category-choice questions, loading the rubric once."""
        if self._questions is None:
            _, version = read_rubric(self._rubric_path)
            self._questions = [
                Question(
                    id=_DIFFICULTY,
                    kind=QuestionKind.score,
                    text=_DIFFICULTY_TEXT,
                    levels=list(_LEVELS),
                ),
                Question(
                    id=_CATEGORY,
                    kind=QuestionKind.choice,
                    text=_CATEGORY_TEXT,
                    options=list(_CATEGORIES),
                ),
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

    def task_state(self, task: Task, variant: str) -> StateRecord:
        """Serialize a task into judge-visible triage text: instruction then policy.

        The text carries no trajectory and no ground truth; the leakage guard
        rejects any required output string that would appear in the instruction.
        """
        text = f"Task: {task.instruction}\n\n{POLICY_SUMMARY}"
        _guard_task_text(text, task)
        return StateRecord(
            variant=variant,
            task_id=task.task_id,
            profile=StateProfile.full,
            injection=Injection.none,
            text=text,
            token_estimate=len(text) // 4,
            state_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Build one triage item per non-excluded record, keyed to its task and reward.

        The truth value is the task's expected-action count and the truth label
        is the run's pass/fail outcome, so the analysis can correlate predicted
        difficulty with both.
        """
        items: list[Item] = []
        for record in records.values():
            if record.excluded:
                continue
            task = tasks[record.task_id]
            state = self.task_state(task, record.variant)
            label = _PASS if record.reward >= 1.0 else _FAIL
            items.append(Item(state=state, truth_label=label, truth_value=float(len(task.actions))))
        return items

    def _write_counts(self) -> dict[str, int]:
        """Return each known task's write-action count, or empty without tasks."""
        if self._tasks is None:
            return {}
        return {task_id: len(write_actions_for(task)) for task_id, task in self._tasks.items()}

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Correlate predicted difficulty with action counts and run failure.

        Per judge and variant, this reports the Spearman correlation of mean
        difficulty with the expected-action count and the write-action count and
        the AUROC of difficulty against run failure, alongside the category
        distribution. It is exploratory and the findings say so.
        """
        by_hash = {item.state.state_hash: item for item in items}
        write_counts = self._write_counts()
        groups: dict[tuple[str, str], list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            item = by_hash.get(verdict.state_hash)
            if item is not None:
                groups[(verdict.judge_id, item.state.variant)].append(verdict)

        summary_rows: list[dict[str, object]] = []
        category_rows: list[dict[str, object]] = []
        points: list[tuple[float, float]] = []
        for judge_id, variant in sorted(groups):
            row, cats, group_points = self._summarize(
                judge_id, variant, groups[(judge_id, variant)], by_hash, write_counts
            )
            summary_rows.append(row)
            category_rows.extend(cats)
            points.extend(group_points)

        summary = pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)
        categories = pd.DataFrame(
            category_rows, columns=["judge_id", "variant", "category", "count"]
        )
        return GateResult(
            tables={"g1_summary": summary, "g1_categories": categories},
            charts={"g1_difficulty_vs_actions": self._scatter(points)},
            findings=self._findings(summary),
        )

    def _summarize(
        self,
        judge_id: str,
        variant: str,
        group: Sequence[Verdict],
        by_hash: Mapping[str, Item],
        write_counts: Mapping[str, int],
    ) -> tuple[dict[str, object], list[dict[str, object]], list[tuple[float, float]]]:
        """Reduce one judge-and-variant group into a summary row, category rows, and points."""
        difficulties: dict[str, list[float]] = defaultdict(list)
        choices: dict[str, list[str]] = defaultdict(list)
        for verdict in group:
            if verdict.error is not None:
                continue
            for answer in verdict.answers:
                if answer.question_id == _DIFFICULTY and answer.score is not None:
                    difficulties[verdict.state_hash].append(answer.score)
                elif answer.question_id == _CATEGORY and answer.choice is not None:
                    choices[verdict.state_hash].append(answer.choice)

        diff: list[float] = []
        expected: list[float] = []
        writes: list[float] = []
        fail: list[float] = []
        category_counter: Counter[str] = Counter()
        points: list[tuple[float, float]] = []
        for state_hash, scores in difficulties.items():
            item = by_hash.get(state_hash)
            if item is None or item.truth_value is None:
                continue
            mean_difficulty = sum(scores) / len(scores)
            diff.append(mean_difficulty)
            expected.append(item.truth_value)
            points.append((item.truth_value, mean_difficulty))
            fail.append(1.0 if item.truth_label == _FAIL else 0.0)
            task_writes = write_counts.get(item.state.task_id)
            writes.append(float(task_writes) if task_writes is not None else _NAN)
            picks = choices.get(state_hash)
            if picks:
                category_counter[Counter(sorted(picks)).most_common(1)[0][0]] += 1

        has_writes = write_counts and not any(value != value for value in writes)
        row: dict[str, object] = {
            "judge_id": judge_id,
            "variant": variant,
            "n": len(diff),
            "spearman_actions": spearman(diff, expected) if len(diff) > 1 else _NAN,
            "spearman_writes": (spearman(diff, writes) if has_writes and len(diff) > 1 else _NAN),
            "auroc_fail": auroc(diff, fail) if diff else _NAN,
        }
        category_rows = [
            {"judge_id": judge_id, "variant": variant, "category": category, "count": count}
            for category, count in sorted(category_counter.items())
        ]
        return row, category_rows, points

    def _scatter(self, points: Sequence[tuple[float, float]]) -> Figure:
        """Return a jittered scatter of expected-action count against difficulty."""
        figure = Figure()
        axes = figure.subplots()
        if points:
            rng = _Jitter()
            xs = [x + rng.next() for x, _ in points]
            ys = [y + rng.next() for _, y in points]
            axes.scatter(xs, ys, alpha=0.6)
        axes.set_xlabel("expected action count")
        axes.set_ylabel("predicted difficulty")
        axes.set_title("G1 predicted difficulty vs expected actions")
        return figure

    def _findings(self, summary: pd.DataFrame) -> str:
        """Return two to three factual sentences opening with the exploratory caveat."""
        if summary.empty:
            return "This gate is exploratory: no verdicts were available to analyze."
        actions = float(summary["spearman_actions"].mean())
        writes = float(summary["spearman_writes"].mean())
        auroc_fail = float(summary["auroc_fail"].mean())
        return (
            "This gate is exploratory: it correlates a pre-run difficulty guess with task "
            "structure and outcome. Predicted difficulty correlated with the expected-action "
            f"count at a mean Spearman of {actions:.2f} and with the write-action count at "
            f"{writes:.2f}. Difficulty separated failing from passing runs with a mean AUROC "
            f"of {auroc_fail:.2f}."
        )


class _Jitter:
    """A tiny deterministic jitter so overlapping points spread without pyplot randomness."""

    def __init__(self, step: float = 0.05) -> None:
        self._step = step
        self._n = 0

    def next(self) -> float:
        """Return the next small offset, alternating sign around zero."""
        self._n += 1
        return self._step * (1 if self._n % 2 else -1) * ((self._n % 5) / 5.0)
