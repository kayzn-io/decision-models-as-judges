"""The G4 decomposition gate: do six atomic questions beat one broad question?

Every judge answers six narrow yes/no questions and one overall question on the
full state. Code-side aggregators then reduce the six atomic probabilities into
a pass score, and each aggregation is scored against the broad question so the
value of decomposition is measured directly.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.judges.llm import prompt_version_for, read_rubric
from decision_judges.metrics import accuracy, auroc
from decision_judges.serialize import StateProfile
from decision_judges.types import Question, QuestionKind, Verdict

_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "config" / "rubrics" / "g4_atomic.md"
_BROAD_ID = "completed"
_ATOMIC_IDS: tuple[str, ...] = (
    "all_actions",
    "no_unrequested_writes",
    "confirmed_writes",
    "identity_verified",
    "final_states_outcome",
    "no_wrong_refusal",
)
_QUESTION_TEXTS: dict[str, str] = {
    _BROAD_ID: "The agent completed the user request in line with the policy.",
    "all_actions": "Every action the user requested was performed.",
    "no_unrequested_writes": "The agent made no order-changing action the user did not request.",
    "confirmed_writes": (
        "The agent obtained explicit confirmation before each order-changing action."
    ),
    "identity_verified": "The agent verified the user identity before accessing account details.",
    "final_states_outcome": "The final message states the outcome of the request.",
    "no_wrong_refusal": "The agent did not refuse a request the policy permits.",
}
_PASS = "pass"


class Aggregator(Protocol):
    """A rule that reduces per-item atomic probabilities into a pass score.

    ``truth`` is supplied so supervised rules can fit; unsupervised rules
    ignore it.
    """

    name: str

    def scores(self, rows: list[dict[str, float]], truth: Sequence[float]) -> list[float]: ...


class MinAggregator:
    """Score each item by the weakest of its six atomic probabilities."""

    name = "min"

    def __init__(self, feature_ids: Sequence[str] = _ATOMIC_IDS) -> None:
        self._feature_ids = tuple(feature_ids)

    def scores(self, rows: list[dict[str, float]], truth: Sequence[float]) -> list[float]:
        """Return the minimum atomic probability per row."""
        return [min(row[f] for f in self._feature_ids) for row in rows]


class MeanAggregator:
    """Score each item by the mean of its six atomic probabilities."""

    name = "mean"

    def __init__(self, feature_ids: Sequence[str] = _ATOMIC_IDS) -> None:
        self._feature_ids = tuple(feature_ids)

    def scores(self, rows: list[dict[str, float]], truth: Sequence[float]) -> list[float]:
        """Return the mean atomic probability per row."""
        return [float(np.mean([row[f] for f in self._feature_ids])) for row in rows]


class BroadAggregator:
    """Score each item by the single broad question probability."""

    name = "broad"

    def __init__(self, broad_id: str = _BROAD_ID) -> None:
        self._broad_id = broad_id

    def scores(self, rows: list[dict[str, float]], truth: Sequence[float]) -> list[float]:
        """Return the broad-question probability per row."""
        return [row[self._broad_id] for row in rows]


class LogRegCVAggregator:
    """Score each item by out-of-fold logistic regression on the atomic probabilities.

    A model is fit on every fold but the one holding an item and used to predict
    that item, so no item is scored by a model trained on it.
    """

    name = "logreg_cv"

    def __init__(self, k: int = 5, seed: int = 0, feature_ids: Sequence[str] = _ATOMIC_IDS) -> None:
        self._k = k
        self._seed = seed
        self._feature_ids = tuple(feature_ids)

    def scores(self, rows: list[dict[str, float]], truth: Sequence[float]) -> list[float]:
        """Return out-of-fold P(pass) for every row."""
        features = np.array([[row[f] for f in self._feature_ids] for row in rows], dtype=float)
        labels = np.asarray(truth, dtype=float)
        n = len(rows)
        preds = np.zeros(n, dtype=float)
        folds = self._fold_assignment(n)
        for fold in range(self._k):
            test_idx = [i for i in range(n) if folds[i] == fold]
            train_idx = [i for i in range(n) if folds[i] != fold]
            if not test_idx:
                continue
            preds[test_idx] = self._fit_predict(features, labels, train_idx, test_idx)
        return preds.tolist()

    def _fold_assignment(self, n: int) -> np.ndarray:
        """Assign each item to one of k folds under a seeded permutation."""
        rng = np.random.default_rng(self._seed)
        folds = np.empty(n, dtype=int)
        for position, index in enumerate(rng.permutation(n)):
            folds[index] = position % self._k
        return folds

    def _fit_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: list[int],
        test_idx: list[int],
    ) -> np.ndarray:
        """Fit on the training folds and return P(pass) for the held-out items."""
        from sklearn.linear_model import LogisticRegression

        y_train = labels[train_idx]
        classes = np.unique(y_train)
        if len(classes) < 2:
            constant = float(y_train.mean()) if len(y_train) else 0.5
            return np.full(len(test_idx), constant)
        model = LogisticRegression()
        model.fit(features[train_idx], y_train)
        positive = list(model.classes_).index(1.0)
        return np.asarray(model.predict_proba(features[test_idx])[:, positive], dtype=float)


class _Rows(NamedTuple):
    """Per-item aggregator inputs, binary truth, and the count of dropped items."""

    rows: list[dict[str, float]]
    truth: list[float]
    dropped: int


class G4Decomposition(Gate):
    """Compare aggregations of six atomic questions against one broad question."""

    gate_id = "g4"
    stage = "g4"

    def __init__(self, rubric_path: Path = _DEFAULT_RUBRIC_PATH) -> None:
        self._rubric_path = rubric_path
        self._outcome = G3Outcome()
        self._questions: list[Question] | None = None
        self._prompt_version: str | None = None

    def questions(self) -> list[Question]:
        """Return the broad and six atomic noul questions, loading the rubric once."""
        if self._questions is None:
            _, version = read_rubric(self._rubric_path)
            self._questions = [
                Question(id=qid, kind=QuestionKind.noul, text=_QUESTION_TEXTS[qid])
                for qid in (_BROAD_ID, *_ATOMIC_IDS)
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
        """Build the G3 outcome items so this gate shares its state cache."""
        return self._outcome.build_items(records, tasks, profile)

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Score every aggregation against the broad question for each judge.

        Rows are built from non-error verdicts, averaging repeated answers per
        question; items missing any question are dropped and counted. AUROC and
        accuracy at a 0.5 threshold are reported per judge and aggregator.
        """
        by_hash = {item.state.state_hash: item for item in items}
        groups: dict[str, list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            if verdict.state_hash in by_hash:
                groups[verdict.judge_id].append(verdict)

        aggregators = self._aggregators()
        rows_out: list[dict[str, object]] = []
        for judge_id in sorted(groups):
            built = self._build_rows(groups[judge_id], by_hash)
            for aggregator in aggregators:
                rows_out.append(self._score_row(judge_id, aggregator, built))
        frame = pd.DataFrame(
            rows_out,
            columns=["judge_id", "aggregator", "auroc", "accuracy", "n", "dropped"],
        )
        return GateResult(
            tables={"g4_summary": frame},
            charts={"g4_auroc_by_aggregator": self._chart(frame)},
            findings=self._findings(frame),
        )

    def _aggregators(self) -> list[Aggregator]:
        """Return the aggregators scored for every judge, in a stable order."""
        return [BroadAggregator(), MinAggregator(), MeanAggregator(), LogRegCVAggregator()]

    def _build_rows(self, group: Sequence[Verdict], by_hash: Mapping[str, Item]) -> _Rows:
        """Collapse a judge's non-error verdicts into per-item probability rows."""
        per_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for verdict in group:
            if verdict.error is not None:
                continue
            for answer in verdict.answers:
                if answer.noul is not None:
                    per_item[verdict.state_hash][answer.question_id].append(answer.noul)

        rows: list[dict[str, float]] = []
        truth: list[float] = []
        dropped = 0
        for state_hash, question_probs in per_item.items():
            item = by_hash.get(state_hash)
            if item is None or item.truth_label is None:
                continue
            row = self._row_or_none(question_probs)
            if row is None:
                dropped += 1
                continue
            rows.append(row)
            truth.append(1.0 if item.truth_label == _PASS else 0.0)
        return _Rows(rows, truth, dropped)

    def _row_or_none(self, question_probs: Mapping[str, list[float]]) -> dict[str, float] | None:
        """Return a mean-per-question row, or None if any question is missing."""
        row: dict[str, float] = {}
        for qid in (_BROAD_ID, *_ATOMIC_IDS):
            probs = question_probs.get(qid)
            if not probs:
                return None
            row[qid] = float(np.mean(probs))
        return row

    def _score_row(self, judge_id: str, aggregator: Aggregator, built: _Rows) -> dict[str, object]:
        """Score one aggregator over one judge's rows into a summary row."""
        if built.rows:
            scores = aggregator.scores(built.rows, built.truth)
            predicted = [1.0 if score >= 0.5 else 0.0 for score in scores]
            au = auroc(scores, built.truth)
            acc = accuracy(predicted, built.truth)
        else:
            au = float("nan")
            acc = float("nan")
        return {
            "judge_id": judge_id,
            "aggregator": aggregator.name,
            "auroc": au,
            "accuracy": acc,
            "n": len(built.rows),
            "dropped": built.dropped,
        }

    def _chart(self, frame: pd.DataFrame) -> Figure:
        """Return grouped AUROC bars per judge and aggregator, built without pyplot."""
        figure = Figure()
        axes = figure.subplots()
        names = [aggregator.name for aggregator in self._aggregators()]
        if not frame.empty:
            judges = sorted(frame["judge_id"].unique().tolist())
            positions = np.arange(len(judges))
            width = 0.8 / len(names)
            for offset, name in enumerate(names):
                values = [self._auroc_for(frame, judge, name) for judge in judges]
                axes.bar(positions + offset * width, values, width, label=name)
            axes.set_xticks(positions + width * (len(names) - 1) / 2)
            axes.set_xticklabels(judges)
            axes.legend()
        axes.set_ylabel("auroc")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G4 AUROC by aggregation")
        return figure

    def _auroc_for(self, frame: pd.DataFrame, judge_id: str, name: str) -> float:
        """Return the AUROC for one judge and aggregator, or 0.0 if absent."""
        subset = frame[(frame["judge_id"] == judge_id) & (frame["aggregator"] == name)]
        if subset.empty:
            return 0.0
        value = float(subset["auroc"].iloc[0])
        return value if not np.isnan(value) else 0.0

    def _findings(self, frame: pd.DataFrame) -> str:
        """Return two to three factual sentences naming the winning aggregation."""
        if frame.empty:
            return "No verdicts were available to analyze."
        broad = frame[frame["aggregator"] == "broad"]
        code = frame[frame["aggregator"] != "broad"]
        means = code.groupby("aggregator")["auroc"].mean().dropna()
        if means.empty:
            return "No aggregation could be scored against the broad question."
        winner = str(means.idxmax())
        winner_auroc = float(means.max())
        broad_mean = float(broad["auroc"].mean())

        judges = sorted(frame["judge_id"].unique().tolist())
        beats = sum(1 for judge in judges if self._code_beats_broad(frame, judge))
        return (
            f"Aggregating the six atomic questions with the {winner!r} rule reached mean AUROC "
            f"{winner_auroc:.2f} against {broad_mean:.2f} for the broad 'completed' question. "
            f"A code aggregation beat the broad question for {beats} of {len(judges)} judges."
        )

    def _code_beats_broad(self, frame: pd.DataFrame, judge_id: str) -> bool:
        """Return whether any code aggregation beat the broad question for one judge."""
        per_judge = frame[frame["judge_id"] == judge_id]
        broad = per_judge[per_judge["aggregator"] == "broad"]["auroc"]
        code = per_judge[per_judge["aggregator"] != "broad"]["auroc"]
        if broad.empty or code.dropna().empty:
            return False
        return float(code.max()) > float(broad.iloc[0]) + 1e-9
