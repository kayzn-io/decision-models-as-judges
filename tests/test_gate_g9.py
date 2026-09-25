"""Tests for the G9 failure-taxonomy gate."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache
from decision_judges.config import load_pricing
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g9_taxonomy import G9Taxonomy, labels_from_store
from decision_judges.judges.base import HasStateText, build_verdict
from decision_judges.labels import TAXONOMY, Label, LabelStore
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Spend
from decision_judges.types import Answer, Question, QuestionKind, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"

_PREMATURE = "premature_end"
_WRONG_ARGS = "wrong_arguments"


# --- builders --------------------------------------------------------------


def _state(state_hash: str) -> StateRecord:
    """Build a minimal serialized state with a chosen hash."""
    return StateRecord(
        variant="baseline",
        task_id=state_hash,
        profile=StateProfile.full,
        text="transcript",
        token_estimate=10,
        state_hash=state_hash,
    )


def _record(task_id: str, reward: float) -> AgentRecord:
    """Build an agent record with a tiny valid trajectory."""
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=[
            {"role": "user", "content": "please help"},
            {"role": "assistant", "content": "done"},
        ],
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
    )


def _task(task_id: str) -> Task:
    """Build a task with no expected actions or outputs."""
    return Task(task_id=task_id, instruction="do the thing", actions=[], outputs=[])


class _Judge:
    """A judge whose failure_type choice or error per state is scripted."""

    model_id = "none"
    prompt_version = "pv"

    def __init__(self, judge_id: str) -> None:
        self.judge_id = judge_id

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:  # noqa: ARG002
        raise NotImplementedError


def _choice_answer(choice: str) -> Answer:
    """Build a failure_type choice answer with a one-hot distribution."""
    return Answer(
        question_id="failure_type",
        kind=QuestionKind.choice,
        choice=choice,
        probabilities={option: (1.0 if option == choice else 0.0) for option in TAXONOMY},
        confidence=1.0,
    )


def _verdict(
    judge_id: str,
    state: StateRecord,
    repeat: int,
    choice: str | None,
    *,
    error: str | None = None,
) -> Verdict:
    """Build a failure_type verdict, or an error verdict when choice is None."""
    judge = _Judge(judge_id)
    answers = [] if choice is None else [_choice_answer(choice)]
    return build_verdict(judge, state, repeat, answers, latency_ms=1, error=error)


# --- questions -------------------------------------------------------------


def test_questions_single_choice_with_taxonomy_in_order() -> None:
    questions = G9Taxonomy().questions()

    assert len(questions) == 1
    question = questions[0]
    assert question.id == "failure_type"
    assert question.kind is QuestionKind.choice
    assert question.options == list(TAXONOMY)
    assert "failure type" in question.text.lower()


# --- build_items -----------------------------------------------------------


def test_build_items_only_labeled_records_with_truth() -> None:
    labels = {("baseline", "retail-1"): _PREMATURE}
    records = {"retail-0": _record("retail-0", 1.0), "retail-1": _record("retail-1", 0.0)}
    tasks = {task_id: _task(task_id) for task_id in records}

    items = G9Taxonomy(labels=labels).build_items(records, tasks, StateProfile.full)

    assert len(items) == 1
    assert items[0].state.task_id == "retail-1"
    assert items[0].truth_label == _PREMATURE


def test_build_items_empty_or_none_labels_returns_empty() -> None:
    records = {"retail-1": _record("retail-1", 0.0)}
    tasks = {"retail-1": _task("retail-1")}

    assert G9Taxonomy(labels={}).build_items(records, tasks, StateProfile.full) == []
    assert G9Taxonomy().build_items(records, tasks, StateProfile.full) == []


# --- analyze ---------------------------------------------------------------


def _analyze_fixture() -> GateResult:
    """Score a good judge and an error-carrying weak judge over four labeled items."""
    hashes = ["h0", "h1", "h2", "h3"]
    truths = [_PREMATURE, _PREMATURE, _WRONG_ARGS, _WRONG_ARGS]
    items = [Item(state=_state(h), truth_label=t) for h, t in zip(hashes, truths, strict=True)]

    verdicts: list[Verdict] = []
    for state_hash, truth in zip(hashes, truths, strict=True):
        verdicts.append(_verdict("good", _state(state_hash), 0, truth))
    for state_hash in hashes:
        verdicts.append(_verdict("weak", _state(state_hash), 0, _PREMATURE))
    verdicts.append(_verdict("weak", _state("h0"), 1, None, error="boom"))

    return G9Taxonomy().analyze(verdicts, items)


def test_analyze_accuracy_and_kappa_hand_checked() -> None:
    result = _analyze_fixture()
    summary = result.tables["g9_summary"]

    good = summary[summary["judge_id"] == "good"].iloc[0]
    assert good["n"] == 4
    assert good["accuracy"] == pytest.approx(1.0)
    assert good["kappa"] == pytest.approx(1.0)
    assert good["error_rate"] == pytest.approx(0.0)

    weak = summary[summary["judge_id"] == "weak"].iloc[0]
    assert weak["n"] == 4
    assert weak["accuracy"] == pytest.approx(0.5)
    assert weak["kappa"] == pytest.approx(0.0)
    assert weak["error_rate"] == pytest.approx(0.2)


def test_analyze_confusion_sums_to_n_per_judge() -> None:
    result = _analyze_fixture()
    summary = result.tables["g9_summary"]
    confusion = result.tables["g9_confusion"]

    assert list(confusion.columns) == ["judge", "truth", "predicted", "count"]
    for judge_id in summary["judge_id"]:
        n = int(summary[summary["judge_id"] == judge_id]["n"].iloc[0])
        total = int(confusion[confusion["judge"] == judge_id]["count"].sum())
        assert total == n


def test_analyze_findings_single_annotator_and_confused_pair() -> None:
    result = _analyze_fixture()

    assert result.findings.startswith("Single annotator:")
    assert "'good'" in result.findings
    assert _WRONG_ARGS in result.findings
    assert _PREMATURE in result.findings
    assert isinstance(result.charts["g9_accuracy"], Figure)


def test_analyze_empty_items_names_label_page() -> None:
    result = G9Taxonomy().analyze([], [])

    assert "Label page" in result.findings
    assert result.tables["g9_summary"].empty
    assert result.tables["g9_confusion"].empty


# --- labels_from_store -----------------------------------------------------


def test_labels_from_store_returns_latest_strings(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(
        Label(
            variant="baseline",
            task_id="retail-1",
            label="other",
            created_at="2020-01-01T00:00:00+00:00",
        )
    )
    store.append(
        Label(
            variant="baseline",
            task_id="retail-1",
            label=_PREMATURE,
            created_at="2021-01-01T00:00:00+00:00",
        )
    )

    assert labels_from_store(store) == {("baseline", "retail-1"): _PREMATURE}


# --- run robustness --------------------------------------------------------


class _RaisingJudge:
    """A judge that rejects any question, as the code judge does off G3/G2."""

    judge_id = "raiser"
    model_id = "none"
    prompt_version = "pv"
    paid = False

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:  # noqa: ARG002
        raise ValueError("cannot answer failure_type")


def test_run_records_error_verdict_when_judge_raises(tmp_path: Path) -> None:
    items = [Item(state=_state("h0"), truth_label=_PREMATURE)]
    cache = Cache(tmp_path / "cache")
    spend = Spend(load_pricing(PRICING_FILE), {"g9": 0.0}, tmp_path / "ledger.json")

    verdicts = G9Taxonomy().run(items, [_RaisingJudge()], cache, spend, repeats=1)

    assert len(verdicts) == 1
    assert verdicts[0].judge_id == "raiser"
    assert verdicts[0].error is not None
    assert verdicts[0].answers == []
