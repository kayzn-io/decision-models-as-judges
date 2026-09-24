"""Tests for the G10 local decision model gate."""

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.gates.g10_local_model import G10LocalModel
from decision_judges.pipeline import gate_registry
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Verdict

# --- fixtures / builders ---------------------------------------------------


def _state(state_hash: str) -> StateRecord:
    """Build a minimal compact state with a chosen hash."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.compact,
        text="transcript",
        token_estimate=50,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str) -> Item:
    """Pair a compact state with its ground-truth label and value."""
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _verdict(
    judge_id: str,
    state_hash: str,
    repeat: int,
    *,
    choice: str | None = None,
    pass_prob: float | None = None,
    error: str | None = None,
    extra: dict[str, str] | None = None,
    latency_ms: int = 1,
) -> Verdict:
    """Build a verdict answering the completed and verdict questions, or an error."""
    answers: list[Answer] = []
    if error is None and choice is not None:
        prob = pass_prob if pass_prob is not None else (1.0 if choice == "pass" else 0.0)
        answers = [
            Answer(question_id="completed", kind=QuestionKind.noul, noul=prob),
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice=choice,
                probabilities={"pass": prob, "fail": 1.0 - prob},
                confidence=max(prob, 1.0 - prob),
            ),
        ]
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=answers,
        latency_ms=latency_ms,
        error=error,
        extra=extra or {},
    )


def _record(task_id: str, reward: float, *, excluded: bool = False) -> AgentRecord:
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
        excluded=excluded,
    )


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, instruction="do the thing", actions=[], outputs=[])


# --- build_items -----------------------------------------------------------


def test_build_items_rejects_non_compact_profile() -> None:
    records = {"retail-0": _record("retail-0", 1.0)}
    tasks = {"retail-0": _task("retail-0")}

    with pytest.raises(ValueError):
        G10LocalModel().build_items(records, tasks, StateProfile.full)


def test_build_items_forces_compact_and_maps_truth() -> None:
    records = {
        "retail-0": _record("retail-0", 1.0),
        "retail-1": _record("retail-1", 0.0),
        "retail-2": _record("retail-2", 1.0, excluded=True),
    }
    tasks = {tid: _task(tid) for tid in records}

    items = G10LocalModel().build_items(records, tasks, StateProfile.compact)

    assert len(items) == 2
    assert all(item.state.profile is StateProfile.compact for item in items)
    by_task = {item.state.task_id: item for item in items}
    assert by_task["retail-0"].truth_label == "pass"
    assert by_task["retail-1"].truth_label == "fail"


# --- delegation ------------------------------------------------------------


def test_questions_and_prompt_version_match_g3() -> None:
    gate = G10LocalModel()
    g3 = G3Outcome()

    assert [q.id for q in gate.questions()] == [q.id for q in g3.questions()]
    assert gate.prompt_version == g3.prompt_version
    assert gate.rubric_path == g3.rubric_path


# --- analyze ---------------------------------------------------------------


def _analysis_fixture() -> tuple[list[Verdict], list[Item]]:
    """Four items and verdicts from laya_base, laya_ft, jev, and llm_cheap."""
    hashes = ["h0", "h1", "h2", "h3"]
    truths = ["pass", "pass", "fail", "fail"]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]

    verdicts: list[Verdict] = []

    # laya_base: near-random, accuracy 0.5, AUROC 0.5, plus one error verdict.
    base_extra = {"checkpoint_hash": "base-ck", "temperature": "1.0"}
    base = [("pass", 0.6), ("fail", 0.4), ("pass", 0.55), ("fail", 0.45)]
    for h, (choice, prob) in zip(hashes, base, strict=True):
        verdicts.append(
            _verdict("laya_base", h, 0, choice=choice, pass_prob=prob, extra=base_extra)
        )
    verdicts.append(_verdict("laya_base", "h0", 1, error="boom", extra=base_extra))

    # laya_ft: accurate, AUROC 1.0, two checkpoint folds.
    ft = [
        ("h0", "pass", 0.9, "ckpt-A"),
        ("h1", "pass", 0.8, "ckpt-A"),
        ("h2", "fail", 0.1, "ckpt-B"),
        ("h3", "fail", 0.2, "ckpt-B"),
    ]
    for h, choice, prob, ckpt in ft:
        verdicts.append(
            _verdict(
                "laya_ft",
                h,
                0,
                choice=choice,
                pass_prob=prob,
                extra={"checkpoint_hash": ckpt, "temperature": "0.7"},
            )
        )

    # jev: overconfident, accuracy 0.75, so temperature fitting lowers ECE.
    jev = [("pass", 0.99), ("pass", 0.99), ("fail", 0.01), ("pass", 0.99)]
    for h, (choice, prob) in zip(hashes, jev, strict=True):
        verdicts.append(_verdict("jev", h, 0, choice=choice, pass_prob=prob))

    # llm_cheap: accuracy 0.75, moderate probabilities.
    cheap = [("pass", 0.7), ("pass", 0.7), ("fail", 0.3), ("pass", 0.6)]
    for h, (choice, prob) in zip(hashes, cheap, strict=True):
        verdicts.append(_verdict("llm_cheap", h, 0, choice=choice, pass_prob=prob))

    return verdicts, items


def test_analyze_summary_rows_and_local_flag() -> None:
    verdicts, items = _analysis_fixture()

    result = G10LocalModel().analyze(verdicts, items)

    assert isinstance(result, GateResult)
    frame = result.tables["g10_summary"]
    assert len(frame) == 4

    by_judge = {row["judge_id"]: row for _, row in frame.iterrows()}
    assert bool(by_judge["laya_base"]["local"]) is True
    assert bool(by_judge["laya_ft"]["local"]) is True
    assert bool(by_judge["jev"]["local"]) is False
    assert bool(by_judge["llm_cheap"]["local"]) is False

    assert by_judge["laya_base"]["checkpoint_hash"] == "base-ck"
    assert by_judge["laya_ft"]["checkpoint_hash"] != ""
    assert by_judge["jev"]["checkpoint_hash"] == ""
    assert by_judge["llm_cheap"]["checkpoint_hash"] == ""
    assert by_judge["laya_base"]["temperature"] == "1.0"


def test_analyze_accuracy_auroc_and_errors() -> None:
    verdicts, items = _analysis_fixture()

    frame = G10LocalModel().analyze(verdicts, items).tables["g10_summary"]
    by_judge = {row["judge_id"]: row for _, row in frame.iterrows()}

    assert by_judge["laya_base"]["accuracy"] == pytest.approx(0.5)
    assert by_judge["laya_ft"]["accuracy"] == pytest.approx(1.0)
    assert by_judge["laya_base"]["n_items"] == 4

    assert by_judge["laya_ft"]["auroc"] > by_judge["laya_base"]["auroc"]

    # The error verdict is excluded from scoring and counted in the error rate.
    assert by_judge["laya_base"]["error_rate"] == pytest.approx(1.0 / 5.0)


def test_analyze_temperature_fit_lowers_ece_for_overconfident_judge() -> None:
    verdicts, items = _analysis_fixture()

    frame = G10LocalModel().analyze(verdicts, items).tables["g10_summary"]
    by_judge = {row["judge_id"]: row for _, row in frame.iterrows()}

    jev = by_judge["jev"]
    assert jev["ece_after"] <= jev["ece"]
    assert jev["ece_after"] < jev["ece"]


def test_analyze_folds_and_chart_and_findings() -> None:
    verdicts, items = _analysis_fixture()

    result = G10LocalModel().analyze(verdicts, items)

    folds = result.tables["g10_folds"]
    assert len(folds) == 2
    fold_hashes = set(folds["checkpoint_hash"])
    assert fold_hashes == {"ckpt-A", "ckpt-B"}
    assert folds["n_items"].tolist() == [2, 2]
    assert folds["accuracy"].tolist() == pytest.approx([1.0, 1.0])

    assert isinstance(result.charts["g10_zero_shot_vs_finetuned"], Figure)
    assert "laya_ft" in result.findings


def test_analyze_omits_folds_without_finetuned_judge() -> None:
    hashes = ["h0", "h1"]
    truths = ["pass", "fail"]
    items = [_item(h, t) for h, t in zip(hashes, truths, strict=True)]
    verdicts = [
        _verdict("jev", "h0", 0, choice="pass", pass_prob=0.8),
        _verdict("jev", "h1", 0, choice="fail", pass_prob=0.2),
    ]

    result = G10LocalModel().analyze(verdicts, items)

    assert "g10_folds" not in result.tables


# --- registry --------------------------------------------------------------


def test_gate_registry_contains_g10() -> None:
    assert gate_registry()["g10"] is G10LocalModel
