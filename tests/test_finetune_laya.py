"""Tests for cross-validated Laya fine-tuning: labels, records, training, calibration."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from safetensors.torch import load_file

from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.training.finetune_laya import (
    Manifest,
    TrainingExample,
    build_training_examples,
    checkpoint_for_task,
    fine_tune,
    fit_temperatures,
    read_manifest,
    run_cross_validation,
    to_vendor_record,
    write_manifest,
)
from decision_judges.types import Question, QuestionKind

STUB = Path(__file__).parent / "fixtures" / "laya_stub"
GITATTRIBUTES = Path(__file__).resolve().parents[1] / ".gitattributes"

VERDICT_Q = Question(
    id="verdict", kind=QuestionKind.choice, text="Pass or fail?", options=["pass", "fail"]
)
COMPLETED_Q = Question(id="completed", kind=QuestionKind.noul, text="Did the agent complete it?")
SCORE_Q = Question(
    id="urgency", kind=QuestionKind.score, text="How urgent?", levels=["low", "high"]
)
QUESTIONS = [VERDICT_Q, COMPLETED_Q]


def _state(task_id: str, text: str, profile: StateProfile = StateProfile.compact) -> StateRecord:
    """Build a serialized state record for tests."""
    return StateRecord(
        variant="baseline",
        task_id=task_id,
        profile=profile,
        text=text,
        token_estimate=len(text) // 4,
        state_hash=f"hash-{task_id}",
    )


# --- labels -----------------------------------------------------------------


def test_build_training_examples_labels_a_pass() -> None:
    states = {"retail-0": _state("retail-0", "the agent verified the account")}
    examples = build_training_examples(states, {"retail-0": 1.0}, QUESTIONS)
    assert len(examples) == 1
    assert examples[0].labels == {"verdict": 0, "completed": 1}


def test_build_training_examples_labels_a_fail() -> None:
    states = {"retail-0": _state("retail-0", "the agent skipped verification")}
    examples = build_training_examples(states, {"retail-0": 0.0}, QUESTIONS)
    assert examples[0].labels == {"verdict": 1, "completed": 0}


def test_build_training_examples_rejects_non_compact_profile() -> None:
    states = {"retail-0": _state("retail-0", "text", profile=StateProfile.full)}
    with pytest.raises(ValueError):
        build_training_examples(states, {"retail-0": 1.0}, QUESTIONS)


# --- vendored record --------------------------------------------------------


def test_to_vendor_record_matches_all_three_kinds() -> None:
    example = TrainingExample(
        state="a compact state",
        questions=[VERDICT_Q, COMPLETED_Q, SCORE_Q],
        labels={"verdict": 0, "completed": 1, "urgency": 1},
    )
    assert to_vendor_record(example) == {
        "state": "a compact state",
        "qs": [
            {"t": "choice", "ins": "Pass or fail?", "crit": {"pass": "", "fail": ""}, "y": 0},
            {"t": "noul", "ins": "Did the agent complete it?", "y": 1},
            {"t": "score", "ins": "How urgent?", "crit": ["low", "high"], "y": 1},
        ],
    }


def test_to_vendor_record_noul_has_no_crit() -> None:
    example = TrainingExample(state="s", questions=[COMPLETED_Q], labels={"completed": 0})
    assert "crit" not in to_vendor_record(example)["qs"][0]


# --- manifest ---------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(
        base_checkpoint_hash="a" * 64,
        fold=1,
        k=3,
        seed=7,
        epochs=2,
        learning_rate=2e-5,
        batch_size=8,
        train_task_ids=["retail-1", "retail-2"],
        test_task_ids=["retail-0"],
        temperature_by_options={"choice:2": 1.5, "noul:2": 2.0},
        n_examples=2,
        created_at=datetime.now(UTC),
    )
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest


# --- training ---------------------------------------------------------------


def _tiny_examples(count: int) -> list[TrainingExample]:
    """Build a handful of short labelled examples over the stub vocabulary."""
    texts = [
        "the agent verified the account and issued store credit",
        "the agent skipped verification and changed the order",
        "the response was polite concise and resolved the problem",
        "the agent asked for a refund on a damaged order",
    ]
    examples: list[TrainingExample] = []
    for index in range(count):
        reward = 1.0 if index % 2 == 0 else 0.0
        labels = {"verdict": 0 if reward >= 1.0 else 1, "completed": 1 if reward >= 1.0 else 0}
        examples.append(
            TrainingExample(state=texts[index % len(texts)], questions=QUESTIONS, labels=labels)
        )
    return examples


def test_fine_tune_produces_reloadable_and_changed_checkpoint(tmp_path: Path) -> None:
    from decision_judges.judges.laya_model import LayaDecisionModel

    out_dir = tmp_path / "ft"
    fine_tune(
        STUB,
        _tiny_examples(3),
        out_dir,
        epochs=1,
        learning_rate=1e-2,
        batch_size=2,
        seed=7,
        max_steps=2,
    )

    reloaded = LayaDecisionModel.from_pretrained(str(out_dir))
    assert len(reloaded.checkpoint_hash) == 64

    base_state = load_file(str(STUB / "model.safetensors"))
    out_state = load_file(str(out_dir / "model.safetensors"))
    assert set(base_state) == set(out_state)
    changed = any(not base_state[key].equal(out_state[key]) for key in base_state)
    assert changed, "at least one tensor must change after training"


def test_fit_temperatures_returns_buckets_in_range(tmp_path: Path) -> None:
    out_dir = tmp_path / "ft"
    fine_tune(
        STUB,
        _tiny_examples(4),
        out_dir,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4,
        seed=7,
        max_steps=1,
    )
    temperatures = fit_temperatures(out_dir, _tiny_examples(4))
    assert temperatures
    for bucket, value in temperatures.items():
        assert ":" in bucket
        assert 0.5 <= value <= 5.0


# --- cross validation -------------------------------------------------------


def test_run_cross_validation_writes_folds_and_isolates_test_ids(tmp_path: Path) -> None:
    task_ids = [f"retail-{index}" for index in range(4)]
    states = {task_id: _state(task_id, f"state for {task_id}") for task_id in task_ids}
    rewards = {task_id: float(index % 2 == 0) for index, task_id in enumerate(task_ids)}
    out_root = tmp_path / "models"

    manifests = run_cross_validation(
        STUB,
        states,
        rewards,
        QUESTIONS,
        out_root,
        k=2,
        seed=7,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4,
        device="cpu",
        max_steps=1,
    )

    assert len(manifests) == 2
    for fold in range(2):
        directory = out_root / f"laya-g3-fold{fold}"
        assert (directory / "manifest.json").is_file()
        manifest = read_manifest(directory)
        assert set(manifest.train_task_ids).isdisjoint(manifest.test_task_ids)

    for task_id in task_ids:
        directory = checkpoint_for_task(out_root, task_id)
        manifest = read_manifest(directory)
        assert task_id in manifest.test_task_ids
        assert task_id not in manifest.train_task_ids


# --- lfs rule ---------------------------------------------------------------


def test_gitattributes_declares_lfs_rule() -> None:
    text = GITATTRIBUTES.read_text(encoding="utf-8")
    assert "models/**/*.safetensors filter=lfs diff=lfs merge=lfs -text" in text
