"""Tests for the fold-routed fine-tuned Laya judge."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel
from safetensors.torch import load_file, save_file

from decision_judges.judges.laya_folds import FoldRoutedLayaJudge
from decision_judges.training.finetune_laya import Manifest, write_manifest
from decision_judges.types import Question, QuestionKind

STUB = Path(__file__).parent / "fixtures" / "laya_stub"

VERDICT_Q = Question(
    id="verdict", kind=QuestionKind.choice, text="Pass or fail?", options=["pass", "fail"]
)
COMPLETED_Q = Question(id="completed", kind=QuestionKind.noul, text="Did the agent complete it?")
QUESTIONS = [VERDICT_Q, COMPLETED_Q]


class StateStub(BaseModel):
    """A minimal state satisfying HasStateText for judge tests."""

    text: str = "the agent verified the account and issued store credit"
    task_id: str = "retail-0"
    state_hash: str = "hash-0"


def _make_fold(root: Path, fold: int, test_ids: list[str], *, perturb: bool = False) -> Path:
    """Copy the stub into a fold directory, optionally perturbing its weights."""
    directory = root / f"laya-g3-fold{fold}"
    shutil.copytree(STUB, directory)
    if perturb:
        weights = directory / "model.safetensors"
        state = load_file(str(weights))
        key = next(iter(state))
        state[key] = state[key] + 1.0
        save_file(state, str(weights))
    manifest = Manifest(
        base_checkpoint_hash="a" * 64,
        fold=fold,
        k=2,
        seed=7,
        epochs=1,
        learning_rate=1e-2,
        batch_size=4,
        train_task_ids=[],
        test_task_ids=test_ids,
        temperature_by_options={"choice:2": 1.5},
        n_examples=1,
        created_at=datetime.now(UTC),
    )
    write_manifest(directory, manifest)
    return directory


def _two_fold_root(tmp_path: Path) -> Path:
    """Build a models root with two folds holding disjoint test task ids."""
    root = tmp_path / "models"
    root.mkdir()
    _make_fold(root, 0, ["retail-0", "retail-1"])
    _make_fold(root, 1, ["retail-2", "retail-3"], perturb=True)
    return root


def test_routes_two_tasks_to_two_checkpoints(tmp_path: Path) -> None:
    root = _two_fold_root(tmp_path)
    judge = FoldRoutedLayaJudge(root, prompt_version="pv")

    first = judge.judge(StateStub(task_id="retail-0", state_hash="h0"), QUESTIONS, repeat=0)
    second = judge.judge(StateStub(task_id="retail-2", state_hash="h2"), QUESTIONS, repeat=0)

    assert first.error is None
    assert second.error is None
    assert first.extra["checkpoint_hash"] != second.extra["checkpoint_hash"]
    assert first.judge_id == "laya_ft"
    assert first.model_id.startswith("laya_ft:")


def test_raises_clear_error_for_task_in_no_fold(tmp_path: Path) -> None:
    root = _two_fold_root(tmp_path)
    judge = FoldRoutedLayaJudge(root, prompt_version="pv")

    with pytest.raises(KeyError, match="retail-99"):
        judge.judge(StateStub(task_id="retail-99", state_hash="h99"), QUESTIONS, repeat=0)


def test_model_id_is_stable_for_the_same_fold_set(tmp_path: Path) -> None:
    root = _two_fold_root(tmp_path)

    first = FoldRoutedLayaJudge(root, prompt_version="pv")
    second = FoldRoutedLayaJudge(root, prompt_version="pv")

    assert first.model_id == second.model_id
    assert first.model_id.startswith("laya_ft:")
