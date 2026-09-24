"""Tests for the Laya judge, its math helpers, and the model adapter."""

import math
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from decision_judges.judges.laya import (
    MAX_STATE_TOKENS,
    LayaJudge,
    StateTooLarge,
    apply_temperature,
    confidence_from_probs,
    expected_level,
    softmax,
)
from decision_judges.judges.laya_model import LayaDecisionModel
from decision_judges.types import Question, QuestionKind

STUB = Path(__file__).parent / "fixtures" / "laya_stub"

VERDICT_Q = Question(
    id="verdict",
    kind=QuestionKind.choice,
    text="Which department?",
    options=["billing", "technical"],
)
URGENCY_Q = Question(
    id="urgency", kind=QuestionKind.score, text="How urgent?", levels=["low", "medium", "high"]
)
FOLLOWED_Q = Question(id="followed", kind=QuestionKind.noul, text="Was the policy followed?")
QUESTIONS: Sequence[Question] = [VERDICT_Q, URGENCY_Q, FOLLOWED_Q]


class StateStub(BaseModel):
    """A minimal state satisfying HasStateText for judge tests."""

    text: str = "the customer asked for a refund on a damaged order"
    task_id: str = "task-0"
    state_hash: str = "hash-0"


class FakeLayaModel:
    """A scripted model returning fixed logits and recording every call."""

    def __init__(
        self,
        logits: dict[str, list[float]] | None = None,
        *,
        token_len: int = 5,
        raises: Exception | None = None,
        checkpoint_hash: str = "abcdef0123456789" + "0" * 48,
    ) -> None:
        self.checkpoint_hash = checkpoint_hash
        self._logits = logits or {}
        self._token_len = token_len
        self._raises = raises
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def option_logits(
        self, state: str, question_type: str, instructions: str, options: Sequence[str]
    ) -> list[float]:
        self.calls.append((question_type, tuple(options)))
        if self._raises is not None:
            raise self._raises
        return self._logits.get(question_type, [0.0] * len(options))

    def tokenizer_len(self, text: str) -> int:
        return self._token_len


def _stub_judge(**kwargs: object) -> LayaJudge:
    model = LayaDecisionModel.from_pretrained(str(STUB))
    return LayaJudge(
        model,
        model_label="laya-stub",
        prompt_version="v1",
        device="cpu",
        **kwargs,  # type: ignore[arg-type]
    )


# --- Math helpers -----------------------------------------------------------


def test_softmax_matches_hand_computation() -> None:
    probs = softmax([0.0, math.log(2.0), math.log(3.0)])
    assert probs == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert sum(probs) == pytest.approx(1.0)


def test_apply_temperature_divides_logits() -> None:
    assert apply_temperature([2.0, -4.0], 2.0) == [1.0, -2.0]


def test_apply_temperature_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        apply_temperature([1.0], 0.0)


def test_confidence_is_one_for_single_option() -> None:
    assert confidence_from_probs([1.0]) == 1.0


def test_confidence_is_zero_for_uniform() -> None:
    assert confidence_from_probs([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_confidence_between_zero_and_one_for_skewed() -> None:
    value = confidence_from_probs([0.7, 0.2, 0.1])
    assert 0.0 < value < 1.0


def test_temperature_two_flattens_distribution() -> None:
    logits = [3.0, 1.0, 0.0]
    sharp = softmax(logits)
    flattened = softmax(apply_temperature(logits, 2.0))
    assert max(flattened) < max(sharp)


def test_expected_level_is_probability_weighted_index() -> None:
    assert expected_level([0.1, 0.1, 0.8]) == pytest.approx(1.7)
    assert expected_level([1.0, 0.0, 0.0]) == pytest.approx(0.0)


# --- Model adapter ----------------------------------------------------------


def test_from_pretrained_loads_and_hashes() -> None:
    model = LayaDecisionModel.from_pretrained(str(STUB))
    assert len(model.checkpoint_hash) == 64
    assert all(character in "0123456789abcdef" for character in model.checkpoint_hash)


def test_checkpoint_hash_is_stable_across_loads() -> None:
    first = LayaDecisionModel.from_pretrained(str(STUB))
    second = LayaDecisionModel.from_pretrained(str(STUB))
    assert first.checkpoint_hash == second.checkpoint_hash


def test_option_logits_returns_one_float_per_option() -> None:
    model = LayaDecisionModel.from_pretrained(str(STUB))
    logits = model.option_logits("a short state", "choice", "which one?", ["a", "b", "c"])
    assert len(logits) == 3
    assert all(isinstance(value, float) for value in logits)


def test_strict_loading_rejects_renamed_key(tmp_path: Path) -> None:
    import shutil

    from safetensors.torch import load_file, save_file

    corrupt = tmp_path / "stub"
    shutil.copytree(STUB, corrupt)
    weights = corrupt / "model.safetensors"
    state = load_file(str(weights))
    key = next(iter(state))
    state[f"renamed.{key}"] = state.pop(key)
    save_file(state, str(weights))

    with pytest.raises(RuntimeError):
        LayaDecisionModel.from_pretrained(str(corrupt))


# --- Judge over the stub ----------------------------------------------------


def test_judge_produces_valid_answers_over_stub() -> None:
    judge = _stub_judge()
    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert verdict.error is None
    by_id = {answer.question_id: answer for answer in verdict.answers}

    choice = by_id["verdict"]
    assert choice.probabilities is not None
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)
    best = max(choice.probabilities, key=lambda key: choice.probabilities[key])
    assert choice.choice == best

    score = by_id["urgency"]
    assert score.probabilities is not None
    assert set(score.probabilities) == {"low", "medium", "high"}
    assert score.score is not None
    assert 0.0 <= score.score <= 2.0

    noul = by_id["followed"]
    assert noul.noul is not None
    assert 0.0 <= noul.noul <= 1.0


def test_judge_stamps_identity_usage_and_provenance() -> None:
    judge = _stub_judge()
    verdict = judge.judge(StateStub(state_hash="h9"), QUESTIONS, repeat=3)

    assert verdict.model_id.startswith("laya:")
    assert verdict.judge_id == "laya"
    assert verdict.state_hash == "h9"
    assert verdict.repeat == 3
    assert verdict.usage.output_tokens == 0
    assert verdict.usage.input_tokens > 0
    assert "checkpoint_hash" in verdict.extra
    assert "device" in verdict.extra
    assert "temperature" in verdict.extra


def test_judge_is_deterministic() -> None:
    judge = _stub_judge()
    first = judge.judge(StateStub(), QUESTIONS, repeat=0)
    second = judge.judge(StateStub(), QUESTIONS, repeat=0)
    assert first.model_dump(exclude={"latency_ms"}) == second.model_dump(exclude={"latency_ms"})


# --- Guard and error handling (fake model) ----------------------------------


def test_state_too_large_raised_before_any_model_call() -> None:
    model = FakeLayaModel(token_len=MAX_STATE_TOKENS + 1)
    judge = LayaJudge(model, model_label="laya", prompt_version="v1", device="cpu")

    with pytest.raises(StateTooLarge):
        judge.judge(StateStub(), QUESTIONS, repeat=0)
    assert model.calls == []


def test_model_exception_becomes_verdict_error() -> None:
    model = FakeLayaModel(raises=RuntimeError("forward failed"))
    judge = LayaJudge(model, model_label="laya", prompt_version="v1", device="cpu")

    verdict = judge.judge(StateStub(), QUESTIONS, repeat=0)

    assert verdict.error is not None
    assert "forward failed" in verdict.error
    assert verdict.answers == []


def test_choice_argmax_follows_logits_with_fake_model() -> None:
    model = FakeLayaModel(logits={"choice": [0.1, 5.0]})
    judge = LayaJudge(model, model_label="laya", prompt_version="v1", device="cpu")

    verdict = judge.judge(StateStub(), [VERDICT_Q], repeat=0)

    answer = verdict.answers[0]
    assert answer.choice == "technical"


def test_noul_uses_true_slot_probability() -> None:
    model = FakeLayaModel(logits={"noul": [0.0, 10.0]})
    judge = LayaJudge(model, model_label="laya", prompt_version="v1", device="cpu")

    verdict = judge.judge(StateStub(), [FOLLOWED_Q], repeat=0)

    answer = verdict.answers[0]
    assert answer.noul is not None
    assert answer.noul > 0.99
    assert model.calls[0][1] == ("false", "true")


# --- Optional real-checkpoint test ------------------------------------------


@pytest.mark.laya_real
@pytest.mark.skipif(os.environ.get("LAYA_REAL") != "1", reason="requires LAYA_REAL=1")
def test_real_checkpoint_makes_one_judgment() -> None:
    from huggingface_hub import snapshot_download

    from decision_judges.config import load_study

    study = load_study("config/study.toml")
    local = snapshot_download(study.models.laya.repo_id, revision=study.models.laya.revision)
    model = LayaDecisionModel.from_pretrained(local)
    judge = LayaJudge(model, model_label="laya", prompt_version="v1", device="cpu")

    verdict = judge.judge(StateStub(), [VERDICT_Q], repeat=0)

    assert verdict.error is None
    assert verdict.answers[0].choice in {"billing", "technical"}
