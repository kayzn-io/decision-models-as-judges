"""Judges declare payment, and gates reserve spend only for paid judges."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from decision_judges.cache import Cache
from decision_judges.config import UnpricedModel, load_pricing
from decision_judges.gates.base import Gate, Item
from decision_judges.gates.g3_outcome import G3Outcome
from decision_judges.judges.base import HasStateText, build_verdict
from decision_judges.judges.code import CodeJudge
from decision_judges.judges.jev import JevJudge
from decision_judges.judges.laya import LayaJudge
from decision_judges.judges.laya_folds import FoldRoutedLayaJudge
from decision_judges.judges.laya_model import LayaDecisionModel
from decision_judges.judges.llm import LlmJudge
from decision_judges.pipeline import FakeJudge
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Spend
from decision_judges.types import Answer, Question, QuestionKind, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = REPO_ROOT / "config" / "pricing.toml"
RUBRIC_FILE = REPO_ROOT / "config" / "rubrics" / "g3_outcome.md"
STUB = Path(__file__).parent / "fixtures" / "laya_stub"


class _FakeChatClient:
    """An OpenAI-like client whose method is never called here."""

    def chat_completions_create(self, **kwargs: Any) -> Any:
        raise AssertionError("client should not be called")


class _FakeJevClient:
    """A TypeSafe-like client whose method is never called here."""

    def system_one(self, state: str, questions: Any, *, model: str) -> Any:
        raise AssertionError("client should not be called")


class _FakeLayaModel:
    """A minimal Laya model exposing the surface a judge reads."""

    checkpoint_hash = "abc123" + "0" * 58

    def option_logits(
        self, state: str, question_type: str, instructions: str, options: Sequence[str]
    ) -> list[float]:
        return [0.0] * len(options)

    def tokenizer_len(self, text: str) -> int:
        return 4


_JUDGE_CASES: list[tuple[str, Callable[[Path], object], bool]] = [
    ("code", lambda tp: CodeJudge({}, {}), False),
    ("llm", lambda tp: LlmJudge("llm", "m", RUBRIC_FILE, _FakeChatClient()), True),
    ("jev", lambda tp: JevJudge("m", _FakeJevClient(), prompt_version="pv"), True),
    (
        "laya",
        lambda tp: LayaJudge(_FakeLayaModel(), model_label="l", prompt_version="pv", device="cpu"),
        False,
    ),
    ("laya_ft", lambda tp: FoldRoutedLayaJudge(tp, prompt_version="pv"), False),
    ("fake", lambda tp: FakeJudge(judge_id="fake"), False),
]


@pytest.mark.parametrize(
    ("build", "expected_paid"),
    [(build, expected) for _, build, expected in _JUDGE_CASES],
    ids=[name for name, _, _ in _JUDGE_CASES],
)
def test_every_judge_declares_paid(
    build: Callable[[Path], object], expected_paid: bool, tmp_path: Path
) -> None:
    judge = build(tmp_path)
    assert isinstance(judge.paid, bool)  # type: ignore[attr-defined]
    assert judge.paid is expected_paid  # type: ignore[attr-defined]


def _state(state_hash: str) -> StateRecord:
    """Build a minimal serialized state with a chosen hash."""
    return StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.full,
        text="transcript",
        token_estimate=100,
        state_hash=state_hash,
    )


def _item(state_hash: str, truth_label: str) -> Item:
    """Pair a state with its ground-truth label and value."""
    return Item(
        state=_state(state_hash),
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "pass" else 0.0,
    )


def _spend(tmp_path: Path, caps: dict[str, float]) -> Spend:
    """Build a real Spend over the repo pricing table and a temp ledger."""
    return Spend(load_pricing(PRICING_FILE), caps, tmp_path / "ledger.json")


def test_local_laya_judge_runs_without_reserving_spend(tmp_path: Path) -> None:
    model = LayaDecisionModel.from_pretrained(str(STUB))
    judge = LayaJudge(model, model_label="laya-stub", prompt_version="v1", device="cpu")
    items = [_item("h0", "pass"), _item("h1", "fail")]
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 0.0})

    verdicts = G3Outcome().run(items, [judge], cache, spend, repeats=1)

    assert len(verdicts) == 2
    assert all(verdict.error is None for verdict in verdicts)
    assert all(verdict.model_id.startswith("laya:") for verdict in verdicts)
    assert spend.total() == 0.0
    assert spend.spent("g3") == 0.0


class _PaidUnpricedJudge:
    """A paid judge whose model id is absent from the pricing table."""

    judge_id = "paid"
    model_id = "model-not-in-pricing"
    prompt_version = "pv"
    paid = True

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        answers = [
            Answer(question_id="completed", kind=QuestionKind.noul, noul=1.0),
            Answer(
                question_id="verdict",
                kind=QuestionKind.choice,
                choice="pass",
                probabilities={"pass": 1.0, "fail": 0.0},
                confidence=1.0,
            ),
        ]
        return build_verdict(self, state, repeat, answers, latency_ms=1)


def test_paid_judge_with_unpriced_model_still_raises(tmp_path: Path) -> None:
    gate: Gate = G3Outcome()
    items = [_item("h0", "pass")]
    cache = Cache(tmp_path / "cache")
    spend = _spend(tmp_path, {"g3": 50.0})

    with pytest.raises(UnpricedModel):
        gate.run(items, [_PaidUnpricedJudge()], cache, spend, repeats=1)
