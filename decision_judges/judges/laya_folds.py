"""A fold-routed Laya judge that picks a fine-tuned checkpoint per task.

Cross-validated fine-tuning produces one checkpoint per fold, each calibrated on
the tasks held out of its training set. This judge routes every state to the
fold that held its task out, so a task is always judged by a checkpoint that did
not train on it. It loads and caches one model per fold on demand and applies
that fold's per-bucket calibration temperatures.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from decision_judges.judges.base import HasStateText, build_verdict, timed
from decision_judges.judges.laya import (
    MAX_STATE_TOKENS,
    StateTooLarge,
    answer_for,
    apply_temperature,
    question_options,
    softmax,
)
from decision_judges.judges.laya_model import LayaDecisionModel
from decision_judges.judges.laya_vendor.rl_common import QTYPES, temp_bucket
from decision_judges.training.finetune_laya import checkpoint_for_task, read_manifest
from decision_judges.types import Answer, Question, QuestionKind, Usage, Verdict

_MANIFEST = "manifest.json"


def _bucket_for(question: Question) -> str:
    """Return the calibration bucket key for a question's type and option count."""
    if question.kind is QuestionKind.choice:
        return temp_bucket(QTYPES["choice"], len(question.options or []))
    if question.kind is QuestionKind.score:
        return temp_bucket(QTYPES["score"], len(question.levels or []))
    return temp_bucket(QTYPES["noul"], 2)


def _safetensors_hash(directory: Path) -> str:
    """Return the sha256 of a checkpoint's ``model.safetensors``."""
    return hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()


def _fold_directories(models_root: Path) -> list[Path]:
    """Return the fold checkpoint directories under a models root, sorted by name."""
    if not models_root.is_dir():
        return []
    return sorted(
        path for path in models_root.iterdir() if path.is_dir() and (path / _MANIFEST).is_file()
    )


def _fold_set_model_id(models_root: Path) -> str:
    """Return a stable model id from the sorted hashes of every fold checkpoint."""
    hashes = sorted(_safetensors_hash(directory) for directory in _fold_directories(models_root))
    digest = hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
    return f"laya_ft:{digest[:12]}"


class FoldRoutedLayaJudge:
    """Route each state to the fine-tuned Laya checkpoint that held its task out."""

    judge_id = "laya_ft"
    paid = False

    def __init__(
        self,
        models_root: Path,
        *,
        prompt_version: str,
        device: str = "cpu",
        model_label: str = "laya_ft",
    ) -> None:
        self._models_root = Path(models_root)
        self.prompt_version = prompt_version
        self.model_label = model_label
        self._device = device
        self.model_id = _fold_set_model_id(self._models_root)
        self._models: dict[Path, LayaDecisionModel] = {}
        self._temperatures: dict[Path, dict[str, float]] = {}

    def judge(self, state: HasStateText, questions: Sequence[Question], repeat: int) -> Verdict:
        """Judge one state with its fold's checkpoint and per-bucket temperatures.

        Routing to a fold whose test set holds the task raises ``KeyError`` when
        no fold does; that surfaces a missing-fold task loudly rather than
        silently misrouting it.
        """
        directory = checkpoint_for_task(self._models_root, state.task_id)
        model = self._model_for(directory)
        temperatures = self._temperatures_for(directory)
        input_tokens = model.tokenizer_len(state.text)
        if input_tokens > MAX_STATE_TOKENS:
            raise StateTooLarge(f"state exceeds {MAX_STATE_TOKENS} tokens")
        with timed() as elapsed:
            try:
                answers = [
                    self._answer(model, temperatures, state.text, question)
                    for question in questions
                ]
            except Exception as error:  # model failures become verdict errors, never raise
                return build_verdict(
                    self,
                    state,
                    repeat,
                    [],
                    latency_ms=elapsed(),
                    error=str(error),
                    extra=self._extra(model),
                )
            usage = Usage(input_tokens=input_tokens, output_tokens=0)
            return build_verdict(
                self,
                state,
                repeat,
                answers,
                usage=usage,
                latency_ms=elapsed(),
                extra=self._extra(model),
            )

    def _answer(
        self,
        model: LayaDecisionModel,
        temperatures: Mapping[str, float],
        state_text: str,
        question: Question,
    ) -> Answer:
        """Answer one question with the fold's checkpoint at its bucket temperature."""
        question_type, options = question_options(question)
        temperature = temperatures.get(_bucket_for(question), 1.0)
        logits = model.option_logits(state_text, question_type, question.text, options)
        probs = softmax(apply_temperature(logits, temperature))
        return answer_for(question, probs)

    def _model_for(self, directory: Path) -> LayaDecisionModel:
        """Load and cache the decision model for one fold directory."""
        key = directory.resolve()
        if key not in self._models:
            self._models[key] = LayaDecisionModel.from_pretrained(
                str(directory), device=self._device
            )
        return self._models[key]

    def _temperatures_for(self, directory: Path) -> dict[str, float]:
        """Read and cache the fold manifest's per-bucket calibration temperatures."""
        key = directory.resolve()
        if key not in self._temperatures:
            self._temperatures[key] = dict(read_manifest(directory).temperature_by_options)
        return self._temperatures[key]

    def _extra(self, model: LayaDecisionModel) -> dict[str, str]:
        """Return the provenance recorded on every verdict for the routed fold."""
        return {
            "checkpoint_hash": model.checkpoint_hash,
            "device": self._device,
            "model_label": self.model_label,
        }
