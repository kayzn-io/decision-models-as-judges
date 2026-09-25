"""Helpers for the live judging page: session keys, judge construction, result shaping.

The page stays thin by delegating every non-Streamlit decision here: constructing
judges from the visitor's own keys, checking whether the local Laya checkpoint is
present, and turning a verdict into the values the page renders. Optional backends
(``huggingface_hub``, the TypeSafe SDK, Laya's torch stack) are imported lazily so
importing this module never pulls a heavy or optional dependency.
"""

import importlib
from collections.abc import Mapping

import pandas as pd
from pydantic import BaseModel

from decision_judges import pipeline
from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import PricingTable, StudyConfig, UnpricedModel
from decision_judges.judges.base import Judge
from decision_judges.judges.llm import LlmJudge, OpenAiClientAdapter
from decision_judges.types import Question, Verdict

LIVE_CALL_CAP = 20

_LLM_JUDGE_ID = "llm_strong"
_JEV_JUDGE_ID = "jev"
_LAYA_JUDGE_ID = "laya"
_VERDICT_ID = "verdict"
_COMPLETED_ID = "completed"
_LAYA_WEIGHTS = "model.safetensors"
_MISSING = "—"


class LiveKeys(BaseModel):
    """The visitor's per-session API keys, held only in memory."""

    openrouter: str | None = None
    typesafe: str | None = None

    def redact(self) -> str:
        """Return a one-line key summary that never contains a key value."""
        return (
            f"OpenRouter key {_presence(self.openrouter)}; TypeSafe key {_presence(self.typesafe)}"
        )


def _presence(value: str | None) -> str:
    """Return whether a key is set without revealing it."""
    return "set" if value else "not set"


def can_call(count: int, cap: int = LIVE_CALL_CAP) -> bool:
    """Return whether another live judgement is allowed under the per-session cap."""
    return count < cap


def laya_available(study: StudyConfig) -> bool:
    """Return whether the study's Laya checkpoint is already in the local cache.

    The lookup only inspects the on-disk Hugging Face cache and never downloads,
    so a missing checkpoint yields ``False`` rather than a network call.
    """
    hub = importlib.import_module("huggingface_hub")
    cached = hub.try_to_load_from_cache(
        study.models.laya.repo_id, _LAYA_WEIGHTS, revision=study.models.laya.revision
    )
    return isinstance(cached, str)


def outcome_questions(tasks: Mapping[str, Task]) -> list[Question]:
    """Return the G3 outcome questions the live judges answer."""
    return pipeline.make_gate("g3", tasks).questions()


def build_live_judges(
    study: StudyConfig,
    pricing: PricingTable,
    tasks: Mapping[str, Task],
    records: Mapping[tuple[str, str], AgentRecord],
    keys: LiveKeys,
    *,
    include_laya: bool,
) -> list[Judge]:
    """Construct the live judges, passing each visitor key explicitly.

    The LLM and Jev judges are always built from the session keys. Laya is added
    only when requested and its checkpoint is already cached locally, so the page
    never triggers a download.
    """
    gate = pipeline.make_gate("g3", tasks)
    rubric_path = pipeline.gate_rubric_path(gate)
    prompt_version = pipeline.gate_prompt_version(gate)
    judges: list[Judge] = [
        LlmJudge(
            _LLM_JUDGE_ID,
            study.models.llm_strong,
            rubric_path,
            OpenAiClientAdapter(study.llm_base_url, api_key=keys.openrouter),
        ),
        _build_jev(study, keys, prompt_version),
    ]
    if include_laya and laya_available(study):
        judges.append(_build_laya(study, prompt_version))
    return judges


def _jev_api_key(study: StudyConfig, keys: LiveKeys) -> str | None:
    """Return the key the Jev route needs: TypeSafe direct, otherwise OpenRouter."""
    if study.jev_route.provider == "typesafe":
        return keys.typesafe
    return keys.openrouter


def _build_jev(study: StudyConfig, keys: LiveKeys, prompt_version: str) -> Judge:
    """Build the Jev judge with a TypeSafe client keyed to the session."""
    from decision_judges.judges.jev import JevJudge

    typesafe = importlib.import_module("typesafe_sdk")
    client = typesafe.TypeSafeClient(
        api_key=_jev_api_key(study, keys), base_url=study.jev_route.base_url
    )
    return JevJudge(study.models.jev, client, judge_id=_JEV_JUDGE_ID, prompt_version=prompt_version)


def _build_laya(study: StudyConfig, prompt_version: str) -> Judge:
    """Load the cached Laya checkpoint and wrap it in a judge."""
    from decision_judges.judges.laya import LayaJudge
    from decision_judges.judges.laya_model import LayaDecisionModel

    model = LayaDecisionModel.from_pretrained(
        study.models.laya.repo_id, revision=study.models.laya.revision, device="cpu"
    )
    return LayaJudge(
        model,
        judge_id=_LAYA_JUDGE_ID,
        model_label=study.models.laya.repo_id,
        prompt_version=prompt_version,
        device="cpu",
    )


def _verdict_answer(verdict: Verdict, question_id: str) -> object | None:
    """Return the answer object for a question id, or None when it is absent."""
    for answer in verdict.answers:
        if answer.question_id == question_id:
            return answer
    return None


def verdict_choice(verdict: Verdict) -> str | None:
    """Return the verdict question's chosen option, or None when it is absent."""
    answer = _verdict_answer(verdict, _VERDICT_ID)
    return getattr(answer, "choice", None)


def confidence_or_noul(verdict: Verdict) -> float | None:
    """Return the verdict confidence, falling back to the completed noul."""
    verdict_answer = _verdict_answer(verdict, _VERDICT_ID)
    confidence = getattr(verdict_answer, "confidence", None)
    if confidence is not None:
        return confidence
    completed = _verdict_answer(verdict, _COMPLETED_ID)
    return getattr(completed, "noul", None)


def verdict_probabilities(verdict: Verdict) -> pd.DataFrame:
    """Return the verdict question's probability distribution as an outcome table."""
    answer = _verdict_answer(verdict, _VERDICT_ID)
    probabilities = getattr(answer, "probabilities", None)
    if not probabilities:
        return pd.DataFrame(columns=["outcome", "probability"])
    rows = [
        {"outcome": outcome, "probability": probability}
        for outcome, probability in probabilities.items()
    ]
    return pd.DataFrame(rows, columns=["outcome", "probability"])


def cost_text(pricing: PricingTable, verdict: Verdict) -> str:
    """Return the verdict's USD cost, or a dash when the model is unpriced."""
    try:
        return f"${pricing.cost(verdict.model_id, verdict.usage):.4f}"
    except UnpricedModel:
        return _MISSING


def _ratio_text(value: float | None) -> str:
    """Render a unit-interval value as a one-decimal percent, dash when missing."""
    if value is None:
        return _MISSING
    return f"{value * 100:.1f}%"


def result_metrics(pricing: PricingTable, verdict: Verdict) -> list[tuple[str, str]]:
    """Return the labeled metrics shown for one judge's verdict."""
    choice = verdict_choice(verdict)
    return [
        ("Verdict", choice if choice is not None else _MISSING),
        ("Confidence", _ratio_text(confidence_or_noul(verdict))),
        ("Latency", f"{verdict.latency_ms} ms"),
        ("Input tokens", str(verdict.usage.input_tokens)),
        ("Cost", cost_text(pricing, verdict)),
    ]


def order_results(
    results: Mapping[tuple[str, str], Verdict], state_hash: str
) -> list[tuple[str, Verdict]]:
    """Return the verdicts recorded for one state, keyed by judge id in stable order."""
    return [
        (judge_id, verdict)
        for (judge_id, recorded_hash), verdict in results.items()
        if recorded_hash == state_hash
    ]
