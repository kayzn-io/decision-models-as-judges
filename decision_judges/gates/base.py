"""Gate framework: build judge-visible items, ask judges, and analyze verdicts.

A gate turns agent runs into serialized items paired with withheld ground
truth, runs a set of judges over them under a spend cap and a content-addressed
cache, and analyzes the resulting verdicts into tables, charts, and findings.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from decision_judges.bench.load import Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.cache import Cache, cache_key
from decision_judges.judges.base import Judge, build_verdict
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.spend import Reservation, Spend
from decision_judges.types import Question, Verdict

_OUTPUT_TOKEN_ESTIMATE = 500


class Item(BaseModel):
    """A judge-visible state paired with the ground truth a gate withholds.

    The truth fields let a gate score verdicts after the fact; a gate fills the
    fields that apply to it and never passes them to a judge, which reads only
    the state text.
    """

    state: StateRecord
    truth_label: str | None = None
    truth_value: float | None = None


class GateResult(BaseModel):
    """Tables, charts, and a prose finding produced by analyzing verdicts.

    pandas DataFrames and matplotlib Figures are not pydantic models and cannot
    be validated, so the model permits arbitrary types and keeps them as opaque
    objects.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tables: dict[str, object]
    charts: dict[str, object]
    findings: str


def _repeats_for(repeats: Mapping[str, int] | int, judge_id: str) -> int:
    """Return the repeat count for a judge from a per-judge mapping or a shared int."""
    if isinstance(repeats, Mapping):
        return repeats[judge_id]
    return repeats


class Gate(ABC):
    """A single evaluation gate over serialized agent states."""

    gate_id: str
    stage: str

    @property
    @abstractmethod
    def rubric_path(self) -> Path:
        """Path of the rubric file whose text every judge shares."""

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """Version string binding the rubric text and question set."""

    @abstractmethod
    def build_items(
        self,
        records: Mapping[str, AgentRecord],
        tasks: Mapping[str, Task],
        profile: StateProfile,
    ) -> list[Item]:
        """Serialize the applicable agent records into items with ground truth."""

    @abstractmethod
    def questions(self) -> list[Question]:
        """Return the typed questions every judge answers for this gate."""

    @abstractmethod
    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Reduce verdicts and their items into tables, charts, and findings."""

    def run(
        self,
        items: Sequence[Item],
        judges: Sequence[Judge],
        cache: Cache,
        spend: Spend,
        *,
        repeats: Mapping[str, int] | int,
    ) -> list[Verdict]:
        """Judge every item with every judge, caching hits and metering paid calls.

        Each judge answers each item a number of times given by ``repeats`` (per
        judge id when a mapping, otherwise shared). Calls run in a deterministic
        order sorted by judge id, state hash, and repeat, and the verdicts are
        returned in that order.
        """
        questions = self.questions()
        plan = sorted(
            (
                (judge, item, repeat)
                for judge in judges
                for item in items
                for repeat in range(_repeats_for(repeats, judge.judge_id))
            ),
            key=lambda entry: (entry[0].judge_id, entry[1].state.state_hash, entry[2]),
        )
        return [
            self._call_or_error(judge, item, repeat, questions, cache, spend)
            for judge, item, repeat in plan
        ]

    def _call_or_error(
        self,
        judge: Judge,
        item: Item,
        repeat: int,
        questions: Sequence[Question],
        cache: Cache,
        spend: Spend,
    ) -> Verdict:
        """Return the judge's verdict, or an error verdict when it rejects a question.

        A judge that cannot answer a gate's questions raises ``ValueError``; that
        is recorded as one error verdict so a single incompatible judge does not
        crash the run for the others.
        """
        try:
            return self._one_call(judge, item, repeat, questions, cache, spend)
        except ValueError as exc:
            return build_verdict(judge, item.state, repeat, [], latency_ms=0, error=str(exc))

    def _one_call(
        self,
        judge: Judge,
        item: Item,
        repeat: int,
        questions: Sequence[Question],
        cache: Cache,
        spend: Spend,
    ) -> Verdict:
        """Return the cached verdict or reserve spend, judge, and settle on a miss."""
        key = cache_key(
            judge.judge_id, judge.model_id, judge.prompt_version, item.state.state_hash, repeat
        )

        def call() -> Verdict:
            return judge.judge(item.state, questions, repeat)

        if cache.exists(key):
            return cache.get_or_call(key, Verdict, call)

        reservation = self._reserve(judge, item, spend)
        try:
            verdict = cache.get_or_call(key, Verdict, call)
        except BaseException:
            if reservation is not None:
                spend.cancel(reservation)
            raise
        if reservation is not None:
            spend.settle(reservation, verdict.usage)
        return verdict

    def _reserve(self, judge: Judge, item: Item, spend: Spend) -> Reservation | None:
        """Reserve estimated spend for a paid judge, or nothing for a local one."""
        if not judge.paid:
            return None
        est_input, est_output = self._estimate(item)
        return spend.reserve(self.stage, judge.model_id, est_input, est_output)

    def _estimate(self, item: Item) -> tuple[int, int]:
        """Return the estimated input and output tokens for one judging call."""
        return item.state.token_estimate, _OUTPUT_TOKEN_ESTIMATE
