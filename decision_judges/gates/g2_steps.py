"""The G2 step gate: is each tool call necessary and are its arguments consistent?

Every tool call in a trajectory becomes its own item. A judge sees the policy
summary and the conversation up to and including that call, then answers two
nouls: whether the call is necessary and whether its arguments are consistent
with what the user gave or the agent retrieved. Ground truth for necessity is
whether the normalized call appears in the task's expected actions.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel

from decision_judges.bench.load import Task, normalize_action
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import StudyConfig
from decision_judges.gates.base import Gate, GateResult, Item
from decision_judges.judges.llm import prompt_version_for, read_rubric
from decision_judges.metrics import (
    PRF,
    LatencySummary,
    auroc,
    latency_summary,
    precision_recall_f1,
)
from decision_judges.serialize import (
    POLICY_GIST,
    POLICY_SUMMARY,
    StateProfile,
    StateRecord,
    assert_no_leakage,
    render_call,
    render_turns,
)
from decision_judges.types import Question, QuestionKind, Verdict

_DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "config" / "rubrics" / "g2_step.md"

_NECESSARY = "necessary"
_UNNECESSARY = "unnecessary"
_ARGS_ID = "arguments_consistent"
_NECESSARY_TEXT = "This tool call is needed to complete the user request under the policy."
_ARGS_TEXT = (
    "The arguments of this tool call match information the user provided or the "
    "agent retrieved earlier in the conversation."
)

_FULL_RESULT_CAP = 600
_SHRUNK_RESULT_CAP = 100
_COMPACT_TOKEN_CAP = 450
_NAN = float("nan")

_COLUMNS = [
    "judge_id",
    "profile",
    "n_steps",
    "precision",
    "recall",
    "f1",
    "auroc",
    "mean_args_necessary",
    "mean_args_unnecessary",
    "mean_nec_necessary",
    "mean_nec_unnecessary",
    "error_rate",
    "latency_p50",
    "latency_p95",
    "mean_input_tokens",
]


class StepRef(BaseModel):
    """One tool call in a trajectory, located by its position and identity."""

    task_id: str
    step_index: int
    tool_call_id: str | None
    name: str
    arguments: dict[str, object]
    malformed: bool = False


def _call_entries(message: dict[str, object]) -> Iterator[dict[str, object]]:
    """Yield each well-formed tool-call entry on an assistant message."""
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if isinstance(function, dict):
            yield entry


def _parse_call(entry: dict[str, object]) -> tuple[str, dict[str, object], bool]:
    """Return a call's name, argument dict, and whether its arguments are malformed."""
    function = entry["function"]
    assert isinstance(function, dict)
    name = str(function.get("name", ""))
    raw = function.get("arguments", {})
    if isinstance(raw, dict):
        return name, dict(raw), False
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return name, {}, True
        if isinstance(parsed, dict):
            return name, parsed, False
    return name, {}, True


def _is_assistant(message: dict[str, object]) -> bool:
    """Return whether a message is an assistant turn."""
    return str(message.get("role", "")) == "assistant"


def enumerate_steps(record: AgentRecord) -> list[StepRef]:
    """Return one StepRef per assistant tool call, in trajectory order."""
    steps: list[StepRef] = []
    index = 0
    for message in record.trajectory:
        if not _is_assistant(message):
            continue
        for entry in _call_entries(message):
            name, arguments, malformed = _parse_call(entry)
            raw_id = entry.get("id")
            steps.append(
                StepRef(
                    task_id=record.task_id,
                    step_index=index,
                    tool_call_id=str(raw_id) if raw_id is not None else None,
                    name=name,
                    arguments=arguments,
                    malformed=malformed,
                )
            )
            index += 1
    return steps


def _messages_through_step(record: AgentRecord, step: StepRef) -> list[dict[str, object]]:
    """Return the trajectory up to and including the message holding this call."""
    count = 0
    for position, message in enumerate(record.trajectory):
        if not _is_assistant(message):
            continue
        n = sum(1 for _ in _call_entries(message))
        if count + n > step.step_index:
            return list(record.trajectory[: position + 1])
        count += n
    return list(record.trajectory)


def _prior_compact_calls(record: AgentRecord, step: StepRef) -> list[str]:
    """Return normalized ``- name(args)`` lines for calls before this step."""
    lines: list[str] = []
    count = 0
    for message in record.trajectory:
        if not _is_assistant(message):
            continue
        for entry in _call_entries(message):
            if count >= step.step_index:
                return lines
            name, arguments, _ = _parse_call(entry)
            norm_name, norm_kwargs = normalize_action(name, arguments)
            args = ", ".join(f"{key}={value}" for key, value in norm_kwargs)
            lines.append(f"- {norm_name}({args})")
            count += 1
    return lines


def _review_line(step: StepRef) -> str:
    """Return the ``step under review`` line naming the call and its sorted args."""
    return f"step under review: {render_call(step.name, dict(step.arguments))}"


def _full_step_text(
    record: AgentRecord, task: Task, step: StepRef, review: str, budget_tokens: int
) -> tuple[str, bool]:
    """Render the full profile through this call, shrinking tool results to fit."""
    messages = _messages_through_step(record, step)
    header = f"{task.instruction}\n{POLICY_SUMMARY}"
    text = "\n".join([header, render_turns(messages, _FULL_RESULT_CAP), review])
    if len(text) // 4 <= budget_tokens:
        return text, False
    text = "\n".join([header, render_turns(messages, _SHRUNK_RESULT_CAP), review])
    return text, True


def _compact_step_text(
    record: AgentRecord, task: Task, step: StepRef, review: str
) -> tuple[str, bool]:
    """Render the compact profile: instruction, gist, prior calls, and this call."""
    head = [task.instruction, f"Policy: {POLICY_GIST}"]
    prior = _prior_compact_calls(record, step)

    def assemble(calls: Sequence[str]) -> str:
        return "\n".join([*head, *calls, review])

    text = assemble(prior)
    if len(text) // 4 <= _COMPACT_TOKEN_CAP:
        return text, False
    kept = list(prior)
    while kept and len(assemble(kept)) // 4 > _COMPACT_TOKEN_CAP:
        kept.pop(0)
    text = assemble(kept)
    if len(text) // 4 > _COMPACT_TOKEN_CAP:
        text = text[: _COMPACT_TOKEN_CAP * 4]
    return text, True


def serialize_step(
    record: AgentRecord,
    task: Task,
    step: StepRef,
    profile: StateProfile,
    *,
    budget_tokens: int = 28_000,
) -> StateRecord:
    """Serialize one tool call into a judge-visible state ending at the call.

    The full profile renders the conversation up to and including the message
    that holds the call; the compact profile renders the instruction, a policy
    gist, the prior tool-call list, and the call under review under a 450-token
    cap. Both end with a ``step under review`` line. The leakage guard runs on
    the result and the step index is recorded on the state.
    """
    review = _review_line(step)
    if profile is StateProfile.full:
        text, truncated = _full_step_text(record, task, step, review, budget_tokens)
    else:
        text, truncated = _compact_step_text(record, task, step, review)
    assert_no_leakage(text, task, record)
    return StateRecord(
        variant=record.variant,
        task_id=record.task_id,
        profile=profile,
        text=text,
        token_estimate=len(text) // 4,
        truncated=truncated,
        state_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        step_index=step.step_index,
    )


def skipped_malformed(records: Mapping[str, AgentRecord]) -> int:
    """Return the count of malformed-argument steps across non-excluded records."""
    total = 0
    for record in records.values():
        if record.excluded:
            continue
        total += sum(1 for step in enumerate_steps(record) if step.malformed)
    return total


def repeats_for(study: StudyConfig, judge_ids: Sequence[str]) -> dict[str, int]:
    """Return per-judge G2 repeat counts, defaulting unknown judges to 1."""
    mapping = study.g2.repeats
    return {judge_id: mapping.get(judge_id, 1) for judge_id in judge_ids}


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


class G2Steps(Gate):
    """Score each tool call for necessity and argument consistency."""

    gate_id = "g2"
    stage = "g2"

    def __init__(self, rubric_path: Path = _DEFAULT_RUBRIC_PATH) -> None:
        self._rubric_path = rubric_path
        self._questions: list[Question] | None = None
        self._prompt_version: str | None = None

    def questions(self) -> list[Question]:
        """Return the necessary and arguments-consistent nouls, loading the rubric once."""
        if self._questions is None:
            _, version = read_rubric(self._rubric_path)
            self._questions = [
                Question(id=_NECESSARY, kind=QuestionKind.noul, text=_NECESSARY_TEXT),
                Question(id=_ARGS_ID, kind=QuestionKind.noul, text=_ARGS_TEXT),
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
        """Serialize every non-malformed step, labeling it by the expected actions."""
        items: list[Item] = []
        for record in records.values():
            if record.excluded:
                continue
            task = tasks[record.task_id]
            expected = {normalize_action(action.name, action.kwargs) for action in task.actions}
            for step in enumerate_steps(record):
                if step.malformed:
                    continue
                state = serialize_step(record, task, step, profile)
                necessary = normalize_action(step.name, step.arguments) in expected
                items.append(
                    Item(
                        state=state,
                        truth_label=_NECESSARY if necessary else _UNNECESSARY,
                        truth_value=1.0 if necessary else 0.0,
                    )
                )
        return items

    def analyze(self, verdicts: Sequence[Verdict], items: Sequence[Item]) -> GateResult:
        """Summarize necessity metrics per judge and profile into a table and chart."""
        by_hash = {item.state.state_hash: item for item in items}
        groups: dict[tuple[str, StateProfile], list[Verdict]] = defaultdict(list)
        for verdict in verdicts:
            item = by_hash.get(verdict.state_hash)
            if item is not None:
                groups[(verdict.judge_id, item.state.profile)].append(verdict)

        rows = [
            self._summarize(judge_id, profile, groups[(judge_id, profile)], by_hash)
            for judge_id, profile in sorted(groups)
        ]
        frame = pd.DataFrame(rows, columns=_COLUMNS)
        return GateResult(
            tables={"g2_summary": frame},
            charts={"g2_auroc": self._auroc_chart(frame)},
            findings=self._findings(frame),
        )

    def _summarize(
        self,
        judge_id: str,
        profile: StateProfile,
        group: Sequence[Verdict],
        by_hash: Mapping[str, Item],
    ) -> dict[str, object]:
        """Reduce one judge-and-profile group of verdicts into a summary row."""
        nec: dict[str, list[float]] = defaultdict(list)
        arg: dict[str, list[float]] = defaultdict(list)
        for verdict in group:
            if verdict.error is not None:
                continue
            for answer in verdict.answers:
                if answer.noul is None:
                    continue
                if answer.question_id == _NECESSARY:
                    nec[verdict.state_hash].append(answer.noul)
                elif answer.question_id == _ARGS_ID:
                    arg[verdict.state_hash].append(answer.noul)

        nec_scores: list[float] = []
        pred: list[str] = []
        truth: list[str] = []
        truth_value: list[float] = []
        args_by_truth: dict[bool, list[float]] = {True: [], False: []}
        nec_by_truth: dict[bool, list[float]] = {True: [], False: []}
        for state_hash, necs in nec.items():
            item = by_hash.get(state_hash)
            args = arg.get(state_hash)
            if item is None or item.truth_label is None or not args:
                continue
            nec_p = _mean(necs)
            arg_p = _mean(args)
            is_necessary = item.truth_label == _NECESSARY
            nec_scores.append(nec_p)
            pred.append(_NECESSARY if nec_p >= 0.5 else _UNNECESSARY)
            truth.append(item.truth_label)
            truth_value.append(1.0 if is_necessary else 0.0)
            args_by_truth[is_necessary].append(arg_p)
            nec_by_truth[is_necessary].append(nec_p)

        prf = precision_recall_f1(pred, truth, _NECESSARY) if pred else PRF(_NAN, _NAN, _NAN)
        total = len(group)
        errors = sum(1 for verdict in group if verdict.error is not None)
        latency = (
            latency_summary([float(verdict.latency_ms) for verdict in group])
            if group
            else LatencySummary(_NAN, _NAN)
        )
        return {
            "judge_id": judge_id,
            "profile": profile.value,
            "n_steps": len(nec_scores),
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "auroc": auroc(nec_scores, truth_value) if nec_scores else _NAN,
            "mean_args_necessary": (_mean(args_by_truth[True]) if args_by_truth[True] else _NAN),
            "mean_args_unnecessary": (
                _mean(args_by_truth[False]) if args_by_truth[False] else _NAN
            ),
            "mean_nec_necessary": _mean(nec_by_truth[True]) if nec_by_truth[True] else _NAN,
            "mean_nec_unnecessary": _mean(nec_by_truth[False]) if nec_by_truth[False] else _NAN,
            "error_rate": errors / total if total else 0.0,
            "latency_p50": latency.p50,
            "latency_p95": latency.p95,
            "mean_input_tokens": (
                sum(verdict.usage.input_tokens for verdict in group) / total if total else 0.0
            ),
        }

    def _auroc_chart(self, frame: pd.DataFrame) -> Figure:
        """Return a bar figure of necessary-call AUROC per judge and profile."""
        figure = Figure()
        axes = figure.subplots()
        if not frame.empty:
            labels = [
                f"{judge}/{profile}"
                for judge, profile in zip(frame["judge_id"], frame["profile"], strict=True)
            ]
            values = [0.0 if pd.isna(value) else float(value) for value in frame["auroc"]]
            axes.bar(labels, values)
        axes.set_ylabel("auroc")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("G2 necessary-call AUROC by judge")
        return figure

    def _findings(self, frame: pd.DataFrame) -> str:
        """Return two to three factual sentences on the best judge and score separation."""
        if frame.empty:
            return "No verdicts were available to analyze."
        best = frame.sort_values("auroc", ascending=False).iloc[0]
        nec_mean = float(frame["mean_nec_necessary"].mean())
        unn_mean = float(frame["mean_nec_unnecessary"].mean())
        direction = "separates" if nec_mean > unn_mean else "does not separate"
        return (
            f"Judge {str(best['judge_id'])!r} on the {str(best['profile'])} profile reached the "
            f"highest necessary-call AUROC at {float(best['auroc']):.2f}. The mean necessary "
            f"probability was {nec_mean:.2f} for expected calls against {unn_mean:.2f} for "
            f"unexpected calls, so the score {direction} expected from unexpected tool calls."
        )
