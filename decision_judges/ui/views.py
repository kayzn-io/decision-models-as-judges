"""Pure presentation helpers for the trajectories page.

These functions shape agent records, tasks, and verdicts into plain values a
Streamlit page renders. They import no Streamlit so they stay unit-testable.
"""

import difflib
import json
from collections import Counter
from collections.abc import Mapping

import pandas as pd
from pydantic import BaseModel, Field

from decision_judges.bench.load import ExpectedAction, Task, normalize_action
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.gates.g2_steps import enumerate_steps
from decision_judges.metrics import mean, modal_agreement
from decision_judges.serialize import StateRecord
from decision_judges.types import Verdict

_VERDICT_ID = "verdict"
_COMPLETED_ID = "completed"
_NECESSARY_ID = "necessary"
_ARGS_ID = "arguments_consistent"

_QUERY_KEYS = ("variant", "task", "injection")
_ROLE_PREFIXES = ("assistant: ", "tool: ", "user: ", "call: ")


def selection_from_query(params: Mapping[str, str], defaults: Mapping[str, str]) -> dict[str, str]:
    """Return the variant, task, and injection selection, preferring query params.

    Each key is taken from ``params`` when present and non-empty, otherwise from
    ``defaults``. Values are returned verbatim; the caller validates them against
    the options actually available for the current trajectory.
    """
    chosen: dict[str, str] = {}
    for key in _QUERY_KEYS:
        value = params.get(key)
        chosen[key] = str(value) if value else str(defaults[key])
    return chosen


def _added_lines(before: list[str], after: list[str]) -> list[str]:
    """Return the lines present in ``after`` but not aligned to ``before``."""
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    added: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(after[j1:j2])
    return added


def injected_sentence(original: StateRecord | None, injected: StateRecord) -> str:
    """Return the evaluator-directed sentence spliced into the injected state.

    The sentence is the text present in the injected state but absent from the
    matching uninjected state, with its rendered role prefix stripped. An empty
    string is returned when no uninjected state is available to compare against.
    """
    if original is None:
        return ""
    added = _added_lines(original.text.splitlines(), injected.text.splitlines())
    sentence = " ".join(line.strip() for line in added).strip()
    for prefix in _ROLE_PREFIXES:
        if sentence.startswith(prefix):
            return sentence[len(prefix) :]
    return sentence


class Turn(BaseModel):
    """One conversation turn, with tool calls already formatted for display."""

    role: str
    content: str = ""
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None


def _function(tool_call: dict) -> dict:
    """Return the function payload of a tool call, tolerating a bare payload."""
    function = tool_call.get("function")
    if isinstance(function, dict):
        return function
    return tool_call


def _parse_arguments(arguments: object) -> object:
    """Parse JSON-string arguments, returning the raw string when it is invalid."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def format_tool_call(tool_call: dict) -> str:
    """Return ``name({sorted json})``, showing malformed arguments raw."""
    function = _function(tool_call)
    name = str(function.get("name", ""))
    parsed = _parse_arguments(function.get("arguments", {}))
    if isinstance(parsed, str):
        return f"{name}({parsed})"
    return f"{name}({json.dumps(parsed, sort_keys=True)})"


def _arguments_dict(arguments: object) -> dict[str, object]:
    """Return tool-call arguments as a dict, empty when they do not parse to one."""
    parsed = _parse_arguments(arguments)
    return parsed if isinstance(parsed, dict) else {}


def _made_actions(record: AgentRecord) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
    """Return the normalized tool calls the assistant made across the trajectory."""
    made: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for message in record.trajectory:
        if str(message.get("role", "")) != "assistant":
            continue
        raw = message.get("tool_calls")
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            function = _function(entry)
            name = str(function.get("name", ""))
            made.add(normalize_action(name, _arguments_dict(function.get("arguments", {}))))
    return made


def matched_expected_actions(task: Task, record: AgentRecord) -> list[tuple[ExpectedAction, bool]]:
    """Pair each expected action with whether the record made a matching call."""
    made = _made_actions(record)
    return [
        (action, normalize_action(action.name, action.kwargs) in made) for action in task.actions
    ]


def turns(record: AgentRecord) -> list[Turn]:
    """Return the trajectory as display turns, skipping the system message."""
    result: list[Turn] = []
    for message in record.trajectory:
        role = str(message.get("role", ""))
        if role == "system":
            continue
        content = message.get("content")
        tool_calls: list[str] = []
        raw = message.get("tool_calls")
        if isinstance(raw, list):
            tool_calls = [format_tool_call(entry) for entry in raw if isinstance(entry, dict)]
        tool_call_id = message.get("tool_call_id")
        result.append(
            Turn(
                role=role,
                content=content if isinstance(content, str) else "",
                tool_calls=tool_calls,
                tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
            )
        )
    return result


def _verdict_choice(verdict: Verdict) -> str | None:
    """Return the verdict question's chosen option, or None when it is absent."""
    for answer in verdict.answers:
        if answer.question_id == _VERDICT_ID:
            return answer.choice
    return None


def _verdict_confidence(verdict: Verdict) -> float | None:
    """Return the verdict question's confidence, or None when it is absent."""
    for answer in verdict.answers:
        if answer.question_id == _VERDICT_ID:
            return answer.confidence
    return None


def _completed_noul(verdict: Verdict) -> float | None:
    """Return the completed question's noul, or None when it is absent."""
    for answer in verdict.answers:
        if answer.question_id == _COMPLETED_ID:
            return answer.noul
    return None


def _mode(choices: list[str]) -> str:
    """Return the most frequent choice, resolving ties by sort order."""
    return Counter(sorted(choices)).most_common(1)[0][0]


def _mean(values: list[float]) -> float:
    """Return the mean of values, or NaN when there are none."""
    return sum(values) / len(values) if values else float("nan")


def _first_rationale(verdicts: list[Verdict]) -> str:
    """Return the first non-empty rationale across verdicts, or an empty string."""
    for verdict in verdicts:
        if verdict.rationale:
            return verdict.rationale
    return ""


_VERDICT_COLUMNS = [
    "judge_id",
    "verdict",
    "confidence",
    "completed",
    "agreement",
    "repeats",
    "errors",
    "rationale",
]


def _verdict_row(judge_id: str, group: list[Verdict]) -> dict[str, object]:
    """Reduce one judge's repeats into a single summary row."""
    choices = [choice for verdict in group if (choice := _verdict_choice(verdict)) is not None]
    confidences = [c for verdict in group if (c := _verdict_confidence(verdict)) is not None]
    completions = [n for verdict in group if (n := _completed_noul(verdict)) is not None]
    return {
        "judge_id": judge_id,
        "verdict": _mode(choices) if choices else None,
        "confidence": _mean(confidences),
        "completed": _mean(completions),
        "agreement": modal_agreement([choices]).per_item[0] if choices else float("nan"),
        "repeats": len(group),
        "errors": sum(1 for verdict in group if verdict.error is not None),
        "rationale": _first_rationale(group),
    }


def verdict_rows(verdicts: list[Verdict]) -> pd.DataFrame:
    """Return one summary row per judge id over the given verdicts."""
    groups: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        groups.setdefault(verdict.judge_id, []).append(verdict)
    rows = [_verdict_row(judge_id, groups[judge_id]) for judge_id in sorted(groups)]
    return pd.DataFrame(rows, columns=_VERDICT_COLUMNS)


_STEP_COLUMNS = ["judge_id", "necessary", "arguments_consistent", "repeats", "errors"]


def _noul_for(verdict: Verdict, question_id: str) -> float | None:
    """Return the noul answer for a question id, or None when it is absent."""
    for answer in verdict.answers:
        if answer.question_id == question_id:
            return answer.noul
    return None


def _step_row(judge_id: str, group: list[Verdict]) -> dict[str, object]:
    """Reduce one judge's step verdicts into a necessity and consistency row."""
    necessary = [n for verdict in group if (n := _noul_for(verdict, _NECESSARY_ID)) is not None]
    arguments = [n for verdict in group if (n := _noul_for(verdict, _ARGS_ID)) is not None]
    return {
        "judge_id": judge_id,
        "necessary": mean(necessary),
        "arguments_consistent": mean(arguments),
        "repeats": len(group),
        "errors": sum(1 for verdict in group if verdict.error is not None),
    }


def step_scores(
    step_states: Mapping[int, StateRecord],
    verdicts_by_hash: Mapping[str, list[Verdict]],
) -> dict[int, pd.DataFrame]:
    """Return one summary frame per step index over its judges' necessity nouls.

    Each frame carries one row per judge with the mean necessary probability,
    the mean arguments-consistent probability, the repeat count, and the error
    count. A step with no verdicts yields an empty frame.
    """
    frames: dict[int, pd.DataFrame] = {}
    for step_index, state in step_states.items():
        groups: dict[str, list[Verdict]] = {}
        for verdict in verdicts_by_hash.get(state.state_hash, []):
            groups.setdefault(verdict.judge_id, []).append(verdict)
        rows = [_step_row(judge_id, groups[judge_id]) for judge_id in sorted(groups)]
        frames[step_index] = pd.DataFrame(rows, columns=_STEP_COLUMNS)
    return frames


def expected_step_flags(task: Task, record: AgentRecord) -> list[bool]:
    """Return whether each enumerated tool call matches a task expected action."""
    expected = {normalize_action(action.name, action.kwargs) for action in task.actions}
    return [
        normalize_action(step.name, step.arguments) in expected for step in enumerate_steps(record)
    ]
