"""Tests for the G2 step gate: enumeration, serialization, items, and analysis."""

import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from decision_judges.bench.load import ExpectedAction, Task
from decision_judges.bench.run_agent import AgentRecord
from decision_judges.config import load_study
from decision_judges.gates.base import GateResult, Item
from decision_judges.gates.g2_steps import (
    G2Steps,
    StepRef,
    enumerate_steps,
    repeats_for,
    serialize_step,
    skipped_malformed,
    step_items,
)
from decision_judges.serialize import LeakageError, StateProfile, StateRecord
from decision_judges.types import Answer, QuestionKind, Usage, Verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_FILE = REPO_ROOT / "config" / "study.toml"

_FIND = "find_user_id_by_name_zip"
_GET = "get_order_details"
_CANCEL = "cancel_pending_order"


# --- builders --------------------------------------------------------------


def _assistant_call(call_id: str, name: str, arguments: object) -> dict[str, object]:
    """Build an assistant message carrying one tool call."""
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }


def _tool(call_id: str, name: str, content: str) -> dict[str, object]:
    """Build a tool-result message."""
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _trajectory() -> list[dict[str, object]]:
    """Build a three-call trajectory: authenticate, read order, then cancel."""
    return [
        {"role": "system", "content": "SYSTEM_POLICY full wiki text"},
        {"role": "user", "content": "Please cancel my order."},
        _assistant_call("c0", _FIND, {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
        _tool("c0", _FIND, "FIND_RESULT_MARKER yusuf_rossi_9620"),
        _assistant_call("c1", _GET, {"order_id": "#W1"}),
        _tool("c1", _GET, "LATER_TOOL_RESULT_MARKER order is pending"),
        _assistant_call("c2", _CANCEL, {"order_id": "#W1"}),
        _tool("c2", _CANCEL, "cancelled"),
        {"role": "assistant", "content": "All set, your order is cancelled."},
    ]


def _record(
    task_id: str = "retail-0",
    *,
    reward: float = 1.0,
    excluded: bool = False,
    trajectory: list[dict[str, object]] | None = None,
) -> AgentRecord:
    """Build an agent record with a chosen trajectory."""
    return AgentRecord(
        variant="baseline",
        task_id=task_id,
        trajectory=trajectory if trajectory is not None else _trajectory(),
        reward=reward,
        harness_info={},
        agent_model="agent",
        user_model="user",
        tau_bench_ref="ref",
        excluded=excluded,
    )


def _task(task_id: str = "retail-0", *, outputs: list[str] | None = None) -> Task:
    """Build a task whose expected actions authenticate and read the order."""
    return Task(
        task_id=task_id,
        instruction="You are Yusuf Rossi in 19122. Cancel my order.",
        actions=[
            ExpectedAction(
                name=_FIND, kwargs={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}
            ),
            ExpectedAction(name=_GET, kwargs={"order_id": "#W1"}),
        ],
        outputs=outputs or [],
    )


# --- enumerate_steps -------------------------------------------------------


def test_enumerate_steps_orders_and_ids() -> None:
    steps = enumerate_steps(_record())

    assert [step.step_index for step in steps] == [0, 1, 2]
    assert [step.name for step in steps] == [_FIND, _GET, _CANCEL]
    assert [step.tool_call_id for step in steps] == ["c0", "c1", "c2"]
    assert steps[1].arguments == {"order_id": "#W1"}
    assert all(not step.malformed for step in steps)


def test_enumerate_steps_flags_malformed_arguments() -> None:
    trajectory = [
        {"role": "user", "content": "hi"},
        _assistant_call("c0", _GET, "{not valid json"),
    ]
    steps = enumerate_steps(_record(trajectory=trajectory))

    assert len(steps) == 1
    assert steps[0].malformed is True
    assert steps[0].arguments == {}


# --- serialize_step --------------------------------------------------------


def test_serialize_step_full_ends_with_review_and_omits_later_results() -> None:
    record = _record()
    step = enumerate_steps(record)[1]  # the get_order_details call

    state = serialize_step(record, _task(), step, StateProfile.full)

    assert state.profile is StateProfile.full
    assert state.step_index == 1
    assert state.text.endswith('step under review: call: get_order_details({"order_id": "#W1"})')
    # The authenticating call and its result precede this call and are present.
    assert "FIND_RESULT_MARKER" in state.text
    assert "call: get_order_details" in state.text
    # The call's own tool result and every later turn are absent.
    assert "LATER_TOOL_RESULT_MARKER" not in state.text
    assert "cancelled" not in state.text


def test_serialize_step_full_keeps_long_tool_results_whole() -> None:
    long_result = "x" * 2_000
    trajectory = [
        {"role": "user", "content": "hi"},
        _assistant_call("c0", _FIND, {"name": "Yusuf Rossi", "zip": "19122"}),
        _tool("c0", _FIND, long_result),
        _assistant_call("c1", _GET, {"order_id": "#W1"}),
    ]
    record = _record(trajectory=trajectory)
    step = enumerate_steps(record)[1]

    state = serialize_step(record, _task(), step, StateProfile.full)

    assert long_result in state.text
    assert "[truncated" not in state.text
    assert not state.truncated


def test_serialize_step_compact_under_cap() -> None:
    record = _record()
    step = enumerate_steps(record)[2]  # the cancel call

    state = serialize_step(record, _task(), step, StateProfile.compact)

    assert state.profile is StateProfile.compact
    assert state.token_estimate <= 450
    assert state.text.count("Policy:") == 1
    # Prior calls appear as a normalized list, the reviewed call as the final line.
    assert f"- {_FIND}(" in state.text
    assert state.text.endswith('step under review: call: cancel_pending_order({"order_id": "#W1"})')


def test_serialize_step_hashes_unique_across_steps() -> None:
    record = _record()
    task = _task()
    hashes = {
        serialize_step(record, task, step, StateProfile.full).state_hash
        for step in enumerate_steps(record)
    }
    assert len(hashes) == 3


def test_serialize_step_runs_leakage_guard() -> None:
    leaky = [
        {"role": "user", "content": "hi"},
        _assistant_call("c0", _GET, {"order_id": "#W1"}),
        _tool("c0", _GET, '{"r_actions": ["cancel"], "reward": 1.0}'),
        _assistant_call("c1", _CANCEL, {"order_id": "#W1"}),
    ]
    record = _record(trajectory=leaky)
    task = _task()
    step = enumerate_steps(record)[1]  # cancel; its state includes the leaked harness output

    with pytest.raises(LeakageError):
        serialize_step(record, task, step, StateProfile.full)


# --- build_items -----------------------------------------------------------


def test_build_items_labels_expected_and_extra_calls() -> None:
    records = {"retail-0": _record()}
    tasks = {"retail-0": _task()}

    items = G2Steps().build_items(records, tasks, StateProfile.full)

    by_step = {item.state.step_index: item for item in items}
    assert set(by_step) == {0, 1, 2}
    assert by_step[0].truth_label == "necessary"  # find_user matches
    assert by_step[1].truth_label == "necessary"  # get_order_details matches
    assert by_step[2].truth_label == "unnecessary"  # cancel is not expected
    assert by_step[0].truth_value == 1.0
    assert by_step[2].truth_value == 0.0


def test_build_items_matches_int_and_str_kwargs() -> None:
    trajectory = [
        {"role": "user", "content": "look it up"},
        _assistant_call("c0", "get_product_details", {"product_id": "123"}),
    ]
    record = _record(trajectory=trajectory)
    task = Task(
        task_id="retail-0",
        instruction="look it up",
        actions=[ExpectedAction(name="get_product_details", kwargs={"product_id": 123})],
        outputs=[],
    )

    items = G2Steps().build_items({"retail-0": record}, {"retail-0": task}, StateProfile.full)

    assert len(items) == 1
    assert items[0].truth_label == "necessary"


def test_build_items_skips_malformed_and_counts_them() -> None:
    trajectory = [
        {"role": "user", "content": "hi"},
        _assistant_call("c0", _FIND, {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
        _assistant_call("c1", _GET, "{broken json"),
    ]
    records = {"retail-0": _record(trajectory=trajectory)}
    tasks = {"retail-0": _task()}

    items = G2Steps().build_items(records, tasks, StateProfile.full)

    assert [item.state.step_index for item in items] == [0]
    assert skipped_malformed(records) == 1


def test_build_items_skips_excluded_record() -> None:
    records = {"retail-0": _record(excluded=True)}
    tasks = {"retail-0": _task()}

    assert G2Steps().build_items(records, tasks, StateProfile.full) == []
    assert skipped_malformed(records) == 0


# --- questions -------------------------------------------------------------


def test_questions_are_two_nouls() -> None:
    questions = G2Steps().questions()
    by_id = {question.id: question for question in questions}

    assert set(by_id) == {"necessary", "arguments_consistent"}
    assert by_id["necessary"].kind is QuestionKind.noul
    assert by_id["arguments_consistent"].kind is QuestionKind.noul
    assert G2Steps().prompt_version == G2Steps().prompt_version


# --- analyze ---------------------------------------------------------------


def _item(state_hash: str, truth_label: str, step_index: int) -> Item:
    """Build an item with a full-profile state and a chosen truth label."""
    state = StateRecord(
        variant="baseline",
        task_id="retail-0",
        profile=StateProfile.full,
        text="transcript",
        token_estimate=10,
        state_hash=state_hash,
        step_index=step_index,
    )
    return Item(
        state=state,
        truth_label=truth_label,
        truth_value=1.0 if truth_label == "necessary" else 0.0,
    )


def _verdict(judge_id: str, state_hash: str, nec: float, arg: float, *, repeat: int = 0) -> Verdict:
    """Build a verdict answering both nouls for one step."""
    return Verdict(
        judge_id=judge_id,
        model_id="none",
        prompt_version="pv",
        state_hash=state_hash,
        repeat=repeat,
        answers=[
            Answer(question_id="necessary", kind=QuestionKind.noul, noul=nec),
            Answer(question_id="arguments_consistent", kind=QuestionKind.noul, noul=arg),
        ],
        usage=Usage(input_tokens=100, output_tokens=10),
        latency_ms=5,
    )


def test_analyze_scores_good_and_random_judges() -> None:
    hashes = ["h0", "h1", "h2", "h3"]
    truths = ["necessary", "necessary", "unnecessary", "unnecessary"]
    items = [_item(h, t, i) for i, (h, t) in enumerate(zip(hashes, truths, strict=True))]

    verdicts: list[Verdict] = []
    for state_hash, truth in zip(hashes, truths, strict=True):
        is_nec = truth == "necessary"
        # Good judge: necessary probability equals truth; args high when necessary.
        verdicts.append(
            _verdict("good", state_hash, 1.0 if is_nec else 0.0, 0.9 if is_nec else 0.2)
        )
        # Random judge: constant necessary probability, no separation.
        verdicts.append(_verdict("random", state_hash, 0.5, 0.5))

    result = G2Steps().analyze(verdicts, items)
    assert isinstance(result, GateResult)

    frame = result.tables["g2_summary"]
    good = frame[frame["judge_id"] == "good"].iloc[0]
    random = frame[frame["judge_id"] == "random"].iloc[0]

    assert good["n_steps"] == 4
    assert good["precision"] == pytest.approx(1.0)
    assert good["recall"] == pytest.approx(1.0)
    assert good["f1"] == pytest.approx(1.0)
    assert good["auroc"] == pytest.approx(1.0)
    assert good["mean_args_necessary"] == pytest.approx(0.9)
    assert good["mean_args_unnecessary"] == pytest.approx(0.2)
    assert good["auroc"] > random["auroc"]
    assert random["auroc"] == pytest.approx(0.5)

    assert isinstance(result.charts["g2_auroc"], Figure)
    assert "good" in result.findings


def test_analyze_averages_repeats_before_scoring() -> None:
    items = [_item("h0", "necessary", 0)]
    # Two repeats averaging to 0.6 -> predicted necessary, correct.
    verdicts = [
        _verdict("jev", "h0", 0.4, 0.8, repeat=0),
        _verdict("jev", "h0", 0.8, 0.6, repeat=1),
    ]

    frame = G2Steps().analyze(verdicts, items).tables["g2_summary"]
    row = frame[frame["judge_id"] == "jev"].iloc[0]

    assert row["n_steps"] == 1
    assert row["precision"] == pytest.approx(1.0)
    assert row["mean_args_necessary"] == pytest.approx(0.7)


def test_analyze_empty_returns_placeholder() -> None:
    result = G2Steps().analyze([], [])
    assert "No verdicts" in result.findings


# --- repeats_for -----------------------------------------------------------


def test_repeats_for_uses_study_and_defaults_unknown() -> None:
    study = load_study(STUDY_FILE)
    plan = repeats_for(study, ["jev", "llm_cheap", "mystery"])

    assert plan == {"jev": 5, "llm_cheap": 1, "mystery": 1}


def test_step_ref_is_pydantic_model() -> None:
    step = StepRef(task_id="retail-0", step_index=0, tool_call_id=None, name=_GET, arguments={})
    assert step.tool_call_id is None
    assert step.malformed is False


def test_step_items_labels_by_expected_actions() -> None:
    record = _record()
    task = _task()
    step_states = {
        (record.task_id, step.step_index): serialize_step(record, task, step, StateProfile.full)
        for step in enumerate_steps(record)
    }

    items = step_items(step_states, {record.task_id: record}, {task.task_id: task})

    by_step = {item.state.step_index: item for item in items}
    assert set(by_step) == {0, 1, 2}
    assert by_step[0].truth_label == "necessary"  # find_user matches
    assert by_step[1].truth_label == "necessary"  # get_order_details matches
    assert by_step[2].truth_label == "unnecessary"  # cancel is not expected
    assert by_step[0].truth_value == 1.0
    assert by_step[2].truth_value == 0.0


def test_step_items_skips_states_without_record() -> None:
    record = _record()
    task = _task()
    step = enumerate_steps(record)[0]
    key = (record.task_id, step.step_index)
    step_states = {key: serialize_step(record, task, step, StateProfile.full)}

    assert step_items(step_states, {}, {task.task_id: task}) == []
