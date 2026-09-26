"""Load tau-bench retail tasks and normalize expected tool calls.

Package facts (tau-bench @ git 59a200c, ``tau_bench.types`` and
``tau_bench.envs.retail``):

- Retail test tasks are a module-level list ``TASKS_TEST`` in
  ``tau_bench.envs.retail.tasks_test`` (dev/train splits live in the sibling
  ``tasks_dev``/``tasks_train`` modules; the env selects one via its
  ``task_split`` constructor argument). ``load_tasks`` reads the test split,
  which holds 115 tasks.
- A tau-bench ``Task`` (pydantic) exposes ``user_id: str``,
  ``actions: list[Action]``, ``instruction: str`` and ``outputs: list[str]``.
  Each ``Action`` has ``name: str`` and ``kwargs: dict``. The task lists also
  pass an ``annotator`` string that the model ignores; tasks carry no id, so
  this module assigns a stable positional id ``"retail-{index}"``.
- Retail tools are registered in ``tau_bench.envs.retail.tools.ALL_TOOLS``.
  The seven that mutate the data store (and so are write actions) are
  ``cancel_pending_order``, ``exchange_delivered_order_items``,
  ``modify_pending_order_address``, ``modify_pending_order_items``,
  ``modify_pending_order_payment``, ``modify_user_address`` and
  ``return_delivered_order_items``. The rest (``calculate``,
  ``find_user_id_by_email``, ``find_user_id_by_name_zip``,
  ``get_order_details``, ``get_product_details``, ``get_user_details``,
  ``list_all_product_types``, ``think``, ``transfer_to_human_agents``) read or
  are no-ops.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

RETAIL_DOMAIN = "retail"

WRITE_ACTIONS: frozenset[str] = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)


class ExpectedAction(BaseModel):
    """A tool call the reference solution takes, by name and keyword arguments."""

    name: str
    kwargs: dict[str, object]


class Task(BaseModel):
    """A retail task: its instruction, expected actions, and required outputs."""

    task_id: str
    instruction: str
    actions: list[ExpectedAction]
    outputs: list[str]


def _canonical_value(value: object) -> str:
    """Stringify a value canonically so equal contents compare equal."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normalize_action(
    name: str, kwargs: Mapping[str, object]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return a comparable form of an action, insensitive to name case,
    keyword order, and scalar value type."""
    normalized_kwargs = tuple((key, _canonical_value(kwargs[key])) for key in sorted(kwargs))
    return name.lower(), normalized_kwargs


def is_write_action(name: str) -> bool:
    """Return whether a tool name mutates retail state."""
    return name in WRITE_ACTIONS


def write_actions_for(task: Task) -> list[ExpectedAction]:
    """Return the task's expected actions that mutate state."""
    return [action for action in task.actions if is_write_action(action.name)]


def _task_from_native(index: int, raw: object) -> Task:
    """Convert one native-shape task mapping into a typed Task."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"task at index {index} is not a mapping")
    raw_actions = raw.get("actions", [])
    if not isinstance(raw_actions, Sequence):
        raise ValueError(f"task at index {index} has non-sequence actions")
    actions = [
        ExpectedAction(name=str(action["name"]), kwargs=dict(action["kwargs"]))
        for action in raw_actions
    ]
    return Task(
        task_id=f"{RETAIL_DOMAIN}-{index}",
        instruction=str(raw["instruction"]),
        actions=actions,
        outputs=[str(output) for output in raw.get("outputs", [])],
    )


def load_tasks(domain: str = RETAIL_DOMAIN, source: Sequence[object] | None = None) -> list[Task]:
    """Load retail tasks as typed Tasks.

    With ``source`` given, convert that parsed native-shape list. With
    ``source=None``, import the tau-bench retail test split and convert it.
    """
    if domain != RETAIL_DOMAIN:
        raise ValueError(f"unsupported domain: {domain!r}")
    if source is None:
        from tau_bench.envs.retail.tasks_test import TASKS_TEST

        source = [task.model_dump() for task in TASKS_TEST]
    return [_task_from_native(index, raw) for index, raw in enumerate(source)]


def _task_to_native(task: Task) -> dict[str, object]:
    """Convert a typed Task back into a native-shape mapping ``load_tasks`` reads."""
    return {
        "instruction": task.instruction,
        "actions": [{"name": action.name, "kwargs": action.kwargs} for action in task.actions],
        "outputs": list(task.outputs),
    }


def export_tasks(path: Path) -> int:
    """Write the retail tasks to a JSON file in native shape and return the count.

    Load the tau-bench retail tasks, render each back to the native record shape
    that ``load_tasks(source=...)`` reads, and write them pretty-printed with
    sorted keys, creating the parent directory when it is absent.
    """
    tasks = load_tasks()
    native = [_task_to_native(task) for task in tasks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(tasks)
