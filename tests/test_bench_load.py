"""Tests for loading and normalizing tau-bench retail tasks.

Unit tests read the checked-in fixture and never import tau_bench. The single
tau_bench-marked test imports the real package and is skipped when it is absent.
"""

import json
from pathlib import Path

import pytest

from decision_judges.bench.load import (
    ExpectedAction,
    Task,
    export_tasks,
    is_write_action,
    load_tasks,
    normalize_action,
    write_actions_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tasks_sample.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_source() -> list[object]:
    """Return the parsed native-shape task list from the fixture file."""
    return json.loads(FIXTURE.read_text())


def test_load_tasks_from_source() -> None:
    tasks = load_tasks(source=_fixture_source())
    assert len(tasks) == 3
    for i, task in enumerate(tasks):
        assert isinstance(task, Task)
        assert task.task_id == f"retail-{i}"
        assert task.instruction.strip()
        assert isinstance(task.outputs, list)
        assert task.actions
        for action in task.actions:
            assert isinstance(action, ExpectedAction)
            assert action.name
            assert isinstance(action.kwargs, dict)


def test_load_tasks_rejects_non_retail() -> None:
    with pytest.raises(ValueError):
        load_tasks(domain="airline", source=_fixture_source())


def test_normalize_action_value_type_insensitive() -> None:
    assert normalize_action("t", {"a": 1}) == normalize_action("t", {"a": "1"})


def test_normalize_action_order_insensitive() -> None:
    assert normalize_action("t", {"a": 1, "b": 2}) == normalize_action("t", {"b": 2, "a": 1})


def test_normalize_action_name_case_insensitive() -> None:
    assert normalize_action("GetOrder", {}) == normalize_action("getorder", {})


def test_normalize_action_stringifies_containers_canonically() -> None:
    left = normalize_action("t", {"ids": {"b": 2, "a": 1}})
    right = normalize_action("t", {"ids": {"a": 1, "b": 2}})
    assert left == right


def test_write_actions_for_returns_only_writes() -> None:
    tasks = load_tasks(source=_fixture_source())
    write_tasks = [t for t in tasks if write_actions_for(t)]
    assert write_tasks, "fixture must contain at least one task with a write action"
    for task in write_tasks:
        writes = write_actions_for(task)
        assert writes
        for action in writes:
            assert is_write_action(action.name)


def test_is_write_action_rejects_read_tool() -> None:
    assert is_write_action("cancel_pending_order")
    assert not is_write_action("get_order_details")
    assert not is_write_action("find_user_id_by_name_zip")


@pytest.mark.tau_bench
def test_load_tasks_from_real_package() -> None:
    pytest.importorskip("tau_bench")
    tasks = load_tasks()
    assert len(tasks) == 115


def test_export_tasks_writes_round_trippable_file(monkeypatch, tmp_path: Path) -> None:
    from decision_judges.bench import load as bench_load

    fixture_tasks = load_tasks(source=_fixture_source())
    monkeypatch.setattr(bench_load, "load_tasks", lambda: fixture_tasks)

    out = tmp_path / "nested" / "tasks.json"
    count = bench_load.export_tasks(out)

    assert count == len(fixture_tasks)
    assert out.is_file()
    reloaded = load_tasks(source=json.loads(out.read_text(encoding="utf-8")))
    assert reloaded == fixture_tasks


@pytest.mark.tau_bench
def test_export_tasks_from_real_package(tmp_path: Path) -> None:
    pytest.importorskip("tau_bench")
    out = tmp_path / "tasks.json"

    count = export_tasks(out)

    assert count == 115
    reloaded = load_tasks(source=json.loads(out.read_text(encoding="utf-8")))
    assert reloaded == load_tasks()


def test_shipped_tasks_file_has_115_entries() -> None:
    path = REPO_ROOT / "data" / "tasks.json"
    assert path.is_file(), "data/tasks.json must ship with the study material"
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 115
    for entry in entries:
        assert {"instruction", "actions", "outputs"} <= set(entry)
        for action in entry["actions"]:
            assert set(action) == {"name", "kwargs"}
