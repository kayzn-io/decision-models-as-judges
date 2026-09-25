"""Tests for the taxonomy label model and JSONL store."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_judges.labels import TAXONOMY, Label, LabelStore


def _label(task_id: str, label: str, created_at: str, variant: str = "baseline") -> Label:
    """Build a label with an explicit timestamp for deterministic ordering."""
    return Label(variant=variant, task_id=task_id, label=label, created_at=created_at)


def test_label_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        Label(variant="baseline", task_id="retail-1", label="not_a_label")


def test_label_defaults() -> None:
    label = Label(variant="baseline", task_id="retail-1", label=TAXONOMY[0])
    assert label.note == ""
    assert label.labeler == "owner"
    assert label.created_at


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_label("retail-1", "other", "2020-01-01T00:00:00+00:00"))
    store.append(_label("retail-2", "premature_end", "2020-01-02T00:00:00+00:00", "degraded"))

    loaded = store.load()
    assert [label.task_id for label in loaded] == ["retail-1", "retail-2"]
    assert loaded[0].label == "other"
    assert loaded[1].variant == "degraded"


def test_latest_keeps_the_newest(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_label("retail-1", "other", "2020-01-01T00:00:00+00:00"))
    store.append(_label("retail-1", "premature_end", "2021-01-01T00:00:00+00:00"))

    latest = store.latest()
    assert list(latest) == [("baseline", "retail-1")]
    assert latest[("baseline", "retail-1")].label == "premature_end"


def test_progress_counts_distinct_pairs(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(_label("retail-1", "other", "2020-01-01T00:00:00+00:00"))
    store.append(_label("retail-1", "other", "2020-01-02T00:00:00+00:00"))
    store.append(_label("retail-2", "other", "2020-01-03T00:00:00+00:00", "degraded"))

    labeled, target = store.progress()
    assert labeled == 2
    assert target == 50


def test_progress_target_is_configurable(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    assert store.progress(target=10) == (0, 10)
