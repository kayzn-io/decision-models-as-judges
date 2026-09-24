"""Deterministic k-fold assignment over task ids.

Ids are sorted, shuffled with a seeded random generator, and dealt round-robin
into folds so fold sizes differ by at most one.
"""

import random
from collections.abc import Mapping, Sequence


def assign_folds(task_ids: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """Assign each task id to a fold in ``[0, k)`` deterministically.

    Ids are sorted for a stable starting order, shuffled with
    ``random.Random(seed)``, then dealt round-robin, so the same inputs always
    produce the same assignment and fold sizes differ by at most one.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    ids = sorted(task_ids)
    random.Random(seed).shuffle(ids)
    return {task_id: index % k for index, task_id in enumerate(ids)}


def fold_of(assignments: Mapping[str, int], task_id: str) -> int:
    """Return the fold a task id was assigned to."""
    return assignments[task_id]


def train_test_split(assignments: Mapping[str, int], fold: int) -> tuple[list[str], list[str]]:
    """Return the ``(train, test)`` id lists for one held-out fold.

    Test ids are those assigned to ``fold``; train ids are all others. Both
    lists are sorted, so they are disjoint and together cover every id.
    """
    train = sorted(task_id for task_id, value in assignments.items() if value != fold)
    test = sorted(task_id for task_id, value in assignments.items() if value == fold)
    return train, test
