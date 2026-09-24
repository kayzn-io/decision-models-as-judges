"""Tests for deterministic k-fold assignment over task ids."""

from decision_judges.training.folds import assign_folds, fold_of, train_test_split

TASK_IDS = [f"retail-{index}" for index in range(10)]


def test_assign_folds_is_deterministic() -> None:
    first = assign_folds(TASK_IDS, k=3, seed=7)
    second = assign_folds(list(reversed(TASK_IDS)), k=3, seed=7)
    assert first == second


def test_assign_folds_places_every_id_in_exactly_one_fold() -> None:
    assignments = assign_folds(TASK_IDS, k=3, seed=7)
    assert set(assignments) == set(TASK_IDS)
    assert all(0 <= fold < 3 for fold in assignments.values())


def test_assign_folds_balanced_within_one() -> None:
    assignments = assign_folds(TASK_IDS, k=3, seed=7)
    sizes = [sum(1 for fold in assignments.values() if fold == index) for index in range(3)]
    assert max(sizes) - min(sizes) <= 1


def test_fold_of_returns_the_assigned_fold() -> None:
    assignments = assign_folds(TASK_IDS, k=4, seed=1)
    for task_id, fold in assignments.items():
        assert fold_of(assignments, task_id) == fold


def test_train_test_split_is_disjoint_and_complete() -> None:
    assignments = assign_folds(TASK_IDS, k=5, seed=3)
    for fold in range(5):
        train, test = train_test_split(assignments, fold)
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(TASK_IDS)
        assert all(assignments[task_id] == fold for task_id in test)
        assert all(assignments[task_id] != fold for task_id in train)
