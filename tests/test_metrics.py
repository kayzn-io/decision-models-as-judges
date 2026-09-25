"""Tests for judge evaluation metrics against hand-computed values."""

import math

import numpy as np

from decision_judges.metrics import (
    accuracy,
    auroc,
    bootstrap_delta,
    brier,
    cohens_kappa,
    cost_per_verdict,
    ece,
    flip_rate,
    isotonic_fit_transform,
    latency_summary,
    mean,
    modal_agreement,
    net_flip_rate,
    percentile,
    precision_recall_f1,
    reliability_bins,
    score_std,
    spearman,
    temperature_fit,
)


def test_mean_arithmetic() -> None:
    assert math.isclose(mean([1.0, 2.0, 3.0]), 2.0)
    assert math.isclose(mean([0.2, 0.4]), 0.3)


def test_mean_empty_is_nan() -> None:
    assert math.isnan(mean([]))


def test_accuracy() -> None:
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0


def test_precision_recall_f1() -> None:
    # TP=2 (idx0,1), FP=1 (idx2), FN=1 (idx4); TN at idx3.
    pred = [1, 1, 1, 0, 0]
    truth = [1, 1, 0, 0, 1]
    res = precision_recall_f1(pred, truth, positive=1)
    assert math.isclose(res.precision, 2 / 3)
    assert math.isclose(res.recall, 2 / 3)
    assert math.isclose(res.f1, 2 / 3)


def test_precision_recall_f1_no_positive_predictions() -> None:
    res = precision_recall_f1([0, 0], [1, 1], positive=1)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0


def test_cohens_kappa_textbook() -> None:
    a = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    # po=0.7, pe=0.5 -> kappa=0.4
    assert math.isclose(cohens_kappa(a, b), 0.4)


def test_cohens_kappa_perfect() -> None:
    assert math.isclose(cohens_kappa([1, 0, 1], [1, 0, 1]), 1.0)


def test_auroc_no_ties() -> None:
    scores = [0.1, 0.4, 0.35, 0.8]
    truth = [0, 0, 1, 1]
    assert math.isclose(auroc(scores, truth), 0.75)


def test_auroc_ties_average_ranks() -> None:
    scores = [1.0, 1.0, 2.0, 2.0]
    truth = [0, 1, 0, 1]
    assert math.isclose(auroc(scores, truth), 0.5)


def test_modal_agreement() -> None:
    res = modal_agreement([["a", "a", "b"], ["x", "x", "x"]])
    assert math.isclose(res.per_item[0], 2 / 3)
    assert math.isclose(res.per_item[1], 1.0)
    assert math.isclose(res.mean, 5 / 6)


def test_score_std() -> None:
    res = score_std([[1.0, 1.0, 1.0], [0.0, 2.0]])
    assert math.isclose(res.per_item[0], 0.0)
    assert math.isclose(res.per_item[1], 1.0)
    assert math.isclose(res.mean, 0.5)


def test_ece_hand_value() -> None:
    probs = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    # each in its own bin: (0.2+0.4+0.4+0.2)/4 = 0.3
    assert math.isclose(ece(probs, labels, bins=10), 0.3)


def test_brier_hand_value() -> None:
    probs = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    # (0.04+0.16+0.16+0.04)/4 = 0.1
    assert math.isclose(brier(probs, labels), 0.1)


def test_reliability_bins() -> None:
    probs = [0.2, 0.4, 0.6, 0.8]
    labels = [0, 0, 1, 1]
    bins = reliability_bins(probs, labels, bins=2)
    assert len(bins) == 2
    lo0, hi0, mean0, frac0, count0 = bins[0]
    assert math.isclose(lo0, 0.0) and math.isclose(hi0, 0.5)
    assert math.isclose(mean0, 0.3)
    assert math.isclose(frac0, 0.0)
    assert count0 == 2
    lo1, hi1, mean1, frac1, count1 = bins[1]
    assert math.isclose(mean1, 0.7)
    assert math.isclose(frac1, 1.0)
    assert count1 == 2


def test_reliability_bins_skips_empty() -> None:
    probs = [0.05, 0.95]
    labels = [0, 1]
    bins = reliability_bins(probs, labels, bins=10)
    assert len(bins) == 2


def test_isotonic_monotone_in_input() -> None:
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 1]
    out = isotonic_fit_transform(probs, labels)
    assert np.all(np.diff(out) >= -1e-9)


def test_temperature_fit_lowers_ece() -> None:
    probs = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
    labels = [1, 1, 0, 0, 1, 1, 0, 0]
    best_t, transformed = temperature_fit(probs, labels)
    assert ece(transformed, labels) < ece(probs, labels)
    assert best_t >= 1.0


def test_cost_per_verdict() -> None:
    assert math.isclose(cost_per_verdict(10.0, 4), 2.5)


def test_percentile_linear() -> None:
    assert math.isclose(percentile([1, 2, 3, 4], 50), 2.5)
    assert math.isclose(percentile([1, 2, 3, 4], 0), 1.0)
    assert math.isclose(percentile([1, 2, 3, 4], 100), 4.0)


def test_latency_summary() -> None:
    res = latency_summary([1, 2, 3, 4])
    assert math.isclose(res.p50, 2.5)
    assert math.isclose(res.p95, 3.85)


def test_bootstrap_delta_point_estimate() -> None:
    res = bootstrap_delta([0, 0, 1, 1], [1, 1, 1, 1], n_boot=500, seed=0)
    assert math.isclose(res.delta, 0.5)
    assert res.lo <= res.hi


def test_bootstrap_delta_identical_contains_zero() -> None:
    labels = [0, 1, 0, 1, 1, 0]
    res = bootstrap_delta(labels, labels, n_boot=2000, seed=0)
    assert math.isclose(res.delta, 0.0)
    assert res.lo <= 0.0 <= res.hi


def test_flip_rate() -> None:
    assert math.isclose(flip_rate([1, 0, 1], [1, 1, 1]), 1 / 3)
    assert flip_rate(["a", "b"], ["a", "b"]) == 0.0


def test_net_flip_rate() -> None:
    # injected flips 2/3, control flips 1/3 -> net 1/3
    net = net_flip_rate([1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 0])
    assert math.isclose(net, 1 / 3)


def test_spearman_perfect() -> None:
    assert math.isclose(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 1.0)


def test_spearman_anti() -> None:
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_spearman_with_ties() -> None:
    x = [1, 2, 2, 3]
    y = [1, 2, 3, 4]
    # monotone non-decreasing relation -> positive, < 1 due to tie
    val = spearman(x, y)
    assert 0.0 < val < 1.0
