"""Pure functions comparing judges against ground truth.

Covers agreement, repeatability, calibration, cost, latency, regression,
robustness, and correlation metrics over sequences of plain Python values.
"""

from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

DEFAULT_TEMP_GRID: tuple[float, ...] = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
)


def _as_array(values: Sequence[object]) -> np.ndarray:
    """Convert a sequence to a numpy array without changing element dtype."""
    return np.asarray(list(values))


def _as_float(values: Sequence[float]) -> np.ndarray:
    """Convert a sequence to a 1-D float64 numpy array."""
    return np.asarray(list(values), dtype=float)


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Return 1-based ranks of values with ties resolved by average rank."""
    arr = _as_float(values)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


class PRF(NamedTuple):
    """Precision, recall, and F1 for a single positive label."""

    precision: float
    recall: float
    f1: float


class ModalAgreement(NamedTuple):
    """Per-item modal agreement fractions and their mean."""

    per_item: list[float]
    mean: float


class ScoreStd(NamedTuple):
    """Per-item score standard deviations and their mean."""

    per_item: list[float]
    mean: float


class ReliabilityBin(NamedTuple):
    """One calibration bin: bounds, mean probability, positive rate, count."""

    lower: float
    upper: float
    mean_prob: float
    frac_positive: float
    size: int


class LatencySummary(NamedTuple):
    """Median and 95th-percentile latency."""

    p50: float
    p95: float


class BootstrapDelta(NamedTuple):
    """Point estimate mean(b)-mean(a) with a 95% percentile interval."""

    delta: float
    lo: float
    hi: float


def accuracy(pred: Sequence[object], truth: Sequence[object]) -> float:
    """Return the fraction of predictions that equal the ground truth."""
    p = _as_array(pred)
    t = _as_array(truth)
    return float(np.mean(p == t))


def precision_recall_f1(pred: Sequence[object], truth: Sequence[object], positive: object) -> PRF:
    """Return precision, recall, and F1 for the positive label.

    precision = TP/(TP+FP), recall = TP/(TP+FN), f1 = 2PR/(P+R).
    """
    p = _as_array(pred)
    t = _as_array(truth)
    tp = int(np.sum((p == positive) & (t == positive)))
    fp = int(np.sum((p == positive) & (t != positive)))
    fn = int(np.sum((p != positive) & (t == positive)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1)


def cohens_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Return Cohen's kappa: (po - pe) / (1 - pe).

    po is observed agreement; pe is the agreement expected from the raters'
    marginal label frequencies.
    """
    aa = _as_array(a)
    bb = _as_array(b)
    po = float(np.mean(aa == bb))
    labels = set(aa.tolist()) | set(bb.tolist())
    pe = 0.0
    for label in labels:
        pe += (np.mean(aa == label)) * (np.mean(bb == label))
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def auroc(scores: Sequence[float], truth: Sequence[object]) -> float:
    """Return the area under the ROC curve via the Mann-Whitney statistic.

    AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg),
    using average ranks so ties count as half.
    """
    t = _as_array(truth)
    ranks = _ranks(scores)
    pos = t == 1
    n_pos = int(np.sum(pos))
    n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = float(np.sum(ranks[pos]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def modal_agreement(verdicts_per_item: Sequence[Sequence[str]]) -> ModalAgreement:
    """Return each item's fraction of repeats equal to its mode, and the mean."""
    per_item: list[float] = []
    for verdicts in verdicts_per_item:
        items = list(verdicts)
        mode_count = Counter(items).most_common(1)[0][1]
        per_item.append(mode_count / len(items))
    mean = float(np.mean(per_item)) if per_item else 0.0
    return ModalAgreement(per_item, mean)


def score_std(scores_per_item: Sequence[Sequence[float]]) -> ScoreStd:
    """Return each item's population standard deviation of scores, and the mean."""
    per_item = [float(np.std(_as_float(scores))) for scores in scores_per_item]
    mean = float(np.mean(per_item)) if per_item else 0.0
    return ScoreStd(per_item, mean)


def _bin_index(probs: np.ndarray, bins: int) -> np.ndarray:
    """Assign each probability to an equal-width bin index in [0, bins-1]."""
    idx = np.floor(probs * bins).astype(int)
    return np.clip(idx, 0, bins - 1)


def ece(probs: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    """Return expected calibration error over equal-width bins.

    ECE = sum_b (n_b/N) * |acc_b - conf_b|.
    """
    p = _as_float(probs)
    y = _as_float(labels)
    idx = _bin_index(p, bins)
    n = len(p)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        total += (count / n) * abs(acc - conf)
    return total


def brier(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Return the Brier score: mean((prob - label)^2)."""
    p = _as_float(probs)
    y = _as_float(labels)
    return float(np.mean((p - y) ** 2))


def reliability_bins(
    probs: Sequence[float], labels: Sequence[float], bins: int = 10
) -> list[ReliabilityBin]:
    """Return non-empty calibration bins as bounds, mean prob, pos rate, count."""
    p = _as_float(probs)
    y = _as_float(labels)
    idx = _bin_index(p, bins)
    width = 1.0 / bins
    out: list[ReliabilityBin] = []
    for b in range(bins):
        mask = idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=b * width,
                upper=(b + 1) * width,
                mean_prob=float(np.mean(p[mask])),
                frac_positive=float(np.mean(y[mask])),
                size=count,
            )
        )
    return out


def isotonic_fit_transform(probs: Sequence[float], labels: Sequence[float]) -> np.ndarray:
    """Return probabilities recalibrated by isotonic regression.

    Fits a monotone non-decreasing map from probs to labels and applies it,
    so the output is non-decreasing in the input.
    """
    p = _as_float(probs)
    y = _as_float(labels)
    model = IsotonicRegression(out_of_bounds="clip")
    return np.asarray(model.fit_transform(p, y), dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid of x."""
    return 1.0 / (1.0 + np.exp(-x))


def temperature_fit(
    probs: Sequence[float],
    labels: Sequence[float],
    grid: Sequence[float] = DEFAULT_TEMP_GRID,
) -> tuple[float, np.ndarray]:
    """Return the grid temperature minimizing ECE and the scaled probabilities.

    Scales logits by 1/T (logit -> logit/T -> sigmoid) and picks the T on the
    grid with the lowest ECE.
    """
    p = np.clip(_as_float(probs), 1e-6, 1 - 1e-6)
    y = _as_float(labels)
    logits = np.log(p / (1 - p))
    best_t = float(grid[0])
    best_probs = _sigmoid(logits / best_t)
    best_ece = ece(best_probs.tolist(), y.tolist())
    for t in grid[1:]:
        scaled = _sigmoid(logits / t)
        current = ece(scaled.tolist(), y.tolist())
        if current < best_ece:
            best_ece = current
            best_t = float(t)
            best_probs = scaled
    return best_t, best_probs


def cost_per_verdict(total_usd: float, n: int) -> float:
    """Return the mean USD cost per verdict."""
    return total_usd / n


def percentile(values: Sequence[float], q: float) -> float:
    """Return the q-th percentile (q in [0, 100]) with linear interpolation."""
    return float(np.percentile(_as_float(values), q))


def latency_summary(values: Sequence[float]) -> LatencySummary:
    """Return the 50th and 95th latency percentiles."""
    arr = _as_float(values)
    return LatencySummary(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
    )


def bootstrap_delta(
    a_labels: Sequence[float],
    b_labels: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapDelta:
    """Return mean(b)-mean(a) and its 95% bootstrap percentile interval."""
    a = _as_float(a_labels)
    b = _as_float(b_labels)
    delta = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    deltas = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDelta(delta, float(lo), float(hi))


def flip_rate(before: Sequence[object], after: Sequence[object]) -> float:
    """Return the fraction of items whose label changed."""
    b = _as_array(before)
    a = _as_array(after)
    return float(np.mean(b != a))


def net_flip_rate(
    before: Sequence[object],
    after: Sequence[object],
    control_before: Sequence[object],
    control_after: Sequence[object],
) -> float:
    """Return the injected flip rate minus the control flip rate."""
    return flip_rate(before, after) - flip_rate(control_before, control_after)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Spearman rank correlation: Pearson correlation of average ranks."""
    rx = _ranks(x)
    ry = _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])
