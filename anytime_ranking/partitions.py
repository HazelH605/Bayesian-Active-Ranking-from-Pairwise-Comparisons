"""Ordered-partition labels and normalized Kemeny distance."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def validate_cut_points(cut_points, n_models):
    if len(cut_points) < 2 or cut_points[-1] != n_models:
        raise ValueError("cut_points must end at n_models")
    if cut_points[0] < 1 or any(a >= b for a, b in zip(cut_points, cut_points[1:])):
        raise ValueError("cut_points must be positive and strictly increasing")


def tied_pair_count(cut_points, n_models):
    validate_cut_points(cut_points, n_models)
    sizes = np.diff(np.asarray((0, *cut_points), dtype=int))
    return int(np.sum(sizes * (sizes - 1) // 2))


def report_from_scores(scores, cut_points):
    """Assign partition labels by descending scores and prescribed group sizes."""
    scores = np.asarray(scores, dtype=float)
    validate_cut_points(cut_points, scores.size)
    # Break score ties by model index.
    order = np.argsort(-scores, kind="stable")
    labels = np.empty(scores.size, dtype=np.int64)
    lower = 0
    for block, upper in enumerate(cut_points):
        labels[order[lower:upper]] = block
        lower = upper
    return labels


def ordered_sets(labels):
    return [np.flatnonzero(labels == block).tolist() for block in range(int(labels.max()) + 1)]


def _sample_report_labels(samples, cut_points):
    """Vectorized true ordered partitions for posterior draws."""
    order = np.argsort(-samples, axis=1, kind="stable")
    labels = np.empty(order.shape, dtype=np.int64)
    lower = 0
    rows = np.arange(len(samples))[:, None]
    for block, upper in enumerate(cut_points):
        labels[rows, order[:, lower:upper]] = block
        lower = upper
    return labels


def consensus_score(report_labels, weak_order):
    """F(S;z) = sum over ordered distinct pairs c_ij w_S(i,j)."""
    return float(np.sum(weak_order * (report_labels[:, None] <= report_labels[None, :])))


