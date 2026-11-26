"""Find Bayes reports with reusable exact optimization certificates."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import factorial
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from .partitions import IntArray, tied_pair_count


@dataclass(frozen=True)
class ReportCertificate:
    """Lower bounds on score gaps for reports moving each model's block.

    These are exact integers for the fixed posterior sample set. They may be
    conservative after a certified restricted solve.
    """

    labels: IntArray
    parameters: IntArray
    gap_by_model: IntArray
    cut_points: tuple[int, ...]
    evaluator_id: tuple[int, int, int]
    n_samples: int
    feature_kind: str
    candidates_evaluated: int


@dataclass(frozen=True)
class _ReportCatalog:
    labels: IntArray
    weights: NDArray[np.int8]
    top_features: NDArray[np.int64] | None
    base_weight: NDArray[np.int8]
    edge_delta: csr_matrix
    path_sum: csr_matrix
    parent: IntArray
    traversal: IntArray


def _report_count(cut_points, n_models):
    sizes = np.diff(np.asarray((0, *cut_points), dtype=int))
    count = factorial(n_models)
    for size in sizes:
        count //= factorial(int(size))
    return count


def _valid_reports(cut_points, n_models):
    sizes = np.diff(np.asarray((0, *cut_points), dtype=int))
    labels = np.empty(n_models, dtype=np.int64)

    def visit(block, remaining):
        if block == len(sizes) - 1:
            labels[list(remaining)] = block
            yield labels.copy()
            return
        for chosen in combinations(remaining, int(sizes[block])):
            labels[list(chosen)] = block
            rest = tuple(item for item in remaining if item not in chosen)
            yield from visit(block + 1, rest)

    yield from visit(0, tuple(range(n_models)))


@lru_cache(maxsize=16)
def _report_catalog(cut_points, n_models):
    """Cache candidate reports when there are at most 5000 possibilities."""
    if _report_count(cut_points, n_models) > 5000:
        return None
    labels = np.stack(tuple(_valid_reports(cut_points, n_models)))
    n_reports = len(labels)
    weights = (labels[:, :, None] <= labels[:, None, :]).astype(np.int8)
    diagonal = np.arange(n_models)
    weights[:, diagonal, diagonal] = 0
    weights = weights.reshape(n_reports, n_models * n_models)

    # Build a spanning tree of the swap graph once. Only relations touching
    # the two swapped models differ, so each edge has O(M) nonzero entries.
    index = {tuple(row): position for position, row in enumerate(labels)}
    parent = np.full(n_reports, -1, dtype=np.int64)
    parent[0] = 0
    traversal = [0]
    edge_rows = []
    edge_cols = []
    edge_values = []
    for position in traversal:
        row = labels[position]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if row[i] == row[j]:
                    continue
                swapped = row.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                child = index[tuple(swapped)]
                if parent[child] >= 0:
                    continue
                parent[child] = position
                traversal.append(child)
                difference = weights[child] - weights[position]
                nonzero = np.flatnonzero(difference)
                edge_rows.extend([child] * len(nonzero))
                edge_cols.extend(nonzero.tolist())
                edge_values.extend(difference[nonzero].tolist())
    if len(traversal) != n_reports:
        raise RuntimeError("the ordered-partition swap graph is disconnected")
    edge_delta = csr_matrix(
        (np.asarray(edge_values, dtype=np.int8), (edge_rows, edge_cols)),
        shape=(n_reports, n_models * n_models),
    )
    path_rows = []
    path_cols = []
    for node in range(1, n_reports):
        ancestor = node
        while ancestor != 0:
            path_rows.append(node)
            path_cols.append(ancestor)
            ancestor = int(parent[ancestor])
    path_sum = csr_matrix(
        (
            np.ones(len(path_rows), dtype=np.int8),
            (path_rows, path_cols),
        ),
        shape=(n_reports, n_reports),
    )

    top_features = None
    if len(cut_points) == 2:
        is_top = (labels == 0).astype(np.int64)
        i, j = np.triu_indices(n_models, 1)
        coefficient = n_models + 1 - 2 * cut_points[0]
        top_features = np.concatenate(
            (coefficient * is_top, 2 * is_top[:, i] * is_top[:, j]),
            axis=1,
        )
    return _ReportCatalog(
        labels, weights, top_features, weights[0],
        edge_delta, path_sum, parent, np.asarray(traversal, dtype=np.int64),
    )


def _subset_dp_report(weak_order, cut_points):
    # Use dynamic programming over group membership for larger report spaces.
    """Exact exponential fallback when the report catalog is too large."""
    n_models = weak_order.shape[0]
    sizes = tuple(int(size) for size in np.diff((0, *cut_points)))

    @lru_cache(maxsize=None)
    def solve(mask, block):
        remaining = tuple(i for i in range(n_models) if mask & (1 << i))
        if block == len(sizes) - 1:
            score = sum(
                float(weak_order[i, j])
                for i in remaining for j in remaining if i != j
            )
            return score, (remaining,)
        best_score = float("-inf")
        best_blocks = None
        for chosen in combinations(remaining, sizes[block]):
            chosen_set = set(chosen)
            rest = tuple(i for i in remaining if i not in chosen_set)
            rest_mask = mask ^ sum(1 << i for i in chosen)
            immediate = sum(
                float(weak_order[i, j])
                for i in chosen for j in chosen if i != j
            ) + sum(float(weak_order[i, j]) for i in chosen for j in rest)
            tail_score, tail_blocks = solve(rest_mask, block + 1)
            score = immediate + tail_score
            if score > best_score:
                best_score = score
                best_blocks = (chosen, *tail_blocks)
        assert best_blocks is not None
        return best_score, best_blocks

    best_score, blocks = solve((1 << n_models) - 1, 0)
    labels = np.empty(n_models, dtype=np.int64)
    for block, members in enumerate(blocks):
        labels[list(members)] = block
    return labels, best_score


def _catalog_scores(catalog, parameters, top_k):
    if top_k:
        assert catalog.top_features is not None
        return catalog.top_features @ parameters
    # F(child) = F(parent) + Delta_{parent,child}. The sparse edge matrix
    # contains only relations touching the two swapped models.
    deltas = catalog.edge_delta @ parameters
    return np.dot(catalog.base_weight, parameters) + catalog.path_sum @ deltas


def _score_change_bound(old, new, n_models, top_k, cut_points):
    difference = np.abs(new - old)
    if top_k:
        coefficient = abs(n_models + 1 - 2 * cut_points[0])
        return int(
            coefficient * np.sum(difference[:n_models])
            + 2 * np.sum(difference[n_models:])
        )
    return int(np.sum(difference))


def _sample_optimal_report(true_labels, cut_points, evaluator_id, previous):
    """Find the globally optimal report for the fixed posterior integration samples.

    Certificates skip candidates only when the unchanged optimum is proven.
    """
    n_samples, n_models = true_labels.shape
    q = n_models * (n_models - 1) // 2
    top_k = len(cut_points) == 2
    if top_k:
        feature_kind = 'top_k'
    else:
        feature_kind = 'weak_order'
    if top_k:
        k = cut_points[0]
        membership = (true_labels == 0).astype(np.int64)
        first = membership.sum(axis=0, dtype=np.int64)
        second = membership.T @ membership
        i, j = np.triu_indices(n_models, 1)
        parameters = np.concatenate((first, second[i, j]))
        # If d = k - |A intersection T|, then Q*d_K = d*(M-d).
        # E[Q*d_K]*N = k*(M-k)*N - J(A), where J is a quadratic
        # function of posterior first and second inclusion counts.
        risk_constant = k * (n_models - k) * n_samples
    else:
        comparison = true_labels[:, :, None] <= true_labels[:, None, :]
        counts = np.sum(comparison, axis=0, dtype=np.int64)
        np.fill_diagonal(counts, 0)
        parameters = counts.ravel()
        risk_constant = (
            q + tied_pair_count(cut_points, n_models)
        ) * n_samples

    catalog = _report_catalog(cut_points, n_models)
    if catalog is None:
        if top_k:
            # The subset-DP fallback needs weak-order counts rather than
            # the top-k sufficient statistics used by the small catalog.
            counts = np.sum(
                true_labels[:, :, None] <= true_labels[:, None, :],
                axis=0, dtype=np.int64,
            )
            np.fill_diagonal(counts, 0)
        else:
            counts = parameters.reshape(n_models, n_models)
        labels, score = _subset_dp_report(counts, cut_points)
        general_constant = (
            q + tied_pair_count(cut_points, n_models)
        ) * n_samples
        return labels, float((general_constant - score) / (n_samples * q)), None

    valid_previous = (
        previous is not None
        and previous.evaluator_id == evaluator_id
        and previous.n_samples == n_samples
        and previous.cut_points == cut_points
        and previous.feature_kind == feature_kind
        and previous.parameters.shape == parameters.shape
    )
    bound = 0
    fixed = np.zeros(n_models, dtype=bool)
    candidate_ids = np.arange(len(catalog.labels))
    if valid_previous:
        assert previous is not None
        bound = _score_change_bound(
            previous.parameters, parameters, n_models, top_k, cut_points
        )
        fixed = previous.gap_by_model > bound
        if np.any(fixed):
            candidate_ids = np.flatnonzero(
                np.all(
                    catalog.labels[:, fixed] == previous.labels[fixed],
                    axis=1,
                )
            )
            if len(candidate_ids) == 0:
                raise RuntimeError("certified report subset must be nonempty")

    if len(candidate_ids) < len(catalog.labels) // 2:
        if top_k:
            assert catalog.top_features is not None
            scores = catalog.top_features[candidate_ids] @ parameters
        else:
            scores = catalog.weights[candidate_ids] @ parameters
        chosen = int(candidate_ids[int(np.argmax(scores))])
        best_score = int(np.max(scores))
        assert previous is not None
        # Moving a fixed model still loses at least old_gap - bound.
        # Zero is a valid, if uninformative, bound for active models.
        gaps = np.where(
            fixed, np.maximum(previous.gap_by_model - bound, 0), 0
        )
        candidates_evaluated = len(candidate_ids)
    else:
        scores = _catalog_scores(catalog, parameters, top_k)
        chosen = int(np.argmax(scores))
        best_score = int(scores[chosen])
        gaps = np.empty(n_models, dtype=np.int64)
        for model in range(n_models):
            alternatives = scores[
                catalog.labels[:, model] != catalog.labels[chosen, model]
            ]
            if len(alternatives):
                gaps[model] = best_score - int(np.max(alternatives))
            else:
                gaps[model] = np.iinfo(np.int64).max // 4
        candidates_evaluated = len(catalog.labels)

    labels = catalog.labels[chosen].copy()
    risk = (risk_constant - best_score) / (n_samples * q)
    certificate = ReportCertificate(
        labels=labels.copy(),
        parameters=parameters.copy(),
        gap_by_model=gaps,
        cut_points=cut_points,
        evaluator_id=evaluator_id,
        n_samples=n_samples,
        feature_kind=feature_kind,
        candidates_evaluated=candidates_evaluated,
    )
    return labels, float(risk), certificate
