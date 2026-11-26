"""Published acquisition rules, with explicit adapters to the shared experiment.

Acquisition rules select pairs, while the shared engine performs inference and reporting. AR/LUCB/AKG/Quicksort/TOP keep their own acquisition state. The JSON state
is saved with the episode RNG after each observed batch. Sources and adapters
are documented in the draft's baseline section and README.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import betainc, expit, xlogy, gammaln, log_expit, logsumexp

from ..belief import GaussianBelief, Pair, all_pairs, _hermgauss
from ..config import PUBLISHED_POLICIES, TOPK_ONLY, ranking_task
from .context import PolicyContext, PolicyDecision
from .heap import TournamentHeap
from .fisher import FisherSelector


@lru_cache(maxsize=None)
def _other_indices(n_models: int, pair: Pair) -> np.ndarray:
    indices = np.array([i for i in range(n_models) if i not in pair], dtype=np.intp)
    indices.setflags(write=False)
    return indices


@lru_cache(maxsize=None)
def _pair_indices(n_models: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.triu_indices(n_models, 1)
    for array in indices:
        array.setflags(write=False)
    return indices


def dirichlet_update(alpha: np.ndarray, pair: Pair, outcome: int) -> np.ndarray:
    """Homogeneous-worker MM update: Chen et al. (2016), Eqs. (20)--(22).

    Match every coordinate mean and the sum of coordinate second moments.
    The observed outcome is 1 iff the *first* coordinate of pair wins.
    """
    i, j = pair
    total, pair_total = float(alpha.sum()), float(alpha[i] + alpha[j])
    updated_pair = np.array([alpha[i] + outcome, alpha[j] + 1 - outcome])
    # Avoid erasing a tiny losing coordinate in (alpha_j + 1) - 1.
    # Retain ordinary-state arithmetic so saved experimental paths replay exactly.
    if outcome == 1 and updated_pair[1] == 0.0:
        updated_pair[1] = alpha[j]
    means = alpha / total
    means[[i, j]] = updated_pair * pair_total / (total * (pair_total + 1.0))
    other = _other_indices(len(alpha), pair)
    second = (
        np.sum(updated_pair * (updated_pair + 1.0)) * pair_total / (pair_total + 2.0)
        + np.sum(alpha[other] * (alpha[other] + 1.0))
    ) / (total * (total + 1.0))
    concentration = (1.0 - second) / (second - float(means @ means))
    result = means * concentration
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("AKG moment matching produced invalid Dirichlet parameters")
    return result


def dirichlet_accuracy(alpha: np.ndarray) -> float:
    """Optimal expected fraction of concordant pairs under Dir(alpha)."""
    order = np.argsort(-alpha, kind="stable")
    a, b = _pair_indices(len(alpha))
    return float(np.mean(betainc(alpha[order[b]], alpha[order[a]], 0.5)))


def log_dirichlet_update(log_alpha: np.ndarray, pair: Pair, outcome: int) -> np.ndarray:
    """The same MM equations, retaining parameters below float underflow.

    Only the representation changes. No pseudocount or parameter floor is used.
    Vanishing coordinates remain finite in log space and keep their ratios.
    """
    i, j = pair
    log_total = float(logsumexp(log_alpha))
    log_pair = float(np.logaddexp(log_alpha[i], log_alpha[j]))
    log_updated = np.array([np.logaddexp(log_alpha[i], 0.) if outcome else log_alpha[i],
                            log_alpha[j] if outcome else np.logaddexp(log_alpha[j], 0.)])
    log_means = log_alpha-log_total
    log_means[[i, j]] = log_updated+log_pair-log_total-np.logaddexp(log_pair, 0.)
    alpha = np.exp(log_alpha)
    total = float(np.exp(log_total))
    pair_total = float(np.exp(log_pair))
    updated = np.exp(log_updated)
    other = _other_indices(len(alpha), pair)
    second = (np.sum(updated*(updated+1.))*pair_total/(pair_total+2.)
              + np.sum(alpha[other]*(alpha[other]+1.)))/(total*(total+1.))
    concentration = (1.-second)/(second-float(np.exp(2.*log_means).sum()))
    if not np.isfinite(concentration) or concentration <= 0:
        raise FloatingPointError("AKG log-space moment matching produced invalid concentration")
    result = log_means+np.log(concentration)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("AKG log-space moment matching produced nonfinite parameters")
    return result


def log_dirichlet_accuracy(log_alpha: np.ndarray) -> np.ndarray:
    """Expected concordance, with subnormal beta shapes resolved in log space.

    If both shapes are below exp(-40), their beta mass is at the endpoints
    to better than double precision: P(theta_i>theta_j)=alpha_i/(alpha_i+alpha_j).
    A single shape below exp(-700) against a nonvanishing shape gives a
    concordance indistinguishable from one. All other cases use betainc.
    """
    values = np.atleast_2d(log_alpha)
    i, j = _pair_indices(values.shape[1])
    high = np.maximum(values[:, i], values[:, j])
    low = np.minimum(values[:, i], values[:, j])
    tiny = high < -40.
    ordinary = (~tiny) & (low >= -700.)
    probabilities = np.ones_like(high)
    probabilities[tiny] = expit(high[tiny]-low[tiny])
    probabilities[ordinary] = betainc(np.exp(low[ordinary]), np.exp(high[ordinary]), .5)
    return probabilities.mean(axis=1)


def log_akg_values(log_alpha: np.ndarray) -> dict[Pair, float]:
    """All original AKG candidate scores, with bounded-memory beta batches."""
    pairs = all_pairs(len(log_alpha))
    current = float(log_dirichlet_accuracy(log_alpha)[0])
    values = {}
    for start in range(0, len(pairs), 32):
        block = pairs[start:start+32]
        children = np.stack([log_dirichlet_update(log_alpha, pair, y)
                             for pair in block for y in (1, 0)])
        accuracy = log_dirichlet_accuracy(children).reshape(-1, 2)
        for pair, v in zip(block, accuracy):
            i, j = pair
            probability = float(expit(log_alpha[i]-log_alpha[j]))
            values[pair] = float(probability*v[0]+(1.-probability)*v[1]-current)
    return values


def lucb_radius(counts: np.ndarray, delta: float) -> np.ndarray:
    """Heckel et al. (2018), Sec. 3.1, with confidence delta/M."""
    u = np.maximum(np.asarray(counts, dtype=float), 1.0)
    log_confidence = np.log(len(u) / delta)
    beta = log_confidence + 0.75 * np.log(log_confidence)
    beta = beta + 1.5 * np.log(1.0 + np.log(u / 2.0))
    if np.any(beta < 0.0):
        raise ValueError("delta is too large for the published Hamming-LUCB radius at these counts")
    return np.sqrt(beta / (2.0 * u))


def ar_radius(t: int, n_models: int, delta: float, variant: str = "theoretical") -> float:
    """AR Algorithm 1 radius, or its smaller-constant variant in Section 5.2."""
    if variant == "empirical":
        return float(0.25 * np.sqrt(np.log(n_models / 3.0 * (np.log(t) + 1.0) / delta) / t))
    if variant == "theoretical":
        return float(np.sqrt(np.log(125.0 * n_models * np.log(1.12 * t) / delta) / t))
    raise ValueError("ar_confidence must be theoretical or empirical")






def topk_information_values(belief: GaussianBelief, context: PolicyContext) -> dict[Pair, float]:
    """Xu et al. (2026), Eq. (5), with both judge bias terms set to zero."""
    k = context.cut_points[0]
    samples = context.evaluator.samples(belief)
    selected = np.argsort(-samples, axis=1, kind="stable")[:, :k]
    membership = np.bincount(selected.ravel(), minlength=belief.n_models) / len(samples)
    entropy = -xlogy(membership, membership) - xlogy(1.0 - membership, 1.0 - membership)
    values = {}
    for i, j in all_pairs(belief.n_models):
        probability = expit(belief.mean[i] - belief.mean[j])
        variance = belief.covariance[i, i] + belief.covariance[j, j] - 2.0 * belief.covariance[i, j]
        values[i, j] = float(probability * (1.0 - probability) * variance * (entropy[i] + entropy[j]))
    return values


def bayesian_entropy_values(belief: GaussianBelief, context: PolicyContext) -> dict[Pair, float]:
    """Maystre Eq. (4), evaluated as BALD with a Gaussian current belief.

    Compute H(C)-E[H(C|theta)] for the exact logistic likelihood, without
    projecting hypothetical posteriors to Gaussians. This is also the expected
    joint-pair KL in Eq. (5), since C depends only on the queried contrast.
    """
    pairs = all_pairs(belief.n_models)
    i, j = _pair_indices(belief.n_models)
    variance = belief.covariance[i, i]+belief.covariance[j, j]-2*belief.covariance[i, j]
    nodes, weights = _hermgauss(max(64, context.inference.quadrature_nodes))
    z = (belief.mean[i]-belief.mean[j])[:, None]+np.sqrt(2*np.maximum(variance, 0))[:, None]*nodes
    batch = context.batch_size
    counts = np.arange(batch+1)
    combinations = gammaln(batch+1)-gammaln(counts+1)-gammaln(batch-counts+1)
    likelihood = np.exp(combinations[None, :, None] + counts[None, :, None]*log_expit(z[:, None, :])
                        + (batch-counts)[None, :, None]*log_expit(-z[:, None, :]))
    prediction = np.einsum("pcq,q->pc", likelihood, weights)
    entropy = -np.sum(xlogy(prediction, prediction), axis=1)
    conditional_entropy = -np.einsum("pq,q->p", np.sum(xlogy(likelihood, likelihood), axis=1), weights)
    scores = np.maximum(entropy-conditional_entropy, 0.)
    return {pair: float(score) for pair, score in zip(pairs, scores)}




class BaselineSelector:
    """One episode's acquisition state. Pair orientation is always canonical."""

    def __init__(self, policy: str, n_models: int, context: PolicyContext,
                 initial_pairs: list[Pair], initial_outcomes: list[int], *, heap_repetitions: int = 5,
                 ar_confidence: str = "empirical"):
        if policy not in PUBLISHED_POLICIES:
            raise ValueError(f"unknown published policy: {policy}")
        if policy in TOPK_ONLY and len(context.cut_points) != 2:
            raise ValueError(f"{policy} supports top-k selection only")
        if policy == "tournament_heap" and ranking_task(context.cut_points, n_models) == "ordered_partition":
            raise ValueError("tournament_heap requires selection, ordered top-k, or full ranking")
        self.policy, self.n_models, self.context = policy, n_models, context
        if ar_confidence not in {"theoretical", "empirical"}:
            raise ValueError("ar_confidence must be theoretical or empirical")
        self.ar_confidence = ar_confidence
        wins, counts = np.zeros(n_models), np.zeros(n_models, dtype=int)
        # Balanced initial data can seed every Borda score. Later observations
        # update ONLY the deliberately selected focal item's score.
        if policy in {"active_ranking", "hamming_lucb"}:
            for (i, j), outcome in zip(initial_pairs, initial_outcomes):
                wins[i] += outcome
                wins[j] += 1 - outcome
                counts[[i, j]] += 1
        if policy in {"active_ranking", "hamming_lucb"} and len(set(counts)) != 1:
            raise ValueError("Borda baselines require balanced initial comparisons")
        self.state: dict[str, Any] = {
            "wins": wins.tolist(), "counts": counts.tolist(), "queue": [],
            "pending_focal": None, "native_certified": False,
            "active": list(range(n_models)),
            "remaining": np.diff((0,) + context.cut_points).tolist(),
            "round": 0, "stack": [], "partition": None, "sort_pass": 0,
        }
        if policy == "akg":
            if n_models >= 32:
                log_alpha = np.zeros(n_models)
                for pair, outcome in zip(initial_pairs, initial_outcomes):
                    log_alpha = log_dirichlet_update(log_alpha, pair, outcome)
                self.state["log_alpha"] = log_alpha.tolist()
            else:
                alpha = np.ones(n_models)
                for pair, outcome in zip(initial_pairs, initial_outcomes):
                    alpha = dirichlet_update(alpha, pair, outcome)
                self.state["alpha"] = alpha.tolist()
        if policy == "tournament_heap":
            cuts = context.cut_points
            k = n_models if len(cuts) == n_models else cuts[0] if len(cuts) == 2 else len(cuts) - 1
            self.state["heap_k"] = k
            self.state["heap_repetitions"] = heap_repetitions
            self.state["heap_state"] = TournamentHeap(n_models, k, heap_repetitions).state
        if policy == "d_opt":
            self.state["fisher_state"] = FisherSelector(n_models, policy, initial_pairs, initial_outcomes).state


    def _heap(self) -> TournamentHeap:
        return TournamentHeap(self.n_models, self.state["heap_k"],
                              self.state["heap_repetitions"], self.state["heap_state"])


    def snapshot(self) -> dict[str, Any]:
        saved = {"version": 1, "policy": self.policy, "n_models": self.n_models,
                 "state": deepcopy(self.state)}
        if self.policy == "active_ranking" and self.ar_confidence != "theoretical":
            saved["ar_confidence"] = self.ar_confidence
        return saved

    def restore(self, saved: dict[str, Any]) -> None:
        if (saved.get("version"), saved.get("policy"), saved.get("n_models")) != (1, self.policy, self.n_models):
            raise ValueError("incompatible baseline acquisition checkpoint")
        if self.policy == "active_ranking" and saved.get("ar_confidence", "theoretical") != self.ar_confidence:
            raise ValueError("incompatible AR confidence variant")
        self.state = deepcopy(saved["state"])

    def _ar_eliminate(self) -> None:
        active = self.state["active"]
        if not active:
            return
        counts = np.asarray(self.state["counts"])
        t = int(counts[active[0]])
        if t == 0:
            return
        scores = np.asarray(self.state["wins"]) / np.maximum(counts, 1)
        ordered = sorted(active, key=lambda i: (-scores[i], i))
        borders = np.cumsum([0] + self.state["remaining"])
        radius = ar_radius(t, self.n_models, self.context.delta, self.ar_confidence)
        assignments = []
        # Test simultaneously against the ranking at this completed round.
        for i in active:
            for group, capacity in enumerate(self.state["remaining"]):
                if capacity == 0:
                    continue
                lower, upper = int(borders[group]), int(borders[group + 1])
                if (lower == 0 or scores[i] < scores[ordered[lower - 1]] - 4.0 * radius) and (
                    upper == len(active) or scores[i] > scores[ordered[upper]] + 4.0 * radius
                ):
                    assignments.append((i, group))
                    break
        removed = {i for i, _ in assignments}
        for _, group in assignments:
            self.state["remaining"][group] -= 1
        self.state["active"] = [i for i in active if i not in removed]
        self.state["native_certified"] = not self.state["active"]

    def _borda_action(self, rng: np.random.Generator) -> tuple[Pair, dict[str, Any]]:
        state = self.state
        if not state["queue"]:
            if self.policy == "active_ranking":
                self._ar_eliminate()
                if not state["active"]:
                    pairs = all_pairs(self.n_models)
                    return pairs[int(rng.integers(len(pairs)))], {
                        "baseline_phase": "uniform_after_certificate", "baseline_certified": 1,
                    }
                state["queue"] = state["active"].copy()
            else:
                counts = np.asarray(state["counts"])
                unobserved = np.flatnonzero(counts == 0)
                if len(unobserved):
                    state["queue"] = unobserved.tolist()
                else:
                    scores = np.asarray(state["wins"]) / counts
                    radius = lucb_radius(counts, self.context.delta)
                    ordered = sorted(range(self.n_models), key=lambda i: (-scores[i], i))
                    k = self.context.cut_points[0]
                    inside = min(ordered[:k], key=lambda i: (scores[i] - radius[i], i))
                    outside = min(ordered[k:], key=lambda i: (-scores[i] - radius[i], i))
                    state["native_certified"] = bool(scores[inside] - radius[inside] >= scores[outside] + radius[outside])
                    # Serialize the paper's TWO focal-vs-uniform comparisons;
                    # retain the second focal until both observations arrive.
                    state["queue"] = [inside, outside]
            state["round"] += 1
        focal = int(state["queue"].pop(0))
        opponent = int(rng.integers(self.n_models - 1))
        opponent += int(opponent >= focal)
        state["pending_focal"] = focal
        return tuple(sorted((focal, opponent))), {
            "baseline_focal": focal, "baseline_round": state["round"],
            "baseline_certified": int(state["native_certified"]),
            "baseline_phase": "borda_sampling",
        }

    def _quicksort_action(self, rng: np.random.Generator) -> tuple[Pair, dict[str, Any]]:
        state = self.state
        while state["partition"] is None:
            if not state["stack"]:
                state["sort_pass"] += 1
                state["stack"] = [list(range(self.n_models))]
            items = state["stack"].pop()
            if len(items) < 2:
                continue
            pivot = items[int(rng.integers(len(items)))]
            state["partition"] = {
                "pivot": pivot, "todo": [i for i in items if i != pivot], "high": [], "low": [],
            }
        frame = state["partition"]
        return tuple(sorted((frame["pivot"], frame["todo"][0]))), {
            "baseline_phase": "quicksort_partition", "quicksort_pass": state["sort_pass"],
        }

    def select(self, belief: GaussianBelief, rng: np.random.Generator) -> PolicyDecision:
        start = perf_counter()
        if self.policy == "d_opt":
            pair, diagnostics = FisherSelector(self.n_models, self.policy,
                state=self.state["fisher_state"]).select(rng, self.context.batch_size)
        elif self.policy in {"active_ranking", "hamming_lucb"}:
            pair, diagnostics = self._borda_action(rng)
        elif self.policy == "quicksort":
            pair, diagnostics = self._quicksort_action(rng)
        elif self.policy == "tournament_heap":
            pair = self._heap().next_pair(rng)
            diagnostics = {"baseline_phase": "tournament_heap", "baseline_round": self.state["heap_state"]["pass"]}
        else:
            if self.policy == "akg":
                if "log_alpha" in self.state:
                    values = log_akg_values(np.asarray(self.state["log_alpha"]))
                else:
                    alpha = np.asarray(self.state["alpha"])
                    current = dirichlet_accuracy(alpha)
                    values = {}
                    for pair in all_pairs(self.n_models):
                        i, j = pair
                        p = alpha[i] / (alpha[i] + alpha[j])
                        values[pair] = (
                            p * dirichlet_accuracy(dirichlet_update(alpha, pair, 1))
                            + (1.0 - p) * dirichlet_accuracy(dirichlet_update(alpha, pair, 0)) - current
                        )
            elif self.policy == "bayes_entropy":
                values = bayesian_entropy_values(belief, self.context)
            else:
                values = topk_information_values(belief, self.context)
            if self.policy == "bayes_entropy":
                best = max(values.values())
                leaders = [q for q, v in values.items() if best-v <= 1e-12]
                pair = leaders[int(rng.integers(len(leaders)))]
            else:
                pair = min(values, key=lambda q: (-values[q], q))
            diagnostics = {"acquisition_score": values[pair], "baseline_phase": self.policy}
        diagnostics["acquisition_seconds"] = perf_counter() - start
        return PolicyDecision(pair, diagnostics)

    def observe(self, pair: Pair, outcomes: list[int], rng: np.random.Generator) -> None:
        """Update acquisition state only; common inference uses every judgment."""
        if self.policy == "d_opt":
            FisherSelector(self.n_models, self.policy, state=self.state["fisher_state"]).observe(pair, outcomes)
            return
        if self.policy == "tournament_heap":
            self._heap().observe(pair, outcomes)
            return
        if self.policy in {"active_ranking", "hamming_lucb"}:
            focal = self.state["pending_focal"]
            if focal is not None:
                wins = sum(outcomes) if focal == pair[0] else len(outcomes) - sum(outcomes)
                self.state["wins"][focal] += wins
                self.state["counts"][focal] += len(outcomes)
                self.state["pending_focal"] = None
        elif self.policy == "akg":
            if "log_alpha" in self.state:
                log_alpha = np.asarray(self.state["log_alpha"])
                for outcome in outcomes:
                    log_alpha = log_dirichlet_update(log_alpha, pair, outcome)
                self.state["log_alpha"] = log_alpha.tolist()
            else:
                alpha = np.asarray(self.state["alpha"])
                for outcome in outcomes:
                    alpha = dirichlet_update(alpha, pair, outcome)
                self.state["alpha"] = alpha.tolist()
        elif self.policy == "quicksort":
            frame = self.state["partition"]
            challenger = frame["todo"].pop(0)
            wins = sum(outcomes)
            first_wins = (wins * 2 > len(outcomes)) if wins * 2 != len(outcomes) else bool(rng.integers(2))
            winner = pair[0] if first_wins else pair[1]
            frame["high" if winner == challenger else "low"].append(challenger)
            if not frame["todo"]:
                self.state["stack"].extend([frame["low"], frame["high"]])
                self.state["partition"] = None
