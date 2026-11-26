"""Synthetic truth generation and comparison data-generating processes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from .belief import all_pairs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticEnvironment:
    """Correctly specified Bradley--Terry environment with unit temperature."""

    theta: FloatArray

    def probability(self, pair):
        i, j = pair
        probability = expit(float(self.theta[i] - self.theta[j]))
        return float(np.clip(probability, 1e-12, 1.0 - 1e-12))

    def observe_batch(self, pair, uniforms=None):
        """Obtain a batch before any posterior or acquisition-policy update."""
        if uniforms is None:
            raise ValueError("uniforms must be supplied to make paired runs reproducible")
        draws = np.asarray(uniforms, dtype=float)
        return (draws < self.probability(pair)).astype(np.int64)


def _fixed_grid_scores(scenario, rng):
    cut_boundaries = set(scenario.cut_points[:-1])
    sorted_scores = np.zeros(scenario.n_models, dtype=float)
    for rank in range(1, scenario.n_models):
        if rank in cut_boundaries:
            gap = scenario.boundary_gap
        else:
            gap = scenario.within_gap
        sorted_scores[rank] = sorted_scores[rank - 1] - gap
    sorted_scores -= np.mean(sorted_scores)
    # Randomizing identities prevents index-based tie breaking from helping a policy.
    theta = np.empty(scenario.n_models, dtype=float)
    theta[rng.permutation(scenario.n_models)] = sorted_scores
    return theta


def generate_truth(scenario, sigma0, seed):
    rng = np.random.default_rng(seed)
    if scenario.truth_mode == "prior_predictive":
        return rng.normal(0.0, sigma0, scenario.n_models)
    return _fixed_grid_scores(scenario, rng)


def build_environment(theta):
    return SyntheticEnvironment(np.asarray(theta, dtype=float))


def generate_initial_history(environment, comparisons_per_pair, seed):
    """Shared BTL observations; each pair's history is nested as its count grows.

    A separate stream per pair keeps both policy identity and the requested
    count from changing the observations already generated for another pair.
    This stream is independent of subsequent active-acquisition observations.
    """
    if type(comparisons_per_pair) is not int or comparisons_per_pair < 0:
        raise ValueError("comparisons_per_pair must be a nonnegative integer")
    pairs = []
    outcomes = []
    if comparisons_per_pair == 0:
        return pairs, outcomes
    for pair in all_pairs(environment.theta.size):
        rng = np.random.default_rng(np.random.SeedSequence([seed, *pair]))
        observations = environment.observe_batch(pair, rng.random(comparisons_per_pair))
        pairs.extend([pair] * comparisons_per_pair)
        outcomes.extend(int(outcome) for outcome in observations)
    return pairs, outcomes
