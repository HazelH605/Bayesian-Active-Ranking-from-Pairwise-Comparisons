"""Gaussian Laplace inference for the Bradley--Terry likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

Pair = tuple[int, int]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianBelief:
    mean: FloatArray
    covariance: FloatArray

    @property
    def n_models(self):
        return int(self.mean.size)


@lru_cache(maxsize=None)
def all_pairs(n_models):
    return tuple((i, j) for i in range(n_models) for j in range(i + 1, n_models))


def _history_arrays(pairs, outcomes, n_models):
    design = np.zeros((len(pairs), n_models), dtype=float)
    for row, (i, j) in enumerate(pairs):
        design[row, i] = 1.0
        design[row, j] = -1.0
    return design, np.asarray(outcomes, dtype=float)


def fit_laplace(n_models, sigma0, pairs, outcomes, initial_mean=None, gtol=1e-08, maxiter=100):
    """Find the posterior mode and approximate covariance by the inverse Hessian."""
    if len(pairs) != len(outcomes):
        raise ValueError("pairs and outcomes must have equal length")
    if not pairs:
        return GaussianBelief(
            np.zeros(n_models, dtype=float), np.eye(n_models, dtype=float) * sigma0**2
        )

    design, y = _history_arrays(pairs, outcomes, n_models)
    prior_precision = 1.0 / sigma0**2
    cached_theta = None
    cached_logits = None
    cached_probabilities = None

    def terms(theta):
        nonlocal cached_theta, cached_logits, cached_probabilities
        if cached_theta is None or not np.array_equal(theta, cached_theta):
            cached_theta = theta.copy()
            cached_logits = design @ theta
            cached_probabilities = expit(cached_logits)
        return cached_logits, cached_probabilities

    # Minimize the negative log posterior with its exact gradient and Hessian.
    def objective(theta):
        logits, _ = terms(theta)
        return float(
            0.5 * prior_precision * np.dot(theta, theta)
            + np.sum(np.logaddexp(0.0, logits) - y * logits)
        )

    def gradient(theta):
        _, probabilities = terms(theta)
        return prior_precision * theta + design.T @ (probabilities - y)

    def hessian(theta):
        _, probabilities = terms(theta)
        weights = probabilities * (1.0 - probabilities)
        return prior_precision * np.eye(n_models) + design.T @ (weights[:, None] * design)

    if initial_mean is None:
        start = np.zeros(n_models, dtype=float)
    else:
        start = np.asarray(initial_mean, dtype=float).copy()
    try:
        result = minimize(
            objective,
            start,
            jac=gradient,
            hess=hessian,
            method="trust-exact",
            options={"gtol": gtol, "maxiter": maxiter},
        )
    except ValueError as error:
        if str(error) != "math domain error":
            raise
        # Preserve the frozen runner's recovery from a rounded trust-region
        # discriminant, using the same convex posterior and residual checks.
        fallback = start if cached_theta is None else cached_theta.copy()
        result = minimize(objective, fallback, jac=gradient, method="BFGS",
                          options={"gtol": gtol, "maxiter": maxiter})
    # SciPy can report a harmless precision warning near an already converged optimum.
    # A genuinely poor solution is detected through the gradient norm.
    grad_norm = float(np.linalg.norm(gradient(result.x), ord=np.inf))
    # The trust-region solver can emit "bad approximation" after reaching a
    # numerically stationary point on nearly separated comparison histories.
    # At this scale, a 1e-5 infinity-norm residual is negligible relative to the
    # posterior uncertainty, while still rejecting genuinely failed fits.
    failure_tolerance = max(1e-5, 100.0 * gtol)
    if not result.success and failure_tolerance < grad_norm <= max(1e-3, 100.0 * failure_tolerance):
        # A trust-region precision failure can leave a tiny residual whose
        # objective improvement is below floating-point resolution. Polish the
        # SAME convex objective with its exact Hessian; keep the old rejection
        # threshold and leave all previously accepted fits unchanged.
        value = objective(result.x)
        for _ in range(3):
            factor = cho_factor(hessian(result.x), lower=True, check_finite=False)
            step = cho_solve(factor, gradient(result.x), check_finite=False)
            candidate = result.x - step
            candidate_norm = float(np.linalg.norm(gradient(candidate), ord=np.inf))
            candidate_value = objective(candidate)
            roundoff = 32.0 * np.finfo(float).eps * max(1.0, abs(value))
            if not (candidate_norm < grad_norm and candidate_value <= value + roundoff):
                break
            result.x = candidate
            grad_norm, value = candidate_norm, candidate_value
            if grad_norm <= gtol:
                break
    if not result.success and grad_norm > failure_tolerance:
        raise RuntimeError(
            f"Laplace optimization failed ({result.message}); gradient inf-norm={grad_norm:.3g}"
        )

    precision = hessian(result.x)
    factor = cho_factor(precision, lower=True, check_finite=False)
    covariance = cho_solve(factor, np.eye(n_models), check_finite=False)
    covariance = 0.5 * (covariance + covariance.T)
    return GaussianBelief(np.asarray(result.x, dtype=float), covariance)


@lru_cache(maxsize=None)
def _hermgauss(n_nodes):
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    nodes.setflags(write=False)
    weights = weights / np.sqrt(np.pi)
    weights.setflags(write=False)
    return nodes, weights


