"""Fewer Battles, More Gain (ICLR 2026), Eqs. (3), (5), (6).

The last item is anchored at zero, as in the paper. Acquisition uses its own
unpenalized BT MLE, not the shared Gaussian posterior precision. Uniform
initialization continues until a finite MLE exists (strong win-graph connectivity).
"""
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, root
from scipy.special import expit

from ..belief import Pair, all_pairs


@lru_cache(maxsize=None)
def anchored_design(n: int) -> np.ndarray:
    design = np.zeros((len(all_pairs(n)), n-1))
    for row, (i, j) in enumerate(all_pairs(n)):
        design[row, i] = 1.
        if j < n-1:
            design[row, j] = -1.
    design.setflags(write=False)
    return design


@lru_cache(maxsize=None)
def _lookup(n: int) -> dict[Pair, int]:
    return {pair: k for k, pair in enumerate(all_pairs(n))}


def strongly_connected(n: int, counts: np.ndarray, wins: np.ndarray) -> bool:
    reach = np.eye(n, dtype=bool)
    for (i, j), c, w in zip(all_pairs(n), counts, wins):
        reach[i, j] = w > 0
        reach[j, i] = c-w > 0
    for k in range(n):
        reach |= reach[:, k, None] & reach[None, k, :]
    return bool(reach.all())


def bt_mle(n: int, counts: np.ndarray, wins: np.ndarray,
           initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the anchored MLE and observed/expected BT information at it."""
    design = anchored_design(n)
    x0 = np.zeros(n-1) if initial is None else np.asarray(initial, dtype=float)

    def objective(x):
        contrast = design @ x
        # Split wins/losses: avoid cancellation for near-deterministic pairs.
        value = np.sum(wins*np.logaddexp(0., -contrast) + (counts-wins)*np.logaddexp(0., contrast))
        gradient = design.T @ (counts*expit(contrast)-wins)
        return float(value), gradient

    def hessian(x):
        contrast = design @ x
        weights = counts*expit(contrast)*expit(-contrast)
        return (design.T*weights) @ design

    result = minimize(objective, x0, jac=True, hess=hessian, method='trust-exact',
                      options={'gtol': 1e-8, 'maxiter': 100})
    # trust-exact can stop on rounding once the objective change is tiny.
    if np.max(np.abs(objective(result.x)[1])) > 2e-6:
        result = minimize(objective, result.x, jac=True, method='BFGS',
                          options={'gtol': 1e-8, 'maxiter': 150})
    # At the minimum, likelihood changes can round to zero before the score
    # equations pass the residual check. Polish those equations directly,
    # retaining the same unpenalized MLE and the same acceptance tolerance.
    if np.all(np.isfinite(result.x)) and np.max(np.abs(objective(result.x)[1])) > 2e-6:
        polished = root(lambda x: objective(x)[1], result.x, jac=hessian,
                        method='hybr', options={'xtol': 1e-10, 'maxfev': 100})
        if (np.all(np.isfinite(polished.x))
                and np.max(np.abs(objective(polished.x)[1]))
                < np.max(np.abs(objective(result.x)[1]))):
            result = polished
    residual = np.max(np.abs(objective(result.x)[1]))
    if not np.all(np.isfinite(result.x)) or residual > 2e-6:
        raise FloatingPointError(f'Fisher baseline BT MLE did not converge: gradient={residual}')
    return result.x, hessian(result.x)


def fisher_values(n: int, estimate: np.ndarray, information: np.ndarray,
                  criterion: str, batch_size: int = 1) -> np.ndarray:
    """D-optimal log-determinant gain for each candidate pair."""
    design = anchored_design(n)
    inverse = cho_solve(cho_factor(information, lower=True, check_finite=False), np.eye(n-1),
                        check_finite=False)
    contrasts = design @ estimate
    weights = batch_size*expit(contrasts)*expit(-contrasts)
    projections = design @ inverse
    variance = np.sum(projections*design, axis=1)
    if criterion == 'd_opt':
        return np.log1p(weights*variance)
    raise ValueError(f'unknown Fisher criterion: {criterion}')


class FisherSelector:
    def __init__(self, n: int, criterion: str, initial_pairs: Sequence[Pair] = (),
                 initial_outcomes: Sequence[int] = (), state: dict[str, Any] | None = None):
        self.n, self.criterion = n, criterion
        self.pairs, self.lookup = all_pairs(n), _lookup(n)
        if state is not None:
            self.state = state
            return
        if len(initial_pairs) != len(initial_outcomes):
            raise ValueError('Fisher initial pairs/outcomes length mismatch')
        self.state = {'counts': [0]*len(self.pairs), 'wins': [0]*len(self.pairs),
                      'estimate': [0.]*(n-1), 'ready': False}
        for pair, outcome in zip(initial_pairs, initial_outcomes):
            self.observe(pair, [outcome])

    def select(self, rng: np.random.Generator, batch_size: int = 1) -> tuple[Pair, dict[str, Any]]:
        s = self.state
        counts, wins = np.asarray(s['counts']), np.asarray(s['wins'])
        if not s['ready']:
            s['ready'] = strongly_connected(self.n, counts, wins)
        if not s['ready']:
            return self.pairs[int(rng.integers(len(self.pairs)))], {
                'baseline_phase': 'fisher_uniform_initialization', 'acquisition_score': 1./len(self.pairs)}
        estimate, information = bt_mle(self.n, counts, wins, np.asarray(s['estimate']))
        s['estimate'] = estimate.tolist()
        values = fisher_values(self.n, estimate, information, self.criterion, batch_size)
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite Fisher acquisition scores')
        leaders = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-15))
        index = int(leaders[int(rng.integers(len(leaders)))])
        return self.pairs[index], {'baseline_phase': self.criterion+'_mle',
                                   'acquisition_score': float(values[index])}

    def observe(self, pair: Pair, outcomes: Sequence[int]) -> None:
        if not outcomes or any(y not in (0, 1) for y in outcomes):
            raise ValueError('Fisher baselines require binary observations')
        index = self.lookup[pair]
        self.state['counts'][index] += len(outcomes)
        self.state['wins'][index] += int(sum(outcomes))
