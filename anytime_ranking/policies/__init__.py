"""Common acquisition interface for SPR, Uniform, and paper baselines."""
from time import perf_counter
import numpy as np
from ..belief import all_pairs
from ..config import POLICIES, PUBLISHED_POLICIES
from .context import PolicyDecision
from .baselines import BaselineSelector

def softmax_pair_risk_probabilities(risks, c):
    """Sample with probabilities proportional to exp(c * pairwise_risk)."""
    values = np.asarray(risks, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("risks must be a nonempty finite vector")
    if not np.isfinite(c) or c < 0:
        raise ValueError("c must be finite and nonnegative")
    if c == 0:
        return np.full(len(values), 1.0 / len(values))
    with np.errstate(over="ignore"):
        logits = c * (values - values.max())
    weights = np.exp(np.maximum(logits, np.log(np.finfo(float).tiny)))
    return weights / weights.sum()

class PolicySelector:
    def __init__(self, policy, context, experiment, initial_pairs, initial_outcomes):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        self.policy, self.context = policy, context
        self.baseline = None
        if policy in PUBLISHED_POLICIES:
            self.baseline = BaselineSelector(
                policy, context.evaluator.n_models, context, initial_pairs, initial_outcomes,
                heap_repetitions=experiment.heap_repetitions, ar_confidence=experiment.ar_confidence,
            )

    def select(self, belief, rng, *, current_summary, reporting_evaluator):
        if self.baseline is not None:
            return self.baseline.select(belief, rng)
        start = perf_counter()
        pairs = all_pairs(belief.n_models)
        diagnostics = {}
        if self.policy == "uniform":
            index = int(rng.integers(len(pairs)))
        else:
            # Reuse the current report and its integration statistics.
            risks = reporting_evaluator.pairwise_risks(belief, self.context.cut_points, current_summary)
            probabilities = softmax_pair_risk_probabilities(risks, self.context.spr_c)
            cumulative = np.cumsum(probabilities)
            cumulative[-1] = 1.0
            index = min(int(np.searchsorted(cumulative, rng.random(), side="right")), len(pairs)-1)
            diagnostics = {
                "acquisition_score": float(risks[index]),
                "spr_selected_probability": float(probabilities[index]),
                "spr_max_probability": float(probabilities.max()),
                "spr_entropy": float(-np.dot(probabilities, np.log(probabilities))),
            }
        diagnostics["acquisition_seconds"] = perf_counter() - start
        return PolicyDecision(pairs[index], diagnostics)

    def observe(self, pair, outcomes, rng):
        if self.baseline is not None:
            self.baseline.observe(pair, outcomes, rng)

    def snapshot(self):
        return self.baseline.snapshot() if self.baseline is not None else None

    def restore(self, state):
        if self.baseline is not None:
            if state is None:
                raise ValueError("missing baseline acquisition checkpoint")
            self.baseline.restore(state)
