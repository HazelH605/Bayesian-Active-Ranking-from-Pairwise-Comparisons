# Baseline references and simulation adapters

The acquisition rules use the shared Gaussian posterior for evaluation and
retain the state required by their own selection algorithms.

| Policy | Original reference |
|---|---|
| AR | Heckel et al. (2019), *Active ranking from pairwise comparisons and when parametric assumptions do not help*. [Paper](https://doi.org/10.1214/18-AOS1772) |
| Hamming-LUCB | Heckel et al. (2018), *Approximate ranking from pairwise comparisons*. [Paper](https://proceedings.mlr.press/v84/heckel18a.html) |
| Top-k-aware | Xu et al. (2026), *When can you debias an LLM judge? Identifiability limits, a test, and designs for top-k ranking*. [Preprint](https://arxiv.org/abs/2607.02104) |
| TOP | Mohajer et al. (2017), *Active learning for top-K rank aggregation from noisy comparisons*. [Paper](https://proceedings.mlr.press/v70/mohajer17a.html) |
| AKG | Chen et al. (2016), *Bayesian decision process for cost-efficient dynamic ranking via crowdsourcing*. [Paper](https://jmlr.org/papers/v17/16-066.html) |
| Quicksort and Bayes-Entropy | Maystre and Grossglauser (2017), *Just sort it! A simple and effective approach to active preference learning*. [Paper](https://proceedings.mlr.press/v70/maystre17a.html) |
| D-optimal | Liu et al. (2026), *Fewer battles, more gain: An information-efficient framework for Arena-based LLM evaluation*. [Paper](https://openreview.net/forum?id=XUVqFRp9oi) |

## Simulation adapters

- **AR** uses the empirical radius from Section 5.2, the four-radius assignment
  threshold, and serialized focal-versus-random queries. Balanced historical
  data initialize both participants' Borda statistics. After certification,
  acquisition is uniform until the budget is exhausted.
- **Hamming-LUCB** uses the original radius with `h=0` and `delta=0.05`,
  serializes its two focal-versus-random comparisons, and continues to the budget.
- **Top-k-aware** sets judge-bias terms to zero and uses the common Laplace
  posterior. Membership entropy uses 1,024 Sobol points. Repeated unordered
  queries are allowed, and score ties follow a fixed pair order.
- **TOP** uses balanced random groups, byes, and five fresh judgments per
  majority decision. It restarts after extracting the requested number of leaders.
- **AKG** starts from a Dirichlet vector of ones and replays historical data
  through the original homogeneous-worker moment-matching updates. It maximizes
  expected full-ranking accuracy gain with fixed pair-order tie-breaking.
- **Quicksort** uses random pivots, a fresh judgment per pivot comparison,
  and repeated passes. Reports come from the shared posterior.
- **Bayes-Entropy** replaces the original EP inference with the shared Gaussian
  Laplace posterior. Mutual information uses at least 64 Gauss–Hermite nodes,
  and maximizing-pair ties are randomized.
- **D-optimal** uses a separate unpenalized BTL fit with the last score anchored
  at zero. It samples uniformly until a finite MLE exists and the directed win
  graph is strongly connected, then maximizes Fisher log-determinant gain.
  Historical and acquired data both enter its acquisition fit.
