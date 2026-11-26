# Configuration reference

A TOML file contains `[experiment]`, `[inference]`, and one or more
`[[scenarios]]` tables. Omitted fields use the defaults below.
The provided example files set some fields explicitly.
Command-line overrides take precedence over the file.

## Experiment

| Field | Default | Meaning and valid values |
|---|---|---|
| `seed` | `42` | Nonnegative master seed. |
| `replicates` | `20` | Positive repetition count per scenario-policy combination. IDs run from 0 to `replicates-1`. |
| `max_comparisons` | `250` | Positive new-judgment budget per episode, additional to initial data. |
| `evaluation_budgets` | `[]` | Strictly increasing positive intermediate budgets up to the maximum. The terminal budget is always included. |
| `policies` | `["softmax_pair_risk", "uniform"]` | Nonempty list of unique policy keys from the README. |
| `initial_comparisons_per_pair` | `0` | Nonnegative number of initial judgments generated for every unordered pair. |
| `spr_c` | `30.0` | Finite, nonnegative SPR inverse temperature. Zero gives uniform selection probabilities. |
| `jobs` | Up to 8 available CPUs | Positive number of independent worker processes. |
| `checkpoint_interval` | `100` | Positive number of acquisition steps between saved within-episode states. The final state is also saved. |
| `batch_size` | `1` | Positive judgments collected for a pair before the next update. Must divide the maximum and every evaluation budget. |
| `heap_repetitions` | `5` | Positive number of fresh comparisons per TOP majority decision. |
| `ar_confidence` | `"empirical"` | `"empirical"` uses the smaller AR confidence constant. `"theoretical"` uses the original theoretical constant. |
| `policy_seed_indices` | Automatic | Optional list of distinct nonnegative selection-stream coordinates, one per global policy in the same order. |
| `timing_replicates` | `20` | Positive number of final replicate IDs used to summarize selection time. At most the available replicate count is used. |
| `timing_queries` | `1000` | Positive number of first acquisition steps used for timing in each selected replicate, clipped to the current budget. |

An acquisition step is one pair choice followed by `batch_size` judgments.
Selection time excludes posterior fitting, outcome generation, and disk writes.

For example, with `replicates=50` and `timing_replicates=10`, timing uses
replicates 40–49. With `batch_size=2` and `timing_queries=100`, the timing
window covers the first 100 pair choices, collecting 200 new judgments.

## Inference

| Field | Default | Meaning and valid values |
|---|---|---|
| `sigma0` | `1.0` | Positive standard deviation of the independent zero-mean Gaussian prior. Also supplies the scale for prior-predictive truth. |
| `posterior_samples` | `4096` | Sobol points used for Bayes reporting and credibility, at least 32. Full-ranking reports and risks are analytic. |
| `acquisition_samples` | `1024` | Sobol points for Top-k-aware membership probabilities, at least 16. |
| `quadrature_nodes` | `20` | Gauss–Hermite nodes, at least 6. Bayes-Entropy uses at least 64. |
| `laplace_gtol` | `1e-8` | Positive gradient tolerance for posterior optimization. |
| `laplace_maxiter` | `100` | Positive posterior optimization iteration limit. |

Changing integration resolution changes numerical approximations.
Higher resolution and larger report spaces can increase computation time.

## Scenarios

| Field | Default | Meaning and valid values |
|---|---|---|
| `name` | Required | Unique nonempty scenario name. Use simple names such as `top3` or `three_tiers` for output filenames. |
| `n_models` | Required | Integer model count, at least 2. |
| `cut_points` | Required | Strictly increasing positive group end positions. At least two groups are required, and the last entry equals `n_models`. |
| `truth_mode` | `"fixed_grid"` | `"prior_predictive"` draws independent Gaussian qualities. `"fixed_grid"` generates spaced qualities and randomly permutes identities. |
| `boundary_gap` | `0.3` | Nonnegative adjacent quality gap across partition boundaries in fixed-grid truth. |
| `within_gap` | `0.3` | Nonnegative adjacent quality gap within groups in fixed-grid truth. |
| `epsilon` | `0.0` | Kemeny-distance tolerance for posterior credibility, between 0 and 1. Zero gives exact-partition credibility. |
| `delta` | `0.05` | Baseline confidence parameter, strictly between 0 and 1. Figures mark `1-delta` as a reference credibility level. |
| `pairing_id` | Scenario index | Optional nonnegative random-stream coordinate. Equal IDs and matching data-generating settings share inputs across targets. |
| `policies` | Global policy list | Optional nonempty subset of `experiment.policies` for this scenario. |
| `inference_sigma0` | Global `sigma0` | Optional positive prior-scale override. Also affects prior-predictive truth for this scenario. |
| `batch_size` | Global `batch_size` | Optional positive batch-size override, dividing all evaluation budgets. |

`boundary_gap` and `within_gap` are used by fixed-grid truth.
`epsilon` controls the posterior event being evaluated.
`delta` supplies baseline confidence parameters and the plotted reference line.
Runs continue to the configured comparison budget.

## Multiple targets and method subsets

~~~toml
[experiment]
replicates = 20
max_comparisons = 200
policies = ["softmax_pair_risk", "uniform", "topk_aware"]

[[scenarios]]
name = "top2"
n_models = 6
cut_points = [2, 6]
truth_mode = "prior_predictive"
pairing_id = 7

[[scenarios]]
name = "full"
n_models = 6
cut_points = [1, 2, 3, 4, 5, 6]
truth_mode = "prior_predictive"
pairing_id = 7
policies = ["softmax_pair_risk", "uniform"]
~~~

The two scenarios share truth and observation streams, while Top-k-aware
is selected only for top-2 identification.

## Command-line options

| Option | Meaning |
|---|---|
| `--config FILE` | Configuration file. Defaults to the included `configs/example.toml`. |
| `--replicates N` | Override the positive repetition count. |
| `--budget N` | Override the positive new-comparison budget and discard evaluation points beyond it. |
| `--spr-c C` | Override the finite, nonnegative SPR inverse temperature. |
| `--seed N` | Override the nonnegative master seed. |
| `--jobs N` | Override the positive worker count. |
| `--output DIR` | Output directory. Defaults to `results/<config-stem>/` in the package. |
| `--smoke` | One replicate and a budget of up to ten new comparisons, adjusted for complete batches. The default output gets a `smoke_` prefix. |
| `--report-only` | Rebuild the completed output's report from saved data. |

All budgets must contain complete batches. When combining `--smoke` with
scientific overrides, the smoke repetition count and short budget take effect
last. For replotting a saved directory directly, use

~~~bash
python make_report.py results/my-experiment --risk current
python make_report.py results/my-experiment --risk average
~~~

## Reproducibility and resumption

Scientific settings are saved with each run. Repeating the same command and
output directory resumes saved work. You may change worker count, checkpoint
cadence, and timing-summary settings. Use a new output directory for other
configuration changes.

The pinned requirements describe the tested numerical environment.
Different numerical libraries or platforms can change decisions near ties.
