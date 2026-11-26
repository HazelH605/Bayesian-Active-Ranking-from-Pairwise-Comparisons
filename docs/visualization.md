# Visualization guide

Completed simulations save figures, plotting data, an HTML report, and a
Markdown report under the selected output directory.

## Generate or refresh a report

~~~bash
python make_report.py results/my-experiment
python make_report.py results/my-experiment --risk average
~~~

The commands read the saved trajectories and regenerate summaries and plots.
The default view shows current-state risk. The average view shows mean
pre-comparison risk up to each budget. Summary tables always use path-average
risk at the reported budget.

## Trajectories

A figure is generated for each truth mode in the configuration.
Rows correspond to scenarios, and columns show ranking risk, ranking
credibility, and exact recovery. Each line represents a policy.

| View | PNG filename under `report/figures/` |
|---|---|
| Current risk | `trajectories_<truth_mode>.png` |
| Path-average risk | `trajectories_<truth_mode>_average_risk.png` |

Each PNG also has a vector PDF version. For the default example,
`truth_mode` is `prior_predictive`.

Lines average the replicates at each budget. Shading gives pointwise 95%
Student-$t$ intervals for risk and credibility and Wilson intervals for recovery.
Student-$t$ intervals require at least two replicates.
The dashed horizontal line shows `1-delta`.

## Query-allocation heatmaps

The filename is `query_heatmaps_<scenario>.png`, with a corresponding PDF.

Every replicate is aligned by true quality rank before accumulating query
counts. Rank 1 is best. Percentages are shares of acquired queries, averaged
across equal-budget replicates. Historical comparisons are excluded.
Each unordered pair appears once in the upper triangle. The lower triangle
and diagonal are masked. Dashed lines mark the ordered-partition boundaries.
For more than 15 models, the heatmap omits cell annotations to keep it readable.

## Use the plotting tables

`report/trajectory_summary.csv` contains

| Column | Meaning |
|---|---|
| `scenario`, `policy` | Scenario and acquisition method. |
| `comparisons` | Number of new judgments collected. |
| `replicates` | Repetitions contributing to this point. |
| `posterior_risk` | Mean current-state ranking risk. |
| `average_ranking_risk` | Mean path-average risk up to this point. Undefined at zero comparisons. |
| `posterior_coverage` | Mean posterior credibility at the chosen `epsilon`. |
| `exact_recovery` | Mean exact recovery indicator. |
| `<metric>_low95`, `<metric>_high95` | Pointwise interval endpoints. |

`report/query_allocation.csv` contains `scenario`, `policy`,
`true_rank_i`, `true_rank_j`, and `mean_query_fraction`.
Multiply the fraction by 100 for percentage labels.

## Make a custom figure

This example plots credibility from an existing run.

~~~python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

run = Path("results/my-experiment")
data = pd.read_csv(run / "report/trajectory_summary.csv")
scenario = data["scenario"].iloc[0]
data = data[data["scenario"] == scenario]

fig, ax = plt.subplots(figsize=(6, 4))
for policy, curve in data.groupby("policy", sort=False):
    curve = curve.sort_values("comparisons")
    ax.plot(curve["comparisons"], curve["posterior_coverage"], label=policy)
    ax.fill_between(
        curve["comparisons"],
        curve["posterior_coverage_low95"],
        curve["posterior_coverage_high95"],
        alpha=0.12,
    )

ax.set(xlabel="New comparisons", ylabel="Ranking credibility", ylim=(0, 1))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(run / "credibility.png", dpi=200)
~~~

For a different response, select `posterior_risk`, `average_ranking_risk`,
or `exact_recovery` and the matching interval columns.
