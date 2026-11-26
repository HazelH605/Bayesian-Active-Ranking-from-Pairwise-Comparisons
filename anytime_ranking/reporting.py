"""Stream simulation trajectories and summarize the three paper metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

KEY = ["scenario", "policy", "replicate"]
METRICS = ["average_ranking_risk", "terminal_ranking_credibility", "terminal_exact_recovery"]


def episode_trajectories(path, chunksize=100000):
    """Stream contiguous episodes, carrying a partial episode across CSV chunks."""
    columns = KEY + ["comparisons", "batch_size", "active_stage", "posterior_risk",
                     "posterior_coverage", "exact_recovery", "action_i", "action_j",
                     "batch_wins", "report", "acquisition_seconds"]
    pending = pd.DataFrame()
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = pd.concat([pending, chunk], ignore_index=True)
        last = chunk[KEY].eq(chunk[KEY].iloc[-1]).all(axis=1)
        yield from chunk.loc[~last].groupby(KEY, sort=False)
        pending = chunk.loc[last]
    if not pending.empty:
        yield from pending.groupby(KEY, sort=False)

def confidence_interval(values, binary=False):
    values = np.asarray(values, dtype=float)
    n, mean = len(values), float(np.mean(values))
    if binary:
        z = 1.959963984540054
        center = (mean + z*z/(2*n)) / (1+z*z/n)
        radius = z*np.sqrt(mean*(1-mean)/n + z*z/(4*n*n)) / (1+z*z/n)
        return mean, float(center-radius), float(center+radius)
    if n == 1:
        return mean, float("nan"), float("nan")
    radius = student_t.ppf(.975, n-1) * np.std(values, ddof=1) / np.sqrt(n)
    return mean, max(0., float(mean-radius)), min(1., float(mean+radius))

def make_reports(destination):
    destination = Path(destination)
    config = json.loads((destination/"manifest.json").read_text(encoding="utf-8"))["config"]
    maximum = config["experiment"]["max_comparisons"]
    budgets = sorted(set(config["experiment"]["evaluation_budgets"]) | {maximum})
    rows = []
    timing_replicates = config["experiment"].get("timing_replicates", 20)
    timing_queries = config["experiment"].get("timing_queries", 1000)
    timing_ids = set(range(max(0, config["experiment"]["replicates"] - timing_replicates),
                           config["experiment"]["replicates"]))
    for key, path in episode_trajectories(destination/"trajectories.csv.gz"):
        counts = path.comparisons.to_numpy(dtype=int)
        batch = int(path.batch_size.iloc[0])
        np.testing.assert_array_equal(counts, np.arange(0, maximum+1, batch))
        risk = path.posterior_risk.to_numpy()
        averages = np.cumsum(risk[:-1]*batch) / counts[1:]
        for budget in budgets:
            index = budget//batch
            terminal = path.iloc[index]
            rows.append(dict(scenario=key[0], policy=key[1], replicate=key[2], budget=budget,
                average_ranking_risk=float(averages[index-1]),
                terminal_ranking_risk=float(terminal.posterior_risk),
                terminal_ranking_credibility=float(terminal.posterior_coverage),
                terminal_exact_recovery=int(terminal.exact_recovery),
                query_selection_ms=(1000 * float(path.loc[
                    path.comparisons.lt(budget), "acquisition_seconds"].head(timing_queries).mean())
                    if key[2] in timing_ids else float("nan")),
                timing_queries=min(timing_queries, index)))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(destination/"budget_runs.csv", index=False)
    summaries = []
    for key, group in per_run.groupby(["scenario", "policy", "budget"], sort=False):
        row = dict(scenario=key[0], policy=key[1], budget=key[2], replicates=len(group))
        for metric in METRICS:
            mean, lo, hi = confidence_interval(group[metric], binary=metric == "terminal_exact_recovery")
            row.update({metric: mean, metric+"_low95": lo, metric+"_high95": hi})
        summaries.append(row)
        times = group.query_selection_ms.dropna()
        row.update(query_selection_ms=float(times.mean()), timing_replicates=len(times),
                   timing_queries=int(group.timing_queries.iloc[0]))
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination/"budget_summary.csv", index=False)
    summary[summary.budget.eq(maximum)].drop(columns="budget").to_csv(destination/"summary.csv", index=False)

