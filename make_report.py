"""Audit and plot results from any supported ranking experiment configuration."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import tempfile

import anytime_ranking  # Import before NumPy to limit worker BLAS threads.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ranking-kit-matplotlib"))
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t

from anytime_ranking.config import ranking_task
from anytime_ranking.reporting import make_reports, episode_trajectories

NAMES = {"softmax_pair_risk": "SPR", "uniform": "Uniform", "active_ranking": "AR",
         "hamming_lucb": "Hamming-LUCB", "akg": "AKG", "quicksort": "Quicksort",
         "tournament_heap": "TOP", "topk_aware": "Top-k-aware",
         "bayes_entropy": "Bayes-Entropy", "d_opt": "D-optimal"}
KEY = ["scenario", "policy", "replicate"]
METRICS = ["average_ranking_risk", "terminal_ranking_credibility", "terminal_exact_recovery"]
CURVES = ["posterior_risk", "average_ranking_risk", "posterior_coverage", "exact_recovery"]
COLORS = {"softmax_pair_risk": "#117733", "uniform": "#7A7A7A", "active_ranking": "#332288",
          "hamming_lucb": "#882255", "akg": "#44AA99", "quicksort": "#AA4499",
          "tournament_heap": "#999933", "topk_aware": "#BB5500",
          "bayes_entropy": "#0072B2", "d_opt": "#EE3377"}


def policy_name(policy, spr_c=None):
    return NAMES.get(policy, policy)




def trajectory_tables(root, manifest, runs, budgets):
    """Verify batches and summarize the observed replicate cohort at each time."""
    maximum = manifest["config"]["experiment"]["max_comparisons"]
    run_index = runs.set_index(KEY)
    if not budgets.empty:
        budget_index = budgets.set_index(KEY + ['budget'])
    else:
        budget_index = None
    seen, acc, initial, allocations = set(), {}, [], {}
    count = 0
    for key, group in episode_trajectories(root / "trajectories.csv.gz"):
        if key in seen or key not in run_index.index:
            raise ValueError(f"Unexpected or repeated trajectory: {key}")
        g = group.sort_values("comparisons")
        row = run_index.loc[key]
        batch, terminal = int(row.batch_size), int(row.stopping_time_comparisons)
        if not 0 <= terminal <= maximum or terminal % batch:
            raise ValueError(f"Invalid terminal comparison count: {key}")
        if terminal != maximum:
            raise ValueError(f"Incomplete fixed-budget episode: {key}")
        times = np.arange(0, terminal + 1, batch)
        np.testing.assert_array_equal(g.comparisons, times)
        np.testing.assert_array_equal(g.batch_size, np.full(len(g), batch))
        np.testing.assert_array_equal(g.active_stage, np.r_[np.ones(len(g)-1), 0])
        observed = g.iloc[:-1]
        i, j = observed.action_i.to_numpy(dtype=int), observed.action_j.to_numpy(dtype=int)
        theta = np.array(json.loads(row.theta))
        ranks = np.empty(len(theta), dtype=int)
        ranks[np.argsort(-theta, kind="stable")] = np.arange(len(theta))
        pair_counts = allocations.setdefault(key[:2], np.zeros((len(theta), len(theta)), dtype=np.int64))
        left, right = ranks[i], ranks[j]
        np.add.at(pair_counts, (np.minimum(left, right), np.maximum(left, right)), 1)
        u = np.random.default_rng(int(row.observation_seed)).random(terminal).reshape(-1, batch)
        p = np.clip(expit(theta[i] - theta[j]), 1e-12, 1-1e-12)
        np.testing.assert_array_equal(observed.batch_wins.to_numpy(dtype=int), (u < p[:, None]).sum(axis=1))
        risk = g.posterior_risk.to_numpy()
        credibility, recovery = g.posterior_coverage.to_numpy(), g.exact_recovery.to_numpy()
        avg = np.r_[np.nan, np.cumsum(risk[:-1] * batch) / times[1:]]
        for n in manifest["config"]["experiment"]["evaluation_budgets"]:
            index = n // batch
            np.testing.assert_allclose(budget_index.loc[(*key, n), METRICS].to_numpy(dtype=float),
                                       [avg[index], credibility[index], recovery[index]], atol=1e-12, rtol=0)
        np.testing.assert_allclose([avg[-1], risk[-1], credibility[-1], recovery[-1]],
            [row.average_posterior_risk, row.terminal_posterior_risk, row.terminal_credibility,
             row.terminal_exact_recovery], atol=1e-12, rtol=0, equal_nan=True)
        if int(json.loads(g.report.iloc[-1]) == json.loads(row.true_report)) != recovery[-1]:
            raise ValueError(f"Terminal recovery disagrees with the report: {key}")
        if key[:2] not in acc:
            shape = (maximum // batch + 1, len(CURVES))
            acc[key[:2]] = [np.zeros(shape), np.zeros(shape), np.zeros(shape[0], dtype=int), batch]
        sums, squares, sizes, _ = acc[key[:2]]
        values = np.c_[risk, np.nan_to_num(avg), credibility, recovery]
        sums[:len(g)] += values
        squares[:len(g)] += values**2
        sizes[:len(g)] += 1
        initial.append(dict(scenario=key[0], policy=key[1], replicate=key[2], risk=risk[0], credibility=credibility[0]))
        seen.add(key)
        count += len(g)
    if seen != set(run_index.index):
        raise ValueError("Trajectories do not cover every completed episode")
    frames = []
    for (scenario, policy), (sums, squares, sizes, batch) in acc.items():
        used = sizes > 0
        sums, squares, n = sums[used], squares[used], sizes[used]
        mean = sums / n[:, None]
        frame = pd.DataFrame(dict(scenario=scenario, policy=policy,
                                  comparisons=np.flatnonzero(used)*batch, replicates=n))
        for k, metric in enumerate(CURVES):
            if metric == "exact_recovery":
                z = 1.959963984540054
                center = (mean[:, k] + z*z/(2*n)) / (1+z*z/n)
                radius = z*np.sqrt(mean[:, k]*(1-mean[:, k])/n + z*z/(4*n*n)) / (1+z*z/n)
                lo, hi = center-radius, center+radius
            else:
                se = np.sqrt(np.maximum(squares[:, k]-sums[:, k]**2/n, 0) / np.maximum(n-1, 1) / n)
                radius = student_t.ppf(.975, np.maximum(n-1, 1))*se
                radius[n == 1] = np.nan
                lo, hi = mean[:, k]-radius, mean[:, k]+radius
            frame[metric] = mean[:, k]
            frame[metric+"_low95"], frame[metric+"_high95"] = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        frame.loc[frame.comparisons.eq(0), [c for c in frame if c.startswith("average_ranking_risk")]] = np.nan
        frames.append(frame)
    shares = []
    for (scenario, policy), counts in allocations.items():
        total = counts.sum()
        for i in range(counts.shape[0]):
            for j in range(i+1, counts.shape[0]):
                shares.append(dict(scenario=scenario, policy=policy, true_rank_i=i+1,
                                   true_rank_j=j+1, mean_query_fraction=counts[i, j]/total))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(initial), count, pd.DataFrame(shares)


def draw_query_heatmaps(shares, scenarios, policies, folder):
    """Plot query fractions after aligning every replicate by true quality rank."""
    paths = []
    for scenario in scenarios:
        data = shares[shares.scenario.eq(scenario["name"])]
        methods = [p for p in policies if p in set(data.policy)]
        rows = (len(methods) + 3) // 4
        n_models = scenario["n_models"]
        maximum = max(.1, float(100 * data.mean_query_fraction.max()))
        norm = LogNorm(vmin=.02, vmax=max(5, np.ceil(maximum/5)*5))
        columns = min(4, len(methods))
        fig, axes = plt.subplots(rows, columns, figsize=(2.5*columns+.6, 2.4*rows), squeeze=False)
        for ax, method in zip(axes.flat, methods):
            matrix = np.zeros((n_models, n_models))
            group = data[data.policy.eq(method)]
            matrix[group.true_rank_i-1, group.true_rank_j-1] = 100*group.mean_query_fraction
            masked = np.ma.array(matrix, mask=np.tri(n_models, dtype=bool))
            cmap = plt.get_cmap("Blues").with_extremes(bad="#f0f0f0")
            im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(NAMES[method], fontsize=9)
            ticks = np.unique(np.linspace(0, n_models-1, min(n_models, 10), dtype=int))
            ax.set_xticks(ticks, ticks+1)
            ax.set_yticks(ticks, ticks+1)
            ax.tick_params(labelsize=6, length=0)
            ax.set_xlabel("True rank (1 = best)", fontsize=7)
            if len(scenario["cut_points"]) != n_models:
                for boundary in scenario["cut_points"][:-1]:
                    ax.axvline(boundary-.5, color="#b16b25", ls="--", lw=.7)
                    ax.axhline(boundary-.5, color="#b16b25", ls="--", lw=.7)
            if n_models <= 15:
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        value = matrix[i,j]
                        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5,
                                color="white" if value > 0 and norm(value) > .63 else "#273344")
        for ax in list(axes.flat)[len(methods):]:
            ax.set_visible(False)
        fig.subplots_adjust(wspace=.4, hspace=.4, right=.91)
        bar = fig.colorbar(im, cax=fig.add_axes([.94, .17, .012, .66]))
        bar.set_label("Share of new queries (%)", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        for extension in ("png", "pdf"):
            path = folder / f'query_heatmaps_{scenario["name"]}.{extension}'
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            paths.append(path)
        plt.close(fig)
    return paths


def task_label(scenario):
    cuts = scenario["cut_points"]
    task = ranking_task(tuple(cuts), scenario["n_models"])
    if task == "top_k_selection":
        return f"Top-{cuts[0]} identification"
    if task == "top_k_ranking":
        return f"Ranking within top-{len(cuts)-1}"
    if task == 'full_ranking':
        return 'Full ranking'
    else:
        return f'Ordered partition {cuts}'


def draw_figures(trajectory, scenarios, policies, folder, maximum, risk="current"):
    figures = []
    colors = COLORS
    with plt.rc_context({"font.size": 8, "legend.fontsize": 7, "pdf.fonttype": 42}):
        for truth in dict.fromkeys(s["truth_mode"] for s in scenarios):
            rows = [s for s in scenarios if s["truth_mode"] == truth]
            fig, axes = plt.subplots(len(rows), 3, figsize=(8, 2.2*len(rows)+1.5), sharex=True, squeeze=False)
            legend = {}
            for r, scenario in enumerate(rows):
                data = trajectory[trajectory.scenario.eq(scenario["name"])]
                risk_metric = "posterior_risk" if risk == "current" else "average_ranking_risk"
                for c, metric in enumerate([risk_metric, "posterior_coverage", "exact_recovery"]):
                    ax = axes[r, c]
                    for policy in policies:
                        g = data[data.policy.eq(policy)].sort_values("comparisons")
                        if g.empty:
                            continue
                        line, = ax.plot(g.comparisons, g[metric], color=colors[policy],
                                        linewidth=1.8 if policy == "softmax_pair_risk" else 1)
                        ax.fill_between(g.comparisons, g[metric+"_low95"], g[metric+"_high95"],
                                        color=colors[policy], alpha=.08, linewidth=0)
                        legend[policy] = line
                    if r == 0:
                        risk_title = "Current ranking risk" if risk == "current" else "Path-average ranking risk"
                        ax.set_title([risk_title, "Ranking credibility", "Exact recovery rate"][c])
                    if c == 0:
                        ax.set_ylabel(f'{scenario["name"]}\nM={scenario["n_models"]}; {task_label(scenario)}')
                    if r == len(rows)-1:
                        ax.set_xlabel("New comparisons")
                    ax.set_xlim(0, maximum)
                    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
                    ax.set_ylim((0, 1) if c else (0, None))
                    if c == 1:
                        ax.axhline(1-scenario["delta"], color="#777", linestyle="--", linewidth=.65)
                    ax.grid(alpha=.15)
                    ax.spines[["top", "right"]].set_visible(False)
            fig.legend(list(legend.values()), [policy_name(p) for p in legend],
                       loc="lower center", ncol=min(4, len(legend)), frameon=False)
            legend_space = .25 * ((len(legend)+3)//4) + .2
            fig.tight_layout(rect=(0, legend_space/fig.get_figheight(), 1, 1), h_pad=1.5)
            for ext in ("png", "pdf"):
                suffix = "" if risk == "current" else "_average_risk"
                path = folder / f"trajectories_{truth}{suffix}.{ext}"
                fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
                figures.append(path)
            plt.close(fig)
    return figures


def make_report(result_dir, rebuild_summaries=True, risk="current"):
    """Audit outputs and draw trajectories from completed runs."""
    if risk not in {"current", "average"}:
        raise ValueError("risk must be current or average")
    root = Path(result_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config, experiment = manifest["config"], manifest["config"]["experiment"]
    if rebuild_summaries:
        make_reports(root)
    runs = pd.read_csv(root / "runs.csv")
    if (root / "budget_runs.csv").exists():
        budgets = pd.read_csv(root / 'budget_runs.csv')
    else:
        budgets = pd.DataFrame()
    if runs.duplicated(KEY).any() or (not budgets.empty and budgets.duplicated(KEY+["budget"]).any()):
        raise ValueError("Duplicate episode or budget metrics")
    cohorts = {s["name"]: range(experiment["replicates"]) for s in config["scenarios"]}
    expected = {(s["name"], p, r) for s in config["scenarios"]
                for p in (s.get("policies") or experiment["policies"]) for r in cohorts[s["name"]]}
    if set(runs[KEY].itertuples(index=False, name=None)) != expected:
        raise ValueError("Completed episodes do not match the configured scenarios and replicate cohort")
    for _, group in runs.groupby(["scenario", "replicate"]):
        for col in ("theta", "truth_seed", "observation_seed", "initial_data_seed", "initial_pair_wins"):
            if group[col].nunique() != 1:
                raise ValueError(f"Methods do not share {col}")
    folder = root / "report"
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    trajectories, initial, count, shares = trajectory_tables(root, manifest, runs, budgets)
    trajectories.to_csv(folder / "trajectory_summary.csv", index=False)
    initial.to_csv(folder / "initial_metrics.csv", index=False)
    shares.to_csv(folder / "query_allocation.csv", index=False)
    terminal = pd.read_csv(root / "summary.csv")
    terminal.to_csv(folder / "terminal_metrics.csv", index=False)
    maximum = experiment["max_comparisons"]
    spr_c = experiment.get("spr_c", 30.0)
    figures = draw_figures(trajectories, config["scenarios"], experiment["policies"], folder / "figures", maximum, risk)
    figures += draw_query_heatmaps(shares, config["scenarios"], experiment["policies"], folder / "figures")
    audit = dict(status="passed", episodes=len(runs), trajectory_rows=count,
                 new_outcomes_verified=int(runs.stopping_time_comparisons.sum()),
                 budget_records_verified=len(budgets), replicates=experiment["replicates"],
                 shared_truth_initial_data_and_observation_seeds_verified=True,
                 manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
    (folder / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    title = "Ranking simulation results"
    description = (f'{experiment["replicates"]} paired replicates per scenario; '
                   f'{maximum} new comparisons; posterior Kemeny reports.')
    if "softmax_pair_risk" in experiment["policies"]:
        description += f" SPR inverse temperature c={spr_c:g}."
    body = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            f'<title>{title}</title>',
            '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#172b3a}table{border-collapse:collapse;width:100%;margin-bottom:30px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dce4e8}th:first-child,td:first-child{text-align:left}img{max-width:100%}a{color:#086952}</style>',
            f'<h1>{title}</h1><p>{html.escape(description)}</p>',
            '<p><a href="terminal_metrics.csv">Terminal metrics</a> · <a href="../summary.csv">Summary</a> · <a href="audit.json">Audit</a></p>']
    if not budgets.empty:
        body.append('<p><a href="../budget_summary.csv">All budgets</a></p>')
    markdown = [f'# {title}', '', description, '']
    for scenario in config["scenarios"]:
        g = terminal[terminal.scenario.eq(scenario["name"])].set_index("policy")
        records = []
        markdown += [f'## {scenario["name"]}', '',
                     '| Policy | Average risk [95% CI] | Credibility (%) [95% CI] | Recovery (%) [95% CI] | Select (ms) |',
                     '|---|---:|---:|---:|---:|']
        for policy in experiment["policies"]:
            if policy not in g.index:
                continue
            row = g.loc[policy]
            cells = []
            for metric in METRICS:
                factor, decimals = (1, 4) if metric == METRICS[0] else (100, 1)
                mean, low, high = [row[metric+suffix] * factor for suffix in ("", "_low95", "_high95")]
                interval = "" if pd.isna(low) else f" [{low:.{decimals}f}, {high:.{decimals}f}]"
                cells.append(f"{mean:.{decimals}f}" + interval)
            cells.append(f'{row.query_selection_ms:.3g}')
            records.append([policy_name(policy, spr_c), *cells])
            markdown.append('| ' + ' | '.join(records[-1]) + ' |')
        details = f'M={scenario["n_models"]}; {task_label(scenario)}; delta={scenario["delta"]}; epsilon={scenario["epsilon"]}'
        body += [f'<h2>{html.escape(scenario["name"])}</h2><p>{html.escape(details)}</p>',
                 pd.DataFrame(records, columns=['Policy', 'Average ranking risk', 'Terminal credibility (%)',
                                               'Exact recovery (%)', 'Select (ms)']).to_html(index=False, border=0)]
        markdown.append('')
    for figure in figures:
        if figure.suffix == '.png':
            body.append(f'<img alt="Ranking trajectories" src="figures/{figure.name}">')
            markdown.append(f'![Trajectories](figures/{figure.name})')
    risk_note = ("Risk curves show current posterior risk after each number of new comparisons."
                 if risk == "current" else
                 "Risk curves show mean pre-comparison risk up to each new-comparison budget.")
    note = (risk_note + ' Curves average across replicates. '
            'Average ranking risk in the tables is the mean pre-comparison risk along each path. '
            'Shading uses pointwise 95% Student-t intervals for risk and credibility, and Wilson intervals for recovery.')
    body.append(f'<p>{note}</p>')
    markdown += ['', note]
    (folder / "index.html").write_text('\n'.join(body), encoding='utf-8')
    (folder / "results.md").write_text('\n'.join(markdown), encoding='utf-8')
    return folder / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--risk", choices=("current", "average"), default="current",
                        help="Risk trajectory to display; tables always use path-average risk")
    args = parser.parse_args()
    print(make_report(args.results, risk=args.risk))
