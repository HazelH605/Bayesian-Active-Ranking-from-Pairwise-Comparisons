"""Run ranking simulations from TOML configurations."""
from dataclasses import replace
from pathlib import Path
import argparse
from multiprocessing import freeze_support

# Import the numerical package before NumPy to limit each worker's BLAS threads.
from anytime_ranking.config import load_config, scenario_policy_indices, _validate
from anytime_ranking.runner import run_resumable
from make_report import make_report

KIT = Path(__file__).resolve().parent


def smoke_config(config):
    """Shorten the budget while preserving complete batches in every scenario."""
    from math import lcm

    batches = tuple(s.batch_size or config.experiment.batch_size for s in config.scenarios)
    batch = lcm(*batches)
    maximum = max(batch, (min(10, config.experiment.max_comparisons) // batch) * batch)
    return replace(config, experiment=replace(
        config.experiment, replicates=1, max_comparisons=maximum,
        evaluation_budgets=(maximum,),
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=KIT/"configs/example.toml",
                        help="Experiment TOML (default: configs/example.toml)")
    parser.add_argument("--jobs", type=int, help="Override the configured number of worker processes")
    parser.add_argument("--replicates", type=int, help="Override the number of repetitions")
    parser.add_argument("--budget", type=int, help="Override the new-comparison budget per episode")
    parser.add_argument("--spr-c", type=float, help="Override the nonnegative SPR inverse temperature")
    parser.add_argument("--seed", type=int, help="Override the nonnegative master seed")
    parser.add_argument("--output", type=Path, help="Result directory (default: results/<config stem>)")
    parser.add_argument("--smoke", action="store_true", help="One replicate and a short budget of complete batches")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports from completed results")
    args = parser.parse_args(argv)
    for name in ("jobs", "replicates", "budget"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.report_only and any(getattr(args, name) is not None for name in ("replicates", "budget", "spr_c", "seed")):
        parser.error("use make_report.py RESULT_DIRECTORY to replot completed results")
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    overrides = {name: getattr(args, name) for name in ("jobs", "replicates", "spr_c", "seed")
                 if getattr(args, name) is not None}
    if args.budget is not None:
        overrides["max_comparisons"] = args.budget
        overrides["evaluation_budgets"] = tuple(n for n in config.experiment.evaluation_budgets if n <= args.budget)
    config = replace(config, experiment=replace(config.experiment, **overrides))
    if args.smoke:
        config = smoke_config(config)
    try:
        _validate(config)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    folder = ("smoke_" if args.smoke else "") + args.config.stem
    output = (args.output or KIT/"results"/folder).resolve()
    if args.report_only:
        if not (output/"manifest.json").exists():
            parser.error(f"No completed experiment at {output}")
    else:
        episodes = sum(len(scenario_policy_indices(config,s)) for s in config.scenarios)*config.experiment.replicates
        print(f"{args.config}: {episodes} episodes, up to {config.experiment.max_comparisons} new comparisons each", flush=True)
        run_resumable(config, output)
    # Rebuild summaries only for an explicit report-only request.
    print(f"Report: {make_report(output, rebuild_summaries=args.report_only)}", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
