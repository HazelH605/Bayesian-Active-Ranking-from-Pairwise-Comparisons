"""One simulation episode, from initial observations to sequential ranking reports."""
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from .belief import fit_laplace
from .checkpointing import atomic_json, restore_summary
from .config import POLICY_SEEDS, scientific_config
from .environment import build_environment, generate_initial_history, generate_truth
from .policies import PolicySelector
from .policies.context import PolicyContext
from .ranking import GaussianRankingEvaluator
from .partitions import ordered_sets, report_from_scores

@dataclass
class EpisodeResult:
    run: dict
    trajectory: list[dict]

def _derived_seed(master, *coordinates):
    sequence = np.random.SeedSequence([master, *coordinates])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def run_episode(config, scenario, policy, scenario_index, policy_index, replicate,
                *, query_checkpoint_path=None):
    exp = config.experiment
    sigma0 = scenario.inference_sigma0 if scenario.inference_sigma0 is not None else config.inference.sigma0
    batch = scenario.batch_size if scenario.batch_size is not None else exp.batch_size
    inference = replace(config.inference, sigma0=sigma0)
    coordinate = scenario.pairing_id if scenario.pairing_id is not None else scenario_index
    policy_coordinate = POLICY_SEEDS[policy] if exp.policy_seed_indices is None else exp.policy_seed_indices[policy_index]
    truth_seed = _derived_seed(exp.seed, coordinate, replicate, 11)
    observation_seed = _derived_seed(exp.seed, coordinate, replicate, 29)
    initial_data_seed = _derived_seed(exp.seed, coordinate, replicate, 53)
    policy_seed = _derived_seed(exp.seed, coordinate, policy_coordinate, replicate, 31)

    # Share truth, historical data, and observation uniforms across methods.
    theta = generate_truth(scenario, sigma0, truth_seed)
    environment = build_environment(theta)
    uniforms = np.random.default_rng(observation_seed).random(exp.max_comparisons)
    rng = np.random.default_rng(policy_seed)
    evaluator = GaussianRankingEvaluator(scenario.n_models, inference.posterior_samples,
                                         _derived_seed(exp.seed, coordinate, replicate, 41))
    acquisition_evaluator = GaussianRankingEvaluator(scenario.n_models, inference.acquisition_samples,
                                                     _derived_seed(exp.seed, coordinate, replicate, 43))
    context = PolicyContext(scenario.cut_points, scenario.epsilon, scenario.delta,
                            inference, acquisition_evaluator, batch, exp.spr_c)
    pairs, outcomes = generate_initial_history(environment, exp.initial_comparisons_per_pair, initial_data_seed)
    initial_count = len(outcomes)
    initial_pair_wins = []
    if exp.initial_comparisons_per_pair:
        step = exp.initial_comparisons_per_pair
        initial_pair_wins = [sum(outcomes[k:k+step]) for k in range(0, initial_count, step)]
    path = Path(query_checkpoint_path) if query_checkpoint_path is not None else None
    resuming = path is not None and path.exists()
    selector = PolicySelector(policy, context, exp, [] if resuming else pairs, [] if resuming else outcomes)
    identity = json.loads(json.dumps(dict(config=scientific_config(config), scenario_index=scenario_index,
                                         policy_index=policy_index, replicate=replicate)))
    count, cumulative_risk = 0, 0.0
    warm_mean, previous = None, None
    trajectory = []
    inference_seconds, acquisition_seconds = 0.0, 0.0

    if resuming:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("schema") != 2 or saved["identity"] != identity:
            raise ValueError("checkpoint belongs to a different experiment")
        pairs = [tuple(pair) for pair in saved["pairs"]]
        outcomes, trajectory, count = saved["outcomes"], saved["trajectory"], saved["comparisons"]
        if count % batch or count > exp.max_comparisons or len(trajectory) != count // batch or len(pairs) != len(outcomes) or len(outcomes) != initial_count+count:
            raise ValueError("inconsistent checkpoint counters")
        warm_mean = np.asarray(saved["warm_mean"]) if saved["warm_mean"] is not None else None
        previous = restore_summary(saved["previous"])
        cumulative_risk = saved["cumulative_risk"]
        inference_seconds, acquisition_seconds = saved["inference_seconds"], saved["acquisition_seconds"]
        rng.bit_generator.state = saved["rng_state"]
        selector.restore(saved["baseline_state"])

    def save():
        if path is not None:
            atomic_json(path, dict(schema=2, identity=identity, comparisons=count, pairs=pairs,
                outcomes=outcomes, trajectory=trajectory, warm_mean=warm_mean, previous=previous,
                cumulative_risk=cumulative_risk, rng_state=rng.bit_generator.state,
                baseline_state=selector.snapshot(), inference_seconds=inference_seconds,
                acquisition_seconds=acquisition_seconds))

    true_labels = report_from_scores(theta, scenario.cut_points)
    i, j = evaluator.pair_i, evaluator.pair_j
    truth_relation = np.sign(true_labels[j]-true_labels[i])
    while True:
        started = perf_counter()
        belief = fit_laplace(scenario.n_models, sigma0, pairs, outcomes, warm_mean,
                             inference.laplace_gtol, inference.laplace_maxiter)
        inference_seconds += perf_counter()-started
        warm_mean = belief.mean
        summary = evaluator.summarize(belief, scenario.cut_points, scenario.epsilon, previous=previous)
        previous = summary
        relation = np.sign(summary.labels[j]-summary.labels[i])
        realized = float(np.abs(truth_relation-relation).sum() / (2*len(i)))
        exact = int(np.array_equal(summary.labels, true_labels))
        active = count < exp.max_comparisons
        if active:
            # Accumulate pre-comparison risk, weighted by the number of new judgments.
            cumulative_risk += batch * summary.risk
        row = dict(scenario=scenario.name, policy=policy, replicate=replicate,
            acquisition_round=count//batch, comparisons=count, initial_comparisons=initial_count,
            total_comparisons=initial_count+count, batch_size=batch, posterior_risk=summary.risk,
            posterior_coverage=summary.coverage, realized_loss=realized, exact_recovery=exact,
            active_stage=int(active), report=json.dumps(ordered_sets(summary.labels), separators=(",", ":")),
            action_i=float("nan"), action_j=float("nan"), batch_wins=float("nan"),
            batch_losses=float("nan"), acquisition_seconds=0.0)
        if not active:
            trajectory.append(row)
            break
        selection_started = perf_counter()
        decision = selector.select(belief, rng, current_summary=summary, reporting_evaluator=evaluator)
        selection_seconds = perf_counter() - selection_started
        observed = environment.observe_batch(decision.pair, uniforms[count:count+batch]).tolist()
        selector.observe(decision.pair, observed, rng)
        pairs.extend([decision.pair]*batch)
        outcomes.extend(observed)
        row.update(decision.diagnostics)
        row["acquisition_seconds"] = selection_seconds
        row["action_i"], row["action_j"] = decision.pair
        row["batch_wins"], row["batch_losses"] = sum(observed), batch-sum(observed)
        acquisition_seconds += row["acquisition_seconds"]
        trajectory.append(row)
        count += batch
        if count//batch % exp.checkpoint_interval == 0 or count == exp.max_comparisons:
            save()

    run = dict(scenario=scenario.name, policy=policy, replicate=replicate, n_models=scenario.n_models,
        cut_points=json.dumps(scenario.cut_points), epsilon=scenario.epsilon, delta=scenario.delta,
        batch_size=batch, initial_comparisons_per_pair=exp.initial_comparisons_per_pair,
        initial_comparisons=initial_count, total_comparisons=initial_count+count,
        stopping_time_comparisons=count, initial_pair_wins=json.dumps(initial_pair_wins, separators=(",", ":")),
        inference_sigma0=sigma0, truth_mode=scenario.truth_mode,
        average_posterior_risk=cumulative_risk/count, terminal_posterior_risk=summary.risk,
        terminal_credibility=summary.coverage, terminal_exact_recovery=exact,
        terminal_realized_loss=realized, inference_seconds=inference_seconds,
        acquisition_seconds=acquisition_seconds, theta=json.dumps(theta.tolist(), separators=(",", ":")),
        true_report=json.dumps(ordered_sets(true_labels), separators=(",", ":")),
        truth_seed=truth_seed, observation_seed=observation_seed, initial_data_seed=initial_data_seed,
        policy_seed=policy_seed, spr_c=exp.spr_c if policy == "softmax_pair_risk" else float("nan"))
    return EpisodeResult(run, trajectory)
