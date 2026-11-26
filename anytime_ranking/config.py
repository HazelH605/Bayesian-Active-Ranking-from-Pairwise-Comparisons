"""Read and validate fixed-budget simulation configurations."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import os
import tomllib

# Stable method coordinates preserve random streams when policies are reordered.
POLICY_SEEDS = {
    "softmax_pair_risk": 25, "uniform": 0, "active_ranking": 5,
    "hamming_lucb": 9, "akg": 6, "quicksort": 7,
    "tournament_heap": 14, "topk_aware": 10,
    "bayes_entropy": 16, "d_opt": 23,
}
POLICIES = frozenset(POLICY_SEEDS)
PUBLISHED_POLICIES = POLICIES - {"softmax_pair_risk", "uniform"}
TOPK_ONLY = frozenset({"hamming_lucb", "topk_aware"})

@dataclass(frozen=True)
class InferenceConfig:
    sigma0: float = 1.0
    posterior_samples: int = 4096
    acquisition_samples: int = 1024
    quadrature_nodes: int = 20
    laplace_gtol: float = 1e-8
    laplace_maxiter: int = 100

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    replicates: int = 20
    max_comparisons: int = 250
    batch_size: int = 1
    policies: tuple[str, ...] = ("softmax_pair_risk", "uniform")
    jobs: int = field(default_factory=lambda: max(1, min(8, os.cpu_count() or 1)))
    initial_comparisons_per_pair: int = 0
    policy_seed_indices: tuple[int, ...] | None = None
    heap_repetitions: int = 5
    ar_confidence: str = "empirical"
    evaluation_budgets: tuple[int, ...] = ()
    checkpoint_interval: int = 100
    spr_c: float = 30.0
    timing_replicates: int = 20
    timing_queries: int = 1000

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_models: int
    cut_points: tuple[int, ...]
    epsilon: float = 0.0
    delta: float = 0.05
    truth_mode: str = "fixed_grid"
    boundary_gap: float = 0.3
    within_gap: float = 0.3
    inference_sigma0: float | None = None
    batch_size: int | None = None
    pairing_id: int | None = None
    policies: tuple[str, ...] | None = None

@dataclass(frozen=True)
class Config:
    inference: InferenceConfig
    experiment: ExperimentConfig
    scenarios: tuple[ScenarioConfig, ...]

    def to_dict(self):
        return asdict(self)

def scientific_config(config):
    """Exclude execution and timing-summary settings from the scientific identity."""
    value = config.to_dict()
    value["experiment"].pop("jobs")
    value["experiment"].pop("checkpoint_interval")
    value["experiment"].pop("timing_replicates")
    value["experiment"].pop("timing_queries")
    return value


def scenario_policy_indices(config, scenario):
    if scenario.policies is not None:
        selected = scenario.policies
    else:
        selected = config.experiment.policies
    return tuple(i for i, policy in enumerate(config.experiment.policies) if policy in selected)

def ranking_task(cut_points, n_models):
    if cut_points == tuple(range(1, n_models + 1)):
        return "full_ranking"
    if len(cut_points) == 2:
        return "top_k_selection"
    if cut_points[:-1] == tuple(range(1, len(cut_points))):
        return "top_k_ranking"
    return "ordered_partition"

def _valid_type(value, annotation):
    if value is None:
        return "None" in annotation
    kind = annotation.split(" | ")[0]
    if kind.startswith("tuple["):
        if '[int,' in kind:
            element_type = int
        else:
            element_type = str
        return isinstance(value, tuple) and all(type(v) is element_type for v in value)
    if kind == "float":
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is {"int": int, "bool": bool, "str": str}[kind]

def _section(cls, raw, label):
    """Use the dataclass schema for all TOML sections, including numeric settings."""
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a table")
    values = dict(raw)
    schema = {item.name: item for item in fields(cls)}
    unknown = set(values) - schema.keys()
    if unknown:
        raise ValueError(f"unknown {label} parameters: {sorted(unknown)}")
    for name, value in values.items():
        if schema[name].type.startswith("tuple["):
            if not isinstance(value, list):
                raise ValueError(f"{label}.{name} must be an array")
            values[name] = tuple(value)
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"{label}: {error}") from error

def _validate(config):
    sections = [("inference", config.inference), ("experiment", config.experiment)]
    sections += [(s.name, s) for s in config.scenarios]
    for label, section in sections:
        for item in fields(section):
            if not _valid_type(getattr(section, item.name), item.type):
                raise ValueError(f"{label}.{item.name} must have type {item.type}; numbers must be finite")
    inf, exp = config.inference, config.experiment
    if exp.seed < 0 or exp.initial_comparisons_per_pair < 0 or exp.spr_c < 0:
        raise ValueError("seed, initial_comparisons_per_pair and spr_c must be nonnegative")
    for name in ("replicates", "max_comparisons", "batch_size", "jobs", "heap_repetitions",
                 "checkpoint_interval", "timing_replicates", "timing_queries"):
        if getattr(exp, name) < 1:
            raise ValueError(f"{name} must be positive")
    if inf.sigma0 <= 0 or inf.laplace_gtol <= 0 or inf.laplace_maxiter < 1:
        raise ValueError("sigma0, laplace_gtol and laplace_maxiter must be positive")
    if inf.posterior_samples < 32 or inf.acquisition_samples < 16 or inf.quadrature_nodes < 6:
        raise ValueError("require posterior_samples >= 32, acquisition_samples >= 16, quadrature_nodes >= 6")
    if not exp.policies or len(set(exp.policies)) != len(exp.policies) or set(exp.policies) - POLICIES:
        raise ValueError(f"policies must be unique and belong to {sorted(POLICIES)}")
    indices = exp.policy_seed_indices
    if indices is not None and (len(indices) != len(exp.policies) or len(set(indices)) != len(indices) or min(indices) < 0):
        raise ValueError("policy_seed_indices must give one distinct nonnegative integer per policy")
    budgets = exp.evaluation_budgets
    if tuple(sorted(set(budgets))) != budgets or any(n < 1 or n > exp.max_comparisons for n in budgets):
        raise ValueError("evaluation_budgets must increase and lie within max_comparisons")
    if exp.ar_confidence not in {"empirical", "theoretical"}:
        raise ValueError("ar_confidence must be empirical or theoretical")
    if not config.scenarios or len({s.name for s in config.scenarios}) != len(config.scenarios):
        raise ValueError("at least one scenario is required; names must be unique")
    for s in config.scenarios:
        selected = s.policies if s.policies is not None else exp.policies
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(exp.policies):
            raise ValueError(f"{s.name}: policies must be a nonempty unique subset of experiment.policies")
        if not s.name.strip() or s.n_models < 2:
            raise ValueError("scenario names must be nonempty; n_models >= 2")
        cuts = s.cut_points
        if len(cuts) < 2 or cuts[0] < 1 or cuts[-1] != s.n_models or tuple(sorted(set(cuts))) != cuts:
            raise ValueError(f"{s.name}: cut_points must increase from a positive integer to n_models")
        if TOPK_ONLY.intersection(selected) and len(cuts) != 2:
            raise ValueError(f"{s.name}: hamming_lucb and topk_aware require top-k selection")
        if "tournament_heap" in selected and ranking_task(cuts, s.n_models) == "ordered_partition":
            raise ValueError("TOP supports selection, ranking within top-k and full ranking")
        if not 0 <= s.epsilon <= 1 or not 0 < s.delta < 1:
            raise ValueError("require 0 <= epsilon <= 1 and 0 < delta < 1")
        if "hamming_lucb" in selected:
            logc = math.log(s.n_models / s.delta)
            if logc + .75 * math.log(logc) + 1.5 * math.log(1 + math.log(.5)) < 0:
                raise ValueError("delta is too large for the Hamming-LUCB confidence radius")
        if s.truth_mode not in {"fixed_grid", "prior_predictive"}:
            raise ValueError("truth_mode must be fixed_grid or prior_predictive")
        if s.boundary_gap < 0 or s.within_gap < 0:
            raise ValueError("truth gaps must be nonnegative")
        if s.inference_sigma0 is not None and s.inference_sigma0 <= 0:
            raise ValueError("inference_sigma0 must be positive")
        if s.pairing_id is not None and s.pairing_id < 0:
            raise ValueError("pairing_id must be nonnegative")
        batch = s.batch_size if s.batch_size is not None else exp.batch_size
        if batch < 1 or exp.max_comparisons % batch or any(n % batch for n in budgets):
            raise ValueError("max_comparisons and evaluation_budgets must contain complete positive batches")

def load_config(path):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - {"inference", "experiment", "scenarios"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    scenarios = raw.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("use [[scenarios]] tables")
    config = Config(
        _section(InferenceConfig, raw.get("inference", {}), "inference"),
        _section(ExperimentConfig, raw.get("experiment", {}), "experiment"),
        tuple(_section(ScenarioConfig, s, "scenario") for s in scenarios),
    )
    _validate(config)
    return config
