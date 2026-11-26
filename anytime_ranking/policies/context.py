"""Inputs and return values shared by acquisition policies."""
from dataclasses import dataclass, field
from ..config import InferenceConfig
from ..ranking import GaussianRankingEvaluator

@dataclass(frozen=True)
class PolicyContext:
    cut_points: tuple[int, ...]
    epsilon: float
    delta: float
    inference: InferenceConfig
    evaluator: GaussianRankingEvaluator
    batch_size: int
    spr_c: float = 30.0

@dataclass
class PolicyDecision:
    pair: tuple[int, int]
    diagnostics: dict = field(default_factory=dict)
