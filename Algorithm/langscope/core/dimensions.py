"""
10-Dimensional Rating System for LangScope.

Defines the 10 rating dimensions and their scoring formulas:
1. Raw Quality - Judge-ranked quality
2. Cost-Adjusted - Quality per cost
3. Latency - Response time scoring
4. TTFT - Time to first token
5. Consistency - Response variance
6. Token Efficiency - Quality per token
7. Instruction Following - Format compliance
8. Hallucination Resistance - Factual accuracy
9. Long Context - Context length handling
10. Combined - Weighted aggregate
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import math


class Dimension(str, Enum):
    """Rating dimension identifiers."""
    RAW_QUALITY = "raw_quality"
    COST_ADJUSTED = "cost_adjusted"
    LATENCY = "latency"
    TTFT = "ttft"
    CONSISTENCY = "consistency"
    TOKEN_EFFICIENCY = "token_efficiency"
    INSTRUCTION_FOLLOWING = "instruction_following"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    LONG_CONTEXT = "long_context"
    COMBINED = "combined"


@dataclass
class DimensionConfig:
    """Configuration for a rating dimension."""
    name: str
    description: str
    formula: str
    unit: str = ""
    higher_is_better: bool = True
    requires_metrics: List[str] = None
    
    def __post_init__(self):
        if self.requires_metrics is None:
            self.requires_metrics = []


# Default dimension weights for Combined score
DEFAULT_COMBINED_WEIGHTS: Dict[str, float] = {
    Dimension.RAW_QUALITY.value: 0.20,
    Dimension.COST_ADJUSTED.value: 0.10,
    Dimension.LATENCY.value: 0.10,
    Dimension.TTFT.value: 0.05,
    Dimension.CONSISTENCY.value: 0.10,
    Dimension.TOKEN_EFFICIENCY.value: 0.10,
    Dimension.INSTRUCTION_FOLLOWING.value: 0.15,
    Dimension.HALLUCINATION_RESISTANCE.value: 0.15,
    Dimension.LONG_CONTEXT.value: 0.05,
}


# Dimension configurations with formulas and metadata
DIMENSION_CONFIGS: Dict[Dimension, DimensionConfig] = {
    Dimension.RAW_QUALITY: DimensionConfig(
        name="Raw Quality",
        description="Quality rating from judge rankings",
        formula="μ_raw (from TrueSkill updates based on judge rankings)",
        requires_metrics=[],
    ),
    Dimension.COST_ADJUSTED: DimensionConfig(
        name="Cost-Adjusted",
        description="Quality normalized by cost efficiency",
        formula="μ_raw / log(1 + cost_per_million)",
        unit="quality/cost",
        requires_metrics=["cost_usd"],
    ),
    Dimension.LATENCY: DimensionConfig(
        name="Latency",
        description="Response time scoring (lower latency = higher score)",
        formula="S_lat = 1 / (1 + L / τ_L) where τ_L = 1000ms",
        unit="ms",
        higher_is_better=False,
        requires_metrics=["latency_ms"],
    ),
    Dimension.TTFT: DimensionConfig(
        name="Time to First Token",
        description="Time to first token scoring",
        formula="S_ttft = 1 / (1 + T / τ_T) where τ_T = 200ms",
        unit="ms",
        higher_is_better=False,
        requires_metrics=["ttft_ms"],
    ),
    Dimension.CONSISTENCY: DimensionConfig(
        name="Consistency",
        description="Response consistency across repeated runs",
        formula="1 / (1 + σ_responses)",
        requires_metrics=["response_variance"],
    ),
    Dimension.TOKEN_EFFICIENCY: DimensionConfig(
        name="Token Efficiency",
        description="Quality per token used",
        formula="μ_raw / log(1 + n_tokens)",
        unit="quality/token",
        requires_metrics=["output_tokens"],
    ),
    Dimension.INSTRUCTION_FOLLOWING: DimensionConfig(
        name="Instruction Following",
        description="Format and constraint compliance",
        formula="constraints_satisfied / total_constraints",
        requires_metrics=["constraints_satisfied", "total_constraints"],
    ),
    Dimension.HALLUCINATION_RESISTANCE: DimensionConfig(
        name="Hallucination Resistance",
        description="Factual accuracy and grounding",
        formula="1 - (hallucination_count / verifiable_claims)",
        requires_metrics=["hallucination_count", "verifiable_claims"],
    ),
    Dimension.LONG_CONTEXT: DimensionConfig(
        name="Long Context",
        description="Quality retention at long context lengths",
        formula="μ_quality@max / μ_quality@4K",
        requires_metrics=["quality_at_lengths"],
    ),
    Dimension.COMBINED: DimensionConfig(
        name="Combined",
        description="User-customizable weighted aggregate of all dimensions",
        formula="Σ w_d × μ_d",
        requires_metrics=[],
    ),
}


# =============================================================================
# Scoring Functions
# =============================================================================

def compute_latency_score(
    latency_ms: float,
    tau_latency: float = 1000.0
) -> float:
    """
    Compute latency score.
    
    Formula: S_lat = 1 / (1 + L / τ_L)
    
    Args:
        latency_ms: Latency in milliseconds
        tau_latency: Latency temperature (default 1000ms)
    
    Returns:
        Score between 0 and 1 (higher is better)
    """
    if latency_ms < 0:
        latency_ms = 0
    return 1.0 / (1.0 + latency_ms / tau_latency)


def compute_ttft_score(
    ttft_ms: float,
    tau_ttft: float = 200.0
) -> float:
    """
    Compute time-to-first-token score.
    
    Formula: S_ttft = 1 / (1 + T / τ_T)
    
    Args:
        ttft_ms: TTFT in milliseconds
        tau_ttft: TTFT temperature (default 200ms)
    
    Returns:
        Score between 0 and 1 (higher is better)
    """
    if ttft_ms < 0:
        ttft_ms = 0
    return 1.0 / (1.0 + ttft_ms / tau_ttft)


def compute_consistency_score(response_variance: float) -> float:
    """
    Compute consistency score from response variance.
    
    Formula: 1 / (1 + σ_responses)
    
    Args:
        response_variance: Standard deviation of responses across repeated runs
    
    Returns:
        Score between 0 and 1 (higher is better)
    """
    if response_variance < 0:
        response_variance = 0
    return 1.0 / (1.0 + response_variance)


def compute_token_efficiency_score(
    mu_raw: float,
    output_tokens: int
) -> float:
    """
    Compute token efficiency score.
    
    Formula: μ_raw / log(1 + n_tokens)
    
    Args:
        mu_raw: Raw quality rating
        output_tokens: Number of output tokens
    
    Returns:
        Token efficiency score
    """
    if output_tokens <= 0:
        return mu_raw  # No penalty for zero tokens
    return mu_raw / math.log(1 + output_tokens)


def compute_cost_adjusted_score(
    mu_raw: float,
    cost_per_million: float
) -> float:
    """
    Compute cost-adjusted quality score.
    
    Formula: μ_raw / log(1 + cost_per_million)
    
    Args:
        mu_raw: Raw quality rating
        cost_per_million: Cost per million tokens
    
    Returns:
        Cost-adjusted quality score
    """
    if cost_per_million <= 0:
        return mu_raw  # No cost adjustment for free models
    return mu_raw / math.log(1 + cost_per_million)


def compute_instruction_following_score(
    constraints_satisfied: int,
    total_constraints: int
) -> float:
    """
    Compute instruction following score.
    
    Formula: constraints_satisfied / total_constraints
    
    Args:
        constraints_satisfied: Number of constraints satisfied
        total_constraints: Total number of constraints
    
    Returns:
        Score between 0 and 1
    """
    if total_constraints <= 0:
        return 1.0  # Perfect score if no constraints
    return constraints_satisfied / total_constraints


def compute_hallucination_resistance_score(
    hallucination_count: int,
    verifiable_claims: int
) -> float:
    """
    Compute hallucination resistance score.
    
    Formula: 1 - (hallucination_count / verifiable_claims)
    
    Args:
        hallucination_count: Number of detected hallucinations
        verifiable_claims: Total verifiable claims
    
    Returns:
        Score between 0 and 1 (higher is better)
    """
    if verifiable_claims <= 0:
        return 1.0  # Perfect score if no claims to verify
    ratio = hallucination_count / verifiable_claims
    return max(0.0, 1.0 - ratio)


def compute_long_context_score(
    quality_at_max: float,
    quality_at_baseline: float
) -> float:
    """
    Compute long context handling score.
    
    Formula: μ_quality@max / μ_quality@4K
    
    Args:
        quality_at_max: Quality at maximum context length
        quality_at_baseline: Quality at baseline (4K) context
    
    Returns:
        Ratio (1.0 means no degradation)
    """
    if quality_at_baseline <= 0:
        return 1.0  # No comparison possible
    return quality_at_max / quality_at_baseline


def compute_combined_score(
    dimension_mus: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """
    Compute combined weighted score.
    
    Formula: Σ w_d × μ_d
    
    Args:
        dimension_mus: Dictionary of dimension name -> mu value
        weights: Optional custom weights (uses defaults if None)
    
    Returns:
        Weighted combined score
    """
    if weights is None:
        weights = DEFAULT_COMBINED_WEIGHTS
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    
    combined = 0.0
    for dim, weight in weights.items():
        if dim in dimension_mus:
            combined += (weight / total_weight) * dimension_mus[dim]
    
    return combined


# =============================================================================
# Dimension Score Calculator
# =============================================================================

@dataclass
class DimensionScores:
    """Container for all dimension scores."""
    raw_quality: float = 0.0
    cost_adjusted: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    consistency: float = 0.0
    token_efficiency: float = 0.0
    instruction_following: float = 0.0
    hallucination_resistance: float = 0.0
    long_context: float = 0.0
    combined: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            Dimension.RAW_QUALITY.value: self.raw_quality,
            Dimension.COST_ADJUSTED.value: self.cost_adjusted,
            Dimension.LATENCY.value: self.latency,
            Dimension.TTFT.value: self.ttft,
            Dimension.CONSISTENCY.value: self.consistency,
            Dimension.TOKEN_EFFICIENCY.value: self.token_efficiency,
            Dimension.INSTRUCTION_FOLLOWING.value: self.instruction_following,
            Dimension.HALLUCINATION_RESISTANCE.value: self.hallucination_resistance,
            Dimension.LONG_CONTEXT.value: self.long_context,
            Dimension.COMBINED.value: self.combined,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionScores":
        """Create from dictionary."""
        return cls(
            raw_quality=data.get(Dimension.RAW_QUALITY.value, 0.0),
            cost_adjusted=data.get(Dimension.COST_ADJUSTED.value, 0.0),
            latency=data.get(Dimension.LATENCY.value, 0.0),
            ttft=data.get(Dimension.TTFT.value, 0.0),
            consistency=data.get(Dimension.CONSISTENCY.value, 0.0),
            token_efficiency=data.get(Dimension.TOKEN_EFFICIENCY.value, 0.0),
            instruction_following=data.get(Dimension.INSTRUCTION_FOLLOWING.value, 0.0),
            hallucination_resistance=data.get(Dimension.HALLUCINATION_RESISTANCE.value, 0.0),
            long_context=data.get(Dimension.LONG_CONTEXT.value, 0.0),
            combined=data.get(Dimension.COMBINED.value, 0.0),
        )
    
    def get(self, dimension: Dimension) -> float:
        """Get score for a specific dimension."""
        return getattr(self, dimension.value, 0.0)
    
    def set(self, dimension: Dimension, value: float) -> None:
        """Set score for a specific dimension."""
        setattr(self, dimension.value, value)


def calculate_dimension_scores(
    mu_raw: float,
    cost_per_million: float = 0.0,
    latency_ms: float = None,
    ttft_ms: float = None,
    response_variance: float = None,
    output_tokens: int = None,
    constraints_satisfied: int = None,
    total_constraints: int = None,
    hallucination_count: int = None,
    verifiable_claims: int = None,
    quality_at_max: float = None,
    quality_at_baseline: float = None,
    weights: Dict[str, float] = None,
    tau_latency: float = 1000.0,
    tau_ttft: float = 200.0,
) -> DimensionScores:
    """
    Calculate all dimension scores from metrics.
    
    Args:
        mu_raw: Raw quality rating
        cost_per_million: Cost per million tokens
        latency_ms: Response latency in ms
        ttft_ms: Time to first token in ms
        response_variance: Variance across repeated runs
        output_tokens: Number of output tokens
        constraints_satisfied: Constraints satisfied
        total_constraints: Total constraints
        hallucination_count: Detected hallucinations
        verifiable_claims: Total verifiable claims
        quality_at_max: Quality at max context
        quality_at_baseline: Quality at baseline context
        weights: Custom dimension weights
        tau_latency: Latency temperature
        tau_ttft: TTFT temperature
    
    Returns:
        DimensionScores with all calculated values
    """
    scores = DimensionScores()
    
    # Raw Quality (directly from TrueSkill)
    scores.raw_quality = mu_raw
    
    # Cost-Adjusted
    if cost_per_million is not None:
        scores.cost_adjusted = compute_cost_adjusted_score(mu_raw, cost_per_million)
    
    # Latency
    if latency_ms is not None:
        scores.latency = compute_latency_score(latency_ms, tau_latency)
    
    # TTFT
    if ttft_ms is not None:
        scores.ttft = compute_ttft_score(ttft_ms, tau_ttft)
    
    # Consistency
    if response_variance is not None:
        scores.consistency = compute_consistency_score(response_variance)
    
    # Token Efficiency
    if output_tokens is not None:
        scores.token_efficiency = compute_token_efficiency_score(mu_raw, output_tokens)
    
    # Instruction Following
    if constraints_satisfied is not None and total_constraints is not None:
        scores.instruction_following = compute_instruction_following_score(
            constraints_satisfied, total_constraints
        )
    
    # Hallucination Resistance
    if hallucination_count is not None and verifiable_claims is not None:
        scores.hallucination_resistance = compute_hallucination_resistance_score(
            hallucination_count, verifiable_claims
        )
    
    # Long Context
    if quality_at_max is not None and quality_at_baseline is not None:
        scores.long_context = compute_long_context_score(
            quality_at_max, quality_at_baseline
        )
    
    # Combined score
    scores.combined = compute_combined_score(scores.to_dict(), weights)
    
    return scores


# =============================================================================
# Utility Functions
# =============================================================================

def get_dimension_list() -> List[Dimension]:
    """Get list of all dimensions."""
    return list(Dimension)


def get_scorable_dimensions() -> List[Dimension]:
    """Get dimensions that can be independently scored (excludes COMBINED)."""
    return [d for d in Dimension if d != Dimension.COMBINED]


def get_dimension_config(dimension: Dimension) -> DimensionConfig:
    """Get configuration for a dimension."""
    return DIMENSION_CONFIGS[dimension]


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


