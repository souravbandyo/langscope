"""
Core module for LangScope.

Contains fundamental classes and constants for the rating system.
Supports 10-dimensional TrueSkill ratings.
"""

from langscope.core.constants import (
    TRUESKILL_MU_0,
    TRUESKILL_SIGMA_0,
    TRUESKILL_BETA,
    TRUESKILL_TAU,
    PLAYERS_PER_MATCH,
    MIN_PLAYERS,
    MAX_PLAYERS,
    SWISS_DELTA,
    COST_TEMP,
    RATING_TEMP,
)
from langscope.core.rating import (
    TrueSkillRating,
    DualTrueSkill,
    MultiDimensionalTrueSkill,
    DIMENSION_NAMES,
    create_initial_rating,
    create_initial_dual_rating,
    create_initial_multi_dimensional_rating,
)
from langscope.core.model import LLMModel, PerformanceMetrics, MatchIds
from langscope.core.dimensions import (
    Dimension,
    DimensionConfig,
    DimensionScores,
    DIMENSION_CONFIGS,
    DEFAULT_COMBINED_WEIGHTS,
    compute_latency_score,
    compute_ttft_score,
    compute_consistency_score,
    compute_token_efficiency_score,
    compute_cost_adjusted_score,
    compute_instruction_following_score,
    compute_hallucination_resistance_score,
    compute_long_context_score,
    compute_combined_score,
    calculate_dimension_scores,
)
from langscope.core.base_model import (
    BaseModel,
    Architecture,
    Capabilities,
    ContextWindow,
    License,
    QuantizationOption,
    BenchmarkScore,
    BenchmarkAggregates,
)
from langscope.core.deployment import (
    ModelDeployment,
    Provider,
    ProviderType,
    DeploymentConfig,
    Pricing,
    Performance,
    RateLimits,
    Availability,
    AvailabilityStatus,
)
from langscope.core.self_hosted import (
    SelfHostedDeployment,
    HardwareConfig,
    SoftwareConfig,
    SelfHostedCosts,
    Owner,
)

__all__ = [
    # Constants
    "TRUESKILL_MU_0",
    "TRUESKILL_SIGMA_0", 
    "TRUESKILL_BETA",
    "TRUESKILL_TAU",
    "PLAYERS_PER_MATCH",
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "SWISS_DELTA",
    "COST_TEMP",
    "RATING_TEMP",
    # Rating classes
    "TrueSkillRating",
    "DualTrueSkill",
    "MultiDimensionalTrueSkill",
    "DIMENSION_NAMES",
    "create_initial_rating",
    "create_initial_dual_rating",
    "create_initial_multi_dimensional_rating",
    # Model
    "LLMModel",
    "PerformanceMetrics",
    "MatchIds",
    # Dimensions
    "Dimension",
    "DimensionConfig",
    "DimensionScores",
    "DIMENSION_CONFIGS",
    "DEFAULT_COMBINED_WEIGHTS",
    "compute_latency_score",
    "compute_ttft_score",
    "compute_consistency_score",
    "compute_token_efficiency_score",
    "compute_cost_adjusted_score",
    "compute_instruction_following_score",
    "compute_hallucination_resistance_score",
    "compute_long_context_score",
    "compute_combined_score",
    "calculate_dimension_scores",
    # Base Model (Phase 11)
    "BaseModel",
    "Architecture",
    "Capabilities",
    "ContextWindow",
    "License",
    "QuantizationOption",
    "BenchmarkScore",
    "BenchmarkAggregates",
    # Deployment (Phase 11)
    "ModelDeployment",
    "Provider",
    "ProviderType",
    "DeploymentConfig",
    "Pricing",
    "Performance",
    "RateLimits",
    "Availability",
    "AvailabilityStatus",
    # Self-Hosted (Phase 11)
    "SelfHostedDeployment",
    "HardwareConfig",
    "SoftwareConfig",
    "SelfHostedCosts",
    "Owner",
]


