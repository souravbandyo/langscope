"""
Parameter Pydantic models for LangScope.

Defines all parameter dataclasses for the dynamic parameter management system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ParamType(str, Enum):
    """Parameter type identifiers."""
    TRUESKILL = "trueskill"
    STRATA = "strata"
    MATCH = "match"
    TEMPERATURE = "temperature"
    DIMENSION_WEIGHTS = "dimension_weights"
    TRANSFER = "transfer"
    FEEDBACK = "feedback"
    PENALTY = "penalty"
    CONSISTENCY = "consistency"
    LONG_CONTEXT = "long_context"


class TrueSkillParams(BaseModel):
    """TrueSkill algorithm parameters."""
    
    mu_0: float = Field(
        default=1500.0,
        description="Default mean rating"
    )
    sigma_0: float = Field(
        default=166.0,
        description="Default uncertainty (standard deviation)"
    )
    beta: float = Field(
        default=83.0,
        description="Performance variability (skill variance in a single game)"
    )
    tau: float = Field(
        default=8.3,
        description="Dynamics factor (σ increase between matches)"
    )
    conservative_k: float = Field(
        default=3.0,
        description="Conservative estimate multiplier (μ - k*σ)"
    )
    
    model_config = {"extra": "forbid"}


class StrataParams(BaseModel):
    """Strata threshold parameters for role assignment."""
    
    elite_threshold: float = Field(
        default=1520.0,
        description="Elite stratum threshold (judging + content creation)"
    )
    high_threshold: float = Field(
        default=1450.0,
        description="High stratum threshold (content creation only)"
    )
    mid_threshold: float = Field(
        default=1400.0,
        description="Mid stratum threshold (competition only)"
    )
    strata_names: Dict[int, str] = Field(
        default={4: "elite", 3: "high", 2: "mid", 1: "low"},
        description="Stratum number to name mapping"
    )
    
    model_config = {"extra": "forbid"}
    
    def get_stratum(self, mu: float) -> int:
        """Get stratum number from TrueSkill μ."""
        if mu >= self.elite_threshold:
            return 4
        elif mu >= self.high_threshold:
            return 3
        elif mu >= self.mid_threshold:
            return 2
        return 1
    
    def get_stratum_name(self, mu: float) -> str:
        """Get stratum name from TrueSkill μ."""
        return self.strata_names.get(self.get_stratum(mu), "unknown")


class MatchParams(BaseModel):
    """Match configuration parameters."""
    
    players_per_match: int = Field(
        default=6,
        ge=2,
        le=10,
        description="Target number of players per match"
    )
    min_players: int = Field(
        default=5,
        ge=2,
        description="Minimum players for a valid match"
    )
    max_players: int = Field(
        default=6,
        le=10,
        description="Maximum players per match"
    )
    judge_count: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of judge models per match"
    )
    max_matches_per_model: int = Field(
        default=50,
        ge=1,
        description="Maximum matches per model (cap for fair distribution)"
    )
    swiss_delta: float = Field(
        default=75.0,
        ge=0,
        description="Maximum TrueSkill μ difference for grouping players"
    )
    
    model_config = {"extra": "forbid"}
    
    @field_validator("min_players")
    @classmethod
    def min_less_than_max(cls, v: int, info) -> int:
        if "max_players" in info.data and v > info.data["max_players"]:
            raise ValueError("min_players cannot exceed max_players")
        return v


class TemperatureParams(BaseModel):
    """Temperature parameters for softmax and scoring."""
    
    cost_temp: float = Field(
        default=0.05,
        gt=0,
        description="Cost temperature for efficiency weighting"
    )
    rating_temp: float = Field(
        default=300.0,
        gt=0,
        description="Rating temperature for softmax weighting (judge/creator selection)"
    )
    latency_temp: float = Field(
        default=1000.0,
        gt=0,
        description="Latency scoring temperature (ms)"
    )
    ttft_temp: float = Field(
        default=200.0,
        gt=0,
        description="Time-to-first-token scoring temperature (ms)"
    )
    
    model_config = {"extra": "forbid"}


class DimensionWeightParams(BaseModel):
    """Dimension weights for Combined score calculation."""
    
    raw_quality: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Weight for Raw Quality dimension"
    )
    cost_adjusted: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Weight for Cost-Adjusted dimension"
    )
    latency: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Weight for Latency dimension"
    )
    ttft: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Weight for Time-to-First-Token dimension"
    )
    consistency: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Weight for Consistency dimension"
    )
    token_efficiency: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Weight for Token Efficiency dimension"
    )
    instruction_following: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Weight for Instruction Following dimension"
    )
    hallucination_resistance: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Weight for Hallucination Resistance dimension"
    )
    long_context: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Weight for Long Context dimension"
    )
    
    model_config = {"extra": "forbid"}
    
    def to_weights_dict(self) -> Dict[str, float]:
        """Convert to dimension name -> weight dict."""
        return {
            "raw_quality": self.raw_quality,
            "cost_adjusted": self.cost_adjusted,
            "latency": self.latency,
            "ttft": self.ttft,
            "consistency": self.consistency,
            "token_efficiency": self.token_efficiency,
            "instruction_following": self.instruction_following,
            "hallucination_resistance": self.hallucination_resistance,
            "long_context": self.long_context,
        }
    
    def total_weight(self) -> float:
        """Get sum of all weights (should be ~1.0)."""
        return sum(self.to_weights_dict().values())
    
    def normalize(self) -> "DimensionWeightParams":
        """Return normalized weights summing to 1.0."""
        total = self.total_weight()
        if total == 0:
            return self
        weights = self.to_weights_dict()
        return DimensionWeightParams(**{k: v / total for k, v in weights.items()})


class TransferParams(BaseModel):
    """Transfer learning parameters."""
    
    correlation_tau: float = Field(
        default=20.0,
        gt=0,
        description="Bayesian smoothing parameter for correlation learning"
    )
    sigma_base: float = Field(
        default=50.0,
        gt=0,
        description="Baseline domain uncertainty for transfer"
    )
    specialist_z_threshold: float = Field(
        default=2.0,
        gt=0,
        description="Z-score threshold for specialist detection"
    )
    
    model_config = {"extra": "forbid"}


class FeedbackParams(BaseModel):
    """User feedback integration parameters."""
    
    user_weight_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=3.0,
        description="α_u: user credibility multiplier for feedback weight"
    )
    user_weight_base: float = Field(
        default=1.0,
        ge=0,
        description="Base weight for comparison"
    )
    use_case_tau: float = Field(
        default=10.0,
        gt=0,
        description="Smoothing parameter for use-case adjustments"
    )
    judge_calibration_gamma: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Weight adjustment factor for calibration"
    )
    user_surprise_threshold: float = Field(
        default=2.0,
        gt=0,
        description="Z-score threshold for user surprise/specialist detection"
    )
    
    model_config = {"extra": "forbid"}


class PenaltyParams(BaseModel):
    """Penalty system parameters."""
    
    judge_penalty_mu: float = Field(
        default=10.0,
        ge=0,
        description="Judge outlier penalty (in μ points)"
    )
    outlier_disagreement_threshold: float = Field(
        default=0.40,
        ge=0,
        le=1,
        description="Outlier disagreement threshold (>40% disagreement with consensus)"
    )
    content_rejection_penalty: float = Field(
        default=5.0,
        ge=0,
        description="Penalty for rejected content (in μ points)"
    )
    
    model_config = {"extra": "forbid"}


class ConsistencyParams(BaseModel):
    """Consistency evaluation parameters."""
    
    n_runs: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of repeated evaluations for consistency"
    )
    temperature_variance: float = Field(
        default=0.0,
        ge=0,
        description="Allowed variance in model temperature for consistency runs"
    )
    
    model_config = {"extra": "forbid"}


class LongContextParams(BaseModel):
    """Long context evaluation parameters."""
    
    test_lengths: List[int] = Field(
        default=[4096, 16384, 32768, 65536, 131072],
        description="Context lengths to test (in tokens)"
    )
    baseline_length: int = Field(
        default=4096,
        ge=1024,
        description="Baseline context length for degradation ratio"
    )
    
    model_config = {"extra": "forbid"}
    
    @field_validator("test_lengths")
    @classmethod
    def sorted_lengths(cls, v: List[int]) -> List[int]:
        return sorted(v)


class SystemParams(BaseModel):
    """Master container for all parameter groups."""
    
    trueskill: TrueSkillParams = Field(default_factory=TrueSkillParams)
    strata: StrataParams = Field(default_factory=StrataParams)
    match: MatchParams = Field(default_factory=MatchParams)
    temperature: TemperatureParams = Field(default_factory=TemperatureParams)
    dimension_weights: DimensionWeightParams = Field(default_factory=DimensionWeightParams)
    transfer: TransferParams = Field(default_factory=TransferParams)
    feedback: FeedbackParams = Field(default_factory=FeedbackParams)
    penalty: PenaltyParams = Field(default_factory=PenaltyParams)
    consistency: ConsistencyParams = Field(default_factory=ConsistencyParams)
    long_context: LongContextParams = Field(default_factory=LongContextParams)
    
    model_config = {"extra": "forbid"}
    
    def get_param_group(self, param_type: ParamType) -> BaseModel:
        """Get parameter group by type."""
        mapping = {
            ParamType.TRUESKILL: self.trueskill,
            ParamType.STRATA: self.strata,
            ParamType.MATCH: self.match,
            ParamType.TEMPERATURE: self.temperature,
            ParamType.DIMENSION_WEIGHTS: self.dimension_weights,
            ParamType.TRANSFER: self.transfer,
            ParamType.FEEDBACK: self.feedback,
            ParamType.PENALTY: self.penalty,
            ParamType.CONSISTENCY: self.consistency,
            ParamType.LONG_CONTEXT: self.long_context,
        }
        return mapping[param_type]


# Mapping from param type to model class
PARAM_TYPE_TO_CLASS: Dict[ParamType, type] = {
    ParamType.TRUESKILL: TrueSkillParams,
    ParamType.STRATA: StrataParams,
    ParamType.MATCH: MatchParams,
    ParamType.TEMPERATURE: TemperatureParams,
    ParamType.DIMENSION_WEIGHTS: DimensionWeightParams,
    ParamType.TRANSFER: TransferParams,
    ParamType.FEEDBACK: FeedbackParams,
    ParamType.PENALTY: PenaltyParams,
    ParamType.CONSISTENCY: ConsistencyParams,
    ParamType.LONG_CONTEXT: LongContextParams,
}


def get_default_params(param_type: ParamType) -> BaseModel:
    """Get default parameters for a given type."""
    return PARAM_TYPE_TO_CLASS[param_type]()


