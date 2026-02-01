"""
TrueSkill rating classes for LangScope.

Provides uncertainty-aware skill estimation with:
- Single dimension rating (TrueSkillRating)
- Dual ratings (DualTrueSkill): raw + cost-adjusted
- 10-dimensional ratings (MultiDimensionalTrueSkill): full dimension support
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional, List
from langscope.core.constants import (
    TRUESKILL_MU_0,
    TRUESKILL_SIGMA_0,
    TRUESKILL_CONSERVATIVE_K,
)


@dataclass
class TrueSkillRating:
    """
    TrueSkill rating with mean (μ) and uncertainty (σ).
    
    The skill of a player is modeled as a Gaussian distribution:
    θ ~ N(μ, σ²)
    
    Attributes:
        mu: Mean skill estimate
        sigma: Uncertainty (standard deviation) of the skill estimate
    """
    mu: float = TRUESKILL_MU_0
    sigma: float = TRUESKILL_SIGMA_0
    
    def conservative_estimate(self, k: float = TRUESKILL_CONSERVATIVE_K) -> float:
        """
        Conservative skill estimate (μ - kσ).
        
        This is used for ranking when we want to be confident about skill.
        Default k=3 gives approximately 99.7% confidence.
        
        Args:
            k: Number of standard deviations to subtract
        
        Returns:
            Conservative skill estimate
        """
        return self.mu - k * self.sigma
    
    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """
        Confidence interval for the skill estimate.
        
        Args:
            z: Z-score for confidence level (1.96 for 95% CI)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return (self.mu - z * self.sigma, self.mu + z * self.sigma)
    
    def variance(self) -> float:
        """Get variance (σ²)."""
        return self.sigma ** 2
    
    def precision(self) -> float:
        """Get precision (1/σ²)."""
        return 1.0 / (self.sigma ** 2) if self.sigma > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {"mu": self.mu, "sigma": self.sigma}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrueSkillRating':
        """Create from dictionary."""
        return cls(
            mu=float(data.get("mu", TRUESKILL_MU_0)),
            sigma=float(data.get("sigma", TRUESKILL_SIGMA_0))
        )
    
    def __repr__(self) -> str:
        return f"TrueSkillRating(μ={self.mu:.1f}, σ={self.sigma:.1f})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrueSkillRating):
            return False
        return abs(self.mu - other.mu) < 1e-6 and abs(self.sigma - other.sigma) < 1e-6


@dataclass
class DualTrueSkill:
    """
    Dual TrueSkill rating system: raw performance + cost-adjusted.
    
    The system maintains two parallel ratings:
    1. Raw: Based purely on response quality
    2. Cost-adjusted: Incorporates efficiency (quality per cost)
    
    This allows evaluation of both absolute performance and value.
    """
    raw: TrueSkillRating = field(default_factory=TrueSkillRating)
    cost_adjusted: TrueSkillRating = field(default_factory=TrueSkillRating)
    
    def conservative_raw(self) -> float:
        """Get conservative estimate for raw rating."""
        return self.raw.conservative_estimate()
    
    def conservative_cost(self) -> float:
        """Get conservative estimate for cost-adjusted rating."""
        return self.cost_adjusted.conservative_estimate()
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary for serialization."""
        return {
            "raw": self.raw.to_dict(),
            "cost_adjusted": self.cost_adjusted.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DualTrueSkill':
        """Create from dictionary."""
        return cls(
            raw=TrueSkillRating.from_dict(data.get("raw", {})),
            cost_adjusted=TrueSkillRating.from_dict(data.get("cost_adjusted", {}))
        )
    
    def __repr__(self) -> str:
        return f"DualTrueSkill(raw={self.raw}, cost={self.cost_adjusted})"


def create_initial_rating(
    mu: float = TRUESKILL_MU_0,
    sigma: float = TRUESKILL_SIGMA_0
) -> TrueSkillRating:
    """
    Create a new initial rating.
    
    Args:
        mu: Initial mean (default: system default)
        sigma: Initial uncertainty (default: system default)
    
    Returns:
        New TrueSkillRating instance
    """
    return TrueSkillRating(mu=mu, sigma=sigma)


def create_initial_dual_rating(
    mu: float = TRUESKILL_MU_0,
    sigma: float = TRUESKILL_SIGMA_0
) -> DualTrueSkill:
    """
    Create a new initial dual rating.
    
    Both raw and cost-adjusted start at the same values.
    
    Args:
        mu: Initial mean (default: system default)
        sigma: Initial uncertainty (default: system default)
    
    Returns:
        New DualTrueSkill instance
    """
    return DualTrueSkill(
        raw=TrueSkillRating(mu=mu, sigma=sigma),
        cost_adjusted=TrueSkillRating(mu=mu, sigma=sigma)
    )


# Dimension names for MultiDimensionalTrueSkill
DIMENSION_NAMES = [
    "raw_quality",
    "cost_adjusted",
    "latency",
    "ttft",
    "consistency",
    "token_efficiency",
    "instruction_following",
    "hallucination_resistance",
    "long_context",
    "combined",
]


@dataclass
class MultiDimensionalTrueSkill:
    """
    10-Dimensional TrueSkill rating system.
    
    Maintains independent TrueSkillRating instances for each dimension:
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
    raw_quality: TrueSkillRating = field(default_factory=TrueSkillRating)
    cost_adjusted: TrueSkillRating = field(default_factory=TrueSkillRating)
    latency: TrueSkillRating = field(default_factory=TrueSkillRating)
    ttft: TrueSkillRating = field(default_factory=TrueSkillRating)
    consistency: TrueSkillRating = field(default_factory=TrueSkillRating)
    token_efficiency: TrueSkillRating = field(default_factory=TrueSkillRating)
    instruction_following: TrueSkillRating = field(default_factory=TrueSkillRating)
    hallucination_resistance: TrueSkillRating = field(default_factory=TrueSkillRating)
    long_context: TrueSkillRating = field(default_factory=TrueSkillRating)
    combined: TrueSkillRating = field(default_factory=TrueSkillRating)
    
    def get_dimension(self, dimension: str) -> TrueSkillRating:
        """
        Get rating for a specific dimension.
        
        Args:
            dimension: Dimension name
        
        Returns:
            TrueSkillRating for the dimension
        
        Raises:
            ValueError: If dimension is invalid
        """
        if dimension not in DIMENSION_NAMES:
            raise ValueError(f"Invalid dimension: {dimension}. Valid: {DIMENSION_NAMES}")
        return getattr(self, dimension)
    
    def set_dimension(
        self,
        dimension: str,
        mu: float = None,
        sigma: float = None,
        rating: TrueSkillRating = None
    ) -> None:
        """
        Set rating for a specific dimension.
        
        Args:
            dimension: Dimension name
            mu: New mu value (optional)
            sigma: New sigma value (optional)
            rating: Full TrueSkillRating to set (optional)
        """
        if dimension not in DIMENSION_NAMES:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        if rating is not None:
            setattr(self, dimension, rating)
        else:
            current = getattr(self, dimension)
            if mu is not None:
                current.mu = mu
            if sigma is not None:
                current.sigma = sigma
    
    def update_combined(
        self,
        weights: Dict[str, float] = None
    ) -> None:
        """
        Update the combined dimension as weighted aggregate of others.
        
        Args:
            weights: Optional custom weights for each dimension
        """
        if weights is None:
            # Default weights from DimensionWeightParams defaults
            weights = {
                "raw_quality": 0.20,
                "cost_adjusted": 0.10,
                "latency": 0.10,
                "ttft": 0.05,
                "consistency": 0.10,
                "token_efficiency": 0.10,
                "instruction_following": 0.15,
                "hallucination_resistance": 0.15,
                "long_context": 0.05,
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return
        
        # Compute weighted mu and sigma
        combined_mu = 0.0
        combined_var = 0.0  # Combine variances for uncertainty
        
        for dim_name, weight in weights.items():
            if dim_name == "combined":
                continue
            rating = self.get_dimension(dim_name)
            norm_weight = weight / total_weight
            combined_mu += norm_weight * rating.mu
            combined_var += (norm_weight ** 2) * (rating.sigma ** 2)
        
        self.combined.mu = combined_mu
        self.combined.sigma = combined_var ** 0.5
    
    def get_all_mus(self) -> Dict[str, float]:
        """Get all dimension mu values."""
        return {dim: self.get_dimension(dim).mu for dim in DIMENSION_NAMES}
    
    def get_all_sigmas(self) -> Dict[str, float]:
        """Get all dimension sigma values."""
        return {dim: self.get_dimension(dim).sigma for dim in DIMENSION_NAMES}
    
    def get_conservative_estimates(
        self,
        k: float = TRUESKILL_CONSERVATIVE_K
    ) -> Dict[str, float]:
        """Get conservative estimates for all dimensions."""
        return {
            dim: self.get_dimension(dim).conservative_estimate(k)
            for dim in DIMENSION_NAMES
        }
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary for serialization."""
        return {
            dim: self.get_dimension(dim).to_dict()
            for dim in DIMENSION_NAMES
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiDimensionalTrueSkill':
        """Create from dictionary."""
        instance = cls()
        for dim in DIMENSION_NAMES:
            if dim in data:
                setattr(instance, dim, TrueSkillRating.from_dict(data[dim]))
        return instance
    
    @classmethod
    def from_dual(cls, dual: DualTrueSkill) -> 'MultiDimensionalTrueSkill':
        """
        Create MultiDimensionalTrueSkill from DualTrueSkill.
        
        Useful for migration from 2D to 10D system.
        Raw and cost_adjusted are copied, others get default values.
        
        Args:
            dual: Existing DualTrueSkill
        
        Returns:
            New MultiDimensionalTrueSkill instance
        """
        instance = cls()
        instance.raw_quality = TrueSkillRating(
            mu=dual.raw.mu,
            sigma=dual.raw.sigma
        )
        instance.cost_adjusted = TrueSkillRating(
            mu=dual.cost_adjusted.mu,
            sigma=dual.cost_adjusted.sigma
        )
        # Update combined based on raw and cost
        instance.update_combined()
        return instance
    
    def to_dual(self) -> DualTrueSkill:
        """
        Convert to DualTrueSkill (for backward compatibility).
        
        Returns:
            DualTrueSkill with raw_quality and cost_adjusted
        """
        return DualTrueSkill(
            raw=TrueSkillRating(
                mu=self.raw_quality.mu,
                sigma=self.raw_quality.sigma
            ),
            cost_adjusted=TrueSkillRating(
                mu=self.cost_adjusted.mu,
                sigma=self.cost_adjusted.sigma
            )
        )
    
    def __repr__(self) -> str:
        return (
            f"MultiDimensionalTrueSkill("
            f"raw={self.raw_quality.mu:.1f}, "
            f"cost={self.cost_adjusted.mu:.1f}, "
            f"combined={self.combined.mu:.1f})"
        )


def create_initial_multi_dimensional_rating(
    mu: float = TRUESKILL_MU_0,
    sigma: float = TRUESKILL_SIGMA_0
) -> MultiDimensionalTrueSkill:
    """
    Create a new initial 10-dimensional rating.
    
    All dimensions start at the same values.
    
    Args:
        mu: Initial mean (default: system default)
        sigma: Initial uncertainty (default: system default)
    
    Returns:
        New MultiDimensionalTrueSkill instance
    """
    return MultiDimensionalTrueSkill(
        raw_quality=TrueSkillRating(mu=mu, sigma=sigma),
        cost_adjusted=TrueSkillRating(mu=mu, sigma=sigma),
        latency=TrueSkillRating(mu=mu, sigma=sigma),
        ttft=TrueSkillRating(mu=mu, sigma=sigma),
        consistency=TrueSkillRating(mu=mu, sigma=sigma),
        token_efficiency=TrueSkillRating(mu=mu, sigma=sigma),
        instruction_following=TrueSkillRating(mu=mu, sigma=sigma),
        hallucination_resistance=TrueSkillRating(mu=mu, sigma=sigma),
        long_context=TrueSkillRating(mu=mu, sigma=sigma),
        combined=TrueSkillRating(mu=mu, sigma=sigma),
    )


