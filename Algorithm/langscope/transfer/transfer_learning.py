"""
Cross-domain transfer learning for LangScope.

Enables knowledge transfer between domains to accelerate convergence
and improve initial estimates for models in new domains.

Key formulas:
- Single source: μ_target = μ₀ + ρ(μ_source - μ₀)
- Uncertainty: σ_target = √(σ_source² + (1-ρ²)σ₀² + σ_base²)
- Multi-source: w_j = (ρ_j/σ_j) / Σ(ρ_k/σ_k)
"""

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from langscope.core.constants import (
    TRUESKILL_MU_0,
    TRUESKILL_SIGMA_0,
    SIGMA_BASE,
)
from langscope.core.rating import TrueSkillRating, DualTrueSkill
from langscope.transfer.correlation import get_correlation, CorrelationLearner

if TYPE_CHECKING:
    from langscope.core.model import LLMModel


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    target_mu: float
    target_sigma: float
    source_domains: List[str]
    source_weights: Dict[str, float]
    correlation_used: float
    confidence: float
    
    def to_rating(self) -> TrueSkillRating:
        """Convert to TrueSkillRating."""
        return TrueSkillRating(mu=self.target_mu, sigma=self.target_sigma)


class TransferLearning:
    """
    Cross-domain transfer learning.
    
    Uses domain correlations to transfer knowledge from established
    domains to new domains, providing better initial estimates.
    """
    
    def __init__(
        self,
        mu_0: float = TRUESKILL_MU_0,
        sigma_0: float = TRUESKILL_SIGMA_0,
        sigma_base: float = SIGMA_BASE,
        correlation_learner: CorrelationLearner = None
    ):
        """
        Initialize transfer learning.
        
        Args:
            mu_0: Default mean rating
            sigma_0: Default uncertainty
            sigma_base: Baseline domain uncertainty
            correlation_learner: Optional correlation learner
        """
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma_base = sigma_base
        self.correlation_learner = correlation_learner
    
    def transfer_single_source(
        self,
        source_rating: TrueSkillRating,
        source_domain: str,
        target_domain: str
    ) -> TransferResult:
        """
        Transfer from a single source domain.
        
        Formula:
        μ_target = μ₀ + ρ(μ_source - μ₀)
        σ_target = √(σ_source² + (1-ρ²)σ₀² + σ_base²)
        
        Args:
            source_rating: Rating in source domain
            source_domain: Source domain name
            target_domain: Target domain name
        
        Returns:
            TransferResult with target rating
        """
        # Get correlation
        if self.correlation_learner:
            rho = self.correlation_learner.get_correlation(source_domain, target_domain)
        else:
            rho = get_correlation(source_domain, target_domain)
        
        # Transfer mean
        mu_target = self.mu_0 + rho * (source_rating.mu - self.mu_0)
        
        # Transfer uncertainty
        # Higher correlation → lower uncertainty increase
        sigma_target = math.sqrt(
            source_rating.sigma ** 2 +
            (1 - rho ** 2) * self.sigma_0 ** 2 +
            self.sigma_base ** 2
        )
        
        # Confidence based on correlation and source certainty
        confidence = rho * (1 - source_rating.sigma / self.sigma_0)
        confidence = max(0, min(1, confidence))
        
        return TransferResult(
            target_mu=mu_target,
            target_sigma=sigma_target,
            source_domains=[source_domain],
            source_weights={source_domain: 1.0},
            correlation_used=rho,
            confidence=confidence,
        )
    
    def transfer_multi_source(
        self,
        source_ratings: Dict[str, TrueSkillRating],  # {domain: rating}
        target_domain: str
    ) -> TransferResult:
        """
        Transfer from multiple source domains.
        
        Uses reliability-weighted combination:
        w_j = (ρ_j/σ_j) / Σ(ρ_k/σ_k)
        
        Args:
            source_ratings: Ratings in source domains
            target_domain: Target domain name
        
        Returns:
            TransferResult with combined target rating
        """
        if not source_ratings:
            return TransferResult(
                target_mu=self.mu_0,
                target_sigma=self.sigma_0,
                source_domains=[],
                source_weights={},
                correlation_used=0.0,
                confidence=0.0,
            )
        
        if len(source_ratings) == 1:
            domain = list(source_ratings.keys())[0]
            return self.transfer_single_source(
                source_ratings[domain], domain, target_domain
            )
        
        # Compute weights for each source
        weights = {}
        total_weight = 0.0
        correlations = {}
        
        for domain, rating in source_ratings.items():
            if self.correlation_learner:
                rho = self.correlation_learner.get_correlation(domain, target_domain)
            else:
                rho = get_correlation(domain, target_domain)
            
            correlations[domain] = rho
            
            # Weight: ρ/σ (reliability-weighted)
            if rating.sigma > 0:
                w = abs(rho) / rating.sigma
            else:
                w = abs(rho) * 100  # High weight for certain ratings
            
            weights[domain] = w
            total_weight += w
        
        # Normalize weights
        if total_weight > 0:
            weights = {d: w / total_weight for d, w in weights.items()}
        else:
            weights = {d: 1.0 / len(source_ratings) for d in source_ratings}
        
        # Weighted combination of transferred means
        mu_target = self.mu_0
        weighted_deviation = 0.0
        
        for domain, rating in source_ratings.items():
            rho = correlations[domain]
            w = weights[domain]
            weighted_deviation += w * rho * (rating.mu - self.mu_0)
        
        mu_target = self.mu_0 + weighted_deviation
        
        # Combined uncertainty
        # Use weighted average of individual transfer uncertainties
        variance_sum = 0.0
        for domain, rating in source_ratings.items():
            rho = correlations[domain]
            w = weights[domain]
            individual_var = (
                rating.sigma ** 2 +
                (1 - rho ** 2) * self.sigma_0 ** 2 +
                self.sigma_base ** 2
            )
            variance_sum += w * individual_var
        
        sigma_target = math.sqrt(variance_sum)
        
        # Average correlation (weighted)
        avg_correlation = sum(
            weights[d] * correlations[d] for d in source_ratings
        )
        
        # Confidence
        avg_sigma = sum(
            weights[d] * source_ratings[d].sigma for d in source_ratings
        )
        confidence = abs(avg_correlation) * (1 - avg_sigma / self.sigma_0)
        confidence = max(0, min(1, confidence))
        
        return TransferResult(
            target_mu=mu_target,
            target_sigma=sigma_target,
            source_domains=list(source_ratings.keys()),
            source_weights=weights,
            correlation_used=avg_correlation,
            confidence=confidence,
        )
    
    def initialize_model_in_domain(
        self,
        model: 'LLMModel',
        target_domain: str
    ) -> TrueSkillRating:
        """
        Initialize a model's rating in a new domain using transfer.
        
        Uses all available domain ratings for the model.
        
        Args:
            model: Model to initialize
            target_domain: Domain to initialize in
        
        Returns:
            Initial TrueSkillRating for the domain
        """
        # Collect existing domain ratings
        source_ratings = {}
        
        for domain, dual_ts in model.trueskill_by_domain.items():
            if domain != target_domain:
                source_ratings[domain] = dual_ts.raw
        
        # Also consider global rating if different from default
        if model.trueskill.raw.sigma < self.sigma_0 * 0.9:
            source_ratings["_global"] = model.trueskill.raw
        
        if not source_ratings:
            return TrueSkillRating(mu=self.mu_0, sigma=self.sigma_0)
        
        result = self.transfer_multi_source(source_ratings, target_domain)
        return result.to_rating()


def transfer_single_source(
    source_mu: float,
    source_sigma: float,
    source_domain: str,
    target_domain: str,
    mu_0: float = TRUESKILL_MU_0,
    sigma_0: float = TRUESKILL_SIGMA_0,
    sigma_base: float = SIGMA_BASE
) -> Tuple[float, float]:
    """
    Convenience function for single-source transfer.
    
    Args:
        source_mu: Source rating mean
        source_sigma: Source rating uncertainty
        source_domain: Source domain
        target_domain: Target domain
        mu_0: Default mean
        sigma_0: Default uncertainty
        sigma_base: Base domain uncertainty
    
    Returns:
        Tuple of (target_mu, target_sigma)
    """
    tl = TransferLearning(mu_0, sigma_0, sigma_base)
    result = tl.transfer_single_source(
        TrueSkillRating(source_mu, source_sigma),
        source_domain,
        target_domain
    )
    return result.target_mu, result.target_sigma


def transfer_multi_source(
    source_ratings: Dict[str, Tuple[float, float]],  # {domain: (mu, sigma)}
    target_domain: str,
    mu_0: float = TRUESKILL_MU_0,
    sigma_0: float = TRUESKILL_SIGMA_0,
    sigma_base: float = SIGMA_BASE
) -> Tuple[float, float, Dict[str, float]]:
    """
    Convenience function for multi-source transfer.
    
    Args:
        source_ratings: Dict of {domain: (mu, sigma)}
        target_domain: Target domain
        mu_0: Default mean
        sigma_0: Default uncertainty
        sigma_base: Base domain uncertainty
    
    Returns:
        Tuple of (target_mu, target_sigma, source_weights)
    """
    tl = TransferLearning(mu_0, sigma_0, sigma_base)
    ratings = {
        domain: TrueSkillRating(mu, sigma)
        for domain, (mu, sigma) in source_ratings.items()
    }
    result = tl.transfer_multi_source(ratings, target_domain)
    return result.target_mu, result.target_sigma, result.source_weights


def should_transfer(
    source_rating: TrueSkillRating,
    source_domain: str,
    target_domain: str,
    min_confidence: float = 0.3
) -> bool:
    """
    Determine if transfer would be beneficial.
    
    Args:
        source_rating: Rating in source domain
        source_domain: Source domain
        target_domain: Target domain
        min_confidence: Minimum confidence threshold
    
    Returns:
        True if transfer is recommended
    """
    tl = TransferLearning()
    result = tl.transfer_single_source(source_rating, source_domain, target_domain)
    return result.confidence >= min_confidence


