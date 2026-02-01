"""
Specialist detection for LangScope.

Identifies models that significantly outperform or underperform
expectations in specific domains (specialists and weak spots).

Key formula:
z = (μ_actual - μ_predicted) / √(σ_actual² + σ_predicted²)
Specialist if |z| > 2
"""

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from langscope.core.constants import SPECIALIST_Z_THRESHOLD
from langscope.core.rating import TrueSkillRating
from langscope.transfer.transfer_learning import TransferLearning

if TYPE_CHECKING:
    from langscope.core.model import LLMModel


@dataclass
class SpecialistResult:
    """Result of specialist detection."""
    model_id: str
    domain: str
    is_specialist: bool
    z_score: float
    actual_mu: float
    predicted_mu: float
    actual_sigma: float
    predicted_sigma: float
    p_value: float
    category: str  # "specialist", "weak_spot", or "normal"
    
    @property
    def deviation(self) -> float:
        """Get absolute deviation from expected."""
        return self.actual_mu - self.predicted_mu
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "domain": self.domain,
            "is_specialist": self.is_specialist,
            "z_score": self.z_score,
            "actual_mu": self.actual_mu,
            "predicted_mu": self.predicted_mu,
            "actual_sigma": self.actual_sigma,
            "predicted_sigma": self.predicted_sigma,
            "p_value": self.p_value,
            "category": self.category,
        }


class SpecialistDetector:
    """
    Detects specialist and weak-spot patterns in model performance.
    
    A model is a specialist in domain D if its actual performance
    significantly exceeds the prediction based on other domains.
    
    A model has a weak spot in domain D if its actual performance
    significantly underperforms the prediction.
    """
    
    def __init__(
        self,
        z_threshold: float = SPECIALIST_Z_THRESHOLD,
        transfer_learner: TransferLearning = None
    ):
        """
        Initialize detector.
        
        Args:
            z_threshold: Z-score threshold for detection (default: 2.0)
            transfer_learner: Transfer learning instance
        """
        self.z_threshold = z_threshold
        self.transfer_learner = transfer_learner or TransferLearning()
    
    def detect(
        self,
        model: 'LLMModel',
        target_domain: str
    ) -> SpecialistResult:
        """
        Detect if model is a specialist or has weak spot in domain.
        
        Args:
            model: Model to analyze
            target_domain: Domain to check
        
        Returns:
            SpecialistResult
        """
        # Get actual performance in target domain
        if target_domain not in model.trueskill_by_domain:
            return SpecialistResult(
                model_id=model.model_id,
                domain=target_domain,
                is_specialist=False,
                z_score=0.0,
                actual_mu=0.0,
                predicted_mu=0.0,
                actual_sigma=0.0,
                predicted_sigma=0.0,
                p_value=1.0,
                category="insufficient_data",
            )
        
        actual = model.trueskill_by_domain[target_domain].raw
        
        # Get predicted performance from other domains
        predicted = self._predict_performance(model, target_domain)
        
        # Compute z-score
        combined_sigma = math.sqrt(actual.sigma ** 2 + predicted.sigma ** 2)
        
        if combined_sigma < 1e-6:
            z_score = 0.0
        else:
            z_score = (actual.mu - predicted.mu) / combined_sigma
        
        # Compute p-value (two-tailed)
        p_value = self._compute_p_value(abs(z_score))
        
        # Determine category
        is_specialist = abs(z_score) >= self.z_threshold
        
        if z_score >= self.z_threshold:
            category = "specialist"
        elif z_score <= -self.z_threshold:
            category = "weak_spot"
        else:
            category = "normal"
        
        return SpecialistResult(
            model_id=model.model_id,
            domain=target_domain,
            is_specialist=is_specialist,
            z_score=z_score,
            actual_mu=actual.mu,
            predicted_mu=predicted.mu,
            actual_sigma=actual.sigma,
            predicted_sigma=predicted.sigma,
            p_value=p_value,
            category=category,
        )
    
    def _predict_performance(
        self,
        model: 'LLMModel',
        target_domain: str
    ) -> TrueSkillRating:
        """Predict performance in target domain using transfer."""
        # Collect ratings from other domains
        source_ratings = {}
        
        for domain, dual_ts in model.trueskill_by_domain.items():
            if domain != target_domain:
                source_ratings[domain] = dual_ts.raw
        
        if not source_ratings:
            # Use global rating
            return model.trueskill.raw
        
        result = self.transfer_learner.transfer_multi_source(
            source_ratings, target_domain
        )
        return result.to_rating()
    
    def _compute_p_value(self, z: float) -> float:
        """Compute two-tailed p-value from z-score."""
        # Use error function approximation
        # P(|Z| > z) = 2 * (1 - Φ(z)) ≈ 2 * (1 - 0.5 * (1 + erf(z/√2)))
        return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    
    def detect_all_domains(
        self,
        model: 'LLMModel'
    ) -> List[SpecialistResult]:
        """
        Detect specialists/weak spots across all domains.
        
        Args:
            model: Model to analyze
        
        Returns:
            List of SpecialistResults for each domain
        """
        results = []
        
        for domain in model.trueskill_by_domain:
            result = self.detect(model, domain)
            results.append(result)
        
        return results
    
    def get_specialists(
        self,
        model: 'LLMModel'
    ) -> List[SpecialistResult]:
        """Get domains where model is a specialist."""
        results = self.detect_all_domains(model)
        return [r for r in results if r.category == "specialist"]
    
    def get_weak_spots(
        self,
        model: 'LLMModel'
    ) -> List[SpecialistResult]:
        """Get domains where model has weak spots."""
        results = self.detect_all_domains(model)
        return [r for r in results if r.category == "weak_spot"]


def detect_specialist(
    actual_mu: float,
    actual_sigma: float,
    predicted_mu: float,
    predicted_sigma: float,
    z_threshold: float = SPECIALIST_Z_THRESHOLD
) -> Tuple[bool, float, str]:
    """
    Convenience function to detect specialist/weak spot.
    
    Args:
        actual_mu: Actual rating mean
        actual_sigma: Actual rating uncertainty
        predicted_mu: Predicted rating mean
        predicted_sigma: Predicted rating uncertainty
        z_threshold: Z-score threshold
    
    Returns:
        Tuple of (is_specialist_or_weak_spot, z_score, category)
    """
    combined_sigma = math.sqrt(actual_sigma ** 2 + predicted_sigma ** 2)
    
    if combined_sigma < 1e-6:
        return False, 0.0, "normal"
    
    z_score = (actual_mu - predicted_mu) / combined_sigma
    
    if z_score >= z_threshold:
        return True, z_score, "specialist"
    elif z_score <= -z_threshold:
        return True, z_score, "weak_spot"
    else:
        return False, z_score, "normal"


def compute_specialization_score(
    model: 'LLMModel'
) -> float:
    """
    Compute overall specialization score for a model.
    
    Score > 1 indicates specialist tendencies (domain-specific strength)
    Score < 1 indicates generalist tendencies (consistent across domains)
    
    Args:
        model: Model to analyze
    
    Returns:
        Specialization score (1.0 = balanced)
    """
    if len(model.trueskill_by_domain) < 2:
        return 1.0
    
    mus = [ts.raw.mu for ts in model.trueskill_by_domain.values()]
    
    mean_mu = sum(mus) / len(mus)
    variance = sum((mu - mean_mu) ** 2 for mu in mus) / len(mus)
    std_dev = math.sqrt(variance)
    
    # Normalize by expected variance (based on sigma)
    avg_sigma = sum(
        ts.raw.sigma for ts in model.trueskill_by_domain.values()
    ) / len(model.trueskill_by_domain)
    
    if avg_sigma < 1e-6:
        return 1.0
    
    # Score: actual variance / expected variance
    return std_dev / avg_sigma


def get_model_profile(
    model: 'LLMModel'
) -> Dict:
    """
    Get comprehensive profile of model's domain performance.
    
    Args:
        model: Model to analyze
    
    Returns:
        Profile dictionary
    """
    detector = SpecialistDetector()
    results = detector.detect_all_domains(model)
    
    specialists = [r for r in results if r.category == "specialist"]
    weak_spots = [r for r in results if r.category == "weak_spot"]
    
    return {
        "model_id": model.model_id,
        "model_name": model.name,
        "domains_evaluated": list(model.trueskill_by_domain.keys()),
        "specialist_domains": [r.domain for r in specialists],
        "weak_spot_domains": [r.domain for r in weak_spots],
        "specialization_score": compute_specialization_score(model),
        "is_generalist": len(specialists) == 0 and len(weak_spots) == 0,
        "detailed_results": [r.to_dict() for r in results],
    }


