"""
Domain correlation learning with Bayesian smoothing.

Estimates correlations between domains to enable knowledge transfer.
Uses a combination of expert priors and observed data.

Key formula:
ρ = α·ρ_prior + (1-α)·ρ_data
where α = 1/(1 + n/τ), τ = 20
"""

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

# Import constant as fallback default
from langscope.core.constants import CORRELATION_TAU

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB
    from langscope.config.params import TransferParams


def _get_transfer_params() -> 'TransferParams':
    """Get transfer params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_transfer_params()
    except ImportError:
        from langscope.config.params.models import TransferParams
        return TransferParams()


@dataclass
class CorrelationData:
    """Correlation data between two domains."""
    domain_a: str
    domain_b: str
    prior_correlation: float = 0.5
    data_correlation: float = 0.0
    blended_correlation: float = 0.5
    sample_count: int = 0
    alpha: float = 1.0
    confidence: float = 0.0
    updated_at: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "_id": f"{self.domain_a}|{self.domain_b}",
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "prior_correlation": self.prior_correlation,
            "data_correlation": self.data_correlation,
            "blended_correlation": self.blended_correlation,
            "sample_count": self.sample_count,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "updated_at": self.updated_at or datetime.utcnow().isoformat() + "Z",
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CorrelationData':
        """Create from dictionary."""
        return cls(
            domain_a=data.get("domain_a", ""),
            domain_b=data.get("domain_b", ""),
            prior_correlation=data.get("prior_correlation", 0.5),
            data_correlation=data.get("data_correlation", 0.0),
            blended_correlation=data.get("blended_correlation", 0.5),
            sample_count=data.get("sample_count", 0),
            alpha=data.get("alpha", 1.0),
            confidence=data.get("confidence", 0.0),
            updated_at=data.get("updated_at", ""),
        )


class CorrelationLearner:
    """
    Learns correlations between domains using Bayesian smoothing.
    
    The correlation is estimated as:
    ρ = α·ρ_prior + (1-α)·ρ_data
    
    where:
    - α = 1/(1 + n/τ) is the blending factor
    - n is the number of shared observations
    - τ is the smoothing parameter (default: 20)
    
    As more data is observed, α decreases and the estimate relies
    more on observed data than the prior.
    """
    
    def __init__(
        self,
        tau: float = None,
        db: 'MongoDB' = None
    ):
        """
        Initialize correlation learner.
        
        Args:
            tau: Bayesian smoothing parameter (from ParameterManager if None)
            db: Database instance for persistence
        """
        if tau is None:
            tau = _get_transfer_params().correlation_tau
        self.tau = tau
        self.db = db
        self._correlations: Dict[str, CorrelationData] = {}
    
    def _get_key(self, domain_a: str, domain_b: str) -> str:
        """Get canonical key for domain pair."""
        # Always use sorted order for consistency
        if domain_a < domain_b:
            return f"{domain_a}|{domain_b}"
        else:
            return f"{domain_b}|{domain_a}"
    
    def set_prior(
        self,
        domain_a: str,
        domain_b: str,
        prior_correlation: float
    ):
        """
        Set expert prior for domain correlation.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
            prior_correlation: Expert estimate of correlation (-1 to 1)
        """
        if prior_correlation < -1 or prior_correlation > 1:
            raise ValueError("Correlation must be between -1 and 1")
        
        key = self._get_key(domain_a, domain_b)
        
        if key in self._correlations:
            self._correlations[key].prior_correlation = prior_correlation
            # Recompute blended correlation
            self._update_blend(key)
        else:
            self._correlations[key] = CorrelationData(
                domain_a=min(domain_a, domain_b),
                domain_b=max(domain_a, domain_b),
                prior_correlation=prior_correlation,
                blended_correlation=prior_correlation,
                alpha=1.0,
            )
        
        # Persist if database available
        if self.db:
            self.db.save_correlation(self._correlations[key].to_dict())
    
    def update_correlation(
        self,
        domain_a: str,
        domain_b: str,
        observed_correlation: float,
        sample_size: int = 1
    ):
        """
        Update correlation estimate with new observation.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
            observed_correlation: Observed correlation from data
            sample_size: Number of observations this represents
        """
        key = self._get_key(domain_a, domain_b)
        
        if key not in self._correlations:
            self._correlations[key] = CorrelationData(
                domain_a=min(domain_a, domain_b),
                domain_b=max(domain_a, domain_b),
            )
        
        corr = self._correlations[key]
        
        # Update sample count
        old_n = corr.sample_count
        new_n = old_n + sample_size
        
        # Update data correlation (weighted average)
        if old_n == 0:
            corr.data_correlation = observed_correlation
        else:
            corr.data_correlation = (
                (old_n * corr.data_correlation + sample_size * observed_correlation)
                / new_n
            )
        
        corr.sample_count = new_n
        
        # Update blend
        self._update_blend(key)
        
        # Persist
        if self.db:
            self.db.save_correlation(corr.to_dict())
    
    def _update_blend(self, key: str):
        """Update blended correlation for a key."""
        corr = self._correlations[key]
        
        # Compute alpha: α = 1/(1 + n/τ)
        corr.alpha = 1.0 / (1.0 + corr.sample_count / self.tau)
        
        # Compute blended correlation
        corr.blended_correlation = (
            corr.alpha * corr.prior_correlation +
            (1 - corr.alpha) * corr.data_correlation
        )
        
        # Compute confidence (higher with more samples)
        corr.confidence = 1.0 - corr.alpha
        
        corr.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def get_correlation(
        self,
        domain_a: str,
        domain_b: str
    ) -> float:
        """
        Get correlation between two domains.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
        
        Returns:
            Blended correlation estimate
        """
        if domain_a == domain_b:
            return 1.0
        
        key = self._get_key(domain_a, domain_b)
        
        # Check cache
        if key in self._correlations:
            return self._correlations[key].blended_correlation
        
        # Check database
        if self.db:
            data = self.db.get_correlation(domain_a, domain_b)
            if data:
                self._correlations[key] = CorrelationData.from_dict(data)
                return self._correlations[key].blended_correlation
        
        # Default: moderate positive correlation
        return 0.5
    
    def get_correlation_data(
        self,
        domain_a: str,
        domain_b: str
    ) -> Optional[CorrelationData]:
        """
        Get full correlation data.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
        
        Returns:
            CorrelationData or None
        """
        key = self._get_key(domain_a, domain_b)
        return self._correlations.get(key)
    
    def get_observation_count(
        self,
        domain_a: str,
        domain_b: str
    ) -> int:
        """
        Get the number of observations for a domain pair.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
        
        Returns:
            Number of observations
        """
        if domain_a == domain_b:
            return 0
        
        key = self._get_key(domain_a, domain_b)
        
        # Check cache
        if key in self._correlations:
            return self._correlations[key].sample_count
        
        # Check database
        if self.db:
            data = self.db.get_correlation(domain_a, domain_b)
            if data:
                self._correlations[key] = CorrelationData.from_dict(data)
                return self._correlations[key].sample_count
        
        return 0
    
    def get_alpha(
        self,
        domain_a: str,
        domain_b: str
    ) -> float:
        """
        Get the blending alpha for a domain pair.
        
        Alpha determines how much weight is given to prior vs observed data.
        α = 1/(1 + n/τ)
        
        Args:
            domain_a: First domain
            domain_b: Second domain
        
        Returns:
            Alpha value (1.0 = all prior, 0.0 = all data)
        """
        if domain_a == domain_b:
            return 0.0  # Perfect correlation from data
        
        key = self._get_key(domain_a, domain_b)
        
        # Check cache
        if key in self._correlations:
            return self._correlations[key].alpha
        
        # Check database
        if self.db:
            data = self.db.get_correlation(domain_a, domain_b)
            if data:
                self._correlations[key] = CorrelationData.from_dict(data)
                return self._correlations[key].alpha
        
        return 1.0  # Default to all prior
    
    def estimate_from_performance(
        self,
        ratings_a: List[Tuple[str, float]],  # [(model_id, mu_a), ...]
        ratings_b: List[Tuple[str, float]]   # [(model_id, mu_b), ...]
    ) -> float:
        """
        Estimate correlation from paired performance data.
        
        Args:
            ratings_a: Ratings in domain A
            ratings_b: Ratings in domain B
        
        Returns:
            Estimated Pearson correlation
        """
        # Build lookup
        ratings_a_dict = dict(ratings_a)
        ratings_b_dict = dict(ratings_b)
        
        # Find common models
        common = set(ratings_a_dict.keys()) & set(ratings_b_dict.keys())
        
        if len(common) < 2:
            return 0.0
        
        # Extract paired values
        vals_a = [ratings_a_dict[m] for m in common]
        vals_b = [ratings_b_dict[m] for m in common]
        
        # Compute Pearson correlation
        return self._pearson_correlation(vals_a, vals_b)
    
    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if var_x == 0 or var_y == 0:
            return 0.0
        
        return cov / math.sqrt(var_x * var_y)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_learner: Optional[CorrelationLearner] = None


def get_default_learner() -> CorrelationLearner:
    """Get or create default correlation learner."""
    global _default_learner
    if _default_learner is None:
        _default_learner = CorrelationLearner()
    return _default_learner


def get_correlation(domain_a: str, domain_b: str) -> float:
    """Get correlation between domains using default learner."""
    return get_default_learner().get_correlation(domain_a, domain_b)


def set_prior_correlation(
    domain_a: str,
    domain_b: str,
    correlation: float
):
    """Set prior correlation using default learner."""
    get_default_learner().set_prior(domain_a, domain_b, correlation)


# =============================================================================
# Default Domain Correlations
# =============================================================================

DEFAULT_CORRELATIONS = {
    # Medical domain correlations
    ("general_medical", "clinical_reasoning"): 0.85,
    ("general_medical", "medical_diagnosis"): 0.80,
    ("clinical_reasoning", "medical_diagnosis"): 0.90,
    
    # Language correlations
    ("english", "hindi"): 0.60,
    ("hindi", "hindi_medical"): 0.70,
    ("english", "hindi_medical"): 0.50,
    
    # Coding correlations
    ("python", "javascript"): 0.75,
    ("python", "algorithms"): 0.70,
    ("javascript", "algorithms"): 0.65,
    
    # General vs specialized
    ("general", "medical"): 0.55,
    ("general", "coding"): 0.50,
    ("general", "math"): 0.60,
}


def initialize_default_correlations():
    """Initialize default domain correlations."""
    learner = get_default_learner()
    for (domain_a, domain_b), corr in DEFAULT_CORRELATIONS.items():
        learner.set_prior(domain_a, domain_b, corr)


