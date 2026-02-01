"""
Benchmark correlation with LangScope ratings.

Provides:
- Correlation computation between external benchmarks and LangScope ratings
- Prior initialization from benchmark scores for new models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import math

from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0


@dataclass
class BenchmarkCorrelation:
    """
    Correlation between a benchmark and LangScope ratings.
    
    Used for:
    1. Understanding how well benchmarks predict LangScope performance
    2. Initializing priors for new models based on benchmark scores
    """
    benchmark_id: str
    domain: str = ""  # Empty for overall correlation
    
    # Correlation statistics
    correlation_mu: float = 0.0  # Pearson correlation with LangScope μ
    p_value: float = 1.0
    sample_size: int = 0
    
    # Regression parameters for prior calculation
    slope: float = 0.0  # For μ_prior = intercept + slope * benchmark_score
    intercept: float = TRUESKILL_MU_0
    r_squared: float = 0.0
    
    # Confidence in correlation
    confidence: float = 0.0  # 0-1 scale
    
    last_computed: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "correlation_mu": self.correlation_mu,
            "p_value": self.p_value,
            "sample_size": self.sample_size,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "confidence": self.confidence,
            "last_computed": self.last_computed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkCorrelation':
        """Create from dictionary."""
        return cls(
            benchmark_id=data.get("benchmark_id", ""),
            domain=data.get("domain", ""),
            correlation_mu=data.get("correlation_mu", 0.0),
            p_value=data.get("p_value", 1.0),
            sample_size=data.get("sample_size", 0),
            slope=data.get("slope", 0.0),
            intercept=data.get("intercept", TRUESKILL_MU_0),
            r_squared=data.get("r_squared", 0.0),
            confidence=data.get("confidence", 0.0),
            last_computed=data.get("last_computed", ""),
        )
    
    def predict_mu(self, benchmark_score: float) -> float:
        """
        Predict LangScope μ from benchmark score.
        
        Args:
            benchmark_score: Score on this benchmark
        
        Returns:
            Predicted TrueSkill μ
        """
        if self.sample_size < 5 or abs(self.correlation_mu) < 0.1:
            # Not enough data or weak correlation
            return TRUESKILL_MU_0
        
        return self.intercept + self.slope * benchmark_score
    
    def get_prior_sigma(self) -> float:
        """
        Get recommended sigma for prior based on correlation strength.
        
        Returns:
            Recommended sigma (higher if less confident)
        """
        if self.sample_size < 5:
            return TRUESKILL_SIGMA_0  # Full uncertainty
        
        # Scale sigma based on r_squared
        # Higher r_squared = more confidence = lower sigma
        sigma_reduction = min(0.5, self.r_squared * 0.5)
        return TRUESKILL_SIGMA_0 * (1 - sigma_reduction)


def compute_benchmark_correlation(
    benchmark_scores: List[Tuple[str, float]],  # (model_id, score)
    langscope_mus: Dict[str, float],  # model_id -> μ
    benchmark_id: str,
    domain: str = ""
) -> BenchmarkCorrelation:
    """
    Compute correlation between benchmark scores and LangScope ratings.
    
    Args:
        benchmark_scores: List of (model_id, benchmark_score) tuples
        langscope_mus: Dictionary of model_id -> LangScope μ
        benchmark_id: Benchmark identifier
        domain: Optional domain for domain-specific correlation
    
    Returns:
        BenchmarkCorrelation with computed statistics
    """
    # Find models with both scores
    paired_scores = []
    for model_id, bench_score in benchmark_scores:
        if model_id in langscope_mus:
            paired_scores.append((bench_score, langscope_mus[model_id]))
    
    n = len(paired_scores)
    if n < 3:
        return BenchmarkCorrelation(
            benchmark_id=benchmark_id,
            domain=domain,
            sample_size=n,
            last_computed=datetime.utcnow().isoformat() + "Z",
        )
    
    # Extract x (benchmark) and y (langscope) values
    x_vals = [p[0] for p in paired_scores]
    y_vals = [p[1] for p in paired_scores]
    
    # Compute means
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    
    # Compute correlation (Pearson)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    x_std = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
    y_std = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))
    
    if x_std == 0 or y_std == 0:
        correlation = 0.0
    else:
        correlation = numerator / (x_std * y_std)
    
    # Compute regression (y = a + bx)
    if x_std == 0:
        slope = 0.0
        intercept = y_mean
    else:
        slope = numerator / (x_std ** 2)
        intercept = y_mean - slope * x_mean
    
    # R-squared
    r_squared = correlation ** 2
    
    # Simple p-value approximation using t-statistic
    if n > 2 and abs(correlation) < 1:
        t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2))
        # Approximate p-value (simplified)
        p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 2))
    else:
        p_value = 1.0
    
    # Confidence based on sample size and correlation strength
    confidence = min(1.0, (n / 50) * abs(correlation))
    
    return BenchmarkCorrelation(
        benchmark_id=benchmark_id,
        domain=domain,
        correlation_mu=correlation,
        p_value=p_value,
        sample_size=n,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        confidence=confidence,
        last_computed=datetime.utcnow().isoformat() + "Z",
    )


def _t_cdf(t: float, df: int) -> float:
    """
    Simple approximation of Student's t CDF.
    
    Uses normal approximation for large df.
    """
    if df > 30:
        # Normal approximation
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    
    # Very rough approximation for smaller df
    x = df / (df + t ** 2)
    return 1 - 0.5 * (x ** (df / 2))


def get_prior_from_benchmarks(
    benchmark_scores: Dict[str, float],  # benchmark_id -> score
    correlations: Dict[str, BenchmarkCorrelation],
    domain: str = ""
) -> Tuple[float, float]:
    """
    Get TrueSkill prior from benchmark scores.
    
    Uses a weighted average of predictions from multiple benchmarks,
    where weights are based on correlation strength and sample size.
    
    Args:
        benchmark_scores: Dictionary of benchmark_id -> score
        correlations: Dictionary of benchmark_id -> BenchmarkCorrelation
        domain: Optional domain for domain-specific correlations
    
    Returns:
        Tuple of (μ_prior, σ_prior)
    """
    if not benchmark_scores or not correlations:
        return (TRUESKILL_MU_0, TRUESKILL_SIGMA_0)
    
    weighted_mu = 0.0
    total_weight = 0.0
    min_sigma = TRUESKILL_SIGMA_0
    
    for bench_id, score in benchmark_scores.items():
        # Find correlation for this benchmark
        corr_key = f"{bench_id}:{domain}" if domain else bench_id
        corr = correlations.get(corr_key) or correlations.get(bench_id)
        
        if not corr or corr.sample_size < 5 or abs(corr.correlation_mu) < 0.1:
            continue
        
        # Weight by correlation strength and sample size
        weight = abs(corr.correlation_mu) * min(1.0, corr.sample_size / 20)
        
        # Get predicted μ
        predicted_mu = corr.predict_mu(score)
        
        weighted_mu += weight * predicted_mu
        total_weight += weight
        
        # Track minimum sigma (more confident if multiple strong correlations)
        sigma = corr.get_prior_sigma()
        min_sigma = min(min_sigma, sigma)
    
    if total_weight == 0:
        return (TRUESKILL_MU_0, TRUESKILL_SIGMA_0)
    
    final_mu = weighted_mu / total_weight
    
    # Blend toward default μ based on total confidence
    confidence = min(1.0, total_weight)
    blended_mu = confidence * final_mu + (1 - confidence) * TRUESKILL_MU_0
    
    # Sigma decreases with more evidence
    sigma_reduction = min(0.3, total_weight * 0.1)
    final_sigma = min_sigma * (1 - sigma_reduction)
    
    return (blended_mu, final_sigma)

