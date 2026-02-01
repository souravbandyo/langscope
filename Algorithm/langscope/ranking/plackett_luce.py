"""
Plackett-Luce model for multi-way ranking with TrueSkill integration.

The Plackett-Luce model assigns strength parameters λ_i to each item,
where the probability of a ranking is:

P(r₁ > r₂ > ... > rₙ) = ∏ᵢ λᵣᵢ / Σⱼ≥ᵢ λᵣⱼ

This is a "top-down" model: at each position, the item is chosen
proportionally to its strength among remaining items.

Information content: log₂(n!) bits per full ranking
- 6-way ranking: ~9.5 bits (vs 1 bit for pairwise)
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from langscope.core.constants import (
    PLACKETT_LUCE_MAX_ITER,
    PLACKETT_LUCE_TOL,
    INFO_BITS,
)


@dataclass
class PlackettLuceResult:
    """Result of Plackett-Luce strength estimation."""
    strengths: Dict[str, float]      # λ values for each model
    log_likelihood: float            # Log-likelihood of observed rankings
    iterations: int                  # Number of MM iterations
    converged: bool                  # Whether algorithm converged
    info_bits: float                 # Information content log₂(n!)
    
    def get_ranking(self) -> List[str]:
        """
        Get ranking from strongest to weakest.
        
        Returns:
            List of model IDs sorted by strength (descending)
        """
        return sorted(self.strengths.keys(), key=lambda m: -self.strengths[m])
    
    def win_probability(self, model_a: str, model_b: str) -> float:
        """
        Compute probability that model_a beats model_b.
        
        P(A > B) = λ_A / (λ_A + λ_B)
        
        Args:
            model_a: First model ID
            model_b: Second model ID
        
        Returns:
            Win probability for model_a
        """
        lambda_a = self.strengths.get(model_a, 1.0)
        lambda_b = self.strengths.get(model_b, 1.0)
        return lambda_a / (lambda_a + lambda_b)


class PlackettLuceModel:
    """
    Plackett-Luce model for multi-way ranking estimation.
    
    Uses the Minorization-Maximization (MM) algorithm to estimate
    strength parameters from observed rankings.
    
    The MM algorithm iterates:
    λᵢ^(t+1) = wᵢ / Σⱼ (1 / Σₖ∈Rⱼ λₖ^(t))
    
    where wᵢ is the number of times item i appears in rankings,
    and Rⱼ is the set of items at or after position j.
    """
    
    def __init__(
        self,
        max_iter: int = PLACKETT_LUCE_MAX_ITER,
        tolerance: float = PLACKETT_LUCE_TOL
    ):
        """
        Initialize Plackett-Luce model.
        
        Args:
            max_iter: Maximum iterations for MM algorithm
            tolerance: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def estimate_strengths(
        self,
        rankings: List[List[str]],
        model_ids: List[str] = None,
        initial_strengths: Dict[str, float] = None
    ) -> PlackettLuceResult:
        """
        Estimate strength parameters using MM algorithm.
        
        Args:
            rankings: List of rankings. Each ranking is [1st_place, 2nd_place, ..., last_place]
            model_ids: List of all model IDs (inferred if None)
            initial_strengths: Initial strength values (uniform if None)
        
        Returns:
            PlackettLuceResult with estimated strengths
        
        Example:
            >>> pl = PlackettLuceModel()
            >>> rankings = [
            ...     ["model_a", "model_b", "model_c"],  # a beat b beat c
            ...     ["model_b", "model_a", "model_c"],  # b beat a beat c
            ... ]
            >>> result = pl.estimate_strengths(rankings)
            >>> print(result.strengths)
        """
        if not rankings:
            return PlackettLuceResult(
                strengths={},
                log_likelihood=0.0,
                iterations=0,
                converged=True,
                info_bits=0.0
            )
        
        # Infer model IDs if not provided
        if model_ids is None:
            model_ids = list(set(m for r in rankings for m in r))
        
        n_models = len(model_ids)
        model_to_idx = {m: i for i, m in enumerate(model_ids)}
        
        # Initialize strengths
        if initial_strengths:
            lambdas = np.array([initial_strengths.get(m, 1.0) for m in model_ids])
        else:
            lambdas = np.ones(n_models)
        
        # Count appearances (wins) for each model
        wins = np.zeros(n_models)
        for ranking in rankings:
            for model in ranking:
                if model in model_to_idx:
                    wins[model_to_idx[model]] += 1
        
        # MM algorithm iterations
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iter):
            old_lambdas = lambdas.copy()
            
            # Compute denominators for MM update
            denom_sum = np.zeros(n_models)
            
            for ranking in rankings:
                # Compute cumulative strength from end to start
                remaining_strength = 0.0
                
                for pos in range(len(ranking) - 1, -1, -1):
                    model = ranking[pos]
                    if model in model_to_idx:
                        idx = model_to_idx[model]
                        remaining_strength += lambdas[idx]
                        
                        if remaining_strength > 0:
                            denom_sum[idx] += 1.0 / remaining_strength
            
            # Update lambdas
            for i in range(n_models):
                if denom_sum[i] > 0:
                    lambdas[i] = wins[i] / denom_sum[i]
                else:
                    lambdas[i] = 1.0  # Default for items never ranked
            
            # Normalize (mean = 1)
            mean_lambda = np.mean(lambdas)
            if mean_lambda > 0:
                lambdas = lambdas / mean_lambda
            
            # Check convergence
            max_diff = np.max(np.abs(lambdas - old_lambdas))
            if max_diff < self.tolerance:
                converged = True
                break
        
        # Compute log-likelihood
        log_likelihood = self._compute_log_likelihood(
            rankings, lambdas, model_to_idx
        )
        
        # Compute information bits
        max_ranking_size = max(len(r) for r in rankings) if rankings else 0
        if max_ranking_size in INFO_BITS:
            info_bits = INFO_BITS[max_ranking_size]
        else:
            info_bits = math.log2(math.factorial(max_ranking_size)) if max_ranking_size > 0 else 0.0
        
        return PlackettLuceResult(
            strengths={model_ids[i]: float(lambdas[i]) for i in range(n_models)},
            log_likelihood=log_likelihood,
            iterations=iteration + 1,
            converged=converged,
            info_bits=info_bits
        )
    
    def ranking_probability(
        self,
        ranking: List[str],
        strengths: Dict[str, float]
    ) -> float:
        """
        Compute probability of a specific ranking.
        
        P(r₁ > r₂ > ... > rₙ) = ∏ᵢ λᵣᵢ / Σⱼ≥ᵢ λᵣⱼ
        
        Args:
            ranking: Ordered list [1st_place, 2nd_place, ..., last_place]
            strengths: Strength parameters {model_id: λ}
        
        Returns:
            Probability of the ranking
        """
        if not ranking:
            return 1.0
        
        prob = 1.0
        remaining = sum(strengths.get(m, 1.0) for m in ranking)
        
        for model in ranking[:-1]:  # Last position has probability 1
            lambda_m = strengths.get(model, 1.0)
            if remaining > 0:
                prob *= lambda_m / remaining
                remaining -= lambda_m
            else:
                return 0.0
        
        return prob
    
    def _compute_log_likelihood(
        self,
        rankings: List[List[str]],
        lambdas: np.ndarray,
        model_to_idx: Dict[str, int]
    ) -> float:
        """Compute log-likelihood of rankings given parameters."""
        ll = 0.0
        
        for ranking in rankings:
            remaining = sum(
                lambdas[model_to_idx[m]]
                for m in ranking if m in model_to_idx
            )
            
            for model in ranking[:-1]:
                if model in model_to_idx:
                    idx = model_to_idx[model]
                    if remaining > 0 and lambdas[idx] > 0:
                        ll += math.log(lambdas[idx] / remaining)
                        remaining -= lambdas[idx]
        
        return ll
    
    def top_k_probability(
        self,
        model: str,
        k: int,
        strengths: Dict[str, float]
    ) -> float:
        """
        Compute probability of a model finishing in top k.
        
        Uses Monte Carlo approximation.
        
        Args:
            model: Model ID
            k: Top k positions
            strengths: Strength parameters
        
        Returns:
            Probability of finishing in top k
        """
        if model not in strengths or k <= 0:
            return 0.0
        
        n = len(strengths)
        if k >= n:
            return 1.0
        
        # Approximate using expected rank
        model_lambda = strengths[model]
        total_lambda = sum(strengths.values())
        
        # Expected number of models that beat this one
        expected_rank = 1 + sum(
            strengths[m] / (strengths[m] + model_lambda)
            for m in strengths if m != model
        )
        
        # Rough approximation of P(rank <= k)
        # Using normal approximation
        variance = sum(
            (strengths[m] * model_lambda) / ((strengths[m] + model_lambda) ** 2)
            for m in strengths if m != model
        )
        
        if variance > 0:
            std = math.sqrt(variance)
            # P(rank <= k) ≈ Φ((k - expected_rank) / std)
            z = (k + 0.5 - expected_rank) / std
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            return 1.0 if expected_rank <= k else 0.0


def convert_strengths_to_trueskill_updates(
    pl_strengths: Dict[str, float],
    scale_factor: float = 100.0
) -> Dict[str, float]:
    """
    Convert Plackett-Luce strengths to TrueSkill μ update suggestions.
    
    Higher λ → positive μ update
    Lower λ → negative μ update
    
    Formula: Δμ_i = scale × (log(λ_i) - mean(log(λ)))
    
    This can be used to initialize TrueSkill ratings from Plackett-Luce
    estimates, or to suggest update magnitudes.
    
    Args:
        pl_strengths: Strength parameters {model_id: λ}
        scale_factor: Scaling factor for updates
    
    Returns:
        Suggested μ updates {model_id: Δμ}
    """
    if not pl_strengths:
        return {}
    
    # Convert to log scale
    log_strengths = {
        m: math.log(max(s, 1e-10))
        for m, s in pl_strengths.items()
    }
    
    mean_log = np.mean(list(log_strengths.values()))
    
    updates = {}
    for model_id, log_s in log_strengths.items():
        updates[model_id] = scale_factor * (log_s - mean_log)
    
    return updates


def estimate_from_rankings(
    rankings: List[List[str]],
    model_ids: List[str] = None
) -> PlackettLuceResult:
    """
    Convenience function to estimate strengths from rankings.
    
    Args:
        rankings: List of rankings
        model_ids: Optional list of model IDs
    
    Returns:
        PlackettLuceResult
    """
    model = PlackettLuceModel()
    return model.estimate_strengths(rankings, model_ids)


def compute_info_bits(n_players: int) -> float:
    """
    Compute information content of a full n-way ranking.
    
    Information content: log₂(n!) bits
    
    This represents the maximum information that can be extracted
    from a complete ranking of n items.
    
    Args:
        n_players: Number of players
    
    Returns:
        Information content in bits
    """
    if n_players <= 1:
        return 0.0
    
    if n_players in INFO_BITS:
        return INFO_BITS[n_players]
    
    return math.log2(math.factorial(n_players))


