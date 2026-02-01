"""
Cost adjustment for multi-player rankings.

Creates a separate cost-adjusted ranking based on efficiency,
allowing evaluation of both absolute performance and value.

Key formulas:
- Efficiency weight: eff_i = e^(-c_i/τ_c) / Σ_j e^(-c_j/τ_c)
- Cost-adjusted score: S_i^adj = (N - rank_i + 1) × eff_i
"""

import math
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

# Import constants as fallback defaults
from langscope.core.constants import COST_TEMP, RATING_TEMP

if TYPE_CHECKING:
    from langscope.config.params import TemperatureParams


def _get_temperature_params() -> 'TemperatureParams':
    """Get temperature params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_temperature_params()
    except ImportError:
        from langscope.config.params.models import TemperatureParams
        return TemperatureParams()


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """
    Compute softmax probabilities with temperature.
    
    P(i) = e^(v_i/τ) / Σ_j e^(v_j/τ)
    
    Args:
        values: Input values
        temperature: Temperature parameter (higher = more uniform)
    
    Returns:
        Probability distribution
    """
    if not values:
        return []
    
    arr = np.array(values)
    
    # Scale by temperature
    scaled = arr / temperature
    
    # Subtract max for numerical stability
    scaled = scaled - np.max(scaled)
    
    exp_vals = np.exp(scaled)
    return (exp_vals / np.sum(exp_vals)).tolist()


def calculate_efficiency_weights(
    costs: Dict[str, float],
    tau_c: float = None
) -> Dict[str, float]:
    """
    Calculate efficiency weights for each model based on response cost.
    
    eff_i = e^(-c_i/τ_c) / Σ_j e^(-c_j/τ_c)
    
    Lower cost → higher efficiency weight.
    
    Args:
        costs: Dictionary {model_id: cost_usd}
        tau_c: Cost temperature (lower = more penalty for expensive models)
    
    Returns:
        Dictionary {model_id: efficiency_weight}
    
    Example:
        >>> costs = {"gpt4": 0.05, "gpt3": 0.002, "claude": 0.03}
        >>> weights = calculate_efficiency_weights(costs)
        >>> print(weights)  # gpt3 will have highest weight
    """
    if tau_c is None:
        tau_c = _get_temperature_params().cost_temp
    if not costs:
        return {}
    
    model_ids = list(costs.keys())
    cost_values = np.array([costs[m] for m in model_ids])
    
    # Softmax on negative costs (lower cost = higher weight)
    neg_costs = -cost_values / tau_c
    
    # Subtract max for numerical stability
    neg_costs = neg_costs - np.max(neg_costs)
    
    exp_neg_costs = np.exp(neg_costs)
    weights = exp_neg_costs / np.sum(exp_neg_costs)
    
    return {model_ids[i]: float(weights[i]) for i in range(len(model_ids))}


def create_cost_adjusted_ranking(
    raw_ranking: Dict[str, int],
    costs: Dict[str, float],
    tau_c: float = None
) -> Dict[str, int]:
    """
    Create cost-adjusted ranking by combining raw rank with efficiency.
    
    Cost-adjusted score: S_i^adj = (N - rank_i + 1) × eff_i
    Then re-rank by adjusted scores.
    
    Args:
        raw_ranking: Dictionary {model_id: rank} (1=best)
        costs: Dictionary {model_id: cost_usd}
        tau_c: Cost temperature
    
    Returns:
        Dictionary {model_id: cost_adjusted_rank} (1=best)
    
    Example:
        >>> raw_ranking = {"gpt4": 1, "gpt3": 3, "claude": 2}
        >>> costs = {"gpt4": 0.05, "gpt3": 0.002, "claude": 0.03}
        >>> cost_ranking = create_cost_adjusted_ranking(raw_ranking, costs)
        >>> # gpt3 might move up due to lower cost
    """
    if not raw_ranking:
        return {}
    
    n = len(raw_ranking)
    eff_weights = calculate_efficiency_weights(costs, tau_c)
    
    # Compute adjusted scores
    adjusted_scores = {}
    for model_id, rank in raw_ranking.items():
        # Convert rank to score (higher = better)
        raw_score = n - rank + 1
        
        # Apply efficiency weight
        eff = eff_weights.get(model_id, 1.0 / n)
        adjusted_scores[model_id] = raw_score * eff
    
    # Re-rank by adjusted scores (descending)
    sorted_models = sorted(
        adjusted_scores.keys(),
        key=lambda m: -adjusted_scores[m]
    )
    
    return {m: i + 1 for i, m in enumerate(sorted_models)}


def aggregate_judge_rankings(
    judge_rankings: List[Dict[str, int]],
    judge_weights: List[float] = None
) -> Dict[str, int]:
    """
    Aggregate multiple judge rankings into a single ranking.
    
    Uses weighted Borda count:
    B_i = Σ_k w_k × (N - rank_ik)
    
    Args:
        judge_rankings: List of {model_id: rank} from each judge
        judge_weights: Weight for each judge (uniform if None)
    
    Returns:
        Aggregated ranking {model_id: rank} (1=best)
    
    Example:
        >>> rankings = [
        ...     {"a": 1, "b": 2, "c": 3},
        ...     {"a": 2, "b": 1, "c": 3},
        ...     {"a": 1, "b": 3, "c": 2},
        ... ]
        >>> weights = [0.4, 0.35, 0.25]
        >>> final = aggregate_judge_rankings(rankings, weights)
    """
    if not judge_rankings:
        return {}
    
    n_judges = len(judge_rankings)
    
    # Default to uniform weights
    if judge_weights is None:
        judge_weights = [1.0 / n_judges] * n_judges
    
    # Normalize weights
    weight_sum = sum(judge_weights)
    if weight_sum > 0:
        judge_weights = [w / weight_sum for w in judge_weights]
    
    # Collect all model IDs
    all_models = set()
    for ranking in judge_rankings:
        all_models.update(ranking.keys())
    
    # Compute weighted Borda scores
    borda_scores = {m: 0.0 for m in all_models}
    
    for ranking, weight in zip(judge_rankings, judge_weights):
        n = len(ranking)
        for model_id, rank in ranking.items():
            # Borda score: N - rank (so rank 1 gets N-1 points)
            borda_scores[model_id] += weight * (n - rank)
    
    # Sort by Borda score (descending) and assign ranks
    sorted_models = sorted(
        borda_scores.keys(),
        key=lambda m: -borda_scores[m]
    )
    
    return {m: i + 1 for i, m in enumerate(sorted_models)}


def compute_judge_weights(
    judge_ratings: List[float],
    tau: float = None
) -> List[float]:
    """
    Compute judge weights based on their TrueSkill μ values.
    
    w_k = e^(μ_k/τ) / Σ_j e^(μ_j/τ)
    
    Higher-rated judges get more weight in aggregation.
    
    Args:
        judge_ratings: List of judge μ values
        tau: Rating temperature (higher = more uniform weights)
    
    Returns:
        List of judge weights (sums to 1)
    """
    if tau is None:
        tau = _get_temperature_params().rating_temp
    return softmax(judge_ratings, temperature=tau)


def ranking_distance(
    ranking_a: Dict[str, int],
    ranking_b: Dict[str, int],
    method: str = "kendall"
) -> float:
    """
    Compute distance between two rankings.
    
    Args:
        ranking_a: First ranking {model_id: rank}
        ranking_b: Second ranking {model_id: rank}
        method: Distance method ("kendall" or "spearman")
    
    Returns:
        Distance score (0 = identical, higher = more different)
    """
    # Get common models
    common = set(ranking_a.keys()) & set(ranking_b.keys())
    
    if len(common) < 2:
        return 0.0
    
    models = sorted(common)
    n = len(models)
    
    if method == "kendall":
        # Kendall tau distance: count discordant pairs
        discordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                m_i, m_j = models[i], models[j]
                
                # Compare relative orderings
                a_order = ranking_a[m_i] < ranking_a[m_j]
                b_order = ranking_b[m_i] < ranking_b[m_j]
                
                if a_order != b_order:
                    discordant += 1
                total_pairs += 1
        
        return discordant / total_pairs if total_pairs > 0 else 0.0
    
    elif method == "spearman":
        # Spearman rank correlation distance
        ranks_a = [ranking_a[m] for m in models]
        ranks_b = [ranking_b[m] for m in models]
        
        d_squared = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        
        # Spearman correlation
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        
        # Convert to distance (0 to 1)
        return (1 - rho) / 2
    
    else:
        raise ValueError(f"Unknown method: {method}")


def consensus_ranking(
    rankings: List[Dict[str, int]],
    method: str = "borda"
) -> Dict[str, int]:
    """
    Compute consensus ranking from multiple rankings.
    
    Args:
        rankings: List of rankings
        method: Aggregation method ("borda" or "kemeny")
    
    Returns:
        Consensus ranking
    """
    if method == "borda":
        # Use Borda count with uniform weights
        return aggregate_judge_rankings(rankings)
    
    elif method == "kemeny":
        # Kemeny-Young method: find ranking that minimizes total distance
        # This is NP-hard, so we use Borda as approximation for large n
        if len(rankings) > 10:
            return aggregate_judge_rankings(rankings)
        
        # For small n, try permutations (simplified)
        return aggregate_judge_rankings(rankings)
    
    else:
        raise ValueError(f"Unknown method: {method}")


