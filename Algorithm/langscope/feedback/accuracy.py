"""
Prediction accuracy metrics for evaluating system predictions.

This module provides metrics to measure how well the system's
predictions aligned with actual user preferences:

1. Pairwise Prediction Accuracy:
   What fraction of pairwise orderings did we predict correctly?
   
2. Kendall's Tau (τ):
   Rank correlation coefficient measuring overall ranking agreement.

These metrics help answer: "Did the system correctly predict which
models the user would prefer?"
"""

from typing import Dict, List, Tuple
from itertools import combinations


def compute_prediction_accuracy(
    predicted_ranking: Dict[str, int],
    user_ranking: Dict[str, int]
) -> float:
    """
    Compute pairwise prediction accuracy.
    
    For each pair of models that appear in both rankings, check if
    the system correctly predicted their relative ordering.
    
    Formula:
        Accuracy = (# correct pairwise orderings) / (total pairwise comparisons)
    
    A pairwise ordering is correct if:
        rank_pred(i) < rank_pred(j) AND rank_user(i) < rank_user(j)
        OR
        rank_pred(i) > rank_pred(j) AND rank_user(i) > rank_user(j)
    
    Args:
        predicted_ranking: {model_id: rank} where 1=best, predicted by system
        user_ranking: {model_id: rank} where 1=best, from user feedback
    
    Returns:
        Accuracy ∈ [0, 1] (1.0 = perfect prediction)
    
    Example:
        >>> predicted = {"gpt-4": 1, "claude": 2, "llama": 3}
        >>> actual = {"gpt-4": 1, "claude": 3, "llama": 2}  # Claude/Llama swapped
        >>> compute_prediction_accuracy(predicted, actual)
        0.667  # 2 out of 3 pairs correct
    """
    # Find models that appear in both rankings
    common_models = set(predicted_ranking.keys()) & set(user_ranking.keys())
    models = list(common_models)
    
    if len(models) < 2:
        return 1.0  # No pairs to compare
    
    correct = 0
    total = 0
    
    for i, j in combinations(models, 2):
        # Check if orderings agree
        pred_order = predicted_ranking[i] < predicted_ranking[j]  # i ranked higher?
        user_order = user_ranking[i] < user_ranking[j]  # i ranked higher by user?
        
        if pred_order == user_order:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 1.0


def compute_kendall_tau(
    predicted_ranking: Dict[str, int],
    user_ranking: Dict[str, int]
) -> float:
    """
    Compute Kendall's tau rank correlation coefficient.
    
    Kendall's tau measures the similarity between two rankings:
        τ = (concordant - discordant) / total_pairs
    
    Where:
        - Concordant: pair where both rankings agree on order
        - Discordant: pair where rankings disagree on order
    
    Interpretation:
        τ = +1: Perfect agreement
        τ = 0:  No correlation
        τ = -1: Perfect disagreement (reversed order)
    
    Args:
        predicted_ranking: {model_id: rank} where 1=best
        user_ranking: {model_id: rank} where 1=best
    
    Returns:
        Kendall's tau ∈ [-1, 1]
    
    Example:
        >>> predicted = {"a": 1, "b": 2, "c": 3}
        >>> actual = {"a": 1, "b": 2, "c": 3}  # Perfect match
        >>> compute_kendall_tau(predicted, actual)
        1.0
        
        >>> actual = {"a": 3, "b": 2, "c": 1}  # Reversed
        >>> compute_kendall_tau(predicted, actual)
        -1.0
    """
    # Find common models
    common_models = [m for m in predicted_ranking if m in user_ranking]
    
    if len(common_models) < 2:
        return 1.0  # No pairs to compare
    
    concordant = 0
    discordant = 0
    
    for i, j in combinations(common_models, 2):
        pred_diff = predicted_ranking[i] - predicted_ranking[j]
        user_diff = user_ranking[i] - user_ranking[j]
        
        product = pred_diff * user_diff
        
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
        # If product == 0, it's a tie (neither concordant nor discordant)
    
    total = concordant + discordant
    
    if total == 0:
        return 0.0  # All ties
    
    return (concordant - discordant) / total


def compute_spearman_rho(
    predicted_ranking: Dict[str, int],
    user_ranking: Dict[str, int]
) -> float:
    """
    Compute Spearman's rank correlation coefficient.
    
    Spearman's ρ is the Pearson correlation of the rank values:
        ρ = 1 - (6 × Σ d²) / (n × (n² - 1))
    
    Where d is the difference in ranks for each model.
    
    Args:
        predicted_ranking: {model_id: rank} where 1=best
        user_ranking: {model_id: rank} where 1=best
    
    Returns:
        Spearman's rho ∈ [-1, 1]
    """
    common_models = [m for m in predicted_ranking if m in user_ranking]
    n = len(common_models)
    
    if n < 2:
        return 1.0
    
    # Compute sum of squared differences
    sum_d_sq = 0.0
    for model in common_models:
        d = predicted_ranking[model] - user_ranking[model]
        sum_d_sq += d ** 2
    
    # Spearman's formula
    rho = 1 - (6 * sum_d_sq) / (n * (n ** 2 - 1))
    
    return rho


def compute_top_k_accuracy(
    predicted_ranking: Dict[str, int],
    user_ranking: Dict[str, int],
    k: int = 3
) -> float:
    """
    Compute overlap between top-k models in predicted and user rankings.
    
    Measures whether the system correctly identified the user's top models.
    
    Args:
        predicted_ranking: {model_id: rank} where 1=best
        user_ranking: {model_id: rank} where 1=best
        k: Number of top models to consider
    
    Returns:
        Fraction of top-k overlap ∈ [0, 1]
    
    Example:
        >>> predicted = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        >>> actual = {"a": 2, "b": 1, "c": 3, "d": 5, "e": 4}
        >>> compute_top_k_accuracy(predicted, actual, k=3)
        1.0  # Both have {a, b, c} in top 3
    """
    # Get top-k from each ranking
    pred_top_k = {
        m for m, r in predicted_ranking.items() if r <= k
    }
    user_top_k = {
        m for m, r in user_ranking.items() if r <= k
    }
    
    if not pred_top_k or not user_top_k:
        return 0.0
    
    overlap = len(pred_top_k & user_top_k)
    max_possible = min(len(pred_top_k), len(user_top_k))
    
    return overlap / max_possible if max_possible > 0 else 0.0


def compute_ndcg(
    predicted_ranking: Dict[str, int],
    user_ranking: Dict[str, int]
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG measures ranking quality with position-weighted relevance.
    Higher-ranked positions have more impact on the score.
    
    Args:
        predicted_ranking: {model_id: rank} where 1=best
        user_ranking: {model_id: rank} where 1=best
    
    Returns:
        NDCG score ∈ [0, 1]
    """
    import math
    
    common_models = list(set(predicted_ranking.keys()) & set(user_ranking.keys()))
    
    if not common_models:
        return 1.0
    
    n = len(common_models)
    
    # Get models sorted by predicted rank
    sorted_by_pred = sorted(common_models, key=lambda m: predicted_ranking[m])
    
    # Relevance score: inverse of user rank (higher rank = higher relevance)
    max_user_rank = max(user_ranking[m] for m in common_models)
    
    def relevance(model: str) -> float:
        return max_user_rank - user_ranking[model] + 1
    
    # DCG: Σ (rel_i / log2(i + 1))
    dcg = 0.0
    for i, model in enumerate(sorted_by_pred):
        dcg += relevance(model) / math.log2(i + 2)  # i+2 because log2(1)=0
    
    # Ideal DCG: sort by actual user ranking
    sorted_by_user = sorted(common_models, key=lambda m: user_ranking[m])
    idcg = 0.0
    for i, model in enumerate(sorted_by_user):
        idcg += relevance(model) / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_accuracy_metrics(
    sessions: List[Tuple[Dict[str, int], Dict[str, int]]]
) -> Dict[str, float]:
    """
    Compute aggregate accuracy metrics across multiple sessions.
    
    Args:
        sessions: List of (predicted_ranking, user_ranking) tuples
    
    Returns:
        Dictionary with mean metrics
    """
    if not sessions:
        return {
            "mean_accuracy": 0.0,
            "mean_kendall_tau": 0.0,
            "mean_spearman_rho": 0.0,
            "n_sessions": 0
        }
    
    accuracies = []
    taus = []
    rhos = []
    
    for pred, user in sessions:
        accuracies.append(compute_prediction_accuracy(pred, user))
        taus.append(compute_kendall_tau(pred, user))
        rhos.append(compute_spearman_rho(pred, user))
    
    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "mean_kendall_tau": sum(taus) / len(taus),
        "mean_spearman_rho": sum(rhos) / len(rhos),
        "n_sessions": len(sessions)
    }
