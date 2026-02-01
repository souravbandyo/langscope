"""
Mathematical utility functions for LangScope.

Provides common mathematical operations used throughout the system.
"""

import math
from typing import List, Dict
import numpy as np


def factorial(n: int) -> int:
    """
    Compute factorial n!
    
    Args:
        n: Non-negative integer
    
    Returns:
        n!
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return math.factorial(n)


def log2_factorial(n: int) -> float:
    """
    Compute log₂(n!)
    
    This is the information content of a full n-way ranking in bits.
    
    Args:
        n: Non-negative integer
    
    Returns:
        log₂(n!)
    """
    if n <= 1:
        return 0.0
    return math.log2(factorial(n))


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """
    Compute softmax probabilities.
    
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


def normalize(values: List[float]) -> List[float]:
    """
    Normalize values to sum to 1.
    
    Args:
        values: Input values
    
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    
    return [v / total for v in values]


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    if var_x == 0 or var_y == 0:
        return 0.0
    
    return cov / math.sqrt(var_x * var_y)


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman rank correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    
    # Convert to ranks
    rank_x = _to_ranks(x)
    rank_y = _to_ranks(y)
    
    return pearson_correlation(rank_x, rank_y)


def _to_ranks(values: List[float]) -> List[float]:
    """Convert values to ranks (1-indexed)."""
    n = len(values)
    sorted_indices = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1.0
    return ranks


def kendall_tau_distance(
    ranking_a: Dict[str, int],
    ranking_b: Dict[str, int]
) -> float:
    """
    Compute normalized Kendall tau distance between rankings.
    
    Args:
        ranking_a: First ranking {item: rank}
        ranking_b: Second ranking {item: rank}
    
    Returns:
        Distance (0 = identical, 1 = reversed)
    """
    common = set(ranking_a.keys()) & set(ranking_b.keys())
    
    if len(common) < 2:
        return 0.0
    
    items = list(common)
    n = len(items)
    
    discordant = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            item_i, item_j = items[i], items[j]
            
            # Compare relative orderings
            a_order = ranking_a[item_i] < ranking_a[item_j]
            b_order = ranking_b[item_i] < ranking_b[item_j]
            
            if a_order != b_order:
                discordant += 1
            total_pairs += 1
    
    return discordant / total_pairs if total_pairs > 0 else 0.0


def weighted_average(
    values: List[float],
    weights: List[float] = None
) -> float:
    """
    Compute weighted average.
    
    Args:
        values: Values to average
        weights: Optional weights
    
    Returns:
        Weighted average
    """
    if not values:
        return 0.0
    
    if weights is None:
        return sum(values) / len(values)
    
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def clip(value: float, min_val: float, max_val: float) -> float:
    """
    Clip value to range.
    
    Args:
        value: Value to clip
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation parameter (0 to 1)
    
    Returns:
        Interpolated value
    """
    return a + t * (b - a)


