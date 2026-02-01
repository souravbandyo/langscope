"""
Ranking aggregation methods for LangScope.

Implements various ranking aggregation algorithms for combining
multiple judge rankings into a consensus ranking.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class RankingAggregator:
    """
    Aggregates multiple rankings into a consensus.
    
    Supports multiple aggregation methods:
    - Borda count (weighted)
    - Kemeny-Young approximation
    - Copeland
    """
    
    def __init__(self, method: str = "borda"):
        """
        Initialize aggregator.
        
        Args:
            method: Aggregation method (borda, kemeny, copeland)
        """
        self.method = method
    
    def aggregate(
        self,
        rankings: List[Dict[str, int]],
        weights: List[float] = None
    ) -> Dict[str, int]:
        """
        Aggregate rankings.
        
        Args:
            rankings: List of {model_id: rank} rankings
            weights: Optional weights for each ranking
        
        Returns:
            Consensus ranking
        """
        if not rankings:
            return {}
        
        if self.method == "borda":
            return borda_count(rankings, weights)
        elif self.method == "kemeny":
            return kemeny_young_approximation(rankings, weights)
        elif self.method == "copeland":
            return copeland_method(rankings, weights)
        else:
            return borda_count(rankings, weights)
    
    def rank_correlation(
        self,
        ranking_a: Dict[str, int],
        ranking_b: Dict[str, int]
    ) -> float:
        """
        Compute Spearman rank correlation between two rankings.
        
        Args:
            ranking_a: First ranking
            ranking_b: Second ranking
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        common = set(ranking_a.keys()) & set(ranking_b.keys())
        if len(common) < 2:
            return 0.0
        
        n = len(common)
        d_squared = sum(
            (ranking_a[m] - ranking_b[m]) ** 2
            for m in common
        )
        
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        return rho


def borda_count(
    rankings: List[Dict[str, int]],
    weights: List[float] = None
) -> Dict[str, int]:
    """
    Aggregate rankings using weighted Borda count.
    
    Borda score: B_i = Σ_k w_k × (N - rank_ik)
    
    Args:
        rankings: List of {model_id: rank} rankings (1=best)
        weights: Weights for each ranking (uniform if None)
    
    Returns:
        Consensus ranking {model_id: rank}
    """
    if not rankings:
        return {}
    
    n_rankings = len(rankings)
    
    # Default to uniform weights
    if weights is None:
        weights = [1.0 / n_rankings] * n_rankings
    else:
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
    
    # Collect all items
    all_items = set()
    for ranking in rankings:
        all_items.update(ranking.keys())
    
    # Compute Borda scores
    scores = {item: 0.0 for item in all_items}
    
    for ranking, weight in zip(rankings, weights):
        n_items = len(ranking)
        for item, rank in ranking.items():
            # Higher score for better rank
            scores[item] += weight * (n_items - rank)
    
    # Sort by score (descending) and assign ranks
    sorted_items = sorted(scores.keys(), key=lambda x: -scores[x])
    return {item: i + 1 for i, item in enumerate(sorted_items)}


def kemeny_young_approximation(
    rankings: List[Dict[str, int]],
    weights: List[float] = None
) -> Dict[str, int]:
    """
    Approximate Kemeny-Young optimal ranking.
    
    The Kemeny-Young method finds the ranking that minimizes
    the total Kendall tau distance to all input rankings.
    This is NP-hard, so we use an approximation.
    
    Args:
        rankings: List of rankings
        weights: Optional weights
    
    Returns:
        Approximate optimal ranking
    """
    if not rankings:
        return {}
    
    if len(rankings) == 1:
        return rankings[0].copy()
    
    # Use Borda as starting point
    borda_ranking = borda_count(rankings, weights)
    
    # Local search improvement
    items = list(borda_ranking.keys())
    current_order = sorted(items, key=lambda x: borda_ranking[x])
    current_cost = _kemeny_cost(current_order, rankings, weights)
    
    # Try swapping adjacent items
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(current_order) - 1):
            # Try swapping i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_cost = _kemeny_cost(new_order, rankings, weights)
            
            if new_cost < current_cost:
                current_order = new_order
                current_cost = new_cost
                improved = True
    
    return {item: i + 1 for i, item in enumerate(current_order)}


def _kemeny_cost(
    order: List[str],
    rankings: List[Dict[str, int]],
    weights: List[float] = None
) -> float:
    """Compute Kemeny cost (total Kendall tau distance)."""
    if weights is None:
        weights = [1.0] * len(rankings)
    
    cost = 0.0
    n = len(order)
    
    for ranking, weight in zip(rankings, weights):
        # Count disagreements
        for i in range(n):
            for j in range(i + 1, n):
                # In our order, i comes before j
                item_i, item_j = order[i], order[j]
                
                if item_i in ranking and item_j in ranking:
                    # Disagreement if ranking has j before i
                    if ranking[item_j] < ranking[item_i]:
                        cost += weight
    
    return cost


def copeland_method(
    rankings: List[Dict[str, int]],
    weights: List[float] = None
) -> Dict[str, int]:
    """
    Aggregate rankings using Copeland method.
    
    Score = wins - losses in pairwise comparisons.
    
    Args:
        rankings: List of rankings
        weights: Optional weights
    
    Returns:
        Consensus ranking
    """
    if not rankings:
        return {}
    
    if weights is None:
        weights = [1.0] * len(rankings)
    
    # Collect all items
    all_items = set()
    for ranking in rankings:
        all_items.update(ranking.keys())
    
    items = list(all_items)
    scores = {item: 0.0 for item in items}
    
    # Pairwise comparisons
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_a, item_b = items[i], items[j]
            
            a_wins = 0.0
            b_wins = 0.0
            
            for ranking, weight in zip(rankings, weights):
                if item_a in ranking and item_b in ranking:
                    if ranking[item_a] < ranking[item_b]:
                        a_wins += weight
                    elif ranking[item_b] < ranking[item_a]:
                        b_wins += weight
            
            if a_wins > b_wins:
                scores[item_a] += 1
                scores[item_b] -= 1
            elif b_wins > a_wins:
                scores[item_b] += 1
                scores[item_a] -= 1
    
    # Sort by score and assign ranks
    sorted_items = sorted(scores.keys(), key=lambda x: -scores[x])
    return {item: i + 1 for i, item in enumerate(sorted_items)}


def compute_consensus_score(
    consensus: Dict[str, int],
    rankings: List[Dict[str, int]],
    weights: List[float] = None
) -> float:
    """
    Compute how well the consensus represents input rankings.
    
    Returns average Spearman correlation.
    
    Args:
        consensus: Consensus ranking
        rankings: Input rankings
        weights: Optional weights
    
    Returns:
        Average correlation score
    """
    if not rankings:
        return 1.0
    
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings)
    
    aggregator = RankingAggregator()
    total_corr = 0.0
    
    for ranking, weight in zip(rankings, weights):
        corr = aggregator.rank_correlation(consensus, ranking)
        total_corr += weight * corr
    
    return total_corr


def detect_ranking_anomalies(
    rankings: List[Dict[str, int]],
    threshold: float = 0.3
) -> List[int]:
    """
    Detect anomalous rankings that differ significantly from consensus.
    
    Args:
        rankings: List of rankings
        threshold: Correlation threshold (rankings below this are anomalous)
    
    Returns:
        Indices of anomalous rankings
    """
    if len(rankings) < 2:
        return []
    
    # Compute consensus without each ranking
    aggregator = RankingAggregator()
    anomalies = []
    
    for i in range(len(rankings)):
        # Leave-one-out consensus
        other_rankings = rankings[:i] + rankings[i+1:]
        consensus = borda_count(other_rankings)
        
        # Check correlation
        corr = aggregator.rank_correlation(rankings[i], consensus)
        
        if corr < threshold:
            anomalies.append(i)
    
    return anomalies


