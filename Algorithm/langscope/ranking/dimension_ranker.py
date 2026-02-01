"""
Per-Dimension Ranking for 10-Dimensional TrueSkill.

Provides ranking logic for each of the 10 dimensions:
- Raw Quality (from judge rankings)
- Cost-Adjusted (quality per cost)
- Latency (response time)
- TTFT (time to first token)
- Consistency (response variance)
- Token Efficiency (quality per token)
- Instruction Following (constraint compliance)
- Hallucination Resistance (factual accuracy)
- Long Context (context handling)
- Combined (weighted aggregate)
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from langscope.core.dimensions import (
    Dimension,
    compute_latency_score,
    compute_ttft_score,
    compute_consistency_score,
    compute_token_efficiency_score,
    compute_cost_adjusted_score,
    compute_instruction_following_score,
    compute_hallucination_resistance_score,
    compute_long_context_score,
    compute_combined_score,
    DEFAULT_COMBINED_WEIGHTS,
)
from langscope.evaluation.metrics import BattleMetrics


@dataclass
class DimensionRanking:
    """Ranking result for a single dimension."""
    dimension: Dimension
    rankings: Dict[str, int]  # model_id -> rank (1 = best)
    scores: Dict[str, float]  # model_id -> score
    
    def get_rank(self, model_id: str) -> int:
        """Get rank for a model (1 = best)."""
        return self.rankings.get(model_id, len(self.rankings) + 1)
    
    def get_score(self, model_id: str) -> float:
        """Get score for a model."""
        return self.scores.get(model_id, 0.0)


@dataclass
class MultiDimensionalRanking:
    """Rankings across all dimensions for a match."""
    match_id: str
    participants: List[str]
    dimension_rankings: Dict[Dimension, DimensionRanking]
    
    def get_dimension_ranking(self, dimension: Dimension) -> DimensionRanking:
        """Get ranking for a specific dimension."""
        return self.dimension_rankings.get(dimension)
    
    def get_model_ranks(self, model_id: str) -> Dict[str, int]:
        """Get all ranks for a model across dimensions."""
        return {
            dim.value: ranking.get_rank(model_id)
            for dim, ranking in self.dimension_rankings.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "participants": self.participants,
            "dimension_rankings": {
                dim.value: {
                    "rankings": ranking.rankings,
                    "scores": ranking.scores,
                }
                for dim, ranking in self.dimension_rankings.items()
            },
        }


class DimensionRanker:
    """
    Computes rankings for each dimension based on metrics.
    """
    
    def __init__(
        self,
        tau_latency: float = 1000.0,
        tau_ttft: float = 200.0,
        combined_weights: Dict[str, float] = None
    ):
        """
        Initialize DimensionRanker.
        
        Args:
            tau_latency: Latency temperature (ms)
            tau_ttft: TTFT temperature (ms)
            combined_weights: Custom weights for combined score
        """
        self.tau_latency = tau_latency
        self.tau_ttft = tau_ttft
        self.combined_weights = combined_weights or DEFAULT_COMBINED_WEIGHTS
    
    def compute_all_rankings(
        self,
        match_id: str,
        raw_rankings: Dict[str, int],  # From judges
        metrics: Dict[str, BattleMetrics],
        mu_raws: Dict[str, float] = None,  # Pre-computed raw quality
        cost_per_millions: Dict[str, float] = None,
    ) -> MultiDimensionalRanking:
        """
        Compute rankings for all dimensions.
        
        Args:
            match_id: Match identifier
            raw_rankings: Raw quality rankings from judges (1 = best)
            metrics: Metrics for each participant
            mu_raws: Optional pre-computed raw quality scores
            cost_per_millions: Cost per million tokens for each model
        
        Returns:
            MultiDimensionalRanking with all dimension rankings
        """
        participants = list(raw_rankings.keys())
        dimension_rankings = {}
        
        # 1. Raw Quality - directly from judge rankings
        dimension_rankings[Dimension.RAW_QUALITY] = DimensionRanking(
            dimension=Dimension.RAW_QUALITY,
            rankings=raw_rankings,
            scores={mid: len(raw_rankings) - r + 1 for mid, r in raw_rankings.items()},
        )
        
        # 2. Cost-Adjusted
        if cost_per_millions:
            cost_scores = {}
            for mid in participants:
                mu_raw = mu_raws.get(mid, 1500.0) if mu_raws else 1500.0
                cost = cost_per_millions.get(mid, 1.0)
                cost_scores[mid] = compute_cost_adjusted_score(mu_raw, cost)
            dimension_rankings[Dimension.COST_ADJUSTED] = self._scores_to_ranking(
                Dimension.COST_ADJUSTED, cost_scores
            )
        
        # 3. Latency
        latency_scores = {}
        for mid in participants:
            if mid in metrics:
                latency_scores[mid] = compute_latency_score(
                    metrics[mid].latency_ms, self.tau_latency
                )
        if latency_scores:
            dimension_rankings[Dimension.LATENCY] = self._scores_to_ranking(
                Dimension.LATENCY, latency_scores
            )
        
        # 4. TTFT
        ttft_scores = {}
        for mid in participants:
            if mid in metrics:
                ttft_scores[mid] = compute_ttft_score(
                    metrics[mid].ttft_ms, self.tau_ttft
                )
        if ttft_scores:
            dimension_rankings[Dimension.TTFT] = self._scores_to_ranking(
                Dimension.TTFT, ttft_scores
            )
        
        # 5. Consistency
        consistency_scores = {}
        for mid in participants:
            if mid in metrics and metrics[mid].consistency_runs > 0:
                consistency_scores[mid] = compute_consistency_score(
                    metrics[mid].response_variance
                )
        if consistency_scores:
            dimension_rankings[Dimension.CONSISTENCY] = self._scores_to_ranking(
                Dimension.CONSISTENCY, consistency_scores
            )
        
        # 6. Token Efficiency
        token_eff_scores = {}
        for mid in participants:
            if mid in metrics:
                mu_raw = mu_raws.get(mid, 1500.0) if mu_raws else 1500.0
                token_eff_scores[mid] = compute_token_efficiency_score(
                    mu_raw, metrics[mid].output_tokens
                )
        if token_eff_scores:
            dimension_rankings[Dimension.TOKEN_EFFICIENCY] = self._scores_to_ranking(
                Dimension.TOKEN_EFFICIENCY, token_eff_scores
            )
        
        # 7. Instruction Following
        instr_scores = {}
        for mid in participants:
            if mid in metrics and metrics[mid].total_constraints > 0:
                instr_scores[mid] = compute_instruction_following_score(
                    metrics[mid].constraints_satisfied,
                    metrics[mid].total_constraints
                )
        if instr_scores:
            dimension_rankings[Dimension.INSTRUCTION_FOLLOWING] = self._scores_to_ranking(
                Dimension.INSTRUCTION_FOLLOWING, instr_scores
            )
        
        # 8. Hallucination Resistance
        hal_scores = {}
        for mid in participants:
            if mid in metrics and metrics[mid].verifiable_claims > 0:
                hal_scores[mid] = compute_hallucination_resistance_score(
                    metrics[mid].hallucination_count,
                    metrics[mid].verifiable_claims
                )
        if hal_scores:
            dimension_rankings[Dimension.HALLUCINATION_RESISTANCE] = self._scores_to_ranking(
                Dimension.HALLUCINATION_RESISTANCE, hal_scores
            )
        
        # 9. Long Context
        lc_scores = {}
        for mid in participants:
            if mid in metrics and metrics[mid].context_length > 0:
                # Use quality_at_length / baseline (assuming baseline is 1.0)
                lc_scores[mid] = metrics[mid].quality_at_length
        if lc_scores:
            dimension_rankings[Dimension.LONG_CONTEXT] = self._scores_to_ranking(
                Dimension.LONG_CONTEXT, lc_scores
            )
        
        # 10. Combined (weighted aggregate of available dimensions)
        combined_mus = {}
        for mid in participants:
            dim_mus = {}
            for dim, ranking in dimension_rankings.items():
                if dim != Dimension.COMBINED:
                    # Normalize score to ~1500 scale for combination
                    dim_mus[dim.value] = ranking.get_score(mid) * 1500
            combined_mus[mid] = compute_combined_score(dim_mus, self.combined_weights)
        
        dimension_rankings[Dimension.COMBINED] = self._scores_to_ranking(
            Dimension.COMBINED, combined_mus
        )
        
        return MultiDimensionalRanking(
            match_id=match_id,
            participants=participants,
            dimension_rankings=dimension_rankings,
        )
    
    def _scores_to_ranking(
        self,
        dimension: Dimension,
        scores: Dict[str, float]
    ) -> DimensionRanking:
        """
        Convert scores to rankings (higher score = better rank).
        
        Args:
            dimension: The dimension being ranked
            scores: Model ID -> score mapping
        
        Returns:
            DimensionRanking with ranks and scores
        """
        # Sort by score descending
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {}
        current_rank = 1
        for i, (model_id, score) in enumerate(sorted_models):
            if i > 0 and score < sorted_models[i - 1][1]:
                current_rank = i + 1
            rankings[model_id] = current_rank
        
        return DimensionRanking(
            dimension=dimension,
            rankings=rankings,
            scores=scores,
        )
    
    def compute_single_dimension_ranking(
        self,
        dimension: Dimension,
        participants: List[str],
        metrics: Dict[str, BattleMetrics],
        mu_raws: Dict[str, float] = None,
        cost_per_millions: Dict[str, float] = None,
        raw_rankings: Dict[str, int] = None,
    ) -> DimensionRanking:
        """
        Compute ranking for a single dimension.
        
        Args:
            dimension: The dimension to rank
            participants: List of model IDs
            metrics: Metrics for each participant
            mu_raws: Raw quality mu values
            cost_per_millions: Cost per million tokens
            raw_rankings: Raw rankings from judges (for RAW_QUALITY)
        
        Returns:
            DimensionRanking for the specified dimension
        """
        scores = {}
        
        if dimension == Dimension.RAW_QUALITY:
            if raw_rankings:
                return DimensionRanking(
                    dimension=dimension,
                    rankings=raw_rankings,
                    scores={mid: len(raw_rankings) - r + 1 for mid, r in raw_rankings.items()},
                )
            return DimensionRanking(dimension=dimension, rankings={}, scores={})
        
        for mid in participants:
            if mid not in metrics:
                continue
            
            m = metrics[mid]
            mu_raw = mu_raws.get(mid, 1500.0) if mu_raws else 1500.0
            cost = cost_per_millions.get(mid, 1.0) if cost_per_millions else 1.0
            
            if dimension == Dimension.COST_ADJUSTED:
                scores[mid] = compute_cost_adjusted_score(mu_raw, cost)
            elif dimension == Dimension.LATENCY:
                scores[mid] = compute_latency_score(m.latency_ms, self.tau_latency)
            elif dimension == Dimension.TTFT:
                scores[mid] = compute_ttft_score(m.ttft_ms, self.tau_ttft)
            elif dimension == Dimension.CONSISTENCY:
                if m.consistency_runs > 0:
                    scores[mid] = compute_consistency_score(m.response_variance)
            elif dimension == Dimension.TOKEN_EFFICIENCY:
                scores[mid] = compute_token_efficiency_score(mu_raw, m.output_tokens)
            elif dimension == Dimension.INSTRUCTION_FOLLOWING:
                if m.total_constraints > 0:
                    scores[mid] = compute_instruction_following_score(
                        m.constraints_satisfied, m.total_constraints
                    )
            elif dimension == Dimension.HALLUCINATION_RESISTANCE:
                if m.verifiable_claims > 0:
                    scores[mid] = compute_hallucination_resistance_score(
                        m.hallucination_count, m.verifiable_claims
                    )
            elif dimension == Dimension.LONG_CONTEXT:
                if m.context_length > 0:
                    scores[mid] = m.quality_at_length
        
        return self._scores_to_ranking(dimension, scores)


def aggregate_dimension_rankings(
    rankings_list: List[DimensionRanking],
    weights: List[float] = None
) -> DimensionRanking:
    """
    Aggregate multiple dimension rankings (e.g., from multiple judges).
    
    Uses weighted Borda count aggregation.
    
    Args:
        rankings_list: List of DimensionRanking from different sources
        weights: Optional weights for each ranking source
    
    Returns:
        Aggregated DimensionRanking
    """
    if not rankings_list:
        return DimensionRanking(dimension=Dimension.COMBINED, rankings={}, scores={})
    
    if weights is None:
        weights = [1.0] * len(rankings_list)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Collect all participants
    all_participants = set()
    for ranking in rankings_list:
        all_participants.update(ranking.rankings.keys())
    
    # Compute weighted Borda scores
    borda_scores = {mid: 0.0 for mid in all_participants}
    n_participants = len(all_participants)
    
    for ranking, weight in zip(rankings_list, weights):
        for model_id, rank in ranking.rankings.items():
            # Borda score: n - rank (higher is better)
            borda_scores[model_id] += weight * (n_participants - rank)
    
    # Convert to rankings
    sorted_models = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    final_rankings = {mid: i + 1 for i, (mid, _) in enumerate(sorted_models)}
    
    return DimensionRanking(
        dimension=rankings_list[0].dimension,
        rankings=final_rankings,
        scores=borda_scores,
    )


