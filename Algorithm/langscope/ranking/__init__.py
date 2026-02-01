"""
Ranking module for LangScope.

Implements TrueSkill + Plackett-Luce rating system for multi-player matches
with 10-dimensional rankings.
"""

from langscope.ranking.trueskill import (
    TrueSkillRating,
    MultiPlayerTrueSkillUpdater,
)
from langscope.ranking.plackett_luce import (
    PlackettLuceModel,
    PlackettLuceResult,
)
from langscope.ranking.cost_adjustment import (
    calculate_efficiency_weights,
    create_cost_adjusted_ranking,
    aggregate_judge_rankings,
)
from langscope.ranking.dimension_ranker import (
    DimensionRanker,
    DimensionRanking,
    MultiDimensionalRanking,
    aggregate_dimension_rankings,
)

__all__ = [
    # TrueSkill
    "TrueSkillRating",
    "MultiPlayerTrueSkillUpdater",
    # Plackett-Luce
    "PlackettLuceModel",
    "PlackettLuceResult",
    # Cost adjustment
    "calculate_efficiency_weights",
    "create_cost_adjusted_ranking",
    "aggregate_judge_rankings",
    # 10D Ranking
    "DimensionRanker",
    "DimensionRanking",
    "MultiDimensionalRanking",
    "aggregate_dimension_rankings",
]


