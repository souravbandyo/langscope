"""
Evaluation module for LangScope.

Provides match management, ranking aggregation, penalty systems,
and metrics collection for 10-dimensional evaluation.
"""

from langscope.evaluation.match import (
    Match,
    MatchParticipant,
    MatchResponse,
    create_match,
)
from langscope.evaluation.aggregation import (
    RankingAggregator,
    borda_count,
    kemeny_young_approximation,
)
from langscope.evaluation.penalties import (
    PenaltySystem,
    Penalty,
    apply_judge_penalty,
    apply_content_penalty,
)
from langscope.evaluation.metrics import (
    BattleMetrics,
    LatencyTimer,
    MetricsCollector,
    collect_latency,
    collect_latency_async,
    run_consistency_evaluation,
    validate_instruction_following,
    check_hallucinations,
    run_long_context_evaluation,
    Constraint,
)

__all__ = [
    # Match
    "Match",
    "MatchParticipant",
    "MatchResponse",
    "create_match",
    # Aggregation
    "RankingAggregator",
    "borda_count",
    "kemeny_young_approximation",
    # Penalties
    "PenaltySystem",
    "Penalty",
    "apply_judge_penalty",
    "apply_content_penalty",
    # Metrics (10D)
    "BattleMetrics",
    "LatencyTimer",
    "MetricsCollector",
    "collect_latency",
    "collect_latency_async",
    "run_consistency_evaluation",
    "validate_instruction_following",
    "check_hallucinations",
    "run_long_context_evaluation",
    "Constraint",
]


