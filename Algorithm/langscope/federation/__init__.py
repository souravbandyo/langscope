"""
Federation module for LangScope.

Implements peer-federated evaluation with:
- Strata-based role assignment
- Swiss-style multi-player pairing
- Judge selection and weighting
- Content creator selection
- Complete match workflow
- Match routing (subjective vs ground truth)
"""

from langscope.federation.strata import (
    get_stratum,
    get_stratum_name,
    get_eligible_judges,
    get_eligible_creators,
)
from langscope.federation.selection import (
    MultiPlayerSwissPairing,
    ContentCreatorSelector,
    JudgeSelector,
)
from langscope.federation.judge import (
    JudgeRankingValidator,
    JudgeAggregator,
    detect_outlier_judge,
)
from langscope.federation.router import (
    MatchRouter,
    MatchRouterConfig,
    get_default_router,
    set_default_router,
    route_match,
    get_evaluation_type,
    is_ground_truth_domain,
    is_subjective_domain,
    list_ground_truth_domains,
    list_subjective_domains,
)

__all__ = [
    # Strata
    "get_stratum",
    "get_stratum_name",
    "get_eligible_judges",
    "get_eligible_creators",
    # Selection
    "MultiPlayerSwissPairing",
    "ContentCreatorSelector",
    "JudgeSelector",
    # Judge
    "JudgeRankingValidator",
    "JudgeAggregator",
    "detect_outlier_judge",
    # Router
    "MatchRouter",
    "MatchRouterConfig",
    "get_default_router",
    "set_default_router",
    "route_match",
    "get_evaluation_type",
    "is_ground_truth_domain",
    "is_subjective_domain",
    "list_ground_truth_domains",
    "list_subjective_domains",
]


