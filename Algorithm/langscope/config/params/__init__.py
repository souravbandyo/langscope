"""
Parameter Management System for LangScope.

Provides dynamic, database-backed parameter management with:
- Runtime parameter updates via API
- Per-domain parameter overrides
- In-memory caching with TTL
- Parameter groups: TrueSkill, Strata, Match, Temperature, DimensionWeights, Transfer, Feedback, Penalty
"""

from langscope.config.params.models import (
    TrueSkillParams,
    StrataParams,
    MatchParams,
    TemperatureParams,
    DimensionWeightParams,
    TransferParams,
    FeedbackParams,
    PenaltyParams,
    ConsistencyParams,
    LongContextParams,
    SystemParams,
    ParamType,
)
from langscope.config.params.cache import ParamCache
from langscope.config.params.manager import ParameterManager, get_parameter_manager

__all__ = [
    # Parameter models
    "TrueSkillParams",
    "StrataParams",
    "MatchParams",
    "TemperatureParams",
    "DimensionWeightParams",
    "TransferParams",
    "FeedbackParams",
    "PenaltyParams",
    "ConsistencyParams",
    "LongContextParams",
    "SystemParams",
    "ParamType",
    # Cache
    "ParamCache",
    # Manager
    "ParameterManager",
    "get_parameter_manager",
]


