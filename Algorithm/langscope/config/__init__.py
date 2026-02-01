"""
Configuration module for LangScope.

Provides settings management, API key handling, and dynamic parameter management.
"""

from langscope.config.settings import Settings, get_settings
from langscope.config.api_keys import APIKeyManager, get_api_key
from langscope.config.params import (
    ParameterManager,
    get_parameter_manager,
    ParamType,
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
    ParamCache,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # API Keys
    "APIKeyManager",
    "get_api_key",
    # Parameter Management
    "ParameterManager",
    "get_parameter_manager",
    "ParamType",
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
    "ParamCache",
]


