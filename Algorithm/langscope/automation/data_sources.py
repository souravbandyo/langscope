"""
Data source configuration for automated syncing.

Configures external APIs and sources for pricing, models, and benchmarks.

DEPRECATED: This module is deprecated. Import from langscope.models.sources instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.sources import (
    DataSource,
    DataSourceType,
    SourceConfig,
    SourceMethod,
    AutomationConfig,
    ValidationConfig,
    ReliabilityMetrics,
    PREDEFINED_SOURCES,
)

__all__ = [
    "DataSource",
    "DataSourceType",
    "SourceConfig",
    "SourceMethod",
    "AutomationConfig",
    "ValidationConfig",
    "ReliabilityMetrics",
    "PREDEFINED_SOURCES",
]

