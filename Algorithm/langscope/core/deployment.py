"""
ModelDeployment class for cloud provider instances.

A deployment is a specific hosted version of a base model with its own
pricing, performance characteristics, and TrueSkill ratings.

Example: groq/llama-3.1-70b-versatile is Llama 3.1 70B served by Groq.

DEPRECATED: This module is deprecated. Import from langscope.models.deployments instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.deployments import (
    ModelDeployment,
    Provider,
    ProviderType,
    DeploymentConfig,
    Pricing,
    BatchPricing,
    FreeTier,
    Performance,
    RateLimits,
    Availability,
    AvailabilityStatus,
    PerformanceStats,
)

__all__ = [
    "ModelDeployment",
    "Provider",
    "ProviderType",
    "DeploymentConfig",
    "Pricing",
    "BatchPricing",
    "FreeTier",
    "Performance",
    "RateLimits",
    "Availability",
    "AvailabilityStatus",
    "PerformanceStats",
]

