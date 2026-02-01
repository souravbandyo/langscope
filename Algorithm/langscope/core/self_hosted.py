"""
SelfHostedDeployment class for user-owned infrastructure.

Users can register their own deployments with custom hardware
configurations and cost calculations.

DEPRECATED: This module is deprecated. Import from langscope.models.deployments instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.deployments import (
    SelfHostedDeployment,
    Owner,
    HardwareConfig,
    SoftwareConfig,
    SelfHostedCosts,
    SelfHostedPerformance,
    SelfHostedAvailability,
    CloudProvider,
    ServingFramework,
    CostCalculationMethod,
    CostCalculation,
    MonthlyFixedCosts,
    # Also re-export from cloud.py for compatibility
    Provider,
    ProviderType,
    AvailabilityStatus,
    PerformanceStats,
)

__all__ = [
    "SelfHostedDeployment",
    "Owner",
    "HardwareConfig",
    "SoftwareConfig",
    "SelfHostedCosts",
    "SelfHostedPerformance",
    "SelfHostedAvailability",
    "CloudProvider",
    "ServingFramework",
    "CostCalculationMethod",
    "CostCalculation",
    "MonthlyFixedCosts",
    "Provider",
    "ProviderType",
    "AvailabilityStatus",
    "PerformanceStats",
]

