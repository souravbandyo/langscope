"""
Model deployment definitions.

Contains classes for cloud provider and self-hosted deployments.
"""

from langscope.models.deployments.cloud import (
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
from langscope.models.deployments.self_hosted import (
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
)

__all__ = [
    # Cloud deployments
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
    # Self-hosted deployments
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
]

