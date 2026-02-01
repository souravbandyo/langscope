"""
LangScope Models Package.

Comprehensive data models for model evaluation, multi-provider support,
and external benchmarks.

This package implements the model_requirement.md specification.

Structure:
- base/: Base model definitions (architecture, capabilities, quantizations)
- deployments/: Cloud and self-hosted deployments
- benchmarks/: Benchmark definitions, results, and correlations
- sources/: Data source configuration and sync automation
- timeseries/: Time series data (rating history, performance metrics)
- reference/: Reference data (hardware profiles, quantization methods)
- hashing: Content and price hashing utilities
"""

# Base models
from langscope.models.base import (
    BaseModel,
    Architecture,
    ArchitectureType,
    Capabilities,
    ContextWindow,
    License,
    Modality,
    QuantizationOption,
    BenchmarkScore,
    BenchmarkAggregates,
)

# Deployments
from langscope.models.deployments import (
    ModelDeployment,
    SelfHostedDeployment,
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
    # Self-hosted specific
    Owner,
    HardwareConfig,
    SoftwareConfig,
    SelfHostedCosts,
    SelfHostedPerformance,
    SelfHostedAvailability,
    CloudProvider,
    ServingFramework,
    CostCalculationMethod,
)

# Benchmarks
from langscope.models.benchmarks import (
    BenchmarkDefinition,
    BenchmarkCategory,
    BenchmarkScoring,
    BenchmarkSource,
    BenchmarkAutomation,
    LangScopeCorrelation,
    BenchmarkResult,
    BenchmarkCorrelation,
    PREDEFINED_BENCHMARKS,
    get_benchmark_definition,
    list_benchmarks_by_category,
    compute_benchmark_correlation,
    get_prior_from_benchmarks,
)

# Data sources
from langscope.models.sources import (
    DataSource,
    DataSourceType,
    SourceConfig,
    SourceMethod,
    AutomationConfig,
    ValidationConfig,
    ReliabilityMetrics,
    SyncEngine,
    SyncResult,
    SyncStatus,
    PREDEFINED_SOURCES,
)

# Hashing utilities
from langscope.models.hashing import (
    content_hash,
    price_hash,
    response_cache_key,
    data_hash,
    benchmark_score_hash,
    normalize_text,
    ContentHasher,
)

__all__ = [
    # Base models
    "BaseModel",
    "Architecture",
    "ArchitectureType",
    "Capabilities",
    "ContextWindow",
    "License",
    "Modality",
    "QuantizationOption",
    "BenchmarkScore",
    "BenchmarkAggregates",
    # Deployments
    "ModelDeployment",
    "SelfHostedDeployment",
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
    "Owner",
    "HardwareConfig",
    "SoftwareConfig",
    "SelfHostedCosts",
    "SelfHostedPerformance",
    "SelfHostedAvailability",
    "CloudProvider",
    "ServingFramework",
    "CostCalculationMethod",
    # Benchmarks
    "BenchmarkDefinition",
    "BenchmarkCategory",
    "BenchmarkScoring",
    "BenchmarkSource",
    "BenchmarkAutomation",
    "LangScopeCorrelation",
    "BenchmarkResult",
    "BenchmarkCorrelation",
    "PREDEFINED_BENCHMARKS",
    "get_benchmark_definition",
    "list_benchmarks_by_category",
    "compute_benchmark_correlation",
    "get_prior_from_benchmarks",
    # Data sources
    "DataSource",
    "DataSourceType",
    "SourceConfig",
    "SourceMethod",
    "AutomationConfig",
    "ValidationConfig",
    "ReliabilityMetrics",
    "SyncEngine",
    "SyncResult",
    "SyncStatus",
    "PREDEFINED_SOURCES",
    # Hashing
    "content_hash",
    "price_hash",
    "response_cache_key",
    "data_hash",
    "benchmark_score_hash",
    "normalize_text",
    "ContentHasher",
]

