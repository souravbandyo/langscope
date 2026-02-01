"""
BaseModel class representing the actual neural network.

A BaseModel is the model itself (e.g., Llama 3.1 70B) - its architecture,
capabilities, and benchmark scores. It can have multiple deployments
across different providers with different pricing and performance.

DEPRECATED: This module is deprecated. Import from langscope.models.base instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
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

__all__ = [
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
]

