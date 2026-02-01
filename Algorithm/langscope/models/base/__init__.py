"""
Base model definitions.

Contains the core BaseModel class and related types for representing
the actual neural network (weights, architecture, capabilities).
"""

from langscope.models.base.model import (
    BaseModel,
    BenchmarkScore,
    BenchmarkAggregates,
)
from langscope.models.base.architecture import (
    Architecture,
    ArchitectureType,
)
from langscope.models.base.capabilities import (
    Capabilities,
    Modality,
)
from langscope.models.base.context import ContextWindow
from langscope.models.base.license import License
from langscope.models.base.quantization import QuantizationOption

__all__ = [
    "BaseModel",
    "BenchmarkScore",
    "BenchmarkAggregates",
    "Architecture",
    "ArchitectureType",
    "Capabilities",
    "Modality",
    "ContextWindow",
    "License",
    "QuantizationOption",
]

