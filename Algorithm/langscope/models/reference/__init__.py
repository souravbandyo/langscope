"""
Reference data models.

Contains static reference data:
- Hardware profiles (GPU specs, memory, etc.)
- Quantization profiles (AWQ, GPTQ, GGUF methods)

These help users choose appropriate hardware and understand
quantization trade-offs.
"""

from langscope.models.reference.hardware import (
    HardwareProfile,
    GPUType,
)
from langscope.models.reference.quantization import (
    QuantizationProfile,
    QuantizationMethod,
)

__all__ = [
    "HardwareProfile",
    "GPUType",
    "QuantizationProfile",
    "QuantizationMethod",
]

