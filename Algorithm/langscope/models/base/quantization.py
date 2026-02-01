"""
Quantization option definitions.

Describes different ways to compress a model for efficient serving.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class QuantizationOption:
    """A quantization option for the model."""
    bits: float  # Can be fractional for mixed quantization
    vram_gb: float
    ram_gb: Optional[float] = None  # For CPU inference
    quality_retention: float = 1.0  # 0-1 scale
    huggingface_id: Optional[str] = None
    supported_frameworks: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "bits": self.bits,
            "vram_gb": self.vram_gb,
            "quality_retention": self.quality_retention,
            "supported_frameworks": self.supported_frameworks,
            "notes": self.notes,
        }
        if self.ram_gb is not None:
            result["ram_gb"] = self.ram_gb
        if self.huggingface_id:
            result["huggingface_id"] = self.huggingface_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationOption':
        """Create from dictionary."""
        return cls(
            bits=data.get("bits", 16),
            vram_gb=data.get("vram_gb", 0),
            ram_gb=data.get("ram_gb"),
            quality_retention=data.get("quality_retention", 1.0),
            huggingface_id=data.get("huggingface_id"),
            supported_frameworks=data.get("supported_frameworks", []),
            notes=data.get("notes", ""),
        )

