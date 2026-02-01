"""
Model architecture definitions.

Describes the technical structure of a model (parameters, layers, etc.).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ArchitectureType(str, Enum):
    """Model architecture types."""
    DECODER_ONLY = "decoder-only"
    ENCODER_DECODER = "encoder-decoder"
    MOE = "moe"  # Mixture of Experts


@dataclass
class Architecture:
    """Model architecture details."""
    type: ArchitectureType = ArchitectureType.DECODER_ONLY
    parameters: int = 0  # Raw parameter count
    parameters_display: str = ""  # Human-readable (e.g., "70B")
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0  # For GQA
    vocab_size: int = 0
    max_position_embeddings: int = 0
    native_precision: str = "bfloat16"
    native_size_gb: float = 0.0
    
    # For MoE models
    moe_num_experts: Optional[int] = None
    moe_top_k: Optional[int] = None
    moe_total_params: Optional[int] = None
    moe_active_params: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "parameters": self.parameters,
            "parameters_display": self.parameters_display,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "native_precision": self.native_precision,
            "native_size_gb": self.native_size_gb,
        }
        if self.moe_num_experts is not None:
            result["moe"] = {
                "num_experts": self.moe_num_experts,
                "top_k": self.moe_top_k,
                "total_params": self.moe_total_params,
                "active_params": self.moe_active_params,
            }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Architecture':
        """Create from dictionary."""
        arch_type = data.get("type", "decoder-only")
        if isinstance(arch_type, str):
            try:
                arch_type = ArchitectureType(arch_type)
            except ValueError:
                arch_type = ArchitectureType.DECODER_ONLY
        
        instance = cls(
            type=arch_type,
            parameters=data.get("parameters", 0),
            parameters_display=data.get("parameters_display", ""),
            hidden_size=data.get("hidden_size", 0),
            num_layers=data.get("num_layers", 0),
            num_attention_heads=data.get("num_attention_heads", 0),
            num_kv_heads=data.get("num_kv_heads", 0),
            vocab_size=data.get("vocab_size", 0),
            max_position_embeddings=data.get("max_position_embeddings", 0),
            native_precision=data.get("native_precision", "bfloat16"),
            native_size_gb=data.get("native_size_gb", 0.0),
        )
        
        if "moe" in data and data["moe"]:
            moe = data["moe"]
            instance.moe_num_experts = moe.get("num_experts")
            instance.moe_top_k = moe.get("top_k")
            instance.moe_total_params = moe.get("total_params")
            instance.moe_active_params = moe.get("active_params")
        
        return instance

