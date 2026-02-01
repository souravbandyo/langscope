"""
Quantization profile definitions.

Reference data for quantization methods to help users understand trade-offs.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class QuantizationMethod(str, Enum):
    """Common quantization methods."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    EXLLAMA = "exllama"


@dataclass
class QuantizationProfile:
    """
    Quantization method profile.
    
    Provides reference data for understanding quantization trade-offs.
    """
    id: str  # e.g., "awq-4bit"
    name: str
    method: QuantizationMethod
    bits: float  # Can be fractional for mixed methods
    
    # Quality metrics
    quality_retention: float = 1.0  # 0-1 scale vs full precision
    perplexity_increase_pct: float = 0.0  # Typical increase vs FP16
    
    # Memory impact
    memory_reduction_pct: float = 0.0  # Memory saved vs FP16
    
    # Supported frameworks
    supported_frameworks: List[str] = field(default_factory=list)
    
    # Speed impact
    inference_speedup: float = 1.0  # Relative to FP16 (1.0 = same)
    
    # Use cases
    best_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    
    # Technical details
    description: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "_id": self.id,
            "name": self.name,
            "method": self.method.value if isinstance(self.method, Enum) else self.method,
            "bits": self.bits,
            "quality": {
                "retention": self.quality_retention,
                "perplexity_increase_pct": self.perplexity_increase_pct,
            },
            "memory": {
                "reduction_pct": self.memory_reduction_pct,
            },
            "supported_frameworks": self.supported_frameworks,
            "inference_speedup": self.inference_speedup,
            "best_for": self.best_for,
            "not_recommended_for": self.not_recommended_for,
            "description": self.description,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationProfile':
        """Create from dictionary."""
        method = data.get("method", "bf16")
        if isinstance(method, str):
            try:
                method = QuantizationMethod(method)
            except ValueError:
                method = QuantizationMethod.BF16
        
        quality = data.get("quality", {})
        memory = data.get("memory", {})
        
        return cls(
            id=data.get("_id", ""),
            name=data.get("name", ""),
            method=method,
            bits=data.get("bits", 16.0),
            quality_retention=quality.get("retention", 1.0),
            perplexity_increase_pct=quality.get("perplexity_increase_pct", 0.0),
            memory_reduction_pct=memory.get("reduction_pct", 0.0),
            supported_frameworks=data.get("supported_frameworks", []),
            inference_speedup=data.get("inference_speedup", 1.0),
            best_for=data.get("best_for", []),
            not_recommended_for=data.get("not_recommended_for", []),
            description=data.get("description", ""),
            notes=data.get("notes", ""),
        )


# Predefined quantization profiles
QUANTIZATION_PROFILES: Dict[str, QuantizationProfile] = {
    "bf16": QuantizationProfile(
        id="bf16",
        name="BFloat16",
        method=QuantizationMethod.BF16,
        bits=16.0,
        quality_retention=1.0,
        memory_reduction_pct=0.0,
        supported_frameworks=["transformers", "vllm", "tgi"],
        inference_speedup=1.0,
        best_for=["maximum-quality", "training", "fine-tuning"],
        description="Full precision BFloat16, baseline quality.",
    ),
    "awq-4bit": QuantizationProfile(
        id="awq-4bit",
        name="AWQ 4-bit",
        method=QuantizationMethod.AWQ,
        bits=4.0,
        quality_retention=0.95,
        perplexity_increase_pct=2.0,
        memory_reduction_pct=75.0,
        supported_frameworks=["vllm", "transformers", "exllama"],
        inference_speedup=1.2,
        best_for=["production-inference", "memory-constrained"],
        description="Activation-aware Weight Quantization with 4-bit weights.",
        notes="Best balance of quality and memory for most use cases.",
    ),
    "gptq-4bit": QuantizationProfile(
        id="gptq-4bit",
        name="GPTQ 4-bit",
        method=QuantizationMethod.GPTQ,
        bits=4.0,
        quality_retention=0.94,
        perplexity_increase_pct=3.0,
        memory_reduction_pct=75.0,
        supported_frameworks=["vllm", "transformers", "exllama", "autogptq"],
        inference_speedup=1.1,
        best_for=["production-inference", "memory-constrained"],
        description="Post-training quantization using second-order information.",
    ),
    "gguf-q4_k_m": QuantizationProfile(
        id="gguf-q4_k_m",
        name="GGUF Q4_K_M",
        method=QuantizationMethod.GGUF,
        bits=4.5,  # Mixed quantization
        quality_retention=0.93,
        perplexity_increase_pct=4.0,
        memory_reduction_pct=72.0,
        supported_frameworks=["llama.cpp", "ollama"],
        inference_speedup=1.3,
        best_for=["local-inference", "cpu-inference", "edge-deployment"],
        description="GGML quantization format, works on CPU and GPU.",
        notes="Best for local/edge deployment, can run on consumer hardware.",
    ),
    "fp8": QuantizationProfile(
        id="fp8",
        name="FP8",
        method=QuantizationMethod.FP8,
        bits=8.0,
        quality_retention=0.99,
        perplexity_increase_pct=0.5,
        memory_reduction_pct=50.0,
        supported_frameworks=["vllm", "tensorrt-llm"],
        inference_speedup=1.5,
        best_for=["high-quality", "high-throughput"],
        description="8-bit floating point, minimal quality loss.",
        notes="Requires H100 or newer GPUs for native support.",
    ),
}

