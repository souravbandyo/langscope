"""
BaseModel class representing the actual neural network.

A BaseModel is the model itself (e.g., Llama 3.1 70B) - its architecture,
capabilities, and benchmark scores. It can have multiple deployments
across different providers with different pricing and performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from langscope.models.base.architecture import Architecture
from langscope.models.base.capabilities import Capabilities
from langscope.models.base.context import ContextWindow
from langscope.models.base.license import License
from langscope.models.base.quantization import QuantizationOption


@dataclass
class BenchmarkScore:
    """A single benchmark score."""
    score: float
    variant: str = ""  # e.g., "5-shot", "pass@1"
    percentile: Optional[int] = None
    updated_at: str = ""
    categories: Dict[str, float] = field(default_factory=dict)  # For Arena categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "score": self.score,
            "variant": self.variant,
        }
        if self.percentile is not None:
            result["percentile"] = self.percentile
        if self.updated_at:
            result["updated_at"] = self.updated_at
        if self.categories:
            result["categories"] = self.categories
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkScore':
        """Create from dictionary."""
        return cls(
            score=data.get("score", 0),
            variant=data.get("variant", ""),
            percentile=data.get("percentile"),
            updated_at=data.get("updated_at", ""),
            categories=data.get("categories", {}),
        )


@dataclass
class BenchmarkAggregates:
    """Computed benchmark aggregates."""
    open_llm_average: float = 0.0
    knowledge_average: float = 0.0
    reasoning_average: float = 0.0
    coding_average: float = 0.0
    math_average: float = 0.0
    chat_average: float = 0.0
    overall_rank: int = 0
    total_models_ranked: int = 0
    last_computed: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "open_llm_average": self.open_llm_average,
            "knowledge_average": self.knowledge_average,
            "reasoning_average": self.reasoning_average,
            "coding_average": self.coding_average,
            "math_average": self.math_average,
            "chat_average": self.chat_average,
            "overall_rank": self.overall_rank,
            "total_models_ranked": self.total_models_ranked,
            "last_computed": self.last_computed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkAggregates':
        """Create from dictionary."""
        return cls(
            open_llm_average=data.get("open_llm_average", 0.0),
            knowledge_average=data.get("knowledge_average", 0.0),
            reasoning_average=data.get("reasoning_average", 0.0),
            coding_average=data.get("coding_average", 0.0),
            math_average=data.get("math_average", 0.0),
            chat_average=data.get("chat_average", 0.0),
            overall_rank=data.get("overall_rank", 0),
            total_models_ranked=data.get("total_models_ranked", 0),
            last_computed=data.get("last_computed", ""),
        )


class BaseModel:
    """
    The actual neural network model.
    
    Represents the model itself (e.g., Llama 3.1 70B), not a specific deployment.
    Stores architecture, capabilities, quantization options, and benchmark scores.
    
    Each base model can have multiple deployments across different providers.
    """
    
    def __init__(
        self,
        id: str,  # e.g., "meta-llama/llama-3.1-70b"
        name: str,  # e.g., "Llama 3.1 70B"
        family: str = "",  # e.g., "llama"
        version: str = "",  # e.g., "3.1"
        organization: str = "",  # e.g., "Meta"
        initialize_new: bool = True
    ):
        """
        Initialize a base model.
        
        Args:
            id: Unique identifier (typically HuggingFace convention)
            name: Human-readable name
            family: Model family for grouping
            version: Version within family
            organization: Creating organization
            initialize_new: Whether to initialize with default values
        """
        self.id = id
        self.name = name
        self.family = family
        self.version = version
        self.organization = organization
        
        if initialize_new:
            self.architecture = Architecture()
            self.capabilities = Capabilities()
            self.context = ContextWindow()
            self.license = License()
            
            # Available quantization options
            self.quantizations: Dict[str, QuantizationOption] = {}
            
            # Serving requirements by framework
            self.serving_requirements: Dict[str, Dict[str, Any]] = {}
            
            # Sources (official, huggingface, etc.)
            self.sources: Dict[str, str] = {}
            
            # External benchmark scores
            self.benchmarks: Dict[str, BenchmarkScore] = {}
            self.benchmark_aggregates = BenchmarkAggregates()
            
            # Metadata
            self.released_at: str = ""
            self.created_at: str = datetime.utcnow().isoformat() + "Z"
            self.updated_at: str = datetime.utcnow().isoformat() + "Z"
    
    def get_quantization_options(self) -> List[str]:
        """Get list of available quantization names."""
        return list(self.quantizations.keys())
    
    def get_vram_requirement(self, quantization: str) -> Optional[float]:
        """
        Get VRAM requirement for a specific quantization.
        
        Args:
            quantization: Quantization name (e.g., "bf16", "awq-4bit")
        
        Returns:
            VRAM in GB or None if quantization not found
        """
        if quantization in self.quantizations:
            return self.quantizations[quantization].vram_gb
        return None
    
    def get_best_quantization_for_vram(self, available_vram_gb: float) -> Optional[str]:
        """
        Get the best quality quantization that fits in available VRAM.
        
        Args:
            available_vram_gb: Available GPU memory
        
        Returns:
            Best quantization name or None if none fits
        """
        fitting = []
        for name, quant in self.quantizations.items():
            if quant.vram_gb <= available_vram_gb:
                fitting.append((name, quant.quality_retention, quant.vram_gb))
        
        if not fitting:
            return None
        
        # Sort by quality retention (descending), then by VRAM usage (ascending)
        fitting.sort(key=lambda x: (-x[1], x[2]))
        return fitting[0][0]
    
    def get_benchmark_score(self, benchmark: str) -> Optional[float]:
        """Get score for a specific benchmark."""
        if benchmark in self.benchmarks:
            return self.benchmarks[benchmark].score
        return None
    
    def add_quantization(
        self,
        name: str,
        bits: float,
        vram_gb: float,
        quality_retention: float = 1.0,
        supported_frameworks: List[str] = None,
        notes: str = ""
    ):
        """Add a quantization option."""
        self.quantizations[name] = QuantizationOption(
            bits=bits,
            vram_gb=vram_gb,
            quality_retention=quality_retention,
            supported_frameworks=supported_frameworks or [],
            notes=notes,
        )
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def add_benchmark(
        self,
        benchmark: str,
        score: float,
        variant: str = "",
        percentile: int = None
    ):
        """Add a benchmark score."""
        self.benchmarks[benchmark] = BenchmarkScore(
            score=score,
            variant=variant,
            percentile=percentile,
            updated_at=datetime.utcnow().isoformat() + "Z",
        )
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.id,
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "organization": self.organization,
            "architecture": self.architecture.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "context": self.context.to_dict(),
            "license": self.license.to_dict(),
            "quantizations": {
                name: quant.to_dict()
                for name, quant in self.quantizations.items()
            },
            "serving_requirements": self.serving_requirements,
            "sources": self.sources,
            "benchmarks": {
                name: score.to_dict()
                for name, score in self.benchmarks.items()
            },
            "benchmark_aggregates": self.benchmark_aggregates.to_dict(),
            "released_at": self.released_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create from MongoDB document."""
        model = cls(
            id=data.get("_id", ""),
            name=data.get("name", ""),
            family=data.get("family", ""),
            version=data.get("version", ""),
            organization=data.get("organization", ""),
            initialize_new=False,
        )
        
        # Restore nested objects
        model.architecture = Architecture.from_dict(data.get("architecture", {}))
        model.capabilities = Capabilities.from_dict(data.get("capabilities", {}))
        model.context = ContextWindow.from_dict(data.get("context", {}))
        model.license = License.from_dict(data.get("license", {}))
        
        # Restore quantizations
        model.quantizations = {}
        for name, quant_data in data.get("quantizations", {}).items():
            model.quantizations[name] = QuantizationOption.from_dict(quant_data)
        
        # Restore other fields
        model.serving_requirements = data.get("serving_requirements", {})
        model.sources = data.get("sources", {})
        
        # Restore benchmarks
        model.benchmarks = {}
        for name, score_data in data.get("benchmarks", {}).items():
            model.benchmarks[name] = BenchmarkScore.from_dict(score_data)
        
        model.benchmark_aggregates = BenchmarkAggregates.from_dict(
            data.get("benchmark_aggregates", {})
        )
        
        # Metadata
        model.released_at = data.get("released_at", "")
        model.created_at = data.get("created_at", "")
        model.updated_at = data.get("updated_at", "")
        
        return model
    
    def __repr__(self) -> str:
        return f"BaseModel(id='{self.id}', name='{self.name}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseModel):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)

