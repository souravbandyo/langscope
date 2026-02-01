"""
Benchmark results storage and retrieval.

Stores historical benchmark scores for base models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class BenchmarkResult:
    """
    A single benchmark result for a model.
    
    Stored as a time series to track score changes over time.
    """
    base_model_id: str
    benchmark_id: str
    score: float
    variant: str = ""  # e.g., "5-shot", "pass@1"
    percentile: Optional[int] = None
    evaluated_at: str = ""
    source_url: str = ""
    confidence: Optional[float] = None  # For uncertain/estimated scores
    
    # Category-specific breakdowns (e.g., Arena categories)
    categories: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "base_model_id": self.base_model_id,
            "benchmark_id": self.benchmark_id,
            "score": self.score,
            "variant": self.variant,
            "percentile": self.percentile,
            "evaluated_at": self.evaluated_at,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "categories": self.categories,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(
            base_model_id=data.get("base_model_id", ""),
            benchmark_id=data.get("benchmark_id", ""),
            score=data.get("score", 0.0),
            variant=data.get("variant", ""),
            percentile=data.get("percentile"),
            evaluated_at=data.get("evaluated_at", ""),
            source_url=data.get("source_url", ""),
            confidence=data.get("confidence"),
            categories=data.get("categories", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class BenchmarkAggregates:
    """
    Computed aggregate scores for quick comparisons.
    
    Stored at the base model level for fast access.
    """
    open_llm_average: float = 0.0
    knowledge_average: float = 0.0  # MMLU, GPQA, ARC
    reasoning_average: float = 0.0  # HellaSwag, WinoGrande
    coding_average: float = 0.0  # HumanEval, MBPP
    math_average: float = 0.0  # GSM8K, MATH
    chat_average: float = 0.0  # Arena, MT-Bench
    instruction_average: float = 0.0  # IFEval
    safety_average: float = 0.0  # TruthfulQA
    
    overall_rank: int = 0
    total_models_ranked: int = 0
    last_computed: str = ""
    
    def compute_from_scores(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float] = None
    ):
        """
        Compute aggregates from individual benchmark scores.
        
        Args:
            scores: Dictionary of benchmark_id -> score
            weights: Optional weights for each benchmark
        """
        # Default equal weights
        if weights is None:
            weights = {k: 1.0 for k in scores}
        
        # Category mappings
        knowledge = ["mmlu", "gpqa", "arc_challenge"]
        reasoning = ["hellaswag", "winogrande"]
        coding = ["humaneval", "mbpp"]
        math = ["gsm8k", "math"]
        chat = ["chatbot_arena", "mt_bench"]
        instruction = ["ifeval"]
        safety = ["truthfulqa"]
        
        def avg_category(benchmarks: List[str]) -> float:
            vals = [scores.get(b, 0) for b in benchmarks if b in scores]
            return sum(vals) / len(vals) if vals else 0.0
        
        self.knowledge_average = avg_category(knowledge)
        self.reasoning_average = avg_category(reasoning)
        self.coding_average = avg_category(coding)
        self.math_average = avg_category(math)
        self.chat_average = avg_category(chat)
        self.instruction_average = avg_category(instruction)
        self.safety_average = avg_category(safety)
        
        # Open LLM Leaderboard average (standard benchmarks)
        open_llm = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "truthfulqa", "gsm8k"]
        self.open_llm_average = avg_category(open_llm)
        
        self.last_computed = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "open_llm_average": self.open_llm_average,
            "knowledge_average": self.knowledge_average,
            "reasoning_average": self.reasoning_average,
            "coding_average": self.coding_average,
            "math_average": self.math_average,
            "chat_average": self.chat_average,
            "instruction_average": self.instruction_average,
            "safety_average": self.safety_average,
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
            instruction_average=data.get("instruction_average", 0.0),
            safety_average=data.get("safety_average", 0.0),
            overall_rank=data.get("overall_rank", 0),
            total_models_ranked=data.get("total_models_ranked", 0),
            last_computed=data.get("last_computed", ""),
        )

