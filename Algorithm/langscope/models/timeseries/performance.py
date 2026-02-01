"""
Performance metrics for tracking per-match performance.

Stored as MongoDB time series for detailed analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class PerformanceMetric:
    """
    Performance metrics for a single match participation.
    
    Stored in performance_metrics time series collection.
    """
    timestamp: str
    model_id: str  # Deployment ID
    domain: str
    match_id: str
    
    # Response metrics
    latency_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    
    # Rankings achieved
    raw_rank: int = 0
    cost_rank: int = 0
    
    # Dimension-specific rankings
    dimension_ranks: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    consistency_score: float = 0.0
    constraints_satisfied: int = 0
    total_constraints: int = 0
    hallucination_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "timestamp": self.timestamp,
            "metadata": {
                "model_id": self.model_id,
                "domain": self.domain,
                "match_id": self.match_id,
            },
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "raw_rank": self.raw_rank,
            "cost_rank": self.cost_rank,
            "dimension_ranks": self.dimension_ranks,
            "consistency_score": self.consistency_score,
            "constraints_satisfied": self.constraints_satisfied,
            "total_constraints": self.total_constraints,
            "hallucination_count": self.hallucination_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from MongoDB document."""
        metadata = data.get("metadata", {})
        
        return cls(
            timestamp=data.get("timestamp", ""),
            model_id=metadata.get("model_id", ""),
            domain=metadata.get("domain", ""),
            match_id=metadata.get("match_id", ""),
            latency_ms=data.get("latency_ms", 0.0),
            ttft_ms=data.get("ttft_ms", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            raw_rank=data.get("raw_rank", 0),
            cost_rank=data.get("cost_rank", 0),
            dimension_ranks=data.get("dimension_ranks", {}),
            consistency_score=data.get("consistency_score", 0.0),
            constraints_satisfied=data.get("constraints_satisfied", 0),
            total_constraints=data.get("total_constraints", 0),
            hallucination_count=data.get("hallucination_count", 0),
        )
    
    @classmethod
    def create_now(
        cls,
        model_id: str,
        domain: str,
        match_id: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        raw_rank: int,
        cost_rank: int
    ) -> 'PerformanceMetric':
        """Create a metric with current timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            domain=domain,
            match_id=match_id,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            raw_rank=raw_rank,
            cost_rank=cost_rank,
        )

