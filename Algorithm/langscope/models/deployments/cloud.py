"""
ModelDeployment class for cloud provider instances.

A deployment is a specific hosted version of a base model with its own
pricing, performance characteristics, and TrueSkill ratings.

Example: groq/llama-3.1-70b-versatile is Llama 3.1 70B served by Groq.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from langscope.core.rating import (
    TrueSkillRating,
    DualTrueSkill,
    MultiDimensionalTrueSkill,
)
from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0


class ProviderType(str, Enum):
    """Provider types."""
    CLOUD = "cloud"
    SELF_HOSTED = "self-hosted"
    EDGE = "edge"


class AvailabilityStatus(str, Enum):
    """Deployment availability status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BETA = "beta"
    LIMITED = "limited"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class Provider:
    """Provider information."""
    id: str
    name: str
    type: ProviderType = ProviderType.CLOUD
    api_base: str = ""
    api_compatible: str = "openai"  # API format
    website: str = ""
    docs: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "api_base": self.api_base,
            "api_compatible": self.api_compatible,
            "website": self.website,
            "docs": self.docs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Provider':
        """Create from dictionary."""
        provider_type = data.get("type", "cloud")
        if isinstance(provider_type, str):
            try:
                provider_type = ProviderType(provider_type)
            except ValueError:
                provider_type = ProviderType.CLOUD
        
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=provider_type,
            api_base=data.get("api_base", ""),
            api_compatible=data.get("api_compatible", "openai"),
            website=data.get("website", ""),
            docs=data.get("docs", ""),
        )


@dataclass
class DeploymentConfig:
    """Deployment configuration details."""
    model_id: str  # What to send in API calls
    display_name: str = ""
    quantization: str = ""
    serving_framework: str = ""
    max_context_length: int = 4096
    max_output_tokens: int = 4096
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "quantization": self.quantization,
            "serving_framework": self.serving_framework,
            "max_context_length": self.max_context_length,
            "max_output_tokens": self.max_output_tokens,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """Create from dictionary."""
        return cls(
            model_id=data.get("model_id", ""),
            display_name=data.get("display_name", ""),
            quantization=data.get("quantization", ""),
            serving_framework=data.get("serving_framework", ""),
            max_context_length=data.get("max_context_length", 4096),
            max_output_tokens=data.get("max_output_tokens", 4096),
            notes=data.get("notes", ""),
        )


@dataclass
class BatchPricing:
    """Batch API pricing (if available)."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    min_batch_size: int = 0
    max_latency_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "min_batch_size": self.min_batch_size,
            "max_latency_hours": self.max_latency_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchPricing':
        """Create from dictionary."""
        return cls(
            input_cost_per_million=data.get("input_cost_per_million", 0.0),
            output_cost_per_million=data.get("output_cost_per_million", 0.0),
            min_batch_size=data.get("min_batch_size", 0),
            max_latency_hours=data.get("max_latency_hours", 24),
        )


@dataclass
class FreeTier:
    """Free tier limits (if available)."""
    tokens_per_day: int = 0
    requests_per_minute: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_per_day": self.tokens_per_day,
            "requests_per_minute": self.requests_per_minute,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FreeTier':
        """Create from dictionary."""
        return cls(
            tokens_per_day=data.get("tokens_per_day", 0),
            requests_per_minute=data.get("requests_per_minute", 0),
        )


@dataclass
class Pricing:
    """Pricing information for a deployment."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    currency: str = "USD"
    
    # Optional pricing variants
    batch_pricing: Optional[BatchPricing] = None
    free_tier: Optional[FreeTier] = None
    
    # Source tracking
    source_id: str = ""
    source_url: str = ""
    last_verified: str = ""
    price_hash: str = ""  # For change detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "currency": self.currency,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "last_verified": self.last_verified,
            "price_hash": self.price_hash,
        }
        if self.batch_pricing:
            result["batch_pricing"] = self.batch_pricing.to_dict()
        if self.free_tier:
            result["free_tier"] = self.free_tier.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pricing':
        """Create from dictionary."""
        instance = cls(
            input_cost_per_million=data.get("input_cost_per_million", 0.0),
            output_cost_per_million=data.get("output_cost_per_million", 0.0),
            currency=data.get("currency", "USD"),
            source_id=data.get("source_id", ""),
            source_url=data.get("source_url", ""),
            last_verified=data.get("last_verified", ""),
            price_hash=data.get("price_hash", ""),
        )
        if "batch_pricing" in data and data["batch_pricing"]:
            instance.batch_pricing = BatchPricing.from_dict(data["batch_pricing"])
        if "free_tier" in data and data["free_tier"]:
            instance.free_tier = FreeTier.from_dict(data["free_tier"])
        return instance
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


@dataclass
class Performance:
    """Performance metrics for a deployment."""
    # Latency statistics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Time to First Token
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    
    # Reliability
    uptime_30d: float = 1.0  # 0-1 scale
    error_rate_30d: float = 0.0  # 0-1 scale
    
    # Measurement metadata
    measured_at: str = ""
    measurement_source: str = "automated"  # or "documented", "user-reported"
    measurement_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_ttft_ms": self.avg_ttft_ms,
            "p50_ttft_ms": self.p50_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "tokens_per_second": self.tokens_per_second,
            "uptime_30d": self.uptime_30d,
            "error_rate_30d": self.error_rate_30d,
            "measured_at": self.measured_at,
            "measurement_source": self.measurement_source,
            "measurement_count": self.measurement_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Performance':
        """Create from dictionary."""
        return cls(
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            p50_latency_ms=data.get("p50_latency_ms", 0.0),
            p95_latency_ms=data.get("p95_latency_ms", 0.0),
            p99_latency_ms=data.get("p99_latency_ms", 0.0),
            avg_ttft_ms=data.get("avg_ttft_ms", 0.0),
            p50_ttft_ms=data.get("p50_ttft_ms", 0.0),
            p95_ttft_ms=data.get("p95_ttft_ms", 0.0),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            uptime_30d=data.get("uptime_30d", 1.0),
            error_rate_30d=data.get("error_rate_30d", 0.0),
            measured_at=data.get("measured_at", ""),
            measurement_source=data.get("measurement_source", "automated"),
            measurement_count=data.get("measurement_count", 0),
        )
    
    def update_from_measurement(self, latency_ms: float, ttft_ms: float = None):
        """Update metrics with a new measurement."""
        n = self.measurement_count
        
        # Running average
        self.avg_latency_ms = (self.avg_latency_ms * n + latency_ms) / (n + 1)
        
        if ttft_ms is not None:
            self.avg_ttft_ms = (self.avg_ttft_ms * n + ttft_ms) / (n + 1)
        
        self.measurement_count = n + 1
        self.measured_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class RateLimits:
    """Rate limit configuration."""
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    tokens_per_day: int = 0
    concurrent_requests: int = 0
    
    # Tier-specific limits
    tiers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_day": self.tokens_per_day,
            "concurrent_requests": self.concurrent_requests,
            "tiers": self.tiers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RateLimits':
        """Create from dictionary."""
        return cls(
            requests_per_minute=data.get("requests_per_minute", 0),
            tokens_per_minute=data.get("tokens_per_minute", 0),
            tokens_per_day=data.get("tokens_per_day", 0),
            concurrent_requests=data.get("concurrent_requests", 0),
            tiers=data.get("tiers", {}),
        )


@dataclass
class Availability:
    """Deployment availability status."""
    status: AvailabilityStatus = AvailabilityStatus.ACTIVE
    regions: List[str] = field(default_factory=list)
    requires_waitlist: bool = False
    requires_enterprise: bool = False
    deprecation_date: Optional[str] = None
    deprecation_replacement: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "regions": self.regions,
            "requires_waitlist": self.requires_waitlist,
            "requires_enterprise": self.requires_enterprise,
            "deprecation_date": self.deprecation_date,
            "deprecation_replacement": self.deprecation_replacement,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Availability':
        """Create from dictionary."""
        status = data.get("status", "active")
        if isinstance(status, str):
            try:
                status = AvailabilityStatus(status)
            except ValueError:
                status = AvailabilityStatus.ACTIVE
        
        return cls(
            status=status,
            regions=data.get("regions", []),
            requires_waitlist=data.get("requires_waitlist", False),
            requires_enterprise=data.get("requires_enterprise", False),
            deprecation_date=data.get("deprecation_date"),
            deprecation_replacement=data.get("deprecation_replacement"),
        )
    
    @property
    def is_available(self) -> bool:
        """Check if deployment is currently available."""
        return self.status in (AvailabilityStatus.ACTIVE, AvailabilityStatus.BETA)


@dataclass
class PerformanceStats:
    """Match performance statistics."""
    total_matches_played: int = 0
    avg_rank_raw: float = 0.0
    avg_rank_cost: float = 0.0
    last_match: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_matches_played": self.total_matches_played,
            "avg_rank_raw": self.avg_rank_raw,
            "avg_rank_cost": self.avg_rank_cost,
            "last_match": self.last_match,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceStats':
        """Create from dictionary."""
        return cls(
            total_matches_played=data.get("total_matches_played", 0),
            avg_rank_raw=data.get("avg_rank_raw", 0.0),
            avg_rank_cost=data.get("avg_rank_cost", 0.0),
            last_match=data.get("last_match", ""),
        )


class ModelDeployment:
    """
    A specific hosted version of a base model.
    
    Each provider (Groq, Together, Fireworks, etc.) can have its own
    deployment of the same base model, with different:
    - Pricing
    - Performance (latency, throughput)
    - Quality (due to quantization)
    - TrueSkill ratings
    """
    
    def __init__(
        self,
        id: str,  # Format: provider/model-id (e.g., "groq/llama-3.1-70b-versatile")
        base_model_id: str,  # Link to BaseModel
        provider: Provider,
        initialize_new: bool = True
    ):
        """
        Initialize a model deployment.
        
        Args:
            id: Unique deployment identifier
            base_model_id: ID of the base model
            provider: Provider information
            initialize_new: Whether to initialize with default values
        """
        self.id = id
        self.base_model_id = base_model_id
        self.provider = provider
        
        if initialize_new:
            self.deployment = DeploymentConfig(model_id="")
            self.pricing = Pricing()
            self.performance = Performance()
            self.rate_limits = RateLimits()
            self.availability = Availability()
            
            # TrueSkill ratings for this deployment
            self.trueskill = DualTrueSkill()
            self.trueskill_by_domain: Dict[str, DualTrueSkill] = {}
            self.multi_trueskill = MultiDimensionalTrueSkill()
            self.multi_trueskill_by_domain: Dict[str, MultiDimensionalTrueSkill] = {}
            
            # Performance stats
            self.performance_stats = PerformanceStats()
            
            # Metadata
            self.created_at: str = datetime.utcnow().isoformat() + "Z"
            self.updated_at: str = datetime.utcnow().isoformat() + "Z"
    
    def get_domain_trueskill(self, domain: str) -> DualTrueSkill:
        """Get TrueSkill for a specific domain."""
        if domain not in self.trueskill_by_domain:
            self.trueskill_by_domain[domain] = DualTrueSkill()
        return self.trueskill_by_domain[domain]
    
    def get_domain_multi_trueskill(self, domain: str) -> MultiDimensionalTrueSkill:
        """Get 10D TrueSkill for a specific domain."""
        if domain not in self.multi_trueskill_by_domain:
            self.multi_trueskill_by_domain[domain] = MultiDimensionalTrueSkill()
        return self.multi_trueskill_by_domain[domain]
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        return self.pricing.calculate_cost(input_tokens, output_tokens)
    
    def update_performance(self, latency_ms: float, ttft_ms: float = None):
        """Update performance metrics from a measurement."""
        self.performance.update_from_measurement(latency_ms, ttft_ms)
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def is_available(self) -> bool:
        """Check if deployment is currently available."""
        return self.availability.is_available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.id,
            "base_model_id": self.base_model_id,
            "provider": self.provider.to_dict(),
            "deployment": self.deployment.to_dict(),
            "pricing": self.pricing.to_dict(),
            "performance": self.performance.to_dict(),
            "rate_limits": self.rate_limits.to_dict(),
            "availability": self.availability.to_dict(),
            "trueskill": self.trueskill.to_dict(),
            "trueskill_by_domain": {
                domain: ts.to_dict()
                for domain, ts in self.trueskill_by_domain.items()
            },
            "multi_trueskill": self.multi_trueskill.to_dict(),
            "multi_trueskill_by_domain": {
                domain: mts.to_dict()
                for domain, mts in self.multi_trueskill_by_domain.items()
            },
            "performance_stats": self.performance_stats.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelDeployment':
        """Create from MongoDB document."""
        deployment = cls(
            id=data.get("_id", ""),
            base_model_id=data.get("base_model_id", ""),
            provider=Provider.from_dict(data.get("provider", {})),
            initialize_new=False,
        )
        
        # Restore nested objects
        deployment.deployment = DeploymentConfig.from_dict(data.get("deployment", {}))
        deployment.pricing = Pricing.from_dict(data.get("pricing", {}))
        deployment.performance = Performance.from_dict(data.get("performance", {}))
        deployment.rate_limits = RateLimits.from_dict(data.get("rate_limits", {}))
        deployment.availability = Availability.from_dict(data.get("availability", {}))
        
        # Restore TrueSkill
        deployment.trueskill = DualTrueSkill.from_dict(data.get("trueskill", {}))
        deployment.trueskill_by_domain = {}
        for domain, ts_dict in data.get("trueskill_by_domain", {}).items():
            deployment.trueskill_by_domain[domain] = DualTrueSkill.from_dict(ts_dict)
        
        # Restore 10D TrueSkill
        if "multi_trueskill" in data:
            deployment.multi_trueskill = MultiDimensionalTrueSkill.from_dict(
                data.get("multi_trueskill", {})
            )
        else:
            deployment.multi_trueskill = MultiDimensionalTrueSkill.from_dual(
                deployment.trueskill
            )
        
        deployment.multi_trueskill_by_domain = {}
        for domain, mts_dict in data.get("multi_trueskill_by_domain", {}).items():
            deployment.multi_trueskill_by_domain[domain] = (
                MultiDimensionalTrueSkill.from_dict(mts_dict)
            )
        
        # Restore stats
        deployment.performance_stats = PerformanceStats.from_dict(
            data.get("performance_stats", {})
        )
        
        # Metadata
        deployment.created_at = data.get("created_at", "")
        deployment.updated_at = data.get("updated_at", "")
        
        return deployment
    
    def __repr__(self) -> str:
        return f"ModelDeployment(id='{self.id}', base='{self.base_model_id}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelDeployment):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)

