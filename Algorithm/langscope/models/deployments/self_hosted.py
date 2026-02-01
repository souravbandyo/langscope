"""
SelfHostedDeployment class for user-owned infrastructure.

Users can register their own deployments with custom hardware
configurations and cost calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from langscope.core.rating import (
    DualTrueSkill,
    MultiDimensionalTrueSkill,
)
from langscope.models.deployments.cloud import (
    Provider,
    ProviderType,
    AvailabilityStatus,
    PerformanceStats,
)


class CloudProvider(str, Enum):
    """Cloud infrastructure providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREM = "on-prem"
    OTHER = "other"


class ServingFramework(str, Enum):
    """Serving frameworks."""
    VLLM = "vllm"
    TGI = "tgi"  # Text Generation Inference
    LLAMA_CPP = "llama.cpp"
    OLLAMA = "ollama"
    CT2 = "ctranslate2"
    TRITON = "triton"
    OTHER = "other"


class CostCalculationMethod(str, Enum):
    """How per-token costs were calculated."""
    USER_PROVIDED = "user_provided"
    ESTIMATED = "estimated"


@dataclass
class HardwareConfig:
    """Hardware configuration for self-hosted deployment."""
    # GPU configuration
    gpu_type: str = ""  # e.g., "A100-80GB", "H100"
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0  # Total VRAM
    
    # CPU/RAM
    cpu_cores: int = 0
    ram_gb: float = 0.0
    
    # Infrastructure
    cloud_provider: CloudProvider = CloudProvider.OTHER
    instance_type: str = ""  # e.g., "p4d.24xlarge"
    region: str = ""
    
    # For on-prem
    datacenter: str = ""
    rack: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "cloud_provider": (
                self.cloud_provider.value
                if isinstance(self.cloud_provider, Enum)
                else self.cloud_provider
            ),
            "instance_type": self.instance_type,
            "region": self.region,
            "datacenter": self.datacenter,
            "rack": self.rack,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareConfig':
        """Create from dictionary."""
        cloud = data.get("cloud_provider", "other")
        if isinstance(cloud, str):
            try:
                cloud = CloudProvider(cloud)
            except ValueError:
                cloud = CloudProvider.OTHER
        
        return cls(
            gpu_type=data.get("gpu_type", ""),
            gpu_count=data.get("gpu_count", 0),
            gpu_memory_gb=data.get("gpu_memory_gb", 0.0),
            cpu_cores=data.get("cpu_cores", 0),
            ram_gb=data.get("ram_gb", 0.0),
            cloud_provider=cloud,
            instance_type=data.get("instance_type", ""),
            region=data.get("region", ""),
            datacenter=data.get("datacenter", ""),
            rack=data.get("rack", ""),
        )


@dataclass
class SoftwareConfig:
    """Software stack configuration."""
    serving_framework: ServingFramework = ServingFramework.OTHER
    framework_version: str = ""
    
    # Quantization
    quantization: str = ""  # e.g., "awq-4bit", "bf16"
    quantization_source: str = ""  # HuggingFace repo
    
    # Framework-specific settings (stored as dict for flexibility)
    framework_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Common settings
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "serving_framework": (
                self.serving_framework.value
                if isinstance(self.serving_framework, Enum)
                else self.serving_framework
            ),
            "framework_version": self.framework_version,
            "quantization": self.quantization,
            "quantization_source": self.quantization_source,
            "framework_settings": self.framework_settings,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoftwareConfig':
        """Create from dictionary."""
        framework = data.get("serving_framework", "other")
        if isinstance(framework, str):
            try:
                framework = ServingFramework(framework)
            except ValueError:
                framework = ServingFramework.OTHER
        
        return cls(
            serving_framework=framework,
            framework_version=data.get("framework_version", ""),
            quantization=data.get("quantization", ""),
            quantization_source=data.get("quantization_source", ""),
            framework_settings=data.get("framework_settings", {}),
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            max_model_len=data.get("max_model_len", 4096),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
        )


@dataclass
class CostCalculation:
    """Details of how costs were calculated."""
    method: CostCalculationMethod = CostCalculationMethod.USER_PROVIDED
    
    # If estimated, these are the assumptions
    assumed_utilization: float = 0.7  # 70% of time generating
    assumed_batch_size: int = 16
    assumed_throughput_tps: float = 500.0  # tokens/second
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": (
                self.method.value
                if isinstance(self.method, Enum)
                else self.method
            ),
            "assumed_utilization": self.assumed_utilization,
            "assumed_batch_size": self.assumed_batch_size,
            "assumed_throughput_tps": self.assumed_throughput_tps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostCalculation':
        """Create from dictionary."""
        method = data.get("method", "user_provided")
        if isinstance(method, str):
            try:
                method = CostCalculationMethod(method)
            except ValueError:
                method = CostCalculationMethod.USER_PROVIDED
        
        return cls(
            method=method,
            assumed_utilization=data.get("assumed_utilization", 0.7),
            assumed_batch_size=data.get("assumed_batch_size", 16),
            assumed_throughput_tps=data.get("assumed_throughput_tps", 500.0),
        )


@dataclass
class MonthlyFixedCosts:
    """Fixed monthly costs for self-hosted deployment."""
    storage: float = 0.0
    network: float = 0.0
    monitoring: float = 0.0
    other: float = 0.0
    
    @property
    def total(self) -> float:
        """Total monthly fixed costs."""
        return self.storage + self.network + self.monitoring + self.other
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage": self.storage,
            "network": self.network,
            "monitoring": self.monitoring,
            "other": self.other,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonthlyFixedCosts':
        """Create from dictionary."""
        return cls(
            storage=data.get("storage", 0.0),
            network=data.get("network", 0.0),
            monitoring=data.get("monitoring", 0.0),
            other=data.get("other", 0.0),
        )


@dataclass
class SelfHostedCosts:
    """Cost configuration for self-hosted deployment."""
    # Direct per-token costs (preferred if known)
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    
    # Compute-based calculation
    hourly_compute_cost: float = 0.0
    
    # How costs were calculated
    calculation: CostCalculation = field(default_factory=CostCalculation)
    
    # Fixed costs
    monthly_fixed_costs: MonthlyFixedCosts = field(default_factory=MonthlyFixedCosts)
    
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "hourly_compute_cost": self.hourly_compute_cost,
            "calculation": self.calculation.to_dict(),
            "monthly_fixed_costs": self.monthly_fixed_costs.to_dict(),
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfHostedCosts':
        """Create from dictionary."""
        return cls(
            input_cost_per_million=data.get("input_cost_per_million", 0.0),
            output_cost_per_million=data.get("output_cost_per_million", 0.0),
            hourly_compute_cost=data.get("hourly_compute_cost", 0.0),
            calculation=CostCalculation.from_dict(data.get("calculation", {})),
            monthly_fixed_costs=MonthlyFixedCosts.from_dict(
                data.get("monthly_fixed_costs", {})
            ),
            notes=data.get("notes", ""),
        )
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost
    
    def estimate_costs_from_hourly(
        self,
        hourly_cost: float,
        throughput_tps: float,
        utilization: float = 0.7,
        input_output_ratio: float = 3.0  # 3 input tokens per output token
    ) -> tuple:
        """
        Estimate per-token costs from hourly compute cost.
        
        Args:
            hourly_cost: Cost per hour in USD
            throughput_tps: Output tokens per second
            utilization: Fraction of time actually generating
            input_output_ratio: Ratio of input to output tokens
        
        Returns:
            Tuple of (input_cost_per_million, output_cost_per_million)
        """
        if throughput_tps <= 0 or utilization <= 0:
            return (0.0, 0.0)
        
        # Tokens per hour (output only)
        output_tokens_per_hour = throughput_tps * 3600 * utilization
        
        # Input tokens per hour (estimated)
        input_tokens_per_hour = output_tokens_per_hour * input_output_ratio
        
        # Total tokens
        total_tokens_per_hour = input_tokens_per_hour + output_tokens_per_hour
        
        # Cost per token (average)
        if total_tokens_per_hour > 0:
            cost_per_token = hourly_cost / total_tokens_per_hour
        else:
            cost_per_token = 0.0
        
        # Assume output tokens are 3x more expensive than input
        output_multiplier = 3.0
        
        # Calculate weighted costs
        input_weight = input_output_ratio / (input_output_ratio + output_multiplier)
        output_weight = 1 - input_weight
        
        input_cost_per_million = cost_per_token * 1_000_000 * input_weight * 0.5
        output_cost_per_million = cost_per_token * 1_000_000 * output_weight * 1.5
        
        return (input_cost_per_million, output_cost_per_million)


@dataclass
class SelfHostedPerformance:
    """Performance for self-hosted deployment."""
    # User-specified estimates
    expected_latency_ms: float = 0.0
    expected_ttft_ms: float = 0.0
    expected_tokens_per_second: float = 0.0
    
    # Measured from actual usage
    measured_latency_ms: float = 0.0
    measured_ttft_ms: float = 0.0
    measured_throughput: float = 0.0
    measurement_count: int = 0
    
    last_measured: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expected_latency_ms": self.expected_latency_ms,
            "expected_ttft_ms": self.expected_ttft_ms,
            "expected_tokens_per_second": self.expected_tokens_per_second,
            "measured_latency_ms": self.measured_latency_ms,
            "measured_ttft_ms": self.measured_ttft_ms,
            "measured_throughput": self.measured_throughput,
            "measurement_count": self.measurement_count,
            "last_measured": self.last_measured,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfHostedPerformance':
        """Create from dictionary."""
        return cls(
            expected_latency_ms=data.get("expected_latency_ms", 0.0),
            expected_ttft_ms=data.get("expected_ttft_ms", 0.0),
            expected_tokens_per_second=data.get("expected_tokens_per_second", 0.0),
            measured_latency_ms=data.get("measured_latency_ms", 0.0),
            measured_ttft_ms=data.get("measured_ttft_ms", 0.0),
            measured_throughput=data.get("measured_throughput", 0.0),
            measurement_count=data.get("measurement_count", 0),
            last_measured=data.get("last_measured", ""),
        )
    
    def update_from_measurement(
        self,
        latency_ms: float,
        ttft_ms: float = None,
        throughput: float = None
    ):
        """Update with a new measurement."""
        n = self.measurement_count
        
        self.measured_latency_ms = (
            (self.measured_latency_ms * n + latency_ms) / (n + 1)
        )
        
        if ttft_ms is not None:
            self.measured_ttft_ms = (
                (self.measured_ttft_ms * n + ttft_ms) / (n + 1)
            )
        
        if throughput is not None:
            self.measured_throughput = (
                (self.measured_throughput * n + throughput) / (n + 1)
            )
        
        self.measurement_count = n + 1
        self.last_measured = datetime.utcnow().isoformat() + "Z"


@dataclass
class SelfHostedAvailability:
    """Availability configuration for self-hosted deployment."""
    status: AvailabilityStatus = AvailabilityStatus.ACTIVE
    schedule: str = "24/7"  # or "business_hours", cron expression
    
    # Health checking
    health_check_url: str = ""
    last_health_check: str = ""
    is_healthy: bool = True
    
    # Availability hours (if not 24/7)
    timezone: str = "UTC"
    availability_hours: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": (
                self.status.value
                if isinstance(self.status, Enum)
                else self.status
            ),
            "schedule": self.schedule,
            "health_check_url": self.health_check_url,
            "last_health_check": self.last_health_check,
            "is_healthy": self.is_healthy,
            "timezone": self.timezone,
            "availability_hours": self.availability_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfHostedAvailability':
        """Create from dictionary."""
        status = data.get("status", "active")
        if isinstance(status, str):
            try:
                status = AvailabilityStatus(status)
            except ValueError:
                status = AvailabilityStatus.ACTIVE
        
        return cls(
            status=status,
            schedule=data.get("schedule", "24/7"),
            health_check_url=data.get("health_check_url", ""),
            last_health_check=data.get("last_health_check", ""),
            is_healthy=data.get("is_healthy", True),
            timezone=data.get("timezone", "UTC"),
            availability_hours=data.get("availability_hours", {}),
        )


@dataclass
class Owner:
    """Self-hosted deployment owner information."""
    user_id: str
    organization: str = ""
    email: str = ""
    is_public: bool = False  # Visibility to other users
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "organization": self.organization,
            "email": self.email,
            "is_public": self.is_public,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Owner':
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            organization=data.get("organization", ""),
            email=data.get("email", ""),
            is_public=data.get("is_public", False),
        )


class SelfHostedDeployment:
    """
    A user's own deployment of a model.
    
    Users can specify their hardware, software stack, and compute
    their own per-token costs based on their infrastructure.
    """
    
    def __init__(
        self,
        id: str,  # Format: user_id/deployment_name
        base_model_id: str,
        owner: Owner,
        initialize_new: bool = True
    ):
        """
        Initialize a self-hosted deployment.
        
        Args:
            id: Unique deployment identifier
            base_model_id: ID of the base model
            owner: Owner information
            initialize_new: Whether to initialize with default values
        """
        self.id = id
        self.base_model_id = base_model_id
        self.owner = owner
        
        if initialize_new:
            # Provider is always self-hosted
            self.provider = Provider(
                id="self-hosted",
                name="Self-Hosted",
                type=ProviderType.SELF_HOSTED,
            )
            
            self.hardware = HardwareConfig()
            self.software = SoftwareConfig()
            self.costs = SelfHostedCosts()
            self.performance = SelfHostedPerformance()
            self.availability = SelfHostedAvailability()
            
            # TrueSkill ratings
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
        return self.costs.calculate_cost(input_tokens, output_tokens)
    
    def estimate_costs_from_hourly(self):
        """
        Estimate per-token costs from hourly compute cost.
        
        Uses the expected throughput to calculate.
        """
        if self.costs.hourly_compute_cost > 0:
            input_cost, output_cost = self.costs.estimate_costs_from_hourly(
                self.costs.hourly_compute_cost,
                self.performance.expected_tokens_per_second,
                self.costs.calculation.assumed_utilization,
            )
            self.costs.input_cost_per_million = input_cost
            self.costs.output_cost_per_million = output_cost
            self.costs.calculation.method = CostCalculationMethod.ESTIMATED
            self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def is_visible_to(self, user_id: str) -> bool:
        """Check if deployment is visible to a user."""
        return self.owner.user_id == user_id or self.owner.is_public
    
    def is_owned_by(self, user_id: str) -> bool:
        """Check if deployment is owned by a user."""
        return self.owner.user_id == user_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.id,
            "base_model_id": self.base_model_id,
            "owner": self.owner.to_dict(),
            "provider": self.provider.to_dict(),
            "hardware": self.hardware.to_dict(),
            "software": self.software.to_dict(),
            "costs": self.costs.to_dict(),
            "performance": self.performance.to_dict(),
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
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfHostedDeployment':
        """Create from MongoDB document."""
        deployment = cls(
            id=data.get("_id", ""),
            base_model_id=data.get("base_model_id", ""),
            owner=Owner.from_dict(data.get("owner", {})),
            initialize_new=False,
        )
        
        # Restore nested objects
        deployment.provider = Provider.from_dict(data.get("provider", {}))
        deployment.hardware = HardwareConfig.from_dict(data.get("hardware", {}))
        deployment.software = SoftwareConfig.from_dict(data.get("software", {}))
        deployment.costs = SelfHostedCosts.from_dict(data.get("costs", {}))
        deployment.performance = SelfHostedPerformance.from_dict(
            data.get("performance", {})
        )
        deployment.availability = SelfHostedAvailability.from_dict(
            data.get("availability", {})
        )
        
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
        return (
            f"SelfHostedDeployment(id='{self.id}', "
            f"base='{self.base_model_id}', "
            f"owner='{self.owner.user_id}')"
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelfHostedDeployment):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)

