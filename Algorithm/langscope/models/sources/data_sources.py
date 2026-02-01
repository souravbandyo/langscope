"""
Data source configuration for automated syncing.

Configures external APIs and sources for pricing, models, and benchmarks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class DataSourceType(str, Enum):
    """Types of data sources."""
    PRICING = "pricing"
    MODELS = "models"
    BENCHMARKS = "benchmarks"
    CAPABILITIES = "capabilities"


class SourceMethod(str, Enum):
    """HTTP methods for source requests."""
    GET = "GET"
    POST = "POST"


@dataclass
class SourceConfig:
    """Configuration for how to fetch data from a source."""
    type: str = "api"  # "api", "webpage", "github", "rss"
    url: str = ""
    method: SourceMethod = SourceMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    
    # Response parsing
    response_format: str = "json"  # "json", "csv", "html"
    data_path: str = ""  # JSON path to data (e.g., "data.models")
    
    # Authentication
    auth_type: str = ""  # "bearer", "api_key", "none"
    auth_env_var: str = ""  # Environment variable with auth token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "url": self.url,
            "method": self.method.value if isinstance(self.method, Enum) else self.method,
            "headers": self.headers,
            "params": self.params,
            "response_format": self.response_format,
            "data_path": self.data_path,
            "auth_type": self.auth_type,
            "auth_env_var": self.auth_env_var,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceConfig':
        """Create from dictionary."""
        method = data.get("method", "GET")
        if isinstance(method, str):
            try:
                method = SourceMethod(method)
            except ValueError:
                method = SourceMethod.GET
        
        return cls(
            type=data.get("type", "api"),
            url=data.get("url", ""),
            method=method,
            headers=data.get("headers", {}),
            params=data.get("params", {}),
            response_format=data.get("response_format", "json"),
            data_path=data.get("data_path", ""),
            auth_type=data.get("auth_type", ""),
            auth_env_var=data.get("auth_env_var", ""),
        )


@dataclass
class AutomationConfig:
    """Automation schedule and settings."""
    schedule: str = "0 */6 * * *"  # Cron expression (every 6 hours)
    enabled: bool = True
    retry_count: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30
    alert_on_failure: bool = True
    alert_email: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule": self.schedule,
            "enabled": self.enabled,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "alert_on_failure": self.alert_on_failure,
            "alert_email": self.alert_email,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomationConfig':
        """Create from dictionary."""
        return cls(
            schedule=data.get("schedule", "0 */6 * * *"),
            enabled=data.get("enabled", True),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            timeout_seconds=data.get("timeout_seconds", 30),
            alert_on_failure=data.get("alert_on_failure", True),
            alert_email=data.get("alert_email", ""),
        )


@dataclass
class ValidationConfig:
    """Configuration for validating changes."""
    max_price_change_pct: float = 50.0  # Max % change before requiring approval
    require_approval_for_new: bool = False
    require_approval_for_delete: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_price_change_pct": self.max_price_change_pct,
            "require_approval_for_new": self.require_approval_for_new,
            "require_approval_for_delete": self.require_approval_for_delete,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        """Create from dictionary."""
        return cls(
            max_price_change_pct=data.get("max_price_change_pct", 50.0),
            require_approval_for_new=data.get("require_approval_for_new", False),
            require_approval_for_delete=data.get("require_approval_for_delete", True),
        )


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for a data source."""
    uptime_30d: float = 1.0  # 0-1 scale
    trust_score: float = 1.0  # 0-1 scale
    total_syncs: int = 0
    successful_syncs: int = 0
    last_success: str = ""
    last_failure: str = ""
    last_failure_reason: str = ""
    
    def update_success(self):
        """Record a successful sync."""
        self.total_syncs += 1
        self.successful_syncs += 1
        self.last_success = datetime.utcnow().isoformat() + "Z"
        self._update_uptime()
    
    def update_failure(self, reason: str):
        """Record a failed sync."""
        self.total_syncs += 1
        self.last_failure = datetime.utcnow().isoformat() + "Z"
        self.last_failure_reason = reason
        self._update_uptime()
    
    def _update_uptime(self):
        """Update uptime calculation."""
        if self.total_syncs > 0:
            self.uptime_30d = self.successful_syncs / self.total_syncs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uptime_30d": self.uptime_30d,
            "trust_score": self.trust_score,
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "last_failure_reason": self.last_failure_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReliabilityMetrics':
        """Create from dictionary."""
        return cls(
            uptime_30d=data.get("uptime_30d", 1.0),
            trust_score=data.get("trust_score", 1.0),
            total_syncs=data.get("total_syncs", 0),
            successful_syncs=data.get("successful_syncs", 0),
            last_success=data.get("last_success", ""),
            last_failure=data.get("last_failure", ""),
            last_failure_reason=data.get("last_failure_reason", ""),
        )


class DataSource:
    """
    Configuration for an external data source.
    
    Supports pricing APIs (OpenRouter, LiteLLM), benchmark leaderboards
    (Open LLM, LMSYS Arena), and model information sources.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str = "",
        source_type: DataSourceType = DataSourceType.PRICING,
        provider: str = "",
        initialize_new: bool = True
    ):
        """
        Initialize a data source.
        
        Args:
            id: Unique identifier
            name: Human-readable name
            description: Description of the source
            source_type: Type of data this source provides
            provider: Provider this source covers (if applicable)
            initialize_new: Whether to initialize with defaults
        """
        self.id = id
        self.name = name
        self.description = description
        self.source_type = source_type
        self.provider = provider
        
        if initialize_new:
            self.source = SourceConfig()
            self.automation = AutomationConfig()
            self.validation = ValidationConfig()
            self.reliability = ReliabilityMetrics()
            
            # Field mappings for parsing response
            self.field_mappings: Dict[str, str] = {}
            
            # Metadata
            self.created_at = datetime.utcnow().isoformat() + "Z"
            self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def is_enabled(self) -> bool:
        """Check if source is enabled for automation."""
        return self.automation.enabled
    
    def is_reliable(self, threshold: float = 0.9) -> bool:
        """Check if source is reliable enough to use."""
        return self.reliability.uptime_30d >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.id,
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value if isinstance(self.source_type, Enum) else self.source_type,
            "provider": self.provider,
            "source": self.source.to_dict(),
            "automation": self.automation.to_dict(),
            "validation": self.validation.to_dict(),
            "reliability": self.reliability.to_dict(),
            "field_mappings": self.field_mappings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Create from MongoDB document."""
        source_type = data.get("source_type", "pricing")
        if isinstance(source_type, str):
            try:
                source_type = DataSourceType(source_type)
            except ValueError:
                source_type = DataSourceType.PRICING
        
        ds = cls(
            id=data.get("_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            source_type=source_type,
            provider=data.get("provider", ""),
            initialize_new=False,
        )
        
        ds.source = SourceConfig.from_dict(data.get("source", {}))
        ds.automation = AutomationConfig.from_dict(data.get("automation", {}))
        ds.validation = ValidationConfig.from_dict(data.get("validation", {}))
        ds.reliability = ReliabilityMetrics.from_dict(data.get("reliability", {}))
        ds.field_mappings = data.get("field_mappings", {})
        ds.created_at = data.get("created_at", "")
        ds.updated_at = data.get("updated_at", "")
        
        return ds


# =============================================================================
# Predefined Data Sources
# =============================================================================

def _create_openrouter_source() -> DataSource:
    """Create OpenRouter API data source."""
    ds = DataSource(
        id="openrouter_api",
        name="OpenRouter API",
        description="OpenRouter aggregates many models with unified API and pricing",
        source_type=DataSourceType.PRICING,
        provider="openrouter",
    )
    ds.source = SourceConfig(
        type="api",
        url="https://openrouter.ai/api/v1/models",
        method=SourceMethod.GET,
        response_format="json",
        data_path="data",
    )
    ds.automation = AutomationConfig(
        schedule="0 */6 * * *",  # Every 6 hours
        enabled=True,
    )
    ds.field_mappings = {
        "model_id": "id",
        "name": "name",
        "input_cost": "pricing.prompt",
        "output_cost": "pricing.completion",
        "context_length": "context_length",
    }
    return ds


def _create_open_llm_leaderboard_source() -> DataSource:
    """Create Open LLM Leaderboard data source."""
    ds = DataSource(
        id="open_llm_leaderboard",
        name="Open LLM Leaderboard",
        description="HuggingFace Open LLM Leaderboard scores",
        source_type=DataSourceType.BENCHMARKS,
    )
    ds.source = SourceConfig(
        type="api",
        url="https://huggingface.co/api/spaces/open-llm-leaderboard/open_llm_leaderboard",
        method=SourceMethod.GET,
        response_format="json",
    )
    ds.automation = AutomationConfig(
        schedule="0 0 * * 0",  # Weekly on Sunday
        enabled=True,
    )
    return ds


def _create_lmsys_arena_source() -> DataSource:
    """Create LMSYS Arena data source."""
    ds = DataSource(
        id="lmsys_arena",
        name="LMSYS Chatbot Arena",
        description="LMSYS Arena Elo ratings from human preferences",
        source_type=DataSourceType.BENCHMARKS,
    )
    ds.source = SourceConfig(
        type="api",
        url="https://chat.lmsys.org/api/models",
        method=SourceMethod.GET,
        response_format="json",
    )
    ds.automation = AutomationConfig(
        schedule="0 0 * * *",  # Daily
        enabled=True,
    )
    return ds


PREDEFINED_SOURCES: Dict[str, DataSource] = {
    "openrouter_api": _create_openrouter_source(),
    "open_llm_leaderboard": _create_open_llm_leaderboard_source(),
    "lmsys_arena": _create_lmsys_arena_source(),
}

