"""
User Model - Private model registration and management.

Represents a user's private model deployment that can be evaluated
against public leaderboards. Supports multiple model types (LLM, ASR, TTS, VLM, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

from langscope.core.rating import TrueSkillRating, MultiDimensionalTrueSkill


class ModelType(str, Enum):
    """Supported model types."""
    LLM = "LLM"
    ASR = "ASR"
    TTS = "TTS"
    VLM = "VLM"
    V2V = "V2V"
    STT = "STT"
    IMAGE_GEN = "ImageGen"
    VIDEO_GEN = "VideoGen"
    EMBEDDING = "Embedding"
    RERANKER = "Reranker"


class APIFormat(str, Enum):
    """Supported API formats."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class ModelAPIConfig:
    """API configuration for connecting to the model."""
    endpoint: str
    model_id: str
    api_format: APIFormat = APIFormat.OPENAI
    api_key_hash: Optional[str] = None  # Store hash, not actual key
    has_api_key: bool = False
    headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "api_format": self.api_format.value if isinstance(self.api_format, APIFormat) else self.api_format,
            "api_key_hash": self.api_key_hash,
            "has_api_key": self.has_api_key,
            "headers": self.headers,
            "extra_params": self.extra_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelAPIConfig":
        api_format = data.get("api_format", "openai")
        if isinstance(api_format, str):
            try:
                api_format = APIFormat(api_format)
            except ValueError:
                api_format = APIFormat.CUSTOM
        return cls(
            endpoint=data.get("endpoint", ""),
            model_id=data.get("model_id", ""),
            api_format=api_format,
            api_key_hash=data.get("api_key_hash"),
            has_api_key=data.get("has_api_key", False),
            headers=data.get("headers", {}),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class ModelTypeConfig:
    """Type-specific configuration."""
    language: Optional[str] = None
    sample_rate: Optional[int] = None
    image_detail: Optional[str] = None
    image_size: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    embedding_dimension: Optional[int] = None
    normalize: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "language": self.language,
            "sample_rate": self.sample_rate,
            "image_detail": self.image_detail,
            "image_size": self.image_size,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "embedding_dimension": self.embedding_dimension,
            "normalize": self.normalize,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelTypeConfig":
        return cls(
            language=data.get("language"),
            sample_rate=data.get("sample_rate"),
            image_detail=data.get("image_detail"),
            image_size=data.get("image_size"),
            steps=data.get("steps"),
            guidance_scale=data.get("guidance_scale"),
            embedding_dimension=data.get("embedding_dimension"),
            normalize=data.get("normalize"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )


@dataclass
class ModelCosts:
    """Cost configuration for the model."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    currency: str = "USD"
    is_estimate: bool = True
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "currency": self.currency,
            "is_estimate": self.is_estimate,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCosts":
        return cls(
            input_cost_per_million=data.get("input_cost_per_million", 0.0),
            output_cost_per_million=data.get("output_cost_per_million", 0.0),
            currency=data.get("currency", "USD"),
            is_estimate=data.get("is_estimate", True),
            notes=data.get("notes"),
        )


@dataclass
class UserModel:
    """
    A user's private model registration.
    
    This represents a model deployment that a user wants to evaluate
    against public leaderboards. It stores API credentials securely
    and tracks evaluation results.
    """
    id: str
    user_id: str
    name: str
    model_type: ModelType
    version: str = "1.0"
    description: Optional[str] = None
    base_model_id: Optional[str] = None  # Link to public base model
    api_config: ModelAPIConfig = field(default_factory=lambda: ModelAPIConfig(endpoint="", model_id=""))
    type_config: ModelTypeConfig = field(default_factory=ModelTypeConfig)
    costs: ModelCosts = field(default_factory=ModelCosts)
    is_public: bool = False
    is_active: bool = True
    
    # TrueSkill ratings (for LLM/VLM)
    trueskill: Optional[MultiDimensionalTrueSkill] = None
    trueskill_by_domain: Dict[str, MultiDimensionalTrueSkill] = field(default_factory=dict)
    
    # Ground truth metrics (for ASR, TTS, etc.)
    ground_truth_metrics: Dict[str, float] = field(default_factory=dict)
    ground_truth_by_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Tracking
    total_evaluations: int = 0
    domains_evaluated: List[str] = field(default_factory=list)
    last_evaluated_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    @classmethod
    def create(
        cls,
        user_id: str,
        name: str,
        model_type: ModelType,
        **kwargs
    ) -> "UserModel":
        """Create a new UserModel with generated ID."""
        model_id = f"um_{uuid.uuid4().hex[:12]}"
        return cls(
            id=model_id,
            user_id=user_id,
            name=name,
            model_type=model_type,
            **kwargs
        )
    
    def is_owned_by(self, user_id: str) -> bool:
        """Check if model is owned by user."""
        return self.user_id == user_id
    
    def is_visible_to(self, user_id: str) -> bool:
        """Check if model is visible to user."""
        return self.is_public or self.user_id == user_id
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def record_evaluation(self, domain: str):
        """Record that an evaluation was performed."""
        self.total_evaluations += 1
        if domain not in self.domains_evaluated:
            self.domains_evaluated.append(domain)
        self.last_evaluated_at = datetime.utcnow().isoformat() + "Z"
        self.update_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = {
            "_id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "model_type": self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            "version": self.version,
            "description": self.description,
            "base_model_id": self.base_model_id,
            "api_config": self.api_config.to_dict(),
            "type_config": self.type_config.to_dict(),
            "costs": self.costs.to_dict(),
            "is_public": self.is_public,
            "is_active": self.is_active,
            "ground_truth_metrics": self.ground_truth_metrics,
            "ground_truth_by_domain": self.ground_truth_by_domain,
            "total_evaluations": self.total_evaluations,
            "domains_evaluated": self.domains_evaluated,
            "last_evaluated_at": self.last_evaluated_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        # Add TrueSkill if present
        if self.trueskill:
            data["trueskill"] = self.trueskill.to_dict()
        
        if self.trueskill_by_domain:
            data["trueskill_by_domain"] = {
                domain: rating.to_dict() 
                for domain, rating in self.trueskill_by_domain.items()
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
        """Create from dictionary."""
        model_type = data.get("model_type", "LLM")
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                model_type = ModelType.LLM
        
        # Parse TrueSkill
        trueskill = None
        if "trueskill" in data and data["trueskill"]:
            trueskill = MultiDimensionalTrueSkill.from_dict(data["trueskill"])
        
        trueskill_by_domain = {}
        if "trueskill_by_domain" in data:
            for domain, rating_data in data.get("trueskill_by_domain", {}).items():
                trueskill_by_domain[domain] = MultiDimensionalTrueSkill.from_dict(rating_data)
        
        return cls(
            id=data.get("_id", data.get("id", "")),
            user_id=data.get("user_id", ""),
            name=data.get("name", ""),
            model_type=model_type,
            version=data.get("version", "1.0"),
            description=data.get("description"),
            base_model_id=data.get("base_model_id"),
            api_config=ModelAPIConfig.from_dict(data.get("api_config", {})),
            type_config=ModelTypeConfig.from_dict(data.get("type_config", {})),
            costs=ModelCosts.from_dict(data.get("costs", {})),
            is_public=data.get("is_public", False),
            is_active=data.get("is_active", True),
            trueskill=trueskill,
            trueskill_by_domain=trueskill_by_domain,
            ground_truth_metrics=data.get("ground_truth_metrics", {}),
            ground_truth_by_domain=data.get("ground_truth_by_domain", {}),
            total_evaluations=data.get("total_evaluations", 0),
            domains_evaluated=data.get("domains_evaluated", []),
            last_evaluated_at=data.get("last_evaluated_at"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
        )
    
    def get_public_response(self) -> Dict[str, Any]:
        """Get a public-safe version (no API keys)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "model_type": self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            "version": self.version,
            "description": self.description,
            "base_model_id": self.base_model_id,
            "api_config": {
                "endpoint": self.api_config.endpoint,
                "model_id": self.api_config.model_id,
                "api_format": self.api_config.api_format.value if isinstance(self.api_config.api_format, APIFormat) else self.api_config.api_format,
                "has_api_key": self.api_config.has_api_key,
            },
            "type_config": self.type_config.to_dict(),
            "costs": self.costs.to_dict(),
            "is_public": self.is_public,
            "is_active": self.is_active,
            "total_evaluations": self.total_evaluations,
            "domains_evaluated": self.domains_evaluated,
            "last_evaluated_at": self.last_evaluated_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
