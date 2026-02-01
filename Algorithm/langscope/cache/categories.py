"""
Cache categories and configuration.

Defines cache layers, categories, and their settings.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class CacheLayer(Enum):
    """Available cache layers."""
    LOCAL = "local"
    REDIS = "redis"
    QDRANT = "qdrant"
    MONGO = "mongo"


class CacheCategory(Enum):
    """Cache categories with specific configurations."""
    SESSION = "session"
    LEADERBOARD = "leaderboard"
    RATE_LIMIT = "ratelimit"
    USER_PREF = "user_pref"
    PROMPT_EXACT = "prompt_exact"
    PROMPT_SEMANTIC = "prompt_semantic"
    CORRELATION = "correlation"
    SAMPLE_USAGE = "sample_usage"
    API_RESPONSE = "api_response"
    PARAM = "param"
    MODEL = "model"
    DOMAIN = "domain"
    CENTROID = "centroid"


@dataclass
class CacheCategoryConfig:
    """Configuration for a cache category."""
    
    # Time-to-live in seconds
    ttl: int
    
    # Which layers to use (in order of priority)
    layers: List[CacheLayer]
    
    # Whether to write to MongoDB for persistence
    write_through: bool
    
    # MongoDB collection name for persistent storage
    mongodb_collection: Optional[str] = None
    
    # Qdrant collection name for vector storage
    qdrant_collection: Optional[str] = None


# =============================================================================
# Category Configurations
# =============================================================================

CACHE_CONFIG: Dict[CacheCategory, CacheCategoryConfig] = {
    # Sessions: Local + Redis primary, MongoDB backup (survives Redis restart)
    CacheCategory.SESSION: CacheCategoryConfig(
        ttl=1800,  # 30 minutes
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="sessions",
    ),
    
    # Leaderboards: Local + Redis, periodic MongoDB snapshots
    CacheCategory.LEADERBOARD: CacheCategoryConfig(
        ttl=300,  # 5 minutes
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS],
        write_through=False,
        mongodb_collection="leaderboard_snapshots",
    ),
    
    # Rate limits: Redis only (ephemeral)
    CacheCategory.RATE_LIMIT: CacheCategoryConfig(
        ttl=60,  # 1 minute
        layers=[CacheLayer.REDIS],
        write_through=False,
    ),
    
    # User preferences: Local + Redis + MongoDB (permanent)
    CacheCategory.USER_PREF: CacheCategoryConfig(
        ttl=3600,  # 1 hour
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="user_preferences",
    ),
    
    # Exact prompt cache: Local + Redis + MongoDB
    CacheCategory.PROMPT_EXACT: CacheCategoryConfig(
        ttl=86400,  # 24 hours
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="prompt_cache",
    ),
    
    # Semantic prompt cache: Qdrant + MongoDB
    CacheCategory.PROMPT_SEMANTIC: CacheCategoryConfig(
        ttl=604800,  # 7 days
        layers=[CacheLayer.QDRANT, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="prompt_cache",
        qdrant_collection="prompt_cache",
    ),
    
    # Correlation cache: Local + Redis (can rebuild)
    CacheCategory.CORRELATION: CacheCategoryConfig(
        ttl=3600,  # 1 hour
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS],
        write_through=False,
    ),
    
    # Sample usage stats: Local + Redis + MongoDB (permanent)
    CacheCategory.SAMPLE_USAGE: CacheCategoryConfig(
        ttl=604800,  # 7 days
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="sample_usage",
    ),
    
    # API response cache: Local + Redis
    CacheCategory.API_RESPONSE: CacheCategoryConfig(
        ttl=300,  # 5 minutes
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS],
        write_through=False,
    ),
    
    # Parameters: Local only (from ParameterManager)
    CacheCategory.PARAM: CacheCategoryConfig(
        ttl=300,  # 5 minutes
        layers=[CacheLayer.LOCAL],
        write_through=False,
        mongodb_collection="params",
    ),
    
    # Models: Local only
    CacheCategory.MODEL: CacheCategoryConfig(
        ttl=60,  # 1 minute
        layers=[CacheLayer.LOCAL],
        write_through=False,
        mongodb_collection="models",
    ),
    
    # Domains: Local only
    CacheCategory.DOMAIN: CacheCategoryConfig(
        ttl=3600,  # 1 hour
        layers=[CacheLayer.LOCAL],
        write_through=False,
        mongodb_collection="domains",
    ),
    
    # Classifier centroids: All layers (expensive to compute)
    CacheCategory.CENTROID: CacheCategoryConfig(
        ttl=86400,  # 1 day
        layers=[CacheLayer.LOCAL, CacheLayer.REDIS, CacheLayer.MONGO],
        write_through=True,
        mongodb_collection="domain_centroids",
    ),
}


def get_category_config(category: CacheCategory) -> CacheCategoryConfig:
    """Get configuration for a cache category."""
    return CACHE_CONFIG[category]


def get_ttl(category: CacheCategory) -> int:
    """Get TTL for a cache category."""
    return CACHE_CONFIG[category].ttl

