"""
LangScope Cache Management Module.

Provides unified caching, session management, rate limiting, and semantic cache.
"""

from langscope.cache.categories import (
    CacheLayer,
    CacheCategory,
    CacheCategoryConfig,
    CACHE_CONFIG,
)
from langscope.cache.manager import UnifiedCacheManager
from langscope.cache.session import Session, SessionManager
from langscope.cache.rate_limit import SlidingWindowRateLimiter
from langscope.cache.recovery import CacheRecovery

__all__ = [
    # Categories
    "CacheLayer",
    "CacheCategory",
    "CacheCategoryConfig",
    "CACHE_CONFIG",
    # Manager
    "UnifiedCacheManager",
    # Session
    "Session",
    "SessionManager",
    # Rate Limiting
    "SlidingWindowRateLimiter",
    # Recovery
    "CacheRecovery",
]

