"""
Rate limiting for LangScope API.

Implements sliding window rate limiting using Redis.
"""

import time
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limit Tiers
# =============================================================================

RATE_LIMIT_TIERS: Dict[str, Dict[str, int]] = {
    # Endpoint pattern -> {limit, window}
    "/api/v1/leaderboard": {"limit": 200, "window": 60},
    "/api/v1/models": {"limit": 100, "window": 60},
    "/api/v1/arena": {"limit": 10, "window": 60},
    "/api/v1/ground-truth/evaluate": {"limit": 5, "window": 60},
    "/api/v1/cache/invalidate": {"limit": 10, "window": 60},
    "/api/v1/prompts/process": {"limit": 50, "window": 60},
    "default": {"limit": 100, "window": 60},
}


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    remaining: int
    limit: int
    window: int
    retry_after: int = 0
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Window": str(self.window),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after)
        return headers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "limit": self.limit,
            "window": self.window,
            "retry_after": self.retry_after,
        }


class SlidingWindowRateLimiter:
    """
    Redis-based sliding window rate limiting.
    
    Uses sorted sets for precise counting within time windows.
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Async Redis client
        """
        self._redis = redis_client
    
    @property
    def redis(self):
        """Get Redis client."""
        return self._redis
    
    @redis.setter
    def redis(self, client):
        """Set Redis client."""
        self._redis = client
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "default",
        limit: int = None,
        window: int = None,
    ) -> RateLimitResult:
        """
        Check if request is within rate limit.
        
        Uses sliding window algorithm with Redis sorted sets.
        
        Args:
            identifier: Unique identifier (IP, API key, user ID)
            endpoint: Endpoint being accessed
            limit: Override limit (uses tier default if None)
            window: Override window (uses tier default if None)
        
        Returns:
            RateLimitResult with allowed status and metadata
        """
        # Get limits from tier configuration
        tier = self._get_tier_config(endpoint)
        limit = limit or tier["limit"]
        window = window or tier["window"]
        
        # If no Redis, allow all (graceful degradation)
        if not self._redis:
            return RateLimitResult(
                allowed=True,
                remaining=limit,
                limit=limit,
                window=window,
            )
        
        key = f"ratelimit:{identifier}:{self._normalize_endpoint(endpoint)}"
        now = time.time()
        window_start = now - window
        
        try:
            # Pipeline for atomic operations
            pipe = self._redis.pipeline()
            
            # Remove entries outside window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Count requests in window
            pipe.zcard(key)
            
            # Set key expiry
            pipe.expire(key, window)
            
            results = await pipe.execute()
            request_count = results[2]
            
            allowed = request_count <= limit
            remaining = max(0, limit - request_count)
            
            # Calculate retry_after if rate limited
            retry_after = 0
            if not allowed:
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + window - now) + 1
            
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                limit=limit,
                window=window,
                retry_after=retry_after,
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis fails
            return RateLimitResult(
                allowed=True,
                remaining=limit,
                limit=limit,
                window=window,
            )
    
    async def get_usage(
        self,
        identifier: str,
        endpoint: str = "default",
    ) -> Dict[str, Any]:
        """
        Get current rate limit usage for an identifier.
        
        Args:
            identifier: Unique identifier
            endpoint: Endpoint pattern
        
        Returns:
            Usage statistics
        """
        tier = self._get_tier_config(endpoint)
        
        if not self._redis:
            return {
                "current_count": 0,
                "limit": tier["limit"],
                "window": tier["window"],
                "remaining": tier["limit"],
            }
        
        key = f"ratelimit:{identifier}:{self._normalize_endpoint(endpoint)}"
        now = time.time()
        window_start = now - tier["window"]
        
        try:
            # Count current requests
            await self._redis.zremrangebyscore(key, 0, window_start)
            count = await self._redis.zcard(key)
            
            return {
                "current_count": count,
                "limit": tier["limit"],
                "window": tier["window"],
                "remaining": max(0, tier["limit"] - count),
            }
        except Exception as e:
            logger.error(f"Failed to get usage: {e}")
            return {
                "current_count": 0,
                "limit": tier["limit"],
                "window": tier["window"],
                "remaining": tier["limit"],
                "error": str(e),
            }
    
    async def reset(
        self,
        identifier: str,
        endpoint: str = None,
    ) -> bool:
        """
        Reset rate limit for an identifier.
        
        Args:
            identifier: Unique identifier
            endpoint: Specific endpoint (or all if None)
        
        Returns:
            True if reset
        """
        if not self._redis:
            return False
        
        try:
            if endpoint:
                key = f"ratelimit:{identifier}:{self._normalize_endpoint(endpoint)}"
                await self._redis.delete(key)
            else:
                # Reset all endpoints for this identifier
                pattern = f"ratelimit:{identifier}:*"
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
    
    def _get_tier_config(self, endpoint: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint."""
        # Check for exact match first
        if endpoint in RATE_LIMIT_TIERS:
            return RATE_LIMIT_TIERS[endpoint]
        
        # Check for prefix match
        for pattern, config in RATE_LIMIT_TIERS.items():
            if pattern != "default" and endpoint.startswith(pattern):
                return config
        
        return RATE_LIMIT_TIERS["default"]
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint for use as cache key."""
        # Remove trailing slashes and query strings
        endpoint = endpoint.rstrip("/").split("?")[0]
        
        # Replace path parameters with placeholders
        # e.g., /api/v1/models/123 -> /api/v1/models/*
        parts = endpoint.split("/")
        normalized = []
        for part in parts:
            if part and (part.isdigit() or len(part) == 36):  # UUID or ID
                normalized.append("*")
            else:
                normalized.append(part)
        
        return "/".join(normalized)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get default rate limiter instance."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = SlidingWindowRateLimiter()
    return _default_limiter


def set_rate_limiter_redis(redis_client):
    """Set Redis client for default rate limiter."""
    get_rate_limiter().redis = redis_client

