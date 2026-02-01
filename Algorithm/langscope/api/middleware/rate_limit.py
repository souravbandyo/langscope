"""
Rate Limiting Middleware for FastAPI.

Provides request rate limiting based on IP or API key.
"""

import logging
from typing import Callable, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


# Rate limit configuration by endpoint pattern
RATE_LIMITS: Dict[str, Dict[str, int]] = {
    "/api/v1/leaderboard": {"limit": 200, "window": 60},
    "/api/v1/models": {"limit": 100, "window": 60},
    "/api/v1/arena": {"limit": 10, "window": 60},
    "/api/v1/ground-truth/evaluate": {"limit": 5, "window": 60},
    "/api/v1/cache/invalidate": {"limit": 10, "window": 60},
    "/api/v1/prompts": {"limit": 50, "window": 60},
    "default": {"limit": 100, "window": 60},
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    
    Uses Redis for distributed rate limiting.
    """
    
    def __init__(
        self,
        app,
        cache_manager=None,
        rate_limits: Dict[str, Dict[str, int]] = None,
        enabled: bool = True,
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            cache_manager: UnifiedCacheManager for rate limiting
            rate_limits: Custom rate limit configuration
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self._cache_manager = cache_manager
        self._rate_limits = rate_limits or RATE_LIMITS
        self._enabled = enabled
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled or no cache manager
        if not self._enabled or not self._cache_manager:
            return await call_next(request)
        
        # Get identifier (API key or IP)
        identifier = self._get_identifier(request)
        
        # Get rate limit config for endpoint
        limit_config = self._get_limit_config(request.url.path)
        
        # Check rate limit
        try:
            allowed, info = await self._cache_manager.check_rate_limit(
                identifier=identifier,
                endpoint=request.url.path,
                limit=limit_config["limit"],
                window=limit_config["window"],
            )
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            # Fail open - allow request
            return await call_next(request)
        
        # Build rate limit headers
        headers = {
            "X-RateLimit-Limit": str(info.get("limit", limit_config["limit"])),
            "X-RateLimit-Remaining": str(info.get("remaining", 0)),
            "X-RateLimit-Window": str(info.get("window", limit_config["window"])),
        }
        
        # Return 429 if rate limited
        if not allowed:
            retry_after = info.get("retry_after", limit_config["window"])
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit": limit_config["limit"],
                    "window": limit_config["window"],
                },
                headers={**headers, "Retry-After": str(retry_after)},
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response
    
    def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting.
        
        Priority:
        1. User ID from JWT (if authenticated via Supabase)
        2. API key
        3. Client IP
        """
        # Check for authenticated user (from AuthMiddleware)
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "user_id") and user.user_id:
            return f"user:{user.user_id}"
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to client IP
        client_host = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_host = forwarded.split(",")[0].strip()
        
        return f"ip:{client_host}"
    
    def _get_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for path."""
        # Check for exact match
        if path in self._rate_limits:
            return self._rate_limits[path]
        
        # Check for prefix match
        for pattern, config in self._rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return config
        
        return self._rate_limits.get("default", {"limit": 100, "window": 60})

