"""
Response Caching Middleware for FastAPI.

Caches GET responses for improved performance.
"""

import hashlib
import json
import logging
from typing import Callable, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


# Paths that should be cached
CACHEABLE_PATHS = [
    "/api/v1/leaderboard",
    "/api/v1/models",
    "/api/v1/domains",
    "/api/v1/base-models",
    "/api/v1/deployments",
]


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Response caching middleware for GET requests.
    
    Caches responses in Redis for improved latency.
    """
    
    def __init__(
        self,
        app,
        cache_manager=None,
        cacheable_paths: List[str] = None,
        default_ttl: int = 300,
        enabled: bool = True,
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            cache_manager: UnifiedCacheManager for caching
            cacheable_paths: Paths that can be cached
            default_ttl: Default cache TTL in seconds
            enabled: Whether caching is enabled
        """
        super().__init__(app)
        self._cache_manager = cache_manager
        self._cacheable_paths = cacheable_paths or CACHEABLE_PATHS
        self._default_ttl = default_ttl
        self._enabled = enabled
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with caching."""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip if disabled or no cache manager
        if not self._enabled or not self._cache_manager:
            return await call_next(request)
        
        # Check if path is cacheable
        if not self._is_cacheable(request.url.path):
            return await call_next(request)
        
        # Skip cache if no-cache header
        if request.headers.get("Cache-Control") == "no-cache":
            response = await call_next(request)
            response.headers["X-Cache"] = "BYPASS"
            return response
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        try:
            from langscope.cache.categories import CacheCategory
            
            cached = await self._cache_manager.get(
                CacheCategory.API_RESPONSE,
                cache_key,
            )
            
            if cached:
                response = JSONResponse(
                    content=cached.get("body"),
                    status_code=cached.get("status_code", 200),
                )
                response.headers["X-Cache"] = "HIT"
                response.headers["Content-Type"] = cached.get(
                    "content_type", "application/json"
                )
                return response
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        
        # Process request
        response = await call_next(request)
        response.headers["X-Cache"] = "MISS"
        
        # Cache successful JSON responses
        if (
            response.status_code == 200 and
            "application/json" in response.headers.get("Content-Type", "")
        ):
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Cache the response
                from langscope.cache.categories import CacheCategory
                
                await self._cache_manager.set(
                    CacheCategory.API_RESPONSE,
                    cache_key,
                    {
                        "body": json.loads(body.decode()),
                        "status_code": response.status_code,
                        "content_type": response.headers.get("Content-Type"),
                    },
                    ttl=self._default_ttl,
                )
                
                # Return new response with cached body
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception as e:
                logger.debug(f"Cache write failed: {e}")
        
        return response
    
    def _is_cacheable(self, path: str) -> bool:
        """Check if path should be cached."""
        for cacheable in self._cacheable_paths:
            if path.startswith(cacheable):
                return True
        return False
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate unique cache key for request."""
        # Include path and query params
        key_parts = [
            request.url.path,
            str(sorted(request.query_params.items())),
        ]
        
        # Include API key if present (for user-specific caching)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key_parts.append(api_key[:8])  # Only use prefix
        
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:24]

