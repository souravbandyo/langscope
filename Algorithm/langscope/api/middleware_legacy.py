"""
API middleware for authentication and rate limiting.

Provides:
- API key authentication
- Rate limiting per client
- Request logging
"""

import os
import time
import logging
from typing import Dict, Optional, Callable
from collections import defaultdict
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Authentication
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> Optional[str]:
    """Get the expected API key from environment."""
    return os.getenv("LANGSCOPE_API_KEY")


async def verify_api_key(request: Request, api_key: Optional[str] = None) -> bool:
    """
    Verify the API key.
    
    Args:
        request: The FastAPI request
        api_key: The provided API key
    
    Returns:
        True if valid, raises HTTPException otherwise
    """
    expected_key = get_api_key()
    
    # If no API key is configured, skip authentication
    if not expected_key:
        return True
    
    # Get key from header
    if api_key is None:
        api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "API key"}
        )
    
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.
    
    Checks API key for all requests except health endpoints.
    """
    
    EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Skip auth if no API key is configured
        expected_key = get_api_key()
        if not expected_key:
            return await call_next(request)
        
        # Verify API key
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "API key required", "code": "AUTH_REQUIRED"}
            )
        
        if api_key != expected_key:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Invalid API key", "code": "AUTH_INVALID"}
            )
        
        return await call_next(request)


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.
    
    Allows a configurable number of requests per time window.
    """
    
    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        
        # Track requests per client: {client_id: [(timestamp, count), ...]}
        self.requests: Dict[str, list] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use API key if available, otherwise IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}..."
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Remove requests outside the current window."""
        cutoff = current_time - self.window_seconds
        self.requests[client_id] = [
            (ts, count) for ts, count in self.requests[client_id]
            if ts > cutoff
        ]
    
    def is_allowed(self, request: Request) -> tuple[bool, int, int]:
        """
        Check if request is allowed.
        
        Args:
            request: The FastAPI request
        
        Returns:
            Tuple of (is_allowed, remaining_requests, retry_after_seconds)
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Clean up old requests
        self._cleanup_old_requests(client_id, current_time)
        
        # Count requests in current window
        total_requests = sum(count for _, count in self.requests[client_id])
        
        if total_requests >= self.requests_per_window:
            # Calculate retry-after
            if self.requests[client_id]:
                oldest_ts = min(ts for ts, _ in self.requests[client_id])
                retry_after = int(self.window_seconds - (current_time - oldest_ts))
            else:
                retry_after = self.window_seconds
            
            return False, 0, max(1, retry_after)
        
        # Record this request
        self.requests[client_id].append((current_time, 1))
        
        remaining = self.requests_per_window - total_requests - 1
        return True, remaining, 0
    
    def get_headers(self, remaining: int, retry_after: int = 0) -> Dict[str, str]:
        """Get rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(self.requests_per_window),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Window": str(self.window_seconds),
        }
        
        if retry_after > 0:
            headers["Retry-After"] = str(retry_after)
        
        return headers


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        _rate_limiter = RateLimiter(requests, window)
    
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Applies rate limits to all API endpoints except health.
    """
    
    EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        limiter = get_rate_limiter()
        is_allowed, remaining, retry_after = limiter.is_allowed(request)
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMITED",
                    "retry_after": retry_after
                },
                headers=limiter.get_headers(remaining, retry_after)
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in limiter.get_headers(remaining).items():
            response.headers[key] = value
        
        return response


# =============================================================================
# Request Logging
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.1f}ms"
        )
        
        return response


# =============================================================================
# Dependency for route-level auth
# =============================================================================

async def require_api_key(request: Request):
    """
    Dependency to require API key authentication.
    
    Use in routes that need explicit auth:
        @router.get("/protected", dependencies=[Depends(require_api_key)])
    """
    await verify_api_key(request)
