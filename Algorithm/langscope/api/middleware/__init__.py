"""
LangScope API Middleware.

Provides rate limiting, authentication, logging, and response caching middleware.
"""

from langscope.api.middleware.rate_limit import RateLimitMiddleware
from langscope.api.middleware.response_cache import ResponseCacheMiddleware
from langscope.api.middleware.auth import (
    AuthMiddleware,
    UserContext,
    verify_api_key,
    require_api_key,
    get_current_user,
    get_optional_user,
    verify_supabase_jwt,
    is_supabase_configured,
    get_auth_info,
)
from langscope.api.middleware.logging import RequestLoggingMiddleware

__all__ = [
    # Middleware
    "RateLimitMiddleware",
    "ResponseCacheMiddleware",
    "AuthMiddleware",
    "RequestLoggingMiddleware",
    # Auth types
    "UserContext",
    # Auth dependencies
    "verify_api_key",
    "require_api_key",
    "get_current_user",
    "get_optional_user",
    # Auth utilities
    "verify_supabase_jwt",
    "is_supabase_configured",
    "get_auth_info",
]

