"""
API middleware for authentication.

Provides:
- Stateless JWT authentication via Supabase
- Service role key authentication (for server-to-server)
- Legacy API key authentication (fallback)
- User context extraction from JWT claims
"""

import os
import logging
import httpx
from typing import Optional
from dataclasses import dataclass

import jwt
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Cache for Supabase API client
_supabase_client: Optional[httpx.AsyncClient] = None


# =============================================================================
# User Context (extracted from JWT)
# =============================================================================

@dataclass
class UserContext:
    """
    User context extracted from Supabase JWT.
    
    Attributes:
        user_id: Supabase user ID (sub claim)
        email: User email (if available)
        role: User role (authenticated, anon, service_role)
        app_metadata: Application-specific metadata
        user_metadata: User-specific metadata
        aud: Audience claim (authenticated)
        exp: Token expiration timestamp
    """
    user_id: str
    email: Optional[str] = None
    role: str = "authenticated"
    app_metadata: dict = None
    user_metadata: dict = None
    aud: str = "authenticated"
    exp: Optional[int] = None
    
    def __post_init__(self):
        if self.app_metadata is None:
            self.app_metadata = {}
        if self.user_metadata is None:
            self.user_metadata = {}
    
    @classmethod
    def from_jwt_payload(cls, payload: dict) -> "UserContext":
        """Create UserContext from JWT payload."""
        return cls(
            user_id=payload.get("sub", ""),
            email=payload.get("email"),
            role=payload.get("role", "authenticated"),
            app_metadata=payload.get("app_metadata", {}),
            user_metadata=payload.get("user_metadata", {}),
            aud=payload.get("aud", "authenticated"),
            exp=payload.get("exp"),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "app_metadata": self.app_metadata,
            "user_metadata": self.user_metadata,
        }


# =============================================================================
# Supabase JWT Verification
# =============================================================================

def get_supabase_jwt_secret() -> Optional[str]:
    """Get Supabase JWT secret from environment."""
    return os.getenv("SUPABASE_JWT_SECRET")


def get_supabase_url() -> Optional[str]:
    """Get Supabase URL from environment."""
    return os.getenv("SUPABASE_URL")


def get_supabase_service_role_key() -> Optional[str]:
    """Get Supabase service role key from environment."""
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_anon_key() -> Optional[str]:
    """Get Supabase anon key from environment."""
    return os.getenv("SUPABASE_ANON_KEY")


def verify_supabase_jwt(token: str) -> Optional[UserContext]:
    """
    Verify a Supabase JWT token and extract user context.
    
    This is a stateless verification - the token is cryptographically
    verified using the JWT secret without any database/API calls.
    
    Args:
        token: The JWT access token from Supabase
        
    Returns:
        UserContext if valid, None otherwise
    """
    jwt_secret = get_supabase_jwt_secret()
    
    if not jwt_secret:
        logger.warning("SUPABASE_JWT_SECRET not configured, trying API verification")
        return None
    
    try:
        # Decode and verify the JWT
        # Supabase uses HS256 algorithm with the JWT secret
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={
                "verify_aud": True,
                "verify_exp": True,
                "verify_iss": False,  # Supabase issuer varies by project
            }
        )
        
        # Validate required claims
        if not payload.get("sub"):
            logger.warning("JWT missing 'sub' claim")
            return None
        
        return UserContext.from_jwt_payload(payload)
        
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token expired")
        return None
    except jwt.InvalidAudienceError:
        logger.debug("JWT invalid audience")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"JWT verification failed: {e}")
        return None


async def verify_supabase_jwt_via_api(token: str) -> Optional[UserContext]:
    """
    Verify a Supabase JWT token via the Supabase Auth API.
    
    This is used when SUPABASE_JWT_SECRET is not configured.
    It makes an API call to Supabase to verify the token.
    
    Args:
        token: The JWT access token from Supabase
        
    Returns:
        UserContext if valid, None otherwise
    """
    global _supabase_client
    
    supabase_url = get_supabase_url()
    service_key = get_supabase_service_role_key()
    
    if not supabase_url or not service_key:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not configured")
        return None
    
    try:
        # Create client if not exists
        if _supabase_client is None:
            _supabase_client = httpx.AsyncClient(timeout=10.0)
        
        # Call Supabase Auth API to get user
        response = await _supabase_client.get(
            f"{supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": service_key,
            }
        )
        
        if response.status_code == 200:
            user_data = response.json()
            return UserContext(
                user_id=user_data.get("id", ""),
                email=user_data.get("email"),
                role=user_data.get("role", "authenticated"),
                app_metadata=user_data.get("app_metadata", {}),
                user_metadata=user_data.get("user_metadata", {}),
            )
        else:
            logger.debug(f"Supabase API verification failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Supabase API verification error: {e}")
        return None


def verify_service_role_key(token: str) -> Optional[UserContext]:
    """
    Check if the token is the Supabase service role key.
    
    This allows server-to-server authentication using the service role key
    directly as a Bearer token.
    
    Args:
        token: The token to check
        
    Returns:
        UserContext with service_role if token matches, None otherwise
    """
    service_key = get_supabase_service_role_key()
    
    if service_key and token == service_key:
        # Decode the service role key to extract info (without verification)
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return UserContext(
                user_id="service-role",
                email=None,
                role="service_role",
                app_metadata={"ref": payload.get("ref", "")},
                exp=payload.get("exp"),
            )
        except jwt.InvalidTokenError:
            return UserContext(
                user_id="service-role",
                email=None,
                role="service_role",
            )
    
    return None


# =============================================================================
# Legacy API Key Authentication
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> Optional[str]:
    """Get the expected API key from environment."""
    return os.getenv("LANGSCOPE_API_KEY")


async def verify_api_key(request: Request, api_key: Optional[str] = None) -> bool:
    """
    Verify the API key (legacy authentication method).
    
    Args:
        request: The FastAPI request
        api_key: The provided API key
    
    Returns:
        True if valid, raises HTTPException otherwise
    """
    expected_key = get_api_key()
    
    # If no API key is configured, skip this auth method
    if not expected_key:
        return False
    
    # Get key from header
    if api_key is None:
        api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return False
    
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True


# =============================================================================
# Combined Auth Middleware
# =============================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Stateless authentication middleware.
    
    Supports multiple authentication methods (in order of priority):
    1. Supabase JWT (Authorization: Bearer <token>) - verified locally with JWT secret
    2. Supabase Service Role Key (Authorization: Bearer <service_role_key>)
    3. Supabase API verification (when JWT secret not configured)
    4. Legacy API key (X-API-Key header)
    
    The middleware is stateless - all verification is done cryptographically
    without database lookups. User context is extracted from JWT claims.
    """
    
    EXEMPT_PATHS = {
        "/", "/health", "/docs", "/redoc", "/openapi.json",
        "/auth/info", "/auth/status",  # Auth info endpoints (public)
        "/auth/local/login",  # Local development login
    }
    
    # Path prefixes that don't require authentication
    EXEMPT_PREFIXES = (
        "/uploads/",  # Static file uploads (avatars, logos, etc.)
    )
    
    def _get_cors_headers(self, request: Request) -> dict:
        """Get CORS headers for error responses."""
        origin = request.headers.get("origin", "")
        allowed_origins = os.getenv(
            "CORS_ORIGINS", 
            "http://localhost:3000,http://localhost:3001,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:3001,http://127.0.0.1:5173"
        ).split(",")
        
        if origin in allowed_origins:
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
                "Access-Control-Allow-Headers": "*",
            }
        return {}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Skip auth for exempt path prefixes (like /uploads/)
        if request.url.path.startswith(self.EXEMPT_PREFIXES):
            return await call_next(request)
        
        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check if any auth is configured
        jwt_secret = get_supabase_jwt_secret()
        service_key = get_supabase_service_role_key()
        supabase_url = get_supabase_url()
        api_key = get_api_key()
        
        # Development mode: no auth configured
        if not jwt_secret and not service_key and not api_key:
            return await call_next(request)
        
        # Try Bearer token authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # 1. Check if it's the service role key
            user_context = verify_service_role_key(token)
            if user_context:
                request.state.user = user_context
                return await call_next(request)
            
            # 2. Try local JWT verification (if secret configured)
            if jwt_secret:
                user_context = verify_supabase_jwt(token)
                if user_context:
                    request.state.user = user_context
                    return await call_next(request)
            
            # 3. Try API-based verification (if URL and service key configured)
            if supabase_url and service_key:
                user_context = await verify_supabase_jwt_via_api(token)
                if user_context:
                    request.state.user = user_context
                    return await call_next(request)
            
            # Token provided but invalid
            cors_headers = self._get_cors_headers(request)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid or expired token",
                    "code": "TOKEN_INVALID"
                },
                headers={"WWW-Authenticate": "Bearer", **cors_headers}
            )
        
        # Try legacy API key authentication
        request_api_key = request.headers.get("X-API-Key")
        if request_api_key:
            if api_key and request_api_key == api_key:
                # API key valid - create a service context
                request.state.user = UserContext(
                    user_id="api-key-user",
                    role="service_role",
                    email=None,
                )
                return await call_next(request)
            else:
                cors_headers = self._get_cors_headers(request)
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Invalid API key",
                        "code": "API_KEY_INVALID"
                    },
                    headers=cors_headers
                )
        
        # No authentication provided
        cors_headers = self._get_cors_headers(request)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Authentication required",
                "code": "AUTH_REQUIRED",
                "methods": ["Bearer token", "X-API-Key"]
            },
            headers={"WWW-Authenticate": "Bearer", **cors_headers}
        )


# =============================================================================
# Dependencies for Route-Level Auth
# =============================================================================

http_bearer = HTTPBearer(auto_error=False)


async def get_current_user(request: Request) -> UserContext:
    """
    Dependency to get the current authenticated user.
    
    Use in routes that need user context:
        @router.get("/me")
        async def get_me(user: UserContext = Depends(get_current_user)):
            return user.to_dict()
    
    Raises:
        HTTPException: If user is not authenticated
    """
    user = getattr(request.state, "user", None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_optional_user(request: Request) -> Optional[UserContext]:
    """
    Dependency to optionally get the current user.
    
    Returns None if not authenticated (doesn't raise exception).
    Useful for endpoints that work differently for auth/unauth users.
    """
    return getattr(request.state, "user", None)


async def require_role(required_role: str):
    """
    Factory for role-checking dependencies.
    
    Usage:
        @router.delete("/admin/users/{user_id}", dependencies=[Depends(require_role("service_role"))])
    """
    async def check_role(request: Request):
        user = await get_current_user(request)
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}"
            )
        return user
    return check_role


async def require_api_key(request: Request):
    """
    Dependency to require API key authentication.
    
    Use in routes that need explicit API key auth:
        @router.get("/protected", dependencies=[Depends(require_api_key)])
    """
    user = getattr(request.state, "user", None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # For backward compatibility, allow both service_role and api-key-user
    if user.role not in ("service_role",) and user.user_id != "api-key-user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key or service role required"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def is_supabase_configured() -> bool:
    """Check if Supabase authentication is configured."""
    # Configured if we have JWT secret OR (URL + service key for API verification)
    jwt_secret = get_supabase_jwt_secret()
    service_key = get_supabase_service_role_key()
    supabase_url = get_supabase_url()
    
    return bool(jwt_secret) or bool(service_key and supabase_url)


def get_auth_info() -> dict:
    """Get authentication configuration info (for health/debug endpoints)."""
    jwt_secret = bool(get_supabase_jwt_secret())
    service_key = bool(get_supabase_service_role_key())
    supabase_url = get_supabase_url()
    api_key = bool(get_api_key())
    
    methods = []
    if jwt_secret:
        methods.append("bearer_jwt_local")
    if service_key:
        methods.append("service_role_key")
    if service_key and supabase_url:
        methods.append("bearer_jwt_api")
    if api_key:
        methods.append("api_key")
    
    return {
        "supabase_configured": is_supabase_configured(),
        "supabase_url": supabase_url or "not configured",
        "jwt_secret_configured": jwt_secret,
        "service_role_key_configured": service_key,
        "api_key_configured": api_key,
        "methods": methods or ["none (development mode)"]
    }

