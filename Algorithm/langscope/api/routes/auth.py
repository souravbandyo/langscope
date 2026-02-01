"""
Authentication routes for LangScope API.

Provides endpoints for:
- Getting current user information
- Checking authentication status
- Authentication configuration info
- Local development authentication
"""

import os
import hashlib
from typing import Optional
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, Request, HTTPException, status
from pydantic import BaseModel

from langscope.api.middleware.auth import (
    UserContext,
    get_current_user,
    get_optional_user,
    get_auth_info,
    is_supabase_configured,
    get_supabase_jwt_secret,
)


router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# Response Models
# =============================================================================

class UserResponse(BaseModel):
    """Response model for user information."""
    user_id: str
    email: Optional[str] = None
    role: str
    app_metadata: dict = {}
    user_metadata: dict = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "role": "authenticated",
                "app_metadata": {"provider": "email"},
                "user_metadata": {"name": "John Doe"}
            }
        }


class AuthStatusResponse(BaseModel):
    """Response model for authentication status."""
    authenticated: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    expires_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "authenticated": True,
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "role": "authenticated",
                "expires_at": "2024-12-15T12:00:00Z"
            }
        }


class AuthInfoResponse(BaseModel):
    """Response model for auth configuration info."""
    supabase_configured: bool
    supabase_url: str
    api_key_configured: bool
    methods: list[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "supabase_configured": True,
                "supabase_url": "https://project.supabase.co",
                "api_key_configured": False,
                "methods": ["bearer_jwt"]
            }
        }


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Returns the authenticated user's information extracted from the JWT token.",
)
async def get_me(user: UserContext = Depends(get_current_user)) -> UserResponse:
    """
    Get the current authenticated user.
    
    Requires a valid Bearer token in the Authorization header.
    
    Returns:
        User information including ID, email, role, and metadata
    """
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        role=user.role,
        app_metadata=user.app_metadata,
        user_metadata=user.user_metadata,
    )


@router.get(
    "/status",
    response_model=AuthStatusResponse,
    summary="Check authentication status",
    description="Check if the current request is authenticated and get basic user info.",
)
async def get_auth_status(
    user: Optional[UserContext] = Depends(get_optional_user)
) -> AuthStatusResponse:
    """
    Check authentication status.
    
    This endpoint works with or without authentication.
    Returns whether the request is authenticated and basic user info if so.
    """
    if user:
        # Convert exp timestamp to ISO string if present
        expires_at = None
        if user.exp:
            expires_at = datetime.utcfromtimestamp(user.exp).isoformat() + "Z"
        
        return AuthStatusResponse(
            authenticated=True,
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            expires_at=expires_at,
        )
    
    return AuthStatusResponse(authenticated=False)


@router.get(
    "/info",
    response_model=AuthInfoResponse,
    summary="Get auth configuration",
    description="Returns information about the authentication configuration (no secrets exposed).",
)
async def get_auth_configuration() -> AuthInfoResponse:
    """
    Get authentication configuration info.
    
    Returns information about which authentication methods are configured.
    No sensitive information (secrets, keys) is exposed.
    """
    info = get_auth_info()
    return AuthInfoResponse(**info)


@router.get(
    "/verify",
    summary="Verify token",
    description="Verify the current token is valid. Returns 200 if valid, 401 if invalid.",
)
async def verify_token(user: UserContext = Depends(get_current_user)) -> dict:
    """
    Verify the current token is valid.
    
    This is a simple endpoint that returns success if the token is valid.
    Use this to check token validity before making other API calls.
    
    Returns:
        {"valid": True, "user_id": "...", "expires_at": "..."}
    """
    expires_at = None
    if user.exp:
        expires_at = datetime.utcfromtimestamp(user.exp).isoformat() + "Z"
    
    return {
        "valid": True,
        "user_id": user.user_id,
        "role": user.role,
        "expires_at": expires_at,
    }


# =============================================================================
# Local Development Authentication
# =============================================================================

# Local dev users (for development without Supabase)
LOCAL_DEV_USERS = {
    "test@langscope.dev": {
        "password_hash": hashlib.sha256("TestPassword123!".encode()).hexdigest(),
        "role": "authenticated",
    },
    "admin@langscope.dev": {
        "password_hash": hashlib.sha256("AdminPassword123!".encode()).hexdigest(),
        "role": "authenticated",
    },
}


class LocalLoginRequest(BaseModel):
    """Request model for local login."""
    email: str
    password: str


class LocalLoginResponse(BaseModel):
    """Response model for local login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: int
    user: dict


@router.post(
    "/local/login",
    response_model=LocalLoginResponse,
    summary="Local development login",
    description="Login endpoint for local development. Only works when Supabase URL is localhost.",
)
async def local_login(request: LocalLoginRequest) -> LocalLoginResponse:
    """
    Local development login.
    
    This endpoint provides authentication for local development when
    external Supabase is not available. It generates valid JWT tokens
    using the configured JWT secret.
    
    Only works when SUPABASE_URL points to localhost.
    """
    supabase_url = os.getenv("SUPABASE_URL", "")
    
    # Only allow local login when Supabase URL is localhost
    if "localhost" not in supabase_url and "127.0.0.1" not in supabase_url:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local login only available in development mode"
        )
    
    # Verify credentials
    user_data = LOCAL_DEV_USERS.get(request.email)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    if password_hash != user_data["password_hash"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate JWT token
    jwt_secret = get_supabase_jwt_secret()
    if not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured"
        )
    
    now = datetime.utcnow()
    expires_in = 86400  # 24 hours
    exp = now + timedelta(seconds=expires_in)
    
    user_id = f"local-{request.email.split('@')[0]}"
    
    payload = {
        "sub": user_id,
        "email": request.email,
        "role": user_data["role"],
        "aud": "authenticated",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    
    access_token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    
    return LocalLoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        expires_at=int(exp.timestamp()),
        user={
            "id": user_id,
            "email": request.email,
            "role": user_data["role"],
        }
    )

