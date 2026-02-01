"""
Tests for authentication middleware and routes.

Tests cover:
- JWT token verification
- API key authentication
- Auth routes (/auth/me, /auth/status, etc.)
- Rate limiting by user ID
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import jwt
from fastapi.testclient import TestClient
from starlette.requests import Request

from langscope.api.main import app
from langscope.api.middleware.auth import (
    UserContext,
    verify_supabase_jwt,
    get_current_user,
    get_optional_user,
    get_auth_info,
    is_supabase_configured,
    AuthMiddleware,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def jwt_secret():
    """Test JWT secret."""
    return "test-jwt-secret-for-testing-purposes-only"


@pytest.fixture
def valid_token(jwt_secret):
    """Create a valid JWT token for testing."""
    payload = {
        "sub": "test-user-id-123",
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # 1 hour from now
        "app_metadata": {"provider": "email"},
        "user_metadata": {"name": "Test User"},
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_token(jwt_secret):
    """Create an expired JWT token for testing."""
    payload = {
        "sub": "test-user-id-123",
        "email": "test@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": int(time.time()) - 7200,  # 2 hours ago
        "exp": int(time.time()) - 3600,  # 1 hour ago (expired)
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def wrong_audience_token(jwt_secret):
    """Create a token with wrong audience."""
    payload = {
        "sub": "test-user-id-123",
        "aud": "wrong-audience",
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


# =============================================================================
# UserContext Tests
# =============================================================================

class TestUserContext:
    """Tests for UserContext dataclass."""
    
    def test_create_user_context(self):
        """Test creating a UserContext."""
        user = UserContext(
            user_id="123",
            email="user@example.com",
            role="authenticated",
        )
        
        assert user.user_id == "123"
        assert user.email == "user@example.com"
        assert user.role == "authenticated"
        assert user.app_metadata == {}
        assert user.user_metadata == {}
    
    def test_from_jwt_payload(self):
        """Test creating UserContext from JWT payload."""
        payload = {
            "sub": "user-456",
            "email": "jwt@example.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": 1702656000,
            "app_metadata": {"provider": "google"},
            "user_metadata": {"name": "JWT User"},
        }
        
        user = UserContext.from_jwt_payload(payload)
        
        assert user.user_id == "user-456"
        assert user.email == "jwt@example.com"
        assert user.role == "authenticated"
        assert user.exp == 1702656000
        assert user.app_metadata == {"provider": "google"}
        assert user.user_metadata == {"name": "JWT User"}
    
    def test_to_dict(self):
        """Test converting UserContext to dictionary."""
        user = UserContext(
            user_id="789",
            email="dict@example.com",
            role="service_role",
            app_metadata={"key": "value"},
        )
        
        result = user.to_dict()
        
        assert result["user_id"] == "789"
        assert result["email"] == "dict@example.com"
        assert result["role"] == "service_role"
        assert result["app_metadata"] == {"key": "value"}


# =============================================================================
# JWT Verification Tests
# =============================================================================

class TestJWTVerification:
    """Tests for Supabase JWT verification."""
    
    def test_verify_valid_token(self, jwt_secret, valid_token):
        """Test verifying a valid JWT token."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            user = verify_supabase_jwt(valid_token)
            
            assert user is not None
            assert user.user_id == "test-user-id-123"
            assert user.email == "test@example.com"
            assert user.role == "authenticated"
    
    def test_verify_expired_token(self, jwt_secret, expired_token):
        """Test verifying an expired token returns None."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            user = verify_supabase_jwt(expired_token)
            assert user is None
    
    def test_verify_wrong_audience(self, jwt_secret, wrong_audience_token):
        """Test verifying a token with wrong audience returns None."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            user = verify_supabase_jwt(wrong_audience_token)
            assert user is None
    
    def test_verify_invalid_signature(self, jwt_secret, valid_token):
        """Test verifying a token with wrong secret returns None."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "wrong-secret"}):
            user = verify_supabase_jwt(valid_token)
            assert user is None
    
    def test_verify_missing_secret(self, valid_token):
        """Test verifying without JWT secret configured returns None."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": ""}, clear=False):
            # Clear the env var
            import os
            old_value = os.environ.pop("SUPABASE_JWT_SECRET", None)
            try:
                user = verify_supabase_jwt(valid_token)
                assert user is None
            finally:
                if old_value:
                    os.environ["SUPABASE_JWT_SECRET"] = old_value
    
    def test_verify_malformed_token(self, jwt_secret):
        """Test verifying a malformed token returns None."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            user = verify_supabase_jwt("not.a.valid.jwt.token")
            assert user is None
    
    def test_verify_token_missing_sub(self, jwt_secret):
        """Test verifying a token without 'sub' claim returns None."""
        payload = {
            "email": "nosub@example.com",
            "aud": "authenticated",
            "exp": int(time.time()) + 3600,
        }
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")
        
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            user = verify_supabase_jwt(token)
            assert user is None


# =============================================================================
# Auth Configuration Tests
# =============================================================================

class TestAuthConfiguration:
    """Tests for auth configuration utilities."""
    
    def test_is_supabase_configured_true(self):
        """Test is_supabase_configured returns True when configured."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "test-secret"}):
            assert is_supabase_configured() is True
    
    def test_is_supabase_configured_false(self):
        """Test is_supabase_configured returns False when not configured."""
        import os
        # Need to clear all Supabase-related env vars
        old_jwt = os.environ.pop("SUPABASE_JWT_SECRET", None)
        old_service = os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
        old_url = os.environ.pop("SUPABASE_URL", None)
        try:
            assert is_supabase_configured() is False
        finally:
            if old_jwt:
                os.environ["SUPABASE_JWT_SECRET"] = old_jwt
            if old_service:
                os.environ["SUPABASE_SERVICE_ROLE_KEY"] = old_service
            if old_url:
                os.environ["SUPABASE_URL"] = old_url
    
    def test_get_auth_info(self):
        """Test get_auth_info returns correct structure."""
        with patch.dict("os.environ", {
            "SUPABASE_JWT_SECRET": "test-secret",
            "SUPABASE_URL": "https://test.supabase.co",
            "LANGSCOPE_API_KEY": "",
        }):
            info = get_auth_info()
            
            assert "supabase_configured" in info
            assert "supabase_url" in info
            assert "api_key_configured" in info
            assert "methods" in info
            assert info["supabase_configured"] is True
            # Auth middleware uses "bearer_jwt_local" for local JWT verification
            assert "bearer_jwt_local" in info["methods"]


# =============================================================================
# Auth Routes Tests
# =============================================================================

class TestAuthRoutes:
    """Tests for auth API routes."""
    
    def test_auth_status_unauthenticated(self, client):
        """Test /auth/status without authentication."""
        # Disable auth for this test
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "", "LANGSCOPE_API_KEY": ""}):
            import os
            os.environ.pop("SUPABASE_JWT_SECRET", None)
            os.environ.pop("LANGSCOPE_API_KEY", None)
            
            response = client.get("/auth/status")
            
            # In dev mode (no auth configured), still works
            assert response.status_code == 200
            data = response.json()
            assert data["authenticated"] is False
    
    def test_auth_info(self, client):
        """Test /auth/info endpoint."""
        with patch.dict("os.environ", {
            "SUPABASE_JWT_SECRET": "",
            "LANGSCOPE_API_KEY": "",
        }, clear=False):
            import os
            os.environ.pop("SUPABASE_JWT_SECRET", None)
            os.environ.pop("LANGSCOPE_API_KEY", None)
            
            response = client.get("/auth/info")
            
            assert response.status_code == 200
            data = response.json()
            assert "supabase_configured" in data
            assert "methods" in data
    
    def test_auth_me_requires_auth(self, client):
        """Test /auth/me requires authentication."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "test-secret"}):
            response = client.get("/auth/me")
            
            # Should return 401 without token
            assert response.status_code == 401
    
    def test_auth_me_with_valid_token(self, client, jwt_secret, valid_token):
        """Test /auth/me with valid token."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            response = client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "test-user-id-123"
            assert data["email"] == "test@example.com"
    
    def test_auth_verify_with_valid_token(self, client, jwt_secret, valid_token):
        """Test /auth/verify with valid token."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            response = client.get(
                "/auth/verify",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["user_id"] == "test-user-id-123"
    
    def test_auth_verify_with_expired_token(self, client, jwt_secret, expired_token):
        """Test /auth/verify with expired token."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": jwt_secret}):
            response = client.get(
                "/auth/verify",
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            
            assert response.status_code == 401


# =============================================================================
# API Key Authentication Tests
# =============================================================================

class TestAPIKeyAuth:
    """Tests for legacy API key authentication."""
    
    def test_api_key_auth(self, client):
        """Test authentication with API key."""
        with patch.dict("os.environ", {
            "LANGSCOPE_API_KEY": "test-api-key",
            "SUPABASE_JWT_SECRET": "",
        }):
            import os
            os.environ.pop("SUPABASE_JWT_SECRET", None)
            
            response = client.get(
                "/auth/me",
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "api-key-user"
            assert data["role"] == "service_role"
    
    def test_invalid_api_key(self, client):
        """Test authentication with invalid API key."""
        with patch.dict("os.environ", {"LANGSCOPE_API_KEY": "correct-key"}):
            response = client.get(
                "/auth/me",
                headers={"X-API-Key": "wrong-key"}
            )
            
            assert response.status_code == 403
            assert response.json()["code"] == "API_KEY_INVALID"


# =============================================================================
# Exempt Paths Tests
# =============================================================================

class TestExemptPaths:
    """Tests for authentication exempt paths."""
    
    def test_root_is_exempt(self, client):
        """Test root endpoint doesn't require auth."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "test-secret"}):
            response = client.get("/")
            assert response.status_code == 200
    
    def test_health_is_exempt(self, client):
        """Test health endpoint doesn't require auth."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "test-secret"}):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_docs_is_exempt(self, client):
        """Test docs endpoint doesn't require auth."""
        with patch.dict("os.environ", {"SUPABASE_JWT_SECRET": "test-secret"}):
            response = client.get("/docs")
            # Docs returns HTML, just check it doesn't 401
            assert response.status_code != 401


# =============================================================================
# Development Mode Tests
# =============================================================================

class TestDevelopmentMode:
    """Tests for development mode (no auth configured)."""
    
    def test_dev_mode_allows_requests(self, client):
        """Test requests are allowed when no auth is configured."""
        import os
        
        # Remove all auth config
        old_jwt = os.environ.pop("SUPABASE_JWT_SECRET", None)
        old_key = os.environ.pop("LANGSCOPE_API_KEY", None)
        
        try:
            response = client.get("/auth/status")
            assert response.status_code == 200
        finally:
            if old_jwt:
                os.environ["SUPABASE_JWT_SECRET"] = old_jwt
            if old_key:
                os.environ["LANGSCOPE_API_KEY"] = old_key


