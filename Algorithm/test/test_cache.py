"""
Tests for Cache Management module.

Tests caching, session management, rate limiting, and recovery.
"""

import time
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from langscope.cache.categories import (
    CacheLayer,
    CacheCategory,
    CacheCategoryConfig,
    CACHE_CONFIG,
    get_category_config,
    get_ttl,
)
from langscope.cache.session import (
    Session,
    SESSION_TYPES,
)
from langscope.cache.rate_limit import (
    SlidingWindowRateLimiter,
    RateLimitResult,
    RATE_LIMIT_TIERS,
)
from langscope.cache.manager import UnifiedCacheManager


# =============================================================================
# Cache Categories Tests
# =============================================================================

class TestCacheCategories:
    """Test cache categories and configuration."""
    
    def test_cache_layers_enum(self):
        """Test CacheLayer enum values."""
        assert CacheLayer.LOCAL.value == "local"
        assert CacheLayer.REDIS.value == "redis"
        assert CacheLayer.QDRANT.value == "qdrant"
        assert CacheLayer.MONGO.value == "mongo"
    
    def test_cache_category_enum(self):
        """Test CacheCategory enum values."""
        assert CacheCategory.SESSION.value == "session"
        assert CacheCategory.LEADERBOARD.value == "leaderboard"
        assert CacheCategory.RATE_LIMIT.value == "ratelimit"
        assert CacheCategory.PROMPT_EXACT.value == "prompt_exact"
    
    def test_cache_config_exists(self):
        """Test all categories have configuration."""
        for category in CacheCategory:
            assert category in CACHE_CONFIG
            config = CACHE_CONFIG[category]
            assert isinstance(config, CacheCategoryConfig)
    
    def test_session_config(self):
        """Test session cache configuration."""
        config = CACHE_CONFIG[CacheCategory.SESSION]
        
        assert config.ttl == 1800  # 30 minutes
        assert CacheLayer.REDIS in config.layers
        assert CacheLayer.MONGO in config.layers
        assert config.write_through is True
        assert config.mongodb_collection == "sessions"
    
    def test_rate_limit_config(self):
        """Test rate limit cache configuration."""
        config = CACHE_CONFIG[CacheCategory.RATE_LIMIT]
        
        assert config.ttl == 60  # 1 minute
        assert CacheLayer.REDIS in config.layers
        assert config.write_through is False
        assert config.mongodb_collection is None
    
    def test_get_category_config(self):
        """Test get_category_config helper."""
        config = get_category_config(CacheCategory.SESSION)
        
        assert config.ttl == 1800
    
    def test_get_ttl(self):
        """Test get_ttl helper."""
        ttl = get_ttl(CacheCategory.SESSION)
        assert ttl == 1800
        
        ttl = get_ttl(CacheCategory.RATE_LIMIT)
        assert ttl == 60


# =============================================================================
# Session Tests
# =============================================================================

class TestSession:
    """Test Session dataclass."""
    
    def test_session_creation(self):
        """Test basic session creation."""
        session = Session(
            session_id="test-123",
            session_type="arena",
        )
        
        assert session.session_id == "test-123"
        assert session.session_type == "arena"
        assert session.status == "active"
    
    def test_session_expiry_calculation(self):
        """Test automatic expiry calculation."""
        session = Session(
            session_id="test-123",
            session_type="arena",
        )
        
        # Arena has 30 min TTL
        expected_expiry = session.created_at + timedelta(seconds=1800)
        assert abs((session.expires_at - expected_expiry).total_seconds()) < 1
    
    def test_remaining_ttl(self):
        """Test remaining TTL calculation."""
        session = Session(
            session_id="test-123",
            session_type="arena",
        )
        
        # Should be close to 30 minutes
        assert session.remaining_ttl > 1790
        assert session.remaining_ttl <= 1800
    
    def test_is_expired(self):
        """Test expiry check."""
        session = Session(
            session_id="test-123",
            session_type="arena",
        )
        
        assert session.is_expired is False
        
        # Create expired session
        expired = Session(
            session_id="test-456",
            session_type="arena",
            expires_at=datetime.utcnow() - timedelta(minutes=5),
        )
        
        assert expired.is_expired is True
    
    def test_refresh(self):
        """Test session refresh."""
        session = Session(
            session_id="test-123",
            session_type="arena",
        )
        
        old_activity = session.last_activity
        time.sleep(0.01)
        session.refresh()
        
        assert session.last_activity > old_activity
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        session = Session(
            session_id="test-123",
            session_type="arena",
            user_id="user-1",
            domain="medical",
        )
        
        d = session.to_dict()
        
        assert d["session_id"] == "test-123"
        assert d["session_type"] == "arena"
        assert d["user_id"] == "user-1"
        assert d["domain"] == "medical"
        assert d["status"] == "active"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "session_id": "test-123",
            "session_type": "feedback",
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        }
        
        session = Session.from_dict(data)
        
        assert session.session_id == "test-123"
        assert session.session_type == "feedback"
        assert session.status == "completed"


class TestSessionTypes:
    """Test session type configurations."""
    
    def test_arena_session_type(self):
        """Test arena session type."""
        assert "arena" in SESSION_TYPES
        assert SESSION_TYPES["arena"]["ttl"] == 1800
    
    def test_feedback_session_type(self):
        """Test feedback session type."""
        assert "feedback" in SESSION_TYPES
        assert SESSION_TYPES["feedback"]["ttl"] == 900
    
    def test_api_key_session_type(self):
        """Test API key session type."""
        assert "api_key" in SESSION_TYPES
        assert SESSION_TYPES["api_key"]["ttl"] == 2592000  # 30 days


# =============================================================================
# Rate Limit Tests
# =============================================================================

class TestRateLimitResult:
    """Test RateLimitResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = RateLimitResult(
            allowed=True,
            remaining=99,
            limit=100,
            window=60,
        )
        
        assert result.allowed is True
        assert result.remaining == 99
        assert result.retry_after == 0
    
    def test_to_headers(self):
        """Test header generation."""
        result = RateLimitResult(
            allowed=True,
            remaining=99,
            limit=100,
            window=60,
        )
        
        headers = result.to_headers()
        
        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "99"
        assert headers["X-RateLimit-Window"] == "60"
    
    def test_to_headers_rate_limited(self):
        """Test headers when rate limited."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=100,
            window=60,
            retry_after=30,
        )
        
        headers = result.to_headers()
        
        assert headers["Retry-After"] == "30"


class TestSlidingWindowRateLimiter:
    """Test SlidingWindowRateLimiter."""
    
    def test_creation(self):
        """Test limiter creation."""
        limiter = SlidingWindowRateLimiter()
        
        assert limiter.redis is None
    
    def test_get_tier_config_exact(self):
        """Test getting tier config by exact match."""
        limiter = SlidingWindowRateLimiter()
        
        config = limiter._get_tier_config("/api/v1/leaderboard")
        
        assert config["limit"] == 200
        assert config["window"] == 60
    
    def test_get_tier_config_prefix(self):
        """Test getting tier config by prefix match."""
        limiter = SlidingWindowRateLimiter()
        
        # Should match /api/v1/arena prefix
        config = limiter._get_tier_config("/api/v1/arena/session/start")
        
        assert config["limit"] == 10
    
    def test_get_tier_config_default(self):
        """Test default tier config."""
        limiter = SlidingWindowRateLimiter()
        
        config = limiter._get_tier_config("/unknown/endpoint")
        
        assert config["limit"] == 100  # default
    
    def test_normalize_endpoint(self):
        """Test endpoint normalization."""
        limiter = SlidingWindowRateLimiter()
        
        # Remove trailing slash
        assert limiter._normalize_endpoint("/api/v1/models/") == "/api/v1/models"
        
        # Remove query string
        assert limiter._normalize_endpoint("/api?foo=bar") == "/api"
        
        # Replace numeric IDs
        assert limiter._normalize_endpoint("/api/v1/models/123") == "/api/v1/models/*"
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_no_redis(self):
        """Test rate limit check without Redis (graceful degradation)."""
        limiter = SlidingWindowRateLimiter(redis_client=None)
        
        result = await limiter.check_rate_limit("test-ip", "/api/test")
        
        assert result.allowed is True
        assert result.remaining == result.limit
    
    @pytest.mark.asyncio
    async def test_get_usage_no_redis(self):
        """Test getting usage without Redis."""
        limiter = SlidingWindowRateLimiter(redis_client=None)
        
        usage = await limiter.get_usage("test-ip", "/api/test")
        
        assert usage["current_count"] == 0
        assert usage["limit"] == 100


# =============================================================================
# Unified Cache Manager Tests
# =============================================================================

class TestUnifiedCacheManager:
    """Test UnifiedCacheManager."""
    
    def test_creation(self):
        """Test manager creation."""
        manager = UnifiedCacheManager()
        
        assert manager._redis is None
        assert manager._mongodb is None
        assert len(manager._local) == len(CacheCategory)
    
    def test_local_cache_structure(self):
        """Test local cache is initialized for all categories."""
        manager = UnifiedCacheManager()
        
        for category in CacheCategory:
            assert category in manager._local
            assert isinstance(manager._local[category], dict)
    
    def test_metrics_structure(self):
        """Test metrics are initialized correctly."""
        manager = UnifiedCacheManager()
        
        for category in CacheCategory:
            assert category in manager._metrics["hits"]
            assert category in manager._metrics["misses"]
            assert category in manager._metrics["errors"]
            assert category in manager._metrics["writes"]
    
    @pytest.mark.asyncio
    async def test_local_cache_get_set(self):
        """Test local cache get/set."""
        manager = UnifiedCacheManager()
        
        # Set value
        await manager.set(CacheCategory.PARAM, "test-key", {"foo": "bar"})
        
        # Get value
        value = await manager.get(CacheCategory.PARAM, "test-key")
        
        assert value == {"foo": "bar"}
    
    @pytest.mark.asyncio
    async def test_local_cache_expiry(self):
        """Test local cache respects TTL."""
        manager = UnifiedCacheManager()
        
        # Set with very short TTL
        await manager._set_in_layer(
            CacheLayer.LOCAL,
            CacheCategory.PARAM,
            "expire-key",
            {"test": True},
            ttl=0,  # Immediate expiry
        )
        
        # Wait a tiny bit
        time.sleep(0.01)
        
        # Should be expired
        value = await manager._get_from_layer(
            CacheLayer.LOCAL,
            CacheCategory.PARAM,
            "expire-key",
        )
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test cache deletion."""
        manager = UnifiedCacheManager()
        
        # Set then delete
        await manager.set(CacheCategory.PARAM, "delete-key", {"data": True})
        await manager.delete(CacheCategory.PARAM, "delete-key")
        
        value = await manager.get(CacheCategory.PARAM, "delete-key")
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_exists(self):
        """Test exists check."""
        manager = UnifiedCacheManager()
        
        await manager.set(CacheCategory.PARAM, "exists-key", {"data": True})
        
        assert await manager.exists(CacheCategory.PARAM, "exists-key") is True
        assert await manager.exists(CacheCategory.PARAM, "nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_invalidate_category(self):
        """Test category invalidation."""
        manager = UnifiedCacheManager()
        
        # Set multiple values
        await manager.set(CacheCategory.PARAM, "key1", {"data": 1})
        await manager.set(CacheCategory.PARAM, "key2", {"data": 2})
        
        # Invalidate category
        count = await manager.invalidate_category(CacheCategory.PARAM)
        
        assert count == 2
        assert len(manager._local[CacheCategory.PARAM]) == 0
    
    def test_evict_local_cache(self):
        """Test local cache eviction."""
        manager = UnifiedCacheManager()
        manager._local_max_size = 10
        
        # Fill cache
        for i in range(15):
            manager._local[CacheCategory.PARAM][f"key-{i}"] = {
                "value": i,
                "expires_at": time.time() + 100 + i,  # Later items expire later
            }
        
        # Evict
        manager._evict_local_cache(CacheCategory.PARAM)
        
        # Should have evicted oldest entries
        assert len(manager._local[CacheCategory.PARAM]) < 15
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = UnifiedCacheManager()
        
        # Modify some metrics
        manager._metrics["hits"][CacheCategory.SESSION]["local"] = 10
        manager._metrics["misses"][CacheCategory.SESSION] = 5
        
        stats = manager.get_stats()
        
        assert "by_category" in stats
        assert "totals" in stats
        assert "connections" in stats
        assert stats["connections"]["redis"] is False
    
    def test_reset_stats(self):
        """Test statistics reset."""
        manager = UnifiedCacheManager()
        
        manager._metrics["hits"][CacheCategory.SESSION]["local"] = 100
        manager.reset_stats()
        
        assert manager._metrics["hits"][CacheCategory.SESSION]["local"] == 0


# =============================================================================
# Cache Manager Session Methods Tests
# =============================================================================

class TestCacheManagerSessions:
    """Test session-related methods on UnifiedCacheManager."""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test session creation."""
        manager = UnifiedCacheManager()
        
        session_id = await manager.create_session(
            session_type="arena",
            data={"domain": "medical"},
            user_id="user-123",
        )
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test getting a session."""
        manager = UnifiedCacheManager()
        
        session_id = await manager.create_session(
            session_type="arena",
            data={"test": True},
        )
        
        session = await manager.get_session(session_id)
        
        assert session is not None
        assert session["session_id"] == session_id
        assert session["data"]["test"] is True
    
    @pytest.mark.asyncio
    async def test_update_session(self):
        """Test session update."""
        manager = UnifiedCacheManager()
        
        session_id = await manager.create_session(
            session_type="arena",
            data={"count": 0},
        )
        
        success = await manager.update_session(
            session_id,
            data={"count": 1},
        )
        
        assert success is True
        
        session = await manager.get_session(session_id)
        assert session["data"]["count"] == 1
    
    @pytest.mark.asyncio
    async def test_end_session(self):
        """Test ending a session."""
        manager = UnifiedCacheManager()
        
        session_id = await manager.create_session(session_type="arena")
        
        success = await manager.end_session(session_id)
        
        assert success is True
        
        session = await manager.get_session(session_id)
        # Session should be None because status is not "active"
        assert session is None


# =============================================================================
# Leaderboard Cache Tests
# =============================================================================

class TestCacheManagerLeaderboard:
    """Test leaderboard caching methods."""
    
    @pytest.mark.asyncio
    async def test_set_get_leaderboard(self):
        """Test leaderboard caching."""
        manager = UnifiedCacheManager()
        
        data = {
            "entries": [
                {"rank": 1, "model": "gpt-4", "score": 1720},
                {"rank": 2, "model": "claude-3", "score": 1695},
            ],
        }
        
        await manager.set_leaderboard("medical", "raw_quality", data)
        
        cached = await manager.get_leaderboard("medical", "raw_quality")
        
        assert cached is not None
        assert len(cached["entries"]) == 2
    
    @pytest.mark.asyncio
    async def test_invalidate_domain_leaderboards(self):
        """Test leaderboard invalidation."""
        manager = UnifiedCacheManager()
        
        # Set some leaderboards
        await manager.set_leaderboard("medical", "raw_quality", {"entries": []})
        await manager.set_leaderboard("medical", "latency", {"entries": []})
        
        # Invalidate
        await manager.invalidate_domain_leaderboards("medical")
        
        # Should be None now
        assert await manager.get_leaderboard("medical", "raw_quality") is None
        assert await manager.get_leaderboard("medical", "latency") is None


# =============================================================================
# Rate Limit Integration Tests
# =============================================================================

class TestCacheManagerRateLimit:
    """Test rate limiting through cache manager."""
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_no_redis(self):
        """Test rate limit without Redis."""
        manager = UnifiedCacheManager()
        
        allowed, info = await manager.check_rate_limit(
            identifier="test-ip",
            endpoint="/api/test",
        )
        
        # Should allow when no Redis
        assert allowed is True

