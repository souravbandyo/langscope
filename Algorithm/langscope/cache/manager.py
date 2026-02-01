"""
Unified Cache Manager for LangScope.

Central cache management with multi-layer architecture.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from langscope.cache.categories import (
    CacheLayer,
    CacheCategory,
    CacheCategoryConfig,
    CACHE_CONFIG,
)

logger = logging.getLogger(__name__)


class UnifiedCacheManager:
    """
    Central cache management for all LangScope caching needs.
    
    Features:
    - Multi-layer caching (local → Redis → Qdrant → MongoDB)
    - Write-through for critical data
    - Recovery on Redis restart
    - Graceful degradation
    - Metrics and monitoring
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        mongodb_client: Any = None,
        qdrant_client: Any = None,
        db_name: str = "langscope",
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            mongodb_client: Async MongoDB client (motor)
            qdrant_client: Qdrant client for vector cache
            db_name: MongoDB database name
        """
        # Local caches (per-process)
        self._local: Dict[CacheCategory, Dict[str, Dict]] = {
            cat: {} for cat in CacheCategory
        }
        self._local_max_size = 10000
        
        # Redis connection
        self._redis = None
        self._redis_url = redis_url
        
        # MongoDB (for persistence)
        self._mongodb = mongodb_client
        self._db_name = db_name
        
        # Qdrant (for vector search)
        self._qdrant = qdrant_client
        
        # Metrics
        self._metrics = {
            "hits": {cat: {"local": 0, "redis": 0, "qdrant": 0, "mongo": 0} for cat in CacheCategory},
            "misses": {cat: 0 for cat in CacheCategory},
            "errors": {cat: 0 for cat in CacheCategory},
            "writes": {cat: 0 for cat in CacheCategory},
        }
    
    async def initialize(self) -> bool:
        """
        Initialize connections and recover from MongoDB.
        
        Returns:
            True if initialized successfully
        """
        # Connect to Redis
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self._redis_url)
            await self._redis.ping()
            logger.info("Redis connected")
        except ImportError:
            logger.warning("redis package not installed")
            self._redis = None
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._redis = None
        
        # Recovery from MongoDB if needed
        if self._redis and self._mongodb:
            from langscope.cache.recovery import CacheRecovery
            
            recovery = CacheRecovery(
                redis_client=self._redis,
                mongodb_client=self._mongodb,
                db_name=self._db_name,
            )
            
            if await recovery.check_recovery_needed():
                await recovery.recover_all()
        
        return True
    
    async def close(self):
        """Close all connections."""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    # =========================================================================
    # Core Cache Operations
    # =========================================================================
    
    async def get(
        self,
        category: CacheCategory,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get value from cache, checking layers in order.
        
        Args:
            category: Cache category
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        config = CACHE_CONFIG[category]
        
        for layer in config.layers:
            value = await self._get_from_layer(layer, category, key)
            if value is not None:
                self._metrics["hits"][category][layer.value] += 1
                # Populate earlier layers
                await self._populate_earlier_layers(
                    config.layers, layer, category, key, value
                )
                return value
        
        self._metrics["misses"][category] += 1
        return default
    
    async def set(
        self,
        category: CacheCategory,
        key: str,
        value: Any,
        ttl: int = None,
    ) -> bool:
        """
        Set value in configured layers.
        
        Args:
            category: Cache category
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (uses category default if None)
        
        Returns:
            True if set successfully
        """
        config = CACHE_CONFIG[category]
        ttl = ttl or config.ttl
        
        self._metrics["writes"][category] += 1
        
        # Write to all configured layers
        for layer in config.layers:
            await self._set_in_layer(layer, category, key, value, ttl)
        
        # Write-through to MongoDB if configured
        if config.write_through and config.mongodb_collection and self._mongodb:
            await self._write_to_mongodb(config.mongodb_collection, key, value)
        
        return True
    
    async def delete(
        self,
        category: CacheCategory,
        key: str,
    ) -> bool:
        """
        Delete from all layers.
        
        Args:
            category: Cache category
            key: Cache key
        
        Returns:
            True if deleted
        """
        config = CACHE_CONFIG[category]
        
        for layer in config.layers:
            await self._delete_from_layer(layer, category, key)
        
        # Also delete from MongoDB if write-through
        if config.write_through and config.mongodb_collection and self._mongodb:
            try:
                db = self._mongodb[self._db_name]
                await db[config.mongodb_collection].delete_one({"_id": key})
            except Exception as e:
                logger.warning(f"MongoDB delete failed: {e}")
        
        return True
    
    async def invalidate_category(self, category: CacheCategory) -> int:
        """
        Invalidate all entries for a category.
        
        Args:
            category: Cache category
        
        Returns:
            Number of entries invalidated
        """
        config = CACHE_CONFIG[category]
        count = 0
        
        # Clear local cache
        if CacheLayer.LOCAL in config.layers:
            count += len(self._local[category])
            self._local[category].clear()
        
        # Clear Redis entries
        if CacheLayer.REDIS in config.layers and self._redis:
            pattern = f"{category.value}:*"
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor, match=pattern, count=1000
                )
                if keys:
                    await self._redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
        
        return count
    
    async def exists(
        self,
        category: CacheCategory,
        key: str,
    ) -> bool:
        """Check if key exists in cache."""
        value = await self.get(category, key)
        return value is not None
    
    # =========================================================================
    # Layer Operations
    # =========================================================================
    
    async def _get_from_layer(
        self,
        layer: CacheLayer,
        category: CacheCategory,
        key: str,
    ) -> Any:
        """Get value from a specific layer."""
        try:
            if layer == CacheLayer.LOCAL:
                entry = self._local[category].get(key)
                if entry and entry["expires_at"] > time.time():
                    return entry["value"]
                elif entry:
                    del self._local[category][key]
                return None
            
            elif layer == CacheLayer.REDIS and self._redis:
                redis_key = f"{category.value}:{key}"
                value = await self._redis.get(redis_key)
                if value:
                    return json.loads(value)
                return None
            
            elif layer == CacheLayer.MONGO and self._mongodb:
                config = CACHE_CONFIG[category]
                if config.mongodb_collection:
                    db = self._mongodb[self._db_name]
                    doc = await db[config.mongodb_collection].find_one({"_id": key})
                    if doc:
                        doc.pop("_id", None)
                        return doc
                return None
            
            elif layer == CacheLayer.QDRANT:
                # Semantic search handled separately
                return None
                
        except Exception as e:
            self._metrics["errors"][category] += 1
            logger.debug(f"Cache get error ({layer}): {e}")
        
        return None
    
    async def _set_in_layer(
        self,
        layer: CacheLayer,
        category: CacheCategory,
        key: str,
        value: Any,
        ttl: int,
    ) -> bool:
        """Set value in a specific layer."""
        try:
            if layer == CacheLayer.LOCAL:
                # Enforce max size
                if len(self._local[category]) >= self._local_max_size:
                    self._evict_local_cache(category)
                
                self._local[category][key] = {
                    "value": value,
                    "expires_at": time.time() + ttl,
                }
                return True
            
            elif layer == CacheLayer.REDIS and self._redis:
                redis_key = f"{category.value}:{key}"
                await self._redis.setex(
                    redis_key, ttl, json.dumps(value, default=str)
                )
                return True
            
            elif layer == CacheLayer.MONGO:
                # Handled by write-through
                return True
                
        except Exception as e:
            self._metrics["errors"][category] += 1
            logger.debug(f"Cache set error ({layer}): {e}")
        
        return False
    
    async def _delete_from_layer(
        self,
        layer: CacheLayer,
        category: CacheCategory,
        key: str,
    ) -> bool:
        """Delete from a specific layer."""
        try:
            if layer == CacheLayer.LOCAL:
                self._local[category].pop(key, None)
                return True
            
            elif layer == CacheLayer.REDIS and self._redis:
                redis_key = f"{category.value}:{key}"
                await self._redis.delete(redis_key)
                return True
                
        except Exception as e:
            logger.debug(f"Cache delete error ({layer}): {e}")
        
        return False
    
    async def _write_to_mongodb(
        self,
        collection: str,
        key: str,
        value: Any,
    ):
        """Write-through to MongoDB."""
        if not self._mongodb:
            return
        
        try:
            db = self._mongodb[self._db_name]
            
            # Ensure value is a dict
            if isinstance(value, dict):
                doc = {**value, "_id": key, "updated_at": datetime.utcnow()}
            else:
                doc = {"_id": key, "value": value, "updated_at": datetime.utcnow()}
            
            await db[collection].replace_one(
                {"_id": key},
                doc,
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"MongoDB write error: {e}")
    
    async def _populate_earlier_layers(
        self,
        layers: List[CacheLayer],
        hit_layer: CacheLayer,
        category: CacheCategory,
        key: str,
        value: Any,
    ):
        """Populate faster layers after a cache hit in a slower layer."""
        config = CACHE_CONFIG[category]
        
        for layer in layers:
            if layer == hit_layer:
                break
            await self._set_in_layer(layer, category, key, value, config.ttl)
    
    def _evict_local_cache(self, category: CacheCategory):
        """Evict oldest entries from local cache."""
        cache = self._local[category]
        
        # Sort by expiry time and remove oldest 10%
        sorted_keys = sorted(
            cache.keys(),
            key=lambda k: cache[k]["expires_at"],
        )
        
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            del cache[key]
    
    # =========================================================================
    # Session Management (Convenience Methods)
    # =========================================================================
    
    async def create_session(
        self,
        session_type: str,
        data: Dict = None,
        user_id: str = None,
        ttl: int = None,
    ) -> str:
        """Create a new session."""
        import uuid
        
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        ttl = ttl or CACHE_CONFIG[CacheCategory.SESSION].ttl
        
        session = {
            "session_id": session_id,
            "session_type": session_type,
            "user_id": user_id,
            "data": data or {},
            "status": "active",
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
        }
        
        await self.set(CacheCategory.SESSION, session_id, session, ttl)
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get and refresh session."""
        session = await self.get(CacheCategory.SESSION, session_id)
        
        if session and session.get("status") == "active":
            # Check expiry
            expires_at = session.get("expires_at")
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at)
            
            if datetime.utcnow() > expires_at:
                return None
            
            # Refresh last activity
            session["last_activity"] = datetime.utcnow().isoformat()
            await self.set(CacheCategory.SESSION, session_id, session)
            return session
        
        return None
    
    async def update_session(
        self,
        session_id: str,
        data: Dict = None,
        status: str = None,
    ) -> bool:
        """Update session data."""
        session = await self.get(CacheCategory.SESSION, session_id)
        if not session:
            return False
        
        if data:
            session["data"].update(data)
        if status:
            session["status"] = status
        
        session["last_activity"] = datetime.utcnow().isoformat()
        await self.set(CacheCategory.SESSION, session_id, session)
        return True
    
    async def end_session(self, session_id: str) -> bool:
        """Mark session as completed."""
        return await self.update_session(session_id, status="completed")
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "default",
        limit: int = 100,
        window: int = 60,
    ) -> Tuple[bool, Dict]:
        """
        Check rate limit using sliding window algorithm.
        
        Returns:
            (allowed, rate_limit_info)
        """
        from langscope.cache.rate_limit import SlidingWindowRateLimiter
        
        limiter = SlidingWindowRateLimiter(self._redis)
        result = await limiter.check_rate_limit(identifier, endpoint, limit, window)
        
        return result.allowed, result.to_dict()
    
    # =========================================================================
    # Leaderboard Cache
    # =========================================================================
    
    async def get_leaderboard(
        self,
        domain: Optional[str],
        dimension: str,
    ) -> Optional[Dict]:
        """Get cached leaderboard."""
        key = f"{domain or 'global'}:{dimension}"
        return await self.get(CacheCategory.LEADERBOARD, key)
    
    async def set_leaderboard(
        self,
        domain: Optional[str],
        dimension: str,
        data: Dict,
    ):
        """Cache leaderboard snapshot."""
        key = f"{domain or 'global'}:{dimension}"
        await self.set(CacheCategory.LEADERBOARD, key, data)
    
    async def invalidate_domain_leaderboards(self, domain: str):
        """Invalidate all leaderboards for a domain after match."""
        # Common dimensions to invalidate
        dimensions = [
            "raw_quality", "cost_adjusted", "latency", "ttft",
            "consistency", "token_efficiency", "instruction_following",
            "hallucination_resistance", "long_context", "combined",
        ]
        
        for dim in dimensions:
            await self.delete(CacheCategory.LEADERBOARD, f"{domain}:{dim}")
            await self.delete(CacheCategory.LEADERBOARD, f"global:{dim}")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "by_category": {},
            "totals": {"hits": 0, "misses": 0, "errors": 0, "writes": 0},
            "connections": {
                "redis": self._redis is not None,
                "mongodb": self._mongodb is not None,
                "qdrant": self._qdrant is not None,
            },
        }
        
        for category in CacheCategory:
            cat_hits = sum(self._metrics["hits"][category].values())
            cat_misses = self._metrics["misses"][category]
            cat_errors = self._metrics["errors"][category]
            cat_writes = self._metrics["writes"][category]
            total = cat_hits + cat_misses
            
            stats["by_category"][category.value] = {
                "hits": self._metrics["hits"][category],
                "total_hits": cat_hits,
                "misses": cat_misses,
                "errors": cat_errors,
                "writes": cat_writes,
                "hit_rate": cat_hits / total if total > 0 else 0,
                "local_entries": len(self._local.get(category, {})),
            }
            
            stats["totals"]["hits"] += cat_hits
            stats["totals"]["misses"] += cat_misses
            stats["totals"]["errors"] += cat_errors
            stats["totals"]["writes"] += cat_writes
        
        total_requests = stats["totals"]["hits"] + stats["totals"]["misses"]
        stats["totals"]["hit_rate"] = (
            stats["totals"]["hits"] / total_requests if total_requests > 0 else 0
        )
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self._metrics = {
            "hits": {cat: {"local": 0, "redis": 0, "qdrant": 0, "mongo": 0} for cat in CacheCategory},
            "misses": {cat: 0 for cat in CacheCategory},
            "errors": {cat: 0 for cat in CacheCategory},
            "writes": {cat: 0 for cat in CacheCategory},
        }

