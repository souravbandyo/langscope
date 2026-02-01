"""
Cache recovery from MongoDB.

Restores critical cached data after Redis restart.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheRecovery:
    """
    Recover critical cache data from MongoDB on startup.
    
    Restores:
    - Active sessions
    - Hot prompt cache entries
    - User preferences
    - Domain centroids
    """
    
    def __init__(
        self,
        redis_client=None,
        mongodb_client=None,
        db_name: str = "langscope",
    ):
        """
        Initialize cache recovery.
        
        Args:
            redis_client: Async Redis client
            mongodb_client: Async MongoDB client (motor)
            db_name: MongoDB database name
        """
        self._redis = redis_client
        self._mongodb = mongodb_client
        self._db_name = db_name
    
    async def recover_all(self) -> Dict[str, int]:
        """
        Recover all critical cache data from MongoDB.
        
        Returns:
            Recovery statistics
        """
        stats = {
            "sessions": 0,
            "prompts": 0,
            "preferences": 0,
            "centroids": 0,
            "errors": 0,
        }
        
        if not self._redis or not self._mongodb:
            logger.warning("Redis or MongoDB not available for recovery")
            return stats
        
        try:
            # Recover in parallel for speed
            import asyncio
            
            results = await asyncio.gather(
                self.recover_sessions(),
                self.recover_hot_prompts(),
                self.recover_user_preferences(),
                self.recover_centroids(),
                return_exceptions=True,
            )
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Recovery error: {result}")
                    stats["errors"] += 1
                else:
                    key = list(stats.keys())[i]
                    stats[key] = result
            
            logger.info(f"Cache recovery complete: {stats}")
            
        except Exception as e:
            logger.error(f"Cache recovery failed: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def recover_sessions(self) -> int:
        """
        Recover active sessions from MongoDB.
        
        Returns:
            Number of sessions recovered
        """
        if not self._redis or not self._mongodb:
            return 0
        
        try:
            import json
            
            db = self._mongodb[self._db_name]
            
            # Find active, non-expired sessions
            query = {
                "status": "active",
                "expires_at": {"$gt": datetime.utcnow().isoformat()},
            }
            
            cursor = db.sessions.find(query)
            count = 0
            
            async for session in cursor:
                try:
                    # Calculate remaining TTL
                    expires_at = session.get("expires_at")
                    if isinstance(expires_at, str):
                        expires_at = datetime.fromisoformat(expires_at)
                    
                    remaining = int((expires_at - datetime.utcnow()).total_seconds())
                    
                    if remaining > 0:
                        # Restore to Redis
                        key = f"session:{session['session_id']}"
                        await self._redis.setex(
                            key,
                            remaining,
                            json.dumps(session, default=str),
                        )
                        count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to recover session: {e}")
            
            logger.info(f"Recovered {count} sessions from MongoDB")
            return count
            
        except Exception as e:
            logger.error(f"Session recovery failed: {e}")
            return 0
    
    async def recover_hot_prompts(self) -> int:
        """
        Recover frequently-used prompt cache entries.
        
        Only recovers high-value entries (hit_count >= 5, last 24h).
        
        Returns:
            Number of entries recovered
        """
        if not self._redis or not self._mongodb:
            return 0
        
        try:
            import json
            
            db = self._mongodb[self._db_name]
            
            # Find hot cache entries
            query = {
                "hit_count": {"$gte": 5},
                "created_at": {"$gt": (datetime.utcnow() - timedelta(days=1)).isoformat()},
            }
            
            cursor = db.prompt_cache.find(query).limit(10000)
            count = 0
            
            async for entry in cursor:
                try:
                    key = f"prompt_exact:{entry['prompt_hash']}"
                    await self._redis.setex(
                        key,
                        86400,  # 24 hours
                        json.dumps(entry, default=str),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to recover prompt: {e}")
            
            logger.info(f"Recovered {count} prompt cache entries from MongoDB")
            return count
            
        except Exception as e:
            logger.error(f"Prompt cache recovery failed: {e}")
            return 0
    
    async def recover_user_preferences(self) -> int:
        """
        Recover user preferences from MongoDB.
        
        Returns:
            Number of preferences recovered
        """
        if not self._redis or not self._mongodb:
            return 0
        
        try:
            import json
            
            db = self._mongodb[self._db_name]
            
            # Find recent preferences
            query = {
                "updated_at": {"$gt": (datetime.utcnow() - timedelta(days=7)).isoformat()},
            }
            
            cursor = db.user_preferences.find(query).limit(10000)
            count = 0
            
            async for pref in cursor:
                try:
                    key = f"user_pref:{pref['user_id']}"
                    await self._redis.setex(
                        key,
                        3600,  # 1 hour
                        json.dumps(pref, default=str),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to recover preference: {e}")
            
            logger.info(f"Recovered {count} user preferences from MongoDB")
            return count
            
        except Exception as e:
            logger.error(f"User preference recovery failed: {e}")
            return 0
    
    async def recover_centroids(self) -> int:
        """
        Recover domain classification centroids.
        
        Returns:
            Number of centroids recovered
        """
        if not self._redis or not self._mongodb:
            return 0
        
        try:
            import json
            
            db = self._mongodb[self._db_name]
            
            # Find active centroids
            query = {"is_active": True}
            
            cursor = db.domain_centroids.find(query)
            count = 0
            
            async for centroid in cursor:
                try:
                    key = f"centroid:{centroid['category']}:{centroid['domain']}"
                    await self._redis.setex(
                        key,
                        86400,  # 1 day
                        json.dumps(centroid, default=str),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to recover centroid: {e}")
            
            logger.info(f"Recovered {count} centroids from MongoDB")
            return count
            
        except Exception as e:
            logger.error(f"Centroid recovery failed: {e}")
            return 0
    
    async def check_recovery_needed(self) -> bool:
        """
        Check if recovery is needed (Redis is empty).
        
        Returns:
            True if recovery should be performed
        """
        if not self._redis:
            return False
        
        try:
            # Check if Redis has session keys
            cursor = 0
            cursor, keys = await self._redis.scan(cursor, match="session:*", count=1)
            
            if not keys:
                logger.info("Redis appears empty, recovery may be needed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery check failed: {e}")
            return True  # Err on side of recovery

