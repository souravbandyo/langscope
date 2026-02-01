"""
Session management for LangScope.

Provides session creation, update, and persistence.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Session Types
# =============================================================================

SESSION_TYPES = {
    "arena": {
        "ttl": 1800,      # 30 minutes
        "description": "User running arena battle",
    },
    "feedback": {
        "ttl": 900,       # 15 minutes
        "description": "User providing feedback",
    },
    "evaluation": {
        "ttl": 3600,      # 60 minutes
        "description": "Running GT evaluation",
    },
    "upload": {
        "ttl": 86400,     # 24 hours
        "description": "Self-hosted registration",
    },
    "api_key": {
        "ttl": 2592000,   # 30 days
        "description": "API key session",
    },
}


@dataclass
class Session:
    """
    Session stored in Redis + MongoDB.
    
    Represents an active user session for various workflows.
    """
    
    # Identity
    session_id: str
    session_type: str  # "arena", "feedback", "evaluation", "upload", "api_key"
    
    # User
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    
    # State
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # "active", "completed", "expired", "error"
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = None
    
    # Metadata
    domain: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def __post_init__(self):
        """Set expiry if not provided."""
        if self.expires_at is None:
            ttl = SESSION_TYPES.get(self.session_type, {}).get("ttl", 1800)
            self.expires_at = self.created_at + timedelta(seconds=ttl)
    
    @property
    def remaining_ttl(self) -> int:
        """Get remaining time-to-live in seconds."""
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def refresh(self):
        """Refresh session activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "user_id": self.user_id,
            "api_key_id": self.api_key_id,
            "data": self.data,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "domain": self.domain,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            session_type=data["session_type"],
            user_id=data.get("user_id"),
            api_key_id=data.get("api_key_id"),
            data=data.get("data", {}),
            status=data.get("status", "active"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            last_activity=datetime.fromisoformat(data["last_activity"]) if isinstance(data.get("last_activity"), str) else data.get("last_activity", datetime.utcnow()),
            expires_at=datetime.fromisoformat(data["expires_at"]) if isinstance(data.get("expires_at"), str) else data.get("expires_at"),
            domain=data.get("domain"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
        )


class SessionManager:
    """
    Manages sessions using cache manager.
    
    Provides high-level session operations with automatic
    persistence to Redis and MongoDB.
    """
    
    def __init__(self, cache_manager: 'UnifiedCacheManager'):
        """
        Initialize session manager.
        
        Args:
            cache_manager: UnifiedCacheManager instance
        """
        self.cache = cache_manager
    
    async def create_session(
        self,
        session_type: str,
        data: Dict = None,
        user_id: str = None,
        domain: str = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            session_type: Type of session
            data: Initial session data
            user_id: User identifier
            domain: Domain context
            ip_address: Client IP
            user_agent: Client user agent
        
        Returns:
            Created Session
        """
        from langscope.cache.categories import CacheCategory
        
        session = Session(
            session_id=str(uuid.uuid4()),
            session_type=session_type,
            user_id=user_id,
            data=data or {},
            domain=domain,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        # Store in cache (Redis + MongoDB)
        await self.cache.set(
            CacheCategory.SESSION,
            session.session_id,
            session.to_dict(),
            ttl=session.remaining_ttl,
        )
        
        logger.info(f"Created session: {session.session_id} ({session_type})")
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get and refresh session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session or None
        """
        from langscope.cache.categories import CacheCategory
        
        data = await self.cache.get(CacheCategory.SESSION, session_id)
        if not data:
            return None
        
        session = Session.from_dict(data)
        
        # Check if expired
        if session.is_expired or session.status != "active":
            return None
        
        # Refresh activity
        session.refresh()
        await self.cache.set(
            CacheCategory.SESSION,
            session_id,
            session.to_dict(),
            ttl=session.remaining_ttl,
        )
        
        return session
    
    async def update_session(
        self,
        session_id: str,
        data: Dict = None,
        status: str = None,
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            data: Data to merge
            status: New status
        
        Returns:
            True if updated
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        if data:
            session.data.update(data)
        if status:
            session.status = status
        
        session.refresh()
        
        from langscope.cache.categories import CacheCategory
        await self.cache.set(
            CacheCategory.SESSION,
            session_id,
            session.to_dict(),
            ttl=session.remaining_ttl,
        )
        
        return True
    
    async def end_session(self, session_id: str) -> bool:
        """
        Mark session as completed.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if ended
        """
        return await self.update_session(session_id, status="completed")
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session from all layers.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted
        """
        from langscope.cache.categories import CacheCategory
        return await self.cache.delete(CacheCategory.SESSION, session_id)
    
    async def list_active_sessions(
        self,
        session_type: str = None,
        user_id: str = None,
        limit: int = 100,
    ) -> List[Session]:
        """
        List active sessions.
        
        Note: This requires MongoDB query, not available from Redis.
        
        Args:
            session_type: Filter by type
            user_id: Filter by user
            limit: Maximum results
        
        Returns:
            List of active sessions
        """
        if not self.cache._mongodb:
            return []
        
        try:
            db = self.cache._mongodb[self.cache._db_name]
            
            query = {"status": "active", "expires_at": {"$gt": datetime.utcnow()}}
            if session_type:
                query["session_type"] = session_type
            if user_id:
                query["user_id"] = user_id
            
            cursor = db.sessions.find(query).limit(limit)
            sessions = []
            
            async for doc in cursor:
                try:
                    sessions.append(Session.from_dict(doc))
                except Exception as e:
                    logger.warning(f"Failed to parse session: {e}")
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

