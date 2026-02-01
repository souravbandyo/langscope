"""
In-memory cache with TTL for parameter management.

Provides efficient caching of database parameters with automatic expiration.
"""

import time
import threading
from typing import Dict, Optional, Any, Generic, TypeVar
from dataclasses import dataclass
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: float
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class ParamCache:
    """
    In-memory cache for parameters with TTL-based expiration.
    
    Thread-safe implementation with automatic cleanup of expired entries.
    """
    
    def __init__(self, default_ttl: int = 300, cleanup_interval: int = 60):
        """
        Initialize parameter cache.
        
        Args:
            default_ttl: Default time-to-live in seconds (5 minutes)
            cleanup_interval: Interval for background cleanup (60 seconds)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
    
    def _make_key(self, param_type: str, domain: Optional[str] = None) -> str:
        """Create cache key from param type and optional domain."""
        if domain:
            return f"{param_type}:{domain}"
        return param_type
    
    def get(
        self,
        param_type: str,
        domain: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get cached parameter value.
        
        Args:
            param_type: Parameter type identifier
            domain: Optional domain for domain-specific params
        
        Returns:
            Cached value or None if not found/expired
        """
        key = self._make_key(param_type, domain)
        
        with self._lock:
            self._maybe_cleanup()
            
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(
        self,
        param_type: str,
        value: Any,
        domain: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set cached parameter value.
        
        Args:
            param_type: Parameter type identifier
            value: Value to cache
            domain: Optional domain for domain-specific params
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = self._make_key(param_type, domain)
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
    
    def invalidate(
        self,
        param_type: str,
        domain: Optional[str] = None
    ) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            param_type: Parameter type identifier
            domain: Optional domain for domain-specific params
        
        Returns:
            True if entry was found and removed
        """
        key = self._make_key(param_type, domain)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def invalidate_type(self, param_type: str) -> int:
        """
        Invalidate all cache entries for a parameter type.
        
        Removes both global and all domain-specific entries.
        
        Args:
            param_type: Parameter type identifier
        
        Returns:
            Number of entries removed
        """
        count = 0
        prefix = f"{param_type}:"
        
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key == param_type or key.startswith(prefix)
            ]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
        
        return count
    
    def invalidate_domain(self, domain: str) -> int:
        """
        Invalidate all cache entries for a domain.
        
        Args:
            domain: Domain name
        
        Returns:
            Number of entries removed
        """
        count = 0
        suffix = f":{domain}"
        
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if key.endswith(suffix)
            ]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
        
        return count
    
    def invalidate_all(self) -> int:
        """
        Invalidate all cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def _maybe_cleanup(self) -> None:
        """Cleanup expired entries if cleanup interval has passed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._cleanup_expired()
        self._last_cleanup = now
    
    def _cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        count = 0
        
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            now = time.time()
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired,
                "active_entries": len(self._cache) - expired,
                "default_ttl": self._default_ttl,
                "cleanup_interval": self._cleanup_interval,
            }
    
    def __len__(self) -> int:
        """Get number of entries (including expired)."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (doesn't check expiration)."""
        return key in self._cache


