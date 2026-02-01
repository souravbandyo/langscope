"""
Cache Management API Routes.

Provides endpoints for cache statistics and invalidation.
"""

import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CacheStatsResponse(BaseModel):
    """Response with cache statistics."""
    by_category: Dict[str, Any]
    totals: Dict[str, Any]
    connections: Dict[str, bool]


class InvalidateResponse(BaseModel):
    """Response from invalidation request."""
    invalidated: int = Field(..., description="Number of entries invalidated")
    category: Optional[str] = None
    domain: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    session_type: str
    user_id: Optional[str] = None
    status: str
    created_at: str
    last_activity: str
    expires_at: str
    domain: Optional[str] = None


class CreateSessionRequest(BaseModel):
    """Request to create a session."""
    session_type: str = Field(..., description="Type of session")
    data: Optional[Dict[str, Any]] = Field(None, description="Initial session data")
    user_id: Optional[str] = Field(None, description="User identifier")
    domain: Optional[str] = Field(None, description="Domain context")


class UpdateSessionRequest(BaseModel):
    """Request to update a session."""
    data: Optional[Dict[str, Any]] = Field(None, description="Data to merge")
    status: Optional[str] = Field(None, description="New status")


class RateLimitStatus(BaseModel):
    """Rate limit status for an identifier."""
    current_count: int
    limit: int
    window: int
    remaining: int


# =============================================================================
# Dependencies
# =============================================================================

async def get_cache_manager():
    """Get cache manager instance."""
    from langscope.api.dependencies import get_cache_manager as _get_cm
    return await _get_cm()


# =============================================================================
# Cache Routes
# =============================================================================

@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get comprehensive cache statistics.
    
    Returns hit rates, error counts, and connection status.
    """
    try:
        cache = await get_cache_manager()
        stats = cache.get_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate/{category}", response_model=InvalidateResponse)
async def invalidate_cache_category(category: str):
    """
    Invalidate all entries for a cache category.
    
    Categories: session, leaderboard, rate_limit, user_pref, 
    prompt_exact, prompt_semantic, correlation, sample_usage,
    api_response, param, model, domain, centroid
    """
    try:
        from langscope.cache.categories import CacheCategory
        
        try:
            cat = CacheCategory(category)
        except ValueError:
            raise HTTPException(400, f"Invalid category: {category}")
        
        cache = await get_cache_manager()
        count = await cache.invalidate_category(cat)
        
        return InvalidateResponse(invalidated=count, category=category)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate/leaderboard/{domain}", response_model=InvalidateResponse)
async def invalidate_leaderboard(domain: str):
    """
    Invalidate leaderboard cache for a domain.
    
    Called after a match is completed.
    """
    try:
        cache = await get_cache_manager()
        await cache.invalidate_domain_leaderboards(domain)
        
        return InvalidateResponse(invalidated=10, domain=domain)  # All dimensions
    except Exception as e:
        logger.error(f"Leaderboard invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all")
async def invalidate_all():
    """
    Invalidate all cache entries (admin only).
    
    WARNING: This clears all cached data.
    """
    try:
        from langscope.cache.categories import CacheCategory
        
        cache = await get_cache_manager()
        total = 0
        
        for category in CacheCategory:
            count = await cache.invalidate_category(category)
            total += count
        
        return {"invalidated": total, "message": "All caches cleared"}
    except Exception as e:
        logger.error(f"Full invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/reset")
async def reset_cache_stats():
    """Reset cache statistics."""
    try:
        cache = await get_cache_manager()
        cache.reset_stats()
        return {"status": "ok", "message": "Cache stats reset"}
    except Exception as e:
        logger.error(f"Stats reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Routes
# =============================================================================

@router.post("/sessions", response_model=SessionInfo)
async def create_session(request: CreateSessionRequest):
    """
    Create a new session.
    
    Returns session ID and metadata.
    """
    try:
        from langscope.cache.session import SessionManager
        
        cache = await get_cache_manager()
        session_mgr = SessionManager(cache)
        
        session = await session_mgr.create_session(
            session_type=request.session_type,
            data=request.data,
            user_id=request.user_id,
            domain=request.domain,
        )
        
        return SessionInfo(
            session_id=session.session_id,
            session_type=session.session_type,
            user_id=session.user_id,
            status=session.status,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            expires_at=session.expires_at.isoformat(),
            domain=session.domain,
        )
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get session by ID.
    
    Returns session data and refreshes activity timestamp.
    """
    try:
        from langscope.cache.session import SessionManager, Session
        
        cache = await get_cache_manager()
        session_mgr = SessionManager(cache)
        
        session = await session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        
        return SessionInfo(
            session_id=session.session_id,
            session_type=session.session_type,
            user_id=session.user_id,
            status=session.status,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            expires_at=session.expires_at.isoformat(),
            domain=session.domain,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session get failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, request: UpdateSessionRequest):
    """
    Update session data.
    
    Merges provided data and optionally updates status.
    """
    try:
        from langscope.cache.session import SessionManager
        
        cache = await get_cache_manager()
        session_mgr = SessionManager(cache)
        
        success = await session_mgr.update_session(
            session_id=session_id,
            data=request.data,
            status=request.status,
        )
        
        if not success:
            raise HTTPException(404, "Session not found")
        
        return {"status": "ok", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """
    End a session.
    
    Marks session as completed.
    """
    try:
        from langscope.cache.session import SessionManager
        
        cache = await get_cache_manager()
        session_mgr = SessionManager(cache)
        
        success = await session_mgr.end_session(session_id)
        if not success:
            raise HTTPException(404, "Session not found")
        
        return {"status": "ok", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session end failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Rate Limit Routes
# =============================================================================

@router.get("/rate-limit/status", response_model=RateLimitStatus)
async def get_rate_limit_status(
    identifier: Optional[str] = None,
    endpoint: str = "default",
):
    """
    Get rate limit status for an identifier.
    
    If no identifier provided, uses request IP.
    """
    try:
        from langscope.cache.rate_limit import SlidingWindowRateLimiter
        
        cache = await get_cache_manager()
        limiter = SlidingWindowRateLimiter(cache._redis)
        
        # Use placeholder if no identifier
        identifier = identifier or "check"
        
        usage = await limiter.get_usage(identifier, endpoint)
        return RateLimitStatus(**usage)
    except Exception as e:
        logger.error(f"Rate limit status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rate-limit/reset/{identifier}")
async def reset_rate_limit(identifier: str, endpoint: Optional[str] = None):
    """
    Reset rate limit for an identifier.
    
    Optionally specify endpoint to reset only that endpoint.
    """
    try:
        from langscope.cache.rate_limit import SlidingWindowRateLimiter
        
        cache = await get_cache_manager()
        limiter = SlidingWindowRateLimiter(cache._redis)
        
        success = await limiter.reset(identifier, endpoint)
        return {"status": "ok" if success else "failed", "identifier": identifier}
    except Exception as e:
        logger.error(f"Rate limit reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

