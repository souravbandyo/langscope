"""
Monitoring API Routes.

Provides endpoints for:
- Dashboard data
- Active alerts
- System health
- Coverage monitoring
"""

from typing import List, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from langscope.api.dependencies import get_db
from langscope.monitoring.dashboard import DashboardAggregator, DashboardData
from langscope.monitoring.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    LeaderboardFreshnessAlert,
    ErrorRateAlert,
    CoverageAlert,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# =============================================================================
# Response Models
# =============================================================================

class AlertResponse(BaseModel):
    """Alert response model."""
    alert_id: str
    alert_type: str
    level: str
    message: str
    domain: Optional[str] = None
    details: dict = Field(default_factory=dict)
    timestamp: str
    resolved: bool = False


class DashboardResponse(BaseModel):
    """Dashboard response model."""
    generated_at: str
    match_activity: dict
    coverage: dict
    errors: dict
    leaderboard: dict
    system_health: dict


class HealthResponse(BaseModel):
    """System health response."""
    status: str
    database_connected: bool
    collections: dict = Field(default_factory=dict)
    uptime: Optional[str] = None


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(db=Depends(get_db)):
    """
    Get monitoring dashboard data.
    
    Returns aggregated statistics for:
    - Match activity (subjective and ground truth)
    - Sample coverage
    - Error rates
    - Leaderboard freshness
    - System health
    """
    aggregator = DashboardAggregator(db=db)
    data = aggregator.get_dashboard_data()
    return DashboardResponse(**data.to_dict())


@router.get("/health", response_model=HealthResponse)
async def get_health(db=Depends(get_db)):
    """Get system health status."""
    health = {
        "status": "unknown",
        "database_connected": False,
        "collections": {},
    }
    
    try:
        if db and db.connected:
            health["database_connected"] = True
            health["collections"]["models"] = db.db["models"].count_documents({})
            health["collections"]["matches"] = db.db["matches"].count_documents({})
            health["collections"]["gt_samples"] = db.db["ground_truth_samples"].count_documents({})
            health["collections"]["gt_matches"] = db.db["ground_truth_matches"].count_documents({})
            health["status"] = "healthy"
        else:
            health["status"] = "database_disconnected"
    except Exception as e:
        health["status"] = "error"
        logger.error(f"Health check error: {e}")
    
    return HealthResponse(**health)


# =============================================================================
# Alert Endpoints
# =============================================================================

# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(db=None) -> AlertManager:
    """Get or create alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(db=db)
        _alert_manager.add_default_rules()
    elif db:
        _alert_manager.db = db
    return _alert_manager


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    level: Optional[str] = Query(None, description="Filter by level (info, warning, error, critical)"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    include_resolved: bool = Query(False, description="Include resolved alerts"),
    db=Depends(get_db)
):
    """
    Get active alerts.
    
    Optionally filter by level, domain, or include resolved alerts.
    """
    manager = get_alert_manager(db)
    
    # Run checks to update alerts
    manager.check_all()
    
    # Get alerts
    alert_level = AlertLevel(level) if level else None
    
    if include_resolved:
        alerts = list(manager.active_alerts.values())
        if alert_level:
            alerts = [a for a in alerts if a.level == alert_level]
        if domain:
            alerts = [a for a in alerts if a.domain == domain]
    else:
        alerts = manager.get_active_alerts(level=alert_level, domain=domain)
    
    return [AlertResponse(**a.to_dict()) for a in alerts]


@router.post("/alerts/check")
async def check_alerts(db=Depends(get_db)):
    """
    Manually trigger alert checks.
    
    Returns newly triggered alerts.
    """
    manager = get_alert_manager(db)
    alerts = manager.check_all()
    
    return {
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "alerts_triggered": len(alerts),
        "alerts": [AlertResponse(**a.to_dict()) for a in alerts]
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    db=Depends(get_db)
):
    """Resolve an alert by ID."""
    manager = get_alert_manager(db)
    
    if alert_id not in manager.active_alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    manager.resolve_alert(alert_id)
    manager.persist_alerts()
    
    return {"status": "resolved", "alert_id": alert_id}


# =============================================================================
# Coverage Monitoring
# =============================================================================

@router.get("/coverage")
async def get_coverage_summary(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    db=Depends(get_db)
):
    """
    Get sample coverage summary.
    
    Shows how many ground truth samples have been used in evaluations.
    """
    aggregator = DashboardAggregator(db=db)
    data = aggregator.get_dashboard_data()
    
    coverage = data.coverage
    
    if domain:
        domain_coverage = coverage.by_domain.get(domain, {})
        return {
            "domain": domain,
            "total_samples": domain_coverage.get("total", 0),
            "samples_used": domain_coverage.get("used", 0),
            "coverage_percentage": domain_coverage.get("coverage", 0),
        }
    
    return {
        "total_samples": coverage.total_samples,
        "samples_used": coverage.samples_used,
        "coverage_percentage": coverage.coverage_percentage,
        "by_domain": coverage.by_domain,
        "by_difficulty": coverage.by_difficulty,
        "by_language": coverage.by_language,
    }


@router.get("/coverage/{domain}")
async def get_domain_coverage(
    domain: str,
    db=Depends(get_db)
):
    """Get detailed coverage for a specific domain."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get sample statistics
        total = db.db["ground_truth_samples"].count_documents({"domain": domain})
        used = db.db["ground_truth_samples"].count_documents({
            "domain": domain,
            "usage_count": {"$gt": 0}
        })
        
        coverage = (used / total * 100) if total > 0 else 0
        
        # Get by difficulty
        difficulty_pipeline = [
            {"$match": {"domain": domain}},
            {"$group": {
                "_id": "$difficulty",
                "total": {"$sum": 1},
                "used": {"$sum": {"$cond": [{"$gt": ["$usage_count", 0]}, 1, 0]}}
            }}
        ]
        by_difficulty = {}
        for doc in db.db["ground_truth_samples"].aggregate(difficulty_pipeline):
            diff = doc["_id"] or "medium"
            by_difficulty[diff] = {
                "total": doc["total"],
                "used": doc["used"],
                "coverage": (doc["used"] / doc["total"] * 100) if doc["total"] > 0 else 0
            }
        
        # Get most/least used samples
        most_used = list(db.db["ground_truth_samples"].find(
            {"domain": domain, "usage_count": {"$gt": 0}}
        ).sort("usage_count", -1).limit(5))
        
        unused = list(db.db["ground_truth_samples"].find(
            {"domain": domain, "usage_count": 0}
        ).limit(10))
        
        return {
            "domain": domain,
            "total_samples": total,
            "samples_used": used,
            "samples_unused": total - used,
            "coverage_percentage": round(coverage, 2),
            "by_difficulty": by_difficulty,
            "most_used_samples": [
                {"sample_id": s.get("_id"), "usage_count": s.get("usage_count", 0)}
                for s in most_used
            ],
            "unused_sample_ids": [s.get("_id") for s in unused],
        }
        
    except Exception as e:
        logger.error(f"Error getting domain coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Error Tracking
# =============================================================================

@router.get("/errors")
async def get_error_summary(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    domain: Optional[str] = Query(None),
    db=Depends(get_db)
):
    """
    Get error summary for the specified time window.
    
    Returns error counts, rates, and recent error details.
    """
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    from datetime import timedelta
    
    now = datetime.utcnow()
    window_start = now - timedelta(hours=hours)
    
    try:
        query = {"timestamp": {"$gte": window_start.isoformat()}}
        if domain:
            query["domain"] = domain
        
        total = db.db["ground_truth_matches"].count_documents(query)
        
        query["status"] = "failed"
        failed = db.db["ground_truth_matches"].count_documents(query)
        
        error_rate = (failed / total * 100) if total > 0 else 0
        
        # Get recent errors
        recent_errors = list(db.db["ground_truth_matches"].find(
            query
        ).sort("timestamp", -1).limit(10))
        
        return {
            "time_window_hours": hours,
            "domain": domain,
            "total_matches": total,
            "failed_matches": failed,
            "error_rate_percentage": round(error_rate, 2),
            "recent_errors": [
                {
                    "match_id": e.get("_id"),
                    "domain": e.get("domain"),
                    "timestamp": e.get("timestamp"),
                    "error": e.get("error_message", "Unknown"),
                }
                for e in recent_errors
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting error summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Leaderboard Freshness
# =============================================================================

@router.get("/freshness")
async def get_leaderboard_freshness(
    threshold_hours: float = Query(24.0, ge=1, description="Stale threshold in hours"),
    db=Depends(get_db)
):
    """
    Get leaderboard freshness status for all domains.
    
    Returns when each domain was last updated and flags stale domains.
    """
    aggregator = DashboardAggregator(db=db, stale_threshold_hours=threshold_hours)
    data = aggregator.get_dashboard_data()
    
    return {
        "threshold_hours": threshold_hours,
        "last_updated": data.leaderboard.last_updated,
        "freshness_hours": data.leaderboard.freshness_hours,
        "stale_domains": data.leaderboard.stale_domains,
        "total_domains": len(data.leaderboard.last_updated),
        "stale_count": len(data.leaderboard.stale_domains),
    }


