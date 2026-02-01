"""
Dashboard data aggregation for LangScope monitoring.

Provides real-time statistics for:
- Match activity (subjective and ground truth)
- Sample coverage
- Model performance
- System health
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


@dataclass
class MatchActivityStats:
    """Statistics for match activity."""
    total_matches: int = 0
    matches_last_24h: int = 0
    matches_last_7d: int = 0
    subjective_matches: int = 0
    ground_truth_matches: int = 0
    avg_matches_per_day: float = 0.0
    by_domain: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)


@dataclass
class CoverageStats:
    """Statistics for sample coverage."""
    total_samples: int = 0
    samples_used: int = 0
    coverage_percentage: float = 0.0
    by_domain: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    by_language: Dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorStats:
    """Statistics for error rates."""
    total_errors: int = 0
    errors_last_24h: int = 0
    error_rate_24h: float = 0.0
    by_domain: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LeaderboardStats:
    """Statistics for leaderboard freshness."""
    last_updated: Dict[str, str] = field(default_factory=dict)
    stale_domains: List[str] = field(default_factory=list)
    freshness_hours: Dict[str, float] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Complete dashboard data."""
    generated_at: str = ""
    match_activity: MatchActivityStats = field(default_factory=MatchActivityStats)
    coverage: CoverageStats = field(default_factory=CoverageStats)
    errors: ErrorStats = field(default_factory=ErrorStats)
    leaderboard: LeaderboardStats = field(default_factory=LeaderboardStats)
    system_health: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at,
            "match_activity": {
                "total_matches": self.match_activity.total_matches,
                "matches_last_24h": self.match_activity.matches_last_24h,
                "matches_last_7d": self.match_activity.matches_last_7d,
                "subjective_matches": self.match_activity.subjective_matches,
                "ground_truth_matches": self.match_activity.ground_truth_matches,
                "avg_matches_per_day": self.match_activity.avg_matches_per_day,
                "by_domain": self.match_activity.by_domain,
                "by_status": self.match_activity.by_status,
            },
            "coverage": {
                "total_samples": self.coverage.total_samples,
                "samples_used": self.coverage.samples_used,
                "coverage_percentage": self.coverage.coverage_percentage,
                "by_domain": self.coverage.by_domain,
                "by_difficulty": self.coverage.by_difficulty,
                "by_language": self.coverage.by_language,
            },
            "errors": {
                "total_errors": self.errors.total_errors,
                "errors_last_24h": self.errors.errors_last_24h,
                "error_rate_24h": self.errors.error_rate_24h,
                "by_domain": self.errors.by_domain,
                "by_type": self.errors.by_type,
                "recent_errors": self.errors.recent_errors[:10],  # Last 10 errors
            },
            "leaderboard": {
                "last_updated": self.leaderboard.last_updated,
                "stale_domains": self.leaderboard.stale_domains,
                "freshness_hours": self.leaderboard.freshness_hours,
            },
            "system_health": self.system_health,
        }


class DashboardAggregator:
    """
    Aggregates data for the monitoring dashboard.
    
    Queries the database for match statistics, coverage data,
    error rates, and leaderboard freshness.
    """
    
    def __init__(
        self,
        db: 'MongoDB' = None,
        stale_threshold_hours: float = 24.0
    ):
        """
        Initialize aggregator.
        
        Args:
            db: Database instance
            stale_threshold_hours: Hours after which a leaderboard is considered stale
        """
        self.db = db
        self.stale_threshold_hours = stale_threshold_hours
    
    def get_dashboard_data(self) -> DashboardData:
        """Get complete dashboard data."""
        data = DashboardData()
        
        if self.db and self.db.connected:
            data.match_activity = self._get_match_activity()
            data.coverage = self._get_coverage_stats()
            data.errors = self._get_error_stats()
            data.leaderboard = self._get_leaderboard_stats()
            data.system_health = self._get_system_health()
        
        return data
    
    def _get_match_activity(self) -> MatchActivityStats:
        """Get match activity statistics."""
        stats = MatchActivityStats()
        
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        try:
            # Total subjective matches
            stats.subjective_matches = self.db.db["matches"].count_documents({})
            
            # Total GT matches
            stats.ground_truth_matches = self.db.db["ground_truth_matches"].count_documents({})
            
            stats.total_matches = stats.subjective_matches + stats.ground_truth_matches
            
            # Last 24h
            stats.matches_last_24h = (
                self.db.db["matches"].count_documents({
                    "timestamp": {"$gte": day_ago.isoformat()}
                }) +
                self.db.db["ground_truth_matches"].count_documents({
                    "timestamp": {"$gte": day_ago.isoformat()}
                })
            )
            
            # Last 7 days
            stats.matches_last_7d = (
                self.db.db["matches"].count_documents({
                    "timestamp": {"$gte": week_ago.isoformat()}
                }) +
                self.db.db["ground_truth_matches"].count_documents({
                    "timestamp": {"$gte": week_ago.isoformat()}
                })
            )
            
            # Average per day
            if stats.matches_last_7d > 0:
                stats.avg_matches_per_day = stats.matches_last_7d / 7.0
            
            # By domain (subjective)
            pipeline = [
                {"$group": {"_id": "$domain", "count": {"$sum": 1}}}
            ]
            for doc in self.db.db["matches"].aggregate(pipeline):
                stats.by_domain[doc["_id"]] = doc["count"]
            
            # By domain (GT)
            for doc in self.db.db["ground_truth_matches"].aggregate(pipeline):
                domain = doc["_id"]
                stats.by_domain[domain] = stats.by_domain.get(domain, 0) + doc["count"]
            
            # By status
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            for doc in self.db.db["ground_truth_matches"].aggregate(status_pipeline):
                stats.by_status[doc["_id"] or "completed"] = doc["count"]
                
        except Exception as e:
            logger.error(f"Error getting match activity: {e}")
        
        return stats
    
    def _get_coverage_stats(self) -> CoverageStats:
        """Get sample coverage statistics."""
        stats = CoverageStats()
        
        try:
            # Total samples
            stats.total_samples = self.db.db["ground_truth_samples"].count_documents({})
            
            # Used samples
            stats.samples_used = self.db.db["ground_truth_samples"].count_documents({
                "usage_count": {"$gt": 0}
            })
            
            # Coverage percentage
            if stats.total_samples > 0:
                stats.coverage_percentage = (stats.samples_used / stats.total_samples) * 100
            
            # By domain
            domain_pipeline = [
                {"$group": {
                    "_id": "$domain",
                    "total": {"$sum": 1},
                    "used": {"$sum": {"$cond": [{"$gt": ["$usage_count", 0]}, 1, 0]}}
                }}
            ]
            for doc in self.db.db["ground_truth_samples"].aggregate(domain_pipeline):
                domain = doc["_id"]
                total = doc["total"]
                used = doc["used"]
                stats.by_domain[domain] = {
                    "total": total,
                    "used": used,
                    "coverage": (used / total * 100) if total > 0 else 0
                }
            
            # By difficulty
            difficulty_pipeline = [
                {"$group": {"_id": "$difficulty", "count": {"$sum": 1}}}
            ]
            for doc in self.db.db["ground_truth_samples"].aggregate(difficulty_pipeline):
                stats.by_difficulty[doc["_id"] or "medium"] = doc["count"]
            
            # By language
            language_pipeline = [
                {"$group": {"_id": "$metadata.language", "count": {"$sum": 1}}}
            ]
            for doc in self.db.db["ground_truth_samples"].aggregate(language_pipeline):
                stats.by_language[doc["_id"] or "en"] = doc["count"]
                
        except Exception as e:
            logger.error(f"Error getting coverage stats: {e}")
        
        return stats
    
    def _get_error_stats(self) -> ErrorStats:
        """Get error statistics."""
        stats = ErrorStats()
        
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        
        try:
            # Failed GT matches
            stats.total_errors = self.db.db["ground_truth_matches"].count_documents({
                "status": "failed"
            })
            
            stats.errors_last_24h = self.db.db["ground_truth_matches"].count_documents({
                "status": "failed",
                "timestamp": {"$gte": day_ago.isoformat()}
            })
            
            # Error rate
            total_24h = self.db.db["ground_truth_matches"].count_documents({
                "timestamp": {"$gte": day_ago.isoformat()}
            })
            if total_24h > 0:
                stats.error_rate_24h = (stats.errors_last_24h / total_24h) * 100
            
            # By domain
            error_domain_pipeline = [
                {"$match": {"status": "failed"}},
                {"$group": {"_id": "$domain", "count": {"$sum": 1}}}
            ]
            for doc in self.db.db["ground_truth_matches"].aggregate(error_domain_pipeline):
                stats.by_domain[doc["_id"]] = doc["count"]
            
            # Recent errors
            recent = self.db.db["ground_truth_matches"].find(
                {"status": "failed"}
            ).sort("timestamp", -1).limit(10)
            
            for doc in recent:
                stats.recent_errors.append({
                    "match_id": doc.get("_id"),
                    "domain": doc.get("domain"),
                    "timestamp": doc.get("timestamp"),
                    "error": doc.get("error_message", "Unknown error")
                })
                
        except Exception as e:
            logger.error(f"Error getting error stats: {e}")
        
        return stats
    
    def _get_leaderboard_stats(self) -> LeaderboardStats:
        """Get leaderboard freshness statistics."""
        stats = LeaderboardStats()
        
        now = datetime.utcnow()
        
        try:
            from langscope.domain.domain_config import DOMAIN_TEMPLATES
            
            for domain_name in DOMAIN_TEMPLATES:
                # Get last match timestamp
                last_match = None
                
                # Check subjective matches
                subjective = self.db.db["matches"].find_one(
                    {"domain": domain_name},
                    sort=[("timestamp", -1)]
                )
                if subjective:
                    last_match = subjective.get("timestamp")
                
                # Check GT matches
                gt = self.db.db["ground_truth_matches"].find_one(
                    {"domain": domain_name},
                    sort=[("timestamp", -1)]
                )
                if gt:
                    gt_timestamp = gt.get("timestamp")
                    if not last_match or gt_timestamp > last_match:
                        last_match = gt_timestamp
                
                if last_match:
                    stats.last_updated[domain_name] = last_match
                    
                    # Calculate freshness
                    try:
                        last_dt = datetime.fromisoformat(last_match.replace("Z", "+00:00"))
                        hours_ago = (now - last_dt.replace(tzinfo=None)).total_seconds() / 3600
                        stats.freshness_hours[domain_name] = round(hours_ago, 2)
                        
                        if hours_ago > self.stale_threshold_hours:
                            stats.stale_domains.append(domain_name)
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            logger.error(f"Error getting leaderboard stats: {e}")
        
        return stats
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        health = {
            "database_connected": False,
            "collections": {},
            "status": "unknown"
        }
        
        try:
            if self.db and self.db.connected:
                health["database_connected"] = True
                
                # Collection counts
                health["collections"]["models"] = self.db.db["models"].count_documents({})
                health["collections"]["matches"] = self.db.db["matches"].count_documents({})
                health["collections"]["gt_samples"] = self.db.db["ground_truth_samples"].count_documents({})
                health["collections"]["gt_matches"] = self.db.db["ground_truth_matches"].count_documents({})
                
                health["status"] = "healthy"
            else:
                health["status"] = "database_disconnected"
                
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
        
        return health


# =============================================================================
# Convenience Functions
# =============================================================================

_default_aggregator: Optional[DashboardAggregator] = None


def get_dashboard_data(db: 'MongoDB' = None) -> DashboardData:
    """Get dashboard data using provided or default database."""
    global _default_aggregator
    
    if db:
        aggregator = DashboardAggregator(db=db)
    elif _default_aggregator:
        aggregator = _default_aggregator
    else:
        aggregator = DashboardAggregator()
    
    return aggregator.get_dashboard_data()


def set_default_aggregator(aggregator: DashboardAggregator):
    """Set the default dashboard aggregator."""
    global _default_aggregator
    _default_aggregator = aggregator


