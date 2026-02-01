"""
Alert system for LangScope monitoring.

Provides configurable alerts for:
- Leaderboard freshness (stale domains)
- Error rates exceeding thresholds
- Sample coverage gaps
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert instance."""
    alert_id: str
    alert_type: str
    level: AlertLevel
    message: str
    domain: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    resolved: bool = False
    resolved_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "level": self.level.value,
            "message": self.message,
            "domain": self.domain,
            "details": self.details,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow().isoformat() + "Z"


class AlertRule(ABC):
    """Abstract base class for alert rules."""
    
    def __init__(
        self,
        name: str,
        level: AlertLevel = AlertLevel.WARNING,
        enabled: bool = True
    ):
        self.name = name
        self.level = level
        self.enabled = enabled
    
    @abstractmethod
    def check(self, db: 'MongoDB') -> List[Alert]:
        """
        Check if alert condition is met.
        
        Returns:
            List of Alert instances if condition is met, empty list otherwise
        """
        pass


class LeaderboardFreshnessAlert(AlertRule):
    """Alert when a leaderboard hasn't been updated recently."""
    
    def __init__(
        self,
        threshold_hours: float = 24.0,
        domains: List[str] = None,
        level: AlertLevel = AlertLevel.WARNING
    ):
        """
        Initialize freshness alert.
        
        Args:
            threshold_hours: Hours after which to trigger alert
            domains: Specific domains to monitor (None = all)
            level: Alert severity level
        """
        super().__init__(
            name="leaderboard_freshness",
            level=level
        )
        self.threshold_hours = threshold_hours
        self.domains = domains
    
    def check(self, db: 'MongoDB') -> List[Alert]:
        """Check for stale leaderboards."""
        if not self.enabled or not db or not db.connected:
            return []
        
        alerts = []
        now = datetime.utcnow()
        
        from langscope.domain.domain_config import DOMAIN_TEMPLATES
        
        domains_to_check = self.domains or list(DOMAIN_TEMPLATES.keys())
        
        for domain in domains_to_check:
            last_match = None
            
            # Check subjective matches
            subjective = db.db["matches"].find_one(
                {"domain": domain},
                sort=[("timestamp", -1)]
            )
            if subjective:
                last_match = subjective.get("timestamp")
            
            # Check GT matches
            gt = db.db["ground_truth_matches"].find_one(
                {"domain": domain},
                sort=[("timestamp", -1)]
            )
            if gt:
                gt_timestamp = gt.get("timestamp")
                if not last_match or gt_timestamp > last_match:
                    last_match = gt_timestamp
            
            if last_match:
                try:
                    last_dt = datetime.fromisoformat(last_match.replace("Z", "+00:00"))
                    hours_ago = (now - last_dt.replace(tzinfo=None)).total_seconds() / 3600
                    
                    if hours_ago > self.threshold_hours:
                        alerts.append(Alert(
                            alert_id=f"freshness_{domain}_{now.strftime('%Y%m%d%H')}",
                            alert_type="leaderboard_freshness",
                            level=self.level,
                            message=f"Leaderboard for '{domain}' is stale ({hours_ago:.1f} hours since last update)",
                            domain=domain,
                            details={
                                "last_updated": last_match,
                                "hours_since_update": round(hours_ago, 2),
                                "threshold_hours": self.threshold_hours,
                            }
                        ))
                except (ValueError, TypeError):
                    pass
        
        return alerts


class ErrorRateAlert(AlertRule):
    """Alert when error rate exceeds threshold."""
    
    def __init__(
        self,
        threshold_percentage: float = 5.0,
        window_hours: float = 24.0,
        min_matches: int = 10,
        level: AlertLevel = AlertLevel.ERROR
    ):
        """
        Initialize error rate alert.
        
        Args:
            threshold_percentage: Error rate threshold (0-100)
            window_hours: Time window to check
            min_matches: Minimum matches required to trigger
            level: Alert severity level
        """
        super().__init__(
            name="error_rate",
            level=level
        )
        self.threshold_percentage = threshold_percentage
        self.window_hours = window_hours
        self.min_matches = min_matches
    
    def check(self, db: 'MongoDB') -> List[Alert]:
        """Check for high error rates."""
        if not self.enabled or not db or not db.connected:
            return []
        
        alerts = []
        now = datetime.utcnow()
        window_start = now - timedelta(hours=self.window_hours)
        
        try:
            # Check GT matches
            total = db.db["ground_truth_matches"].count_documents({
                "timestamp": {"$gte": window_start.isoformat()}
            })
            
            if total >= self.min_matches:
                failed = db.db["ground_truth_matches"].count_documents({
                    "timestamp": {"$gte": window_start.isoformat()},
                    "status": "failed"
                })
                
                error_rate = (failed / total) * 100
                
                if error_rate > self.threshold_percentage:
                    alerts.append(Alert(
                        alert_id=f"error_rate_{now.strftime('%Y%m%d%H')}",
                        alert_type="error_rate",
                        level=self.level,
                        message=f"Error rate ({error_rate:.1f}%) exceeds threshold ({self.threshold_percentage}%)",
                        details={
                            "error_rate": round(error_rate, 2),
                            "threshold": self.threshold_percentage,
                            "total_matches": total,
                            "failed_matches": failed,
                            "window_hours": self.window_hours,
                        }
                    ))
            
            # Check by domain
            pipeline = [
                {"$match": {"timestamp": {"$gte": window_start.isoformat()}}},
                {"$group": {
                    "_id": "$domain",
                    "total": {"$sum": 1},
                    "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}}
                }}
            ]
            
            for doc in db.db["ground_truth_matches"].aggregate(pipeline):
                domain = doc["_id"]
                total = doc["total"]
                failed = doc["failed"]
                
                if total >= self.min_matches:
                    error_rate = (failed / total) * 100
                    
                    if error_rate > self.threshold_percentage:
                        alerts.append(Alert(
                            alert_id=f"error_rate_{domain}_{now.strftime('%Y%m%d%H')}",
                            alert_type="error_rate_domain",
                            level=self.level,
                            message=f"Error rate for '{domain}' ({error_rate:.1f}%) exceeds threshold",
                            domain=domain,
                            details={
                                "error_rate": round(error_rate, 2),
                                "threshold": self.threshold_percentage,
                                "total_matches": total,
                                "failed_matches": failed,
                            }
                        ))
                        
        except Exception as e:
            logger.error(f"Error checking error rates: {e}")
        
        return alerts


class CoverageAlert(AlertRule):
    """Alert when sample coverage is low."""
    
    def __init__(
        self,
        min_coverage_percentage: float = 50.0,
        domains: List[str] = None,
        level: AlertLevel = AlertLevel.WARNING
    ):
        """
        Initialize coverage alert.
        
        Args:
            min_coverage_percentage: Minimum coverage required
            domains: Specific domains to monitor (None = all)
            level: Alert severity level
        """
        super().__init__(
            name="coverage",
            level=level
        )
        self.min_coverage_percentage = min_coverage_percentage
        self.domains = domains
    
    def check(self, db: 'MongoDB') -> List[Alert]:
        """Check for low sample coverage."""
        if not self.enabled or not db or not db.connected:
            return []
        
        alerts = []
        now = datetime.utcnow()
        
        try:
            # Aggregate by domain
            pipeline = [
                {"$group": {
                    "_id": "$domain",
                    "total": {"$sum": 1},
                    "used": {"$sum": {"$cond": [{"$gt": ["$usage_count", 0]}, 1, 0]}}
                }}
            ]
            
            for doc in db.db["ground_truth_samples"].aggregate(pipeline):
                domain = doc["_id"]
                
                # Filter by specific domains if set
                if self.domains and domain not in self.domains:
                    continue
                
                total = doc["total"]
                used = doc["used"]
                
                if total > 0:
                    coverage = (used / total) * 100
                    
                    if coverage < self.min_coverage_percentage:
                        alerts.append(Alert(
                            alert_id=f"coverage_{domain}_{now.strftime('%Y%m%d')}",
                            alert_type="coverage",
                            level=self.level,
                            message=f"Sample coverage for '{domain}' ({coverage:.1f}%) is below threshold ({self.min_coverage_percentage}%)",
                            domain=domain,
                            details={
                                "coverage_percentage": round(coverage, 2),
                                "threshold": self.min_coverage_percentage,
                                "total_samples": total,
                                "used_samples": used,
                                "unused_samples": total - used,
                            }
                        ))
                        
        except Exception as e:
            logger.error(f"Error checking coverage: {e}")
        
        return alerts


class AlertManager:
    """
    Manages alert rules and notifications.
    
    Runs alert checks and can persist/notify about alerts.
    """
    
    def __init__(
        self,
        db: 'MongoDB' = None,
        notify_callback: Callable[[Alert], None] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            db: Database instance
            notify_callback: Function to call when alert triggers
        """
        self.db = db
        self.notify_callback = notify_callback
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def add_default_rules(self):
        """Add default alert rules."""
        self.rules = [
            LeaderboardFreshnessAlert(threshold_hours=24.0),
            ErrorRateAlert(threshold_percentage=5.0),
            CoverageAlert(min_coverage_percentage=50.0),
        ]
    
    def check_all(self) -> List[Alert]:
        """
        Run all alert checks.
        
        Returns:
            List of triggered alerts
        """
        all_alerts = []
        
        for rule in self.rules:
            try:
                alerts = rule.check(self.db)
                all_alerts.extend(alerts)
                
                # Track active alerts
                for alert in alerts:
                    if alert.alert_id not in self.active_alerts:
                        self.active_alerts[alert.alert_id] = alert
                        
                        # Notify if callback is set
                        if self.notify_callback:
                            self.notify_callback(alert)
                            
            except Exception as e:
                logger.error(f"Error running alert rule {rule.name}: {e}")
        
        return all_alerts
    
    def get_active_alerts(
        self,
        level: AlertLevel = None,
        domain: str = None
    ) -> List[Alert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            level: Filter by severity level
            domain: Filter by domain
            
        Returns:
            List of active alerts
        """
        alerts = [a for a in self.active_alerts.values() if not a.resolved]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if domain:
            alerts = [a for a in alerts if a.domain == domain]
        
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert by ID."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
    
    def persist_alerts(self):
        """Persist active alerts to database."""
        if not self.db or not self.db.connected:
            return
        
        for alert in self.active_alerts.values():
            self.db.db["alerts"].update_one(
                {"_id": alert.alert_id},
                {"$set": alert.to_dict()},
                upsert=True
            )
    
    def load_alerts(self):
        """Load active alerts from database."""
        if not self.db or not self.db.connected:
            return
        
        cursor = self.db.db["alerts"].find({"resolved": False})
        
        for doc in cursor:
            alert = Alert(
                alert_id=doc.get("_id", doc.get("alert_id")),
                alert_type=doc.get("alert_type", ""),
                level=AlertLevel(doc.get("level", "warning")),
                message=doc.get("message", ""),
                domain=doc.get("domain"),
                details=doc.get("details", {}),
                timestamp=doc.get("timestamp", ""),
                resolved=doc.get("resolved", False),
            )
            self.active_alerts[alert.alert_id] = alert


