"""
Rating trend analysis.

Provides functions for analyzing how ratings evolve over time.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import math

# Import MongoDB type for type hints (optional, for IDE support)
try:
    from langscope.database.mongodb import MongoDB
except ImportError:
    MongoDB = None


@dataclass
class TrendPoint:
    """A single point in a rating trend."""
    timestamp: str
    mu: float
    sigma: float
    matches_played: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "mu": self.mu,
            "sigma": self.sigma,
            "matches_played": self.matches_played,
        }


@dataclass
class RatingTrend:
    """Rating trend over time."""
    deployment_id: str
    domain: str
    dimension: str  # "raw" or "cost_adjusted"
    
    start_date: str
    end_date: str
    granularity: str  # "hour", "day", "week"
    
    points: List[TrendPoint] = field(default_factory=list)
    
    # Summary statistics
    start_mu: float = 0.0
    end_mu: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    peak_mu: float = 0.0
    trough_mu: float = 0.0
    volatility: float = 0.0
    
    def compute_statistics(self):
        """Compute summary statistics from points."""
        if not self.points:
            return
        
        mus = [p.mu for p in self.points]
        
        self.start_mu = mus[0]
        self.end_mu = mus[-1]
        self.change = self.end_mu - self.start_mu
        
        if self.start_mu != 0:
            self.change_pct = (self.change / self.start_mu) * 100
        
        self.peak_mu = max(mus)
        self.trough_mu = min(mus)
        
        # Volatility = standard deviation of changes
        if len(mus) > 1:
            changes = [mus[i+1] - mus[i] for i in range(len(mus)-1)]
            mean_change = sum(changes) / len(changes)
            variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
            self.volatility = math.sqrt(variance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "domain": self.domain,
            "dimension": self.dimension,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "granularity": self.granularity,
            "points": [p.to_dict() for p in self.points],
            "start_mu": self.start_mu,
            "end_mu": self.end_mu,
            "change": self.change,
            "change_pct": self.change_pct,
            "peak_mu": self.peak_mu,
            "trough_mu": self.trough_mu,
            "volatility": self.volatility,
        }


def get_rating_trend(
    db,
    deployment_id: str,
    domain: str = "",
    dimension: str = "raw",
    start_date: datetime = None,
    end_date: datetime = None,
    granularity: str = "day"
) -> RatingTrend:
    """
    Get rating trend for a deployment over time.
    
    Args:
        db: MongoDB instance
        deployment_id: Deployment ID
        domain: Optional domain filter
        dimension: Rating dimension ("raw" or "cost_adjusted")
        start_date: Start of period (default: 30 days ago)
        end_date: End of period (default: now)
        granularity: Time granularity ("hour", "day", "week")
    
    Returns:
        RatingTrend with historical data points
    """
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    trend = RatingTrend(
        deployment_id=deployment_id,
        domain=domain,
        dimension=dimension,
        start_date=start_date.isoformat() + "Z",
        end_date=end_date.isoformat() + "Z",
        granularity=granularity,
    )
    
    # Query from time series collection
    if db is not None and hasattr(db, 'get_rating_history'):
        history = db.get_rating_history(
            deployment_id=deployment_id,
            domain=domain if domain else None,
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Get all points within range
        )
        
        # Convert to TrendPoints with granularity bucketing
        points_by_bucket: Dict[str, List[Dict]] = {}
        
        for record in history:
            ts = record.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif not isinstance(ts, datetime):
                continue
            
            # Bucket by granularity
            if granularity == "hour":
                bucket_key = ts.strftime("%Y-%m-%d %H:00:00")
            elif granularity == "week":
                # Get start of week
                week_start = ts - timedelta(days=ts.weekday())
                bucket_key = week_start.strftime("%Y-%m-%d")
            else:  # day
                bucket_key = ts.strftime("%Y-%m-%d")
            
            if bucket_key not in points_by_bucket:
                points_by_bucket[bucket_key] = []
            points_by_bucket[bucket_key].append(record)
        
        # Convert buckets to TrendPoints (use last value in each bucket)
        for bucket_key in sorted(points_by_bucket.keys()):
            bucket_records = points_by_bucket[bucket_key]
            # Sort by timestamp and take the last one
            bucket_records.sort(
                key=lambda r: r.get("timestamp", ""),
                reverse=True
            )
            record = bucket_records[0]
            
            # Get the right dimension value
            if dimension == "raw":
                ts_data = record.get("trueskill_raw", {})
            else:
                ts_data = record.get("trueskill_cost", {})
            
            mu = ts_data.get("mu", 0.0)
            sigma = ts_data.get("sigma", 0.0)
            
            # Get matches played from stats if available
            stats = record.get("stats", {})
            matches_played = stats.get("matches_played", 0)
            
            trend.points.append(TrendPoint(
                timestamp=bucket_key,
                mu=mu,
                sigma=sigma,
                matches_played=matches_played,
            ))
        
        # Sort points by timestamp (ascending for proper statistics)
        trend.points.sort(key=lambda p: p.timestamp)
        
        # Compute statistics
        trend.compute_statistics()
    
    return trend


def get_top_improvers(
    db,
    domain: str = "",
    period_days: int = 30,
    limit: int = 10,
    dimension: str = "raw"
) -> List[Dict[str, Any]]:
    """
    Get deployments that improved the most over a period.
    
    Args:
        db: MongoDB instance
        domain: Optional domain filter
        period_days: Number of days to look back
        limit: Maximum results
        dimension: Rating dimension
    
    Returns:
        List of {deployment_id, name, start_mu, end_mu, change, change_pct}
    """
    if db is None or not hasattr(db, 'get_all_deployments'):
        return []
    
    # Get all active deployments
    deployments = db.get_all_deployments(limit=500)
    
    improvements = []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=period_days)
    
    for deployment in deployments:
        deployment_id = deployment.get("_id", deployment.get("id", ""))
        if not deployment_id:
            continue
        
        # Get rating at start and end of period
        start_rating = db.get_rating_at_time(deployment_id, domain, start_date)
        
        # Get current rating from deployment document
        if domain:
            current_ts = deployment.get("trueskill_by_domain", {}).get(domain, {})
        else:
            current_ts = deployment.get("trueskill", {})
        
        if dimension == "raw":
            end_mu = current_ts.get("raw", {}).get("mu", 0)
        else:
            end_mu = current_ts.get("cost_adjusted", {}).get("mu", 0)
        
        # Get start mu from historical rating
        if start_rating:
            if dimension == "raw":
                start_mu = start_rating.get("trueskill_raw", {}).get("mu", end_mu)
            else:
                start_mu = start_rating.get("trueskill_cost", {}).get("mu", end_mu)
        else:
            start_mu = end_mu  # No history, no change
        
        change = end_mu - start_mu
        change_pct = (change / start_mu * 100) if start_mu != 0 else 0
        
        improvements.append({
            "deployment_id": deployment_id,
            "name": deployment.get("name", deployment_id),
            "provider": deployment.get("provider", {}).get("name", ""),
            "start_mu": start_mu,
            "end_mu": end_mu,
            "change": change,
            "change_pct": change_pct,
        })
    
    # Sort by change (descending) and take top N
    improvements.sort(key=lambda x: x["change"], reverse=True)
    
    return improvements[:limit]


def compute_volatility(
    db,
    deployment_id: str,
    domain: str = "",
    window_days: int = 30,
    dimension: str = "raw"
) -> Dict[str, Any]:
    """
    Compute rating volatility for a deployment.
    
    Volatility indicates how stable a model's ratings are.
    High volatility may indicate inconsistent performance.
    
    Args:
        db: MongoDB instance
        deployment_id: Deployment ID
        domain: Optional domain filter
        window_days: Analysis window
        dimension: Rating dimension
    
    Returns:
        {volatility, mean_change, max_change, n_matches}
    """
    trend = get_rating_trend(
        db, deployment_id, domain, dimension,
        datetime.utcnow() - timedelta(days=window_days),
        datetime.utcnow(),
        "day"
    )
    
    return {
        "deployment_id": deployment_id,
        "domain": domain,
        "dimension": dimension,
        "window_days": window_days,
        "volatility": trend.volatility,
        "change": trend.change,
        "change_pct": trend.change_pct,
        "n_points": len(trend.points),
    }


def get_rating_at_time(
    db,
    deployment_id: str,
    domain: str,
    timestamp: datetime,
    dimension: str = "raw"
) -> Optional[Dict[str, Any]]:
    """
    Get a deployment's rating at a specific point in time.
    
    Useful for rollback or historical analysis.
    
    Args:
        db: MongoDB instance
        deployment_id: Deployment ID
        domain: Domain name
        timestamp: Point in time to query
        dimension: Rating dimension
    
    Returns:
        {mu, sigma, matches_played} or None
    """
    if db is None or not hasattr(db, 'get_rating_at_time'):
        return None
    
    record = db.get_rating_at_time(deployment_id, domain, timestamp)
    
    if not record:
        return None
    
    # Extract the right dimension
    if dimension == "raw":
        ts_data = record.get("trueskill_raw", {})
    else:
        ts_data = record.get("trueskill_cost", {})
    
    stats = record.get("stats", {})
    
    return {
        "mu": ts_data.get("mu", 0.0),
        "sigma": ts_data.get("sigma", 0.0),
        "matches_played": stats.get("matches_played", 0),
        "timestamp": record.get("timestamp"),
    }

