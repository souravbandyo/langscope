"""
Rating history for tracking TrueSkill changes over time.

Stored as MongoDB time series for efficient queries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class RatingTrigger(str, Enum):
    """What triggered a rating snapshot."""
    MATCH = "match"
    FEEDBACK = "feedback"
    TRANSFER = "transfer"
    CALIBRATION = "calibration"
    MANUAL = "manual"


@dataclass
class RatingSnapshot:
    """
    A snapshot of ratings at a point in time.
    
    Stored in model_ratings_history time series collection.
    """
    timestamp: str
    model_id: str  # Deployment ID
    domain: str = ""  # Empty for global
    
    # TrueSkill ratings
    trueskill_raw_mu: float = 0.0
    trueskill_raw_sigma: float = 0.0
    trueskill_cost_mu: float = 0.0
    trueskill_cost_sigma: float = 0.0
    
    # Multi-dimensional ratings (optional)
    multi_trueskill: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # What triggered this snapshot
    trigger: RatingTrigger = RatingTrigger.MATCH
    trigger_id: str = ""  # Match ID or other trigger ID
    
    # Context at this point
    matches_played: int = 0
    win_rate_last_10: float = 0.0
    avg_rank_last_10: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "timestamp": self.timestamp,
            "metadata": {
                "model_id": self.model_id,
                "domain": self.domain,
            },
            "trueskill_raw": {
                "mu": self.trueskill_raw_mu,
                "sigma": self.trueskill_raw_sigma,
            },
            "trueskill_cost": {
                "mu": self.trueskill_cost_mu,
                "sigma": self.trueskill_cost_sigma,
            },
            "multi_trueskill": self.multi_trueskill,
            "trigger": self.trigger.value if isinstance(self.trigger, Enum) else self.trigger,
            "trigger_id": self.trigger_id,
            "stats": {
                "matches_played": self.matches_played,
                "win_rate_last_10": self.win_rate_last_10,
                "avg_rank_last_10": self.avg_rank_last_10,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RatingSnapshot':
        """Create from MongoDB document."""
        metadata = data.get("metadata", {})
        raw = data.get("trueskill_raw", {})
        cost = data.get("trueskill_cost", {})
        stats = data.get("stats", {})
        
        trigger = data.get("trigger", "match")
        if isinstance(trigger, str):
            try:
                trigger = RatingTrigger(trigger)
            except ValueError:
                trigger = RatingTrigger.MATCH
        
        return cls(
            timestamp=data.get("timestamp", ""),
            model_id=metadata.get("model_id", ""),
            domain=metadata.get("domain", ""),
            trueskill_raw_mu=raw.get("mu", 0.0),
            trueskill_raw_sigma=raw.get("sigma", 0.0),
            trueskill_cost_mu=cost.get("mu", 0.0),
            trueskill_cost_sigma=cost.get("sigma", 0.0),
            multi_trueskill=data.get("multi_trueskill", {}),
            trigger=trigger,
            trigger_id=data.get("trigger_id", ""),
            matches_played=stats.get("matches_played", 0),
            win_rate_last_10=stats.get("win_rate_last_10", 0.0),
            avg_rank_last_10=stats.get("avg_rank_last_10", 0.0),
        )
    
    @classmethod
    def create_now(
        cls,
        model_id: str,
        raw_mu: float,
        raw_sigma: float,
        cost_mu: float,
        cost_sigma: float,
        domain: str = "",
        trigger: RatingTrigger = RatingTrigger.MATCH,
        trigger_id: str = ""
    ) -> 'RatingSnapshot':
        """Create a snapshot with current timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            domain=domain,
            trueskill_raw_mu=raw_mu,
            trueskill_raw_sigma=raw_sigma,
            trueskill_cost_mu=cost_mu,
            trueskill_cost_sigma=cost_sigma,
            trigger=trigger,
            trigger_id=trigger_id,
        )

