"""
Price analysis and impact tracking.

Analyzes how price changes affect cost-adjusted rankings.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


@dataclass
class PriceChange:
    """Record of a price change."""
    deployment_id: str
    effective_from: str
    
    old_input_cost: float
    new_input_cost: float
    old_output_cost: float
    new_output_cost: float
    
    change_pct_input: float = 0.0
    change_pct_output: float = 0.0
    
    source_id: str = ""
    source_url: str = ""
    
    def __post_init__(self):
        """Compute change percentages."""
        if self.old_input_cost > 0:
            self.change_pct_input = (
                (self.new_input_cost - self.old_input_cost) / self.old_input_cost * 100
            )
        if self.old_output_cost > 0:
            self.change_pct_output = (
                (self.new_output_cost - self.old_output_cost) / self.old_output_cost * 100
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "effective_from": self.effective_from,
            "old_input_cost": self.old_input_cost,
            "new_input_cost": self.new_input_cost,
            "old_output_cost": self.old_output_cost,
            "new_output_cost": self.new_output_cost,
            "change_pct_input": self.change_pct_input,
            "change_pct_output": self.change_pct_output,
            "source_id": self.source_id,
            "source_url": self.source_url,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceChange':
        """Create from dictionary."""
        return cls(
            deployment_id=data.get("deployment_id", ""),
            effective_from=data.get("effective_from", ""),
            old_input_cost=data.get("old_input_cost", 0),
            new_input_cost=data.get("new_input_cost", 0),
            old_output_cost=data.get("old_output_cost", 0),
            new_output_cost=data.get("new_output_cost", 0),
            change_pct_input=data.get("change_pct_input", 0),
            change_pct_output=data.get("change_pct_output", 0),
            source_id=data.get("source_id", ""),
            source_url=data.get("source_url", ""),
        )


@dataclass
class PriceImpact:
    """Analysis of how a price change affected rankings."""
    deployment_id: str
    price_change: PriceChange
    
    # Ranking changes
    old_rank_cost: int = 0
    new_rank_cost: int = 0
    rank_change: int = 0
    
    # Rating changes
    old_mu_cost: float = 0.0
    new_mu_cost: float = 0.0
    mu_change: float = 0.0
    
    # Value proposition
    value_improvement: float = 0.0  # Positive = better value
    
    def compute_impact(self):
        """Compute the impact metrics."""
        self.rank_change = self.old_rank_cost - self.new_rank_cost  # Positive = improved
        self.mu_change = self.new_mu_cost - self.old_mu_cost
        
        # Value improvement based on price decrease and rank improvement
        price_change = (
            self.price_change.change_pct_input + self.price_change.change_pct_output
        ) / 2
        
        # Negative price change (decrease) + positive rank change = good
        self.value_improvement = -price_change + (self.rank_change * 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "price_change": self.price_change.to_dict(),
            "old_rank_cost": self.old_rank_cost,
            "new_rank_cost": self.new_rank_cost,
            "rank_change": self.rank_change,
            "old_mu_cost": self.old_mu_cost,
            "new_mu_cost": self.new_mu_cost,
            "mu_change": self.mu_change,
            "value_improvement": self.value_improvement,
        }


def get_price_trend(
    db,
    deployment_id: str,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[PriceChange]:
    """
    Get price change history for a deployment.
    
    Args:
        db: MongoDB instance
        deployment_id: Deployment ID
        start_date: Start of period
        end_date: End of period
    
    Returns:
        List of PriceChange records
    """
    # TODO: Query from price_history collection
    return []


def analyze_price_change_impact(
    db,
    deployment_id: str,
    price_change: PriceChange,
    domain: str = ""
) -> PriceImpact:
    """
    Analyze the impact of a price change on rankings.
    
    Args:
        db: MongoDB instance
        deployment_id: Deployment ID
        price_change: The price change to analyze
        domain: Optional domain for domain-specific analysis
    
    Returns:
        PriceImpact analysis
    """
    impact = PriceImpact(
        deployment_id=deployment_id,
        price_change=price_change,
    )
    
    # TODO: Query before/after rankings from time series
    
    impact.compute_impact()
    return impact


def find_best_value_changes(
    db,
    period_days: int = 30,
    limit: int = 10,
    domain: str = ""
) -> List[Dict[str, Any]]:
    """
    Find deployments with the best value improvements.
    
    Value improvement = price decrease + ranking improvement.
    
    Args:
        db: MongoDB instance
        period_days: Look-back period
        limit: Maximum results
        domain: Optional domain filter
    
    Returns:
        List of {deployment_id, name, value_improvement, price_change_pct, rank_change}
    """
    # TODO: Query price changes and compute value improvements
    return []


def get_price_comparison(
    db,
    base_model_id: str
) -> List[Dict[str, Any]]:
    """
    Compare current prices across all providers for a base model.
    
    Args:
        db: MongoDB instance
        base_model_id: Base model ID
    
    Returns:
        List of {deployment_id, provider, input_cost, output_cost, value_score}
    """
    # TODO: Get all deployments and compare prices
    return []


