"""
Price history for tracking pricing changes over time.

Stored as MongoDB time series for cost analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class PriceHistoryEntry:
    """
    A price change record.
    
    Stored in price_history time series collection.
    """
    effective_from: str  # When this price became effective
    model_id: str
    deployment_id: str
    provider: str
    
    # Current prices
    input_cost_per_million: float
    output_cost_per_million: float
    
    # Previous prices (for change tracking)
    previous_input_cost: Optional[float] = None
    previous_output_cost: Optional[float] = None
    change_pct_input: Optional[float] = None
    change_pct_output: Optional[float] = None
    
    # Source
    source_id: str = ""
    source_url: str = ""
    price_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "effective_from": self.effective_from,
            "metadata": {
                "model_id": self.model_id,
                "deployment_id": self.deployment_id,
                "provider": self.provider,
            },
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "previous_input_cost": self.previous_input_cost,
            "previous_output_cost": self.previous_output_cost,
            "change_pct_input": self.change_pct_input,
            "change_pct_output": self.change_pct_output,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "price_hash": self.price_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceHistoryEntry':
        """Create from MongoDB document."""
        metadata = data.get("metadata", {})
        
        return cls(
            effective_from=data.get("effective_from", ""),
            model_id=metadata.get("model_id", ""),
            deployment_id=metadata.get("deployment_id", ""),
            provider=metadata.get("provider", ""),
            input_cost_per_million=data.get("input_cost_per_million", 0.0),
            output_cost_per_million=data.get("output_cost_per_million", 0.0),
            previous_input_cost=data.get("previous_input_cost"),
            previous_output_cost=data.get("previous_output_cost"),
            change_pct_input=data.get("change_pct_input"),
            change_pct_output=data.get("change_pct_output"),
            source_id=data.get("source_id", ""),
            source_url=data.get("source_url", ""),
            price_hash=data.get("price_hash", ""),
        )
    
    @classmethod
    def create_now(
        cls,
        deployment_id: str,
        provider: str,
        input_cost: float,
        output_cost: float,
        previous_input: Optional[float] = None,
        previous_output: Optional[float] = None,
        source_id: str = "",
        source_url: str = ""
    ) -> 'PriceHistoryEntry':
        """Create a price entry with current timestamp."""
        change_input = None
        change_output = None
        
        if previous_input is not None and previous_input > 0:
            change_input = ((input_cost - previous_input) / previous_input) * 100
        
        if previous_output is not None and previous_output > 0:
            change_output = ((output_cost - previous_output) / previous_output) * 100
        
        # Extract model_id from deployment_id (format: provider/model_id)
        model_id = deployment_id.split("/", 1)[1] if "/" in deployment_id else deployment_id
        
        return cls(
            effective_from=datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            deployment_id=deployment_id,
            provider=provider,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
            previous_input_cost=previous_input,
            previous_output_cost=previous_output,
            change_pct_input=change_input,
            change_pct_output=change_output,
            source_id=source_id,
            source_url=source_url,
        )

