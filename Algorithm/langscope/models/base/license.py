"""
Model license information.

Describes the licensing terms for a model.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class License:
    """Model license information."""
    type: str = "unknown"
    commercial_use: bool = False
    requires_agreement: bool = False
    restrictions: List[str] = field(default_factory=list)
    url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "commercial_use": self.commercial_use,
            "requires_agreement": self.requires_agreement,
            "restrictions": self.restrictions,
            "url": self.url,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'License':
        """Create from dictionary."""
        return cls(
            type=data.get("type", "unknown"),
            commercial_use=data.get("commercial_use", False),
            requires_agreement=data.get("requires_agreement", False),
            restrictions=data.get("restrictions", []),
            url=data.get("url", ""),
        )

