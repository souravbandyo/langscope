"""
Context window configuration.

Defines the context limits and quality degradation characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ContextWindow:
    """Context window configuration."""
    max_context_length: int = 4096
    recommended_context: int = 4096
    max_output_tokens: int = 4096
    quality_at_context: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_context_length": self.max_context_length,
            "recommended_context": self.recommended_context,
            "max_output_tokens": self.max_output_tokens,
            "quality_at_context": self.quality_at_context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextWindow':
        """Create from dictionary."""
        # Convert string keys to int for quality_at_context
        quality = {}
        for k, v in data.get("quality_at_context", {}).items():
            quality[int(k)] = float(v)
        
        return cls(
            max_context_length=data.get("max_context_length", 4096),
            recommended_context=data.get("recommended_context", 4096),
            max_output_tokens=data.get("max_output_tokens", 4096),
            quality_at_context=quality,
        )

