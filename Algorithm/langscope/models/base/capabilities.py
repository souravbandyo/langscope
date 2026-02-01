"""
Model capability definitions.

Describes what a model can do (modalities, languages, features).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class Modality(str, Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Capabilities:
    """Model capabilities."""
    modalities: List[str] = field(default_factory=lambda: ["text"])
    languages: List[str] = field(default_factory=lambda: ["en"])
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    trained_for: List[str] = field(default_factory=lambda: ["chat"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modalities": self.modalities,
            "languages": self.languages,
            "supports_function_calling": self.supports_function_calling,
            "supports_json_mode": self.supports_json_mode,
            "supports_vision": self.supports_vision,
            "supports_audio": self.supports_audio,
            "supports_system_prompt": self.supports_system_prompt,
            "supports_streaming": self.supports_streaming,
            "trained_for": self.trained_for,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Capabilities':
        """Create from dictionary."""
        return cls(
            modalities=data.get("modalities", ["text"]),
            languages=data.get("languages", ["en"]),
            supports_function_calling=data.get("supports_function_calling", False),
            supports_json_mode=data.get("supports_json_mode", False),
            supports_vision=data.get("supports_vision", False),
            supports_audio=data.get("supports_audio", False),
            supports_system_prompt=data.get("supports_system_prompt", True),
            supports_streaming=data.get("supports_streaming", True),
            trained_for=data.get("trained_for", ["chat"]),
        )

