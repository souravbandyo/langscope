"""
API key management for LangScope.

Handles secure storage and retrieval of API keys for various providers.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class APIKeyManager:
    """
    Manages API keys for LLM providers.
    
    Keys can be loaded from environment variables or set programmatically.
    """
    
    _keys: Dict[str, str] = field(default_factory=dict)
    
    # Environment variable names for each provider
    ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "azure": "AZURE_API_KEY",
        "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    }
    
    def __post_init__(self):
        """Load keys from environment variables."""
        self._load_from_env()
    
    def _load_from_env(self):
        """Load API keys from environment variables."""
        for provider, env_var in self.ENV_VARS.items():
            key = os.getenv(env_var)
            if key:
                self._keys[provider] = key
    
    def get_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.
        
        Args:
            provider: Provider name (openai, anthropic, etc.)
        
        Returns:
            API key or None if not found
        """
        provider = provider.lower()
        
        # Check stored keys
        if provider in self._keys:
            return self._keys[provider]
        
        # Check environment
        env_var = self.ENV_VARS.get(provider)
        if env_var:
            key = os.getenv(env_var)
            if key:
                self._keys[provider] = key
                return key
        
        return None
    
    def set_key(self, provider: str, key: str):
        """
        Set API key for a provider.
        
        Args:
            provider: Provider name
            key: API key
        """
        self._keys[provider.lower()] = key
    
    def has_key(self, provider: str) -> bool:
        """Check if key is available for provider."""
        return self.get_key(provider) is not None
    
    def list_providers(self) -> list:
        """List providers with available keys."""
        return [p for p in self.ENV_VARS.keys() if self.has_key(p)]
    
    def get_for_model(self, model_id: str) -> Optional[str]:
        """
        Get API key for a model ID.
        
        Extracts provider from model_id format: provider/model_name
        
        Args:
            model_id: Model identifier
        
        Returns:
            API key or None
        """
        if "/" in model_id:
            provider = model_id.split("/")[0]
            return self.get_key(provider)
        return None


# Global API key manager
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get or create global API key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider using global manager."""
    return get_key_manager().get_key(provider)


def set_api_key(provider: str, key: str):
    """Set API key for a provider using global manager."""
    get_key_manager().set_key(provider, key)


def has_api_key(provider: str) -> bool:
    """Check if API key is available for provider."""
    return get_key_manager().has_key(provider)


