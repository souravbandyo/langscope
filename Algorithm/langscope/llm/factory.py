"""
LLM Provider Factory.

Provides a unified interface for creating and managing LLM provider instances.
"""

import logging
from typing import Dict, Type, Optional, List

from langscope.llm.base import BaseLLMProvider
from langscope.llm.models import ProviderType

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM provider instances.
    
    Provides a registry of available providers and methods to create
    provider instances by name or type.
    
    Example:
        # Get a provider by name
        openai = LLMFactory.get_provider("openai")
        
        # Get a provider with custom configuration
        anthropic = LLMFactory.get_provider(
            "anthropic",
            api_key="sk-ant-...",
            default_model="claude-3-opus-20240229"
        )
        
        # List available providers
        providers = LLMFactory.list_providers()
    """
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _instances: Dict[str, BaseLLMProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a provider class.
        
        Args:
            name: The name to register the provider under.
            provider_class: The provider class to register.
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered LLM provider: {name}")
    
    @classmethod
    def get_provider(
        cls,
        name: str,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 300,
        cached: bool = True,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Get or create a provider instance.
        
        Args:
            name: The provider name (openai, anthropic, groq, xai, huggingface).
            api_key: Optional API key. If not provided, uses environment variable.
            default_model: Optional default model for the provider.
            timeout: Request timeout in seconds.
            cached: If True, return cached instance if available.
            **kwargs: Additional provider-specific arguments.
        
        Returns:
            The provider instance.
        
        Raises:
            ValueError: If the provider is not registered.
        """
        name = name.lower()
        
        # Handle provider type enum
        if isinstance(name, ProviderType):
            name = name.value
        
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {name}. Available providers: {available}"
            )
        
        # Check cache
        cache_key = f"{name}:{default_model or 'default'}"
        if cached and cache_key in cls._instances and api_key is None:
            return cls._instances[cache_key]
        
        # Create new instance
        provider_class = cls._providers[name]
        provider = provider_class(
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
            **kwargs,
        )
        
        # Cache if using default API key
        if cached and api_key is None:
            cls._instances[cache_key] = provider
        
        return provider
    
    @classmethod
    def get_provider_by_type(
        cls,
        provider_type: ProviderType,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Get a provider by its ProviderType enum.
        
        Args:
            provider_type: The ProviderType enum value.
            **kwargs: Additional arguments passed to get_provider.
        
        Returns:
            The provider instance.
        """
        return cls.get_provider(provider_type.value, **kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names.
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            name: The provider name to check.
        
        Returns:
            True if registered, False otherwise.
        """
        return name.lower() in cls._providers
    
    @classmethod
    def get_all_models(cls) -> Dict[str, List[str]]:
        """
        Get all available models from all registered providers.
        
        Returns:
            Dictionary mapping provider names to lists of model names.
        """
        models = {}
        for name in cls._providers:
            try:
                provider = cls.get_provider(name)
                models[name] = provider.get_available_models()
            except Exception as e:
                logger.warning(f"Could not get models for {name}: {e}")
                models[name] = []
        return models
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached provider instances."""
        cls._instances.clear()
        logger.debug("Cleared LLM provider cache")
    
    @classmethod
    def get_configured_providers(cls) -> List[str]:
        """
        Get list of providers that are properly configured with API keys.
        
        Returns:
            List of configured provider names.
        """
        configured = []
        for name in cls._providers:
            try:
                provider = cls.get_provider(name)
                if provider.is_configured():
                    configured.append(name)
            except Exception:
                pass
        return configured


def _register_default_providers():
    """Register all default providers."""
    from langscope.llm.providers.openai import OpenAIProvider
    from langscope.llm.providers.anthropic import AnthropicProvider
    from langscope.llm.providers.groq import GroqProvider
    from langscope.llm.providers.xai import XAIProvider
    from langscope.llm.providers.huggingface import HuggingFaceProvider
    
    LLMFactory.register("openai", OpenAIProvider)
    LLMFactory.register("anthropic", AnthropicProvider)
    LLMFactory.register("groq", GroqProvider)
    LLMFactory.register("xai", XAIProvider)
    LLMFactory.register("grok", XAIProvider)  # Alias
    LLMFactory.register("huggingface", HuggingFaceProvider)
    LLMFactory.register("hf", HuggingFaceProvider)  # Alias


# Register providers on module import
_register_default_providers()


# Convenience functions
def get_provider(name: str, **kwargs) -> BaseLLMProvider:
    """
    Get a provider instance.
    
    Convenience function for LLMFactory.get_provider().
    
    Args:
        name: The provider name.
        **kwargs: Additional arguments.
    
    Returns:
        The provider instance.
    """
    return LLMFactory.get_provider(name, **kwargs)


def list_providers() -> List[str]:
    """
    List all registered provider names.
    
    Convenience function for LLMFactory.list_providers().
    
    Returns:
        List of provider names.
    """
    return LLMFactory.list_providers()

