"""
Base class for LLM providers.

Defines the abstract interface that all LLM provider implementations must follow.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator, Generator

from langscope.llm.models import (
    Message,
    LLMResponse,
    LLMConfig,
    ProviderType,
)

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        provider_type: The type of provider (openai, anthropic, etc.)
        api_key: The API key for authentication
        default_model: The default model to use if none specified
        timeout: Default request timeout in seconds
    """
    
    provider_type: ProviderType
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication. If not provided, will try to
                    load from environment variable.
            default_model: Default model to use for generation.
            timeout: Default request timeout in seconds.
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.default_model = default_model or self._get_default_model()
        self.timeout = timeout
        self._client = None
        
        if not self.api_key:
            logger.warning(
                f"{self.provider_type.value} API key not found. "
                f"Set {self._get_api_key_env_var()} environment variable."
            )
    
    @abstractmethod
    def _get_api_key_env_var(self) -> str:
        """Get the environment variable name for the API key."""
        pass
    
    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Load API key from environment variable."""
        return os.getenv(self._get_api_key_env_var())
    
    @abstractmethod
    def _init_client(self) -> Any:
        """Initialize the provider-specific client."""
        pass
    
    @property
    def client(self) -> Any:
        """Get or create the provider client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages in the conversation.
            config: Generation configuration. If not provided, uses defaults.
        
        Returns:
            LLMResponse containing the generated content and metadata.
        """
        pass
    
    @abstractmethod
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response from the LLM.
        
        Args:
            messages: List of messages in the conversation.
            config: Generation configuration. If not provided, uses defaults.
        
        Returns:
            LLMResponse containing the generated content and metadata.
        """
        pass
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM.
        
        Args:
            messages: List of messages in the conversation.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        # Default implementation - override for streaming support
        response = await self.generate(messages, config)
        yield response.content
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from the LLM.
        
        Args:
            messages: List of messages in the conversation.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        # Default implementation - override for streaming support
        response = self.generate_sync(messages, config)
        yield response.content
    
    def _get_config(self, config: Optional[LLMConfig] = None) -> LLMConfig:
        """Get configuration with defaults applied."""
        if config is None:
            return LLMConfig(model=self.default_model)
        if not config.model:
            config.model = self.default_model
        return config
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the provider API."""
        return [msg.to_dict() for msg in messages]
    
    def is_configured(self) -> bool:
        """Check if the provider is properly configured with an API key."""
        return bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model identifiers.
        """
        return []  # Override in subclasses
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.default_model}, configured={self.is_configured()})"

