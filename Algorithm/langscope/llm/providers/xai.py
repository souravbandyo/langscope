"""
XAI LLM Provider.

Provides integration with xAI's API for Grok models.
"""

import time
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Generator

from langscope.llm.base import BaseLLMProvider
from langscope.llm.models import (
    Message,
    LLMResponse,
    LLMConfig,
    UsageInfo,
    ProviderType,
)

logger = logging.getLogger(__name__)


class XAIProvider(BaseLLMProvider):
    """
    XAI (Grok) LLM Provider.
    
    Supports xAI's Grok models including:
    - grok-4
    - grok-3
    - grok-3-mini
    - grok-2
    - grok-beta
    
    Example:
        provider = XAIProvider()
        response = await provider.generate([
            Message.system("You are Grok, a helpful AI assistant."),
            Message.user("What's the meaning of life?")
        ])
        print(response.content)
    """
    
    provider_type = ProviderType.XAI
    
    # Available models (Updated December 2025)
    MODELS = [
        # Grok 3 Series (Latest Flagship)
        "grok-3",
        "grok-3-fast",
        "grok-3-mini",
        "grok-3-mini-fast",
        # Grok 2 Series
        "grok-2",
        "grok-2-1212",
        "grok-2-latest",
        "grok-2-vision",
        "grok-2-vision-1212",
        "grok-2-vision-latest",
        # Beta/Preview
        "grok-beta",
        "grok-vision-beta",
    ]
    
    # Model pricing per million tokens (USD) - Updated December 2025
    MODEL_PRICING = {
        # Grok 3 Series
        "grok-3": {"input": 3.00, "output": 15.00},
        "grok-3-fast": {"input": 5.00, "output": 25.00},
        "grok-3-mini": {"input": 0.30, "output": 0.50},
        "grok-3-mini-fast": {"input": 0.60, "output": 4.00},
        # Grok 2 Series
        "grok-2": {"input": 2.00, "output": 10.00},
        "grok-2-1212": {"input": 2.00, "output": 10.00},
        "grok-2-latest": {"input": 2.00, "output": 10.00},
        "grok-2-vision": {"input": 2.00, "output": 10.00},
        "grok-2-vision-1212": {"input": 2.00, "output": 10.00},
        "grok-2-vision-latest": {"input": 2.00, "output": 10.00},
        # Beta
        "grok-beta": {"input": 5.00, "output": 15.00},
        "grok-vision-beta": {"input": 5.00, "output": 15.00},
    }
    
    # Model context lengths
    MODEL_CONTEXT_LENGTHS = {
        # Grok 3 Series
        "grok-3": 131072,
        "grok-3-fast": 131072,
        "grok-3-mini": 131072,
        "grok-3-mini-fast": 131072,
        # Grok 2 Series
        "grok-2": 131072,
        "grok-2-1212": 131072,
        "grok-2-latest": 131072,
        "grok-2-vision": 32768,
        "grok-2-vision-1212": 32768,
        "grok-2-vision-latest": 32768,
        # Beta
        "grok-beta": 131072,
        "grok-vision-beta": 8192,
    }
    
    def _get_api_key_env_var(self) -> str:
        return "XAI_API_KEY"
    
    def _get_default_model(self) -> str:
        return "grok-3"
    
    def _init_client(self) -> Any:
        """
        Initialize the XAI client.
        
        XAI uses an OpenAI-compatible API, so we use the OpenAI client
        with a custom base URL.
        """
        try:
            from openai import OpenAI, AsyncOpenAI
            
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",
                timeout=self.timeout,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",
                timeout=self.timeout,
            )
            return self._sync_client
        except ImportError:
            raise ImportError(
                "OpenAI package not installed (required for XAI). "
                "Install with: pip install openai"
            )
    
    @property
    def async_client(self) -> Any:
        """Get the async XAI client."""
        if self._client is None:
            self._init_client()
        return self._async_client
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using XAI's API.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        formatted_messages = self._format_messages(messages)
        
        start_time = time.time()
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            params.update(config.extra_params)
            
            response = await self.async_client.chat.completions.create(**params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage info
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"XAI generation error: {e}")
            raise
    
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response using XAI's API.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        formatted_messages = self._format_messages(messages)
        
        start_time = time.time()
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            params.update(config.extra_params)
            
            response = self.client.chat.completions.create(**params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"XAI generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from XAI.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        config = self._get_config(config)
        formatted_messages = self._format_messages(messages)
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "stream": True,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            params.update(config.extra_params)
            
            stream = await self.async_client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"XAI streaming error: {e}")
            raise
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from XAI.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        config = self._get_config(config)
        formatted_messages = self._format_messages(messages)
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "stream": True,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            params.update(config.extra_params)
            
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"XAI streaming error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available XAI models."""
        return self.MODELS.copy()
    
    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dict with 'input' and 'output' costs per million tokens
        """
        return self.MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    
    def get_model_context_length(self, model: str) -> int:
        """
        Get context length for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Maximum context length in tokens
        """
        return self.MODEL_CONTEXT_LENGTHS.get(model, 8192)

