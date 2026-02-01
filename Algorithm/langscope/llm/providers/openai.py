"""
OpenAI LLM Provider.

Provides integration with OpenAI's API for models like GPT-4, GPT-4o, GPT-5, etc.
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


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider.
    
    Supports all OpenAI chat completion models including:
    - gpt-4o, gpt-4o-mini
    - gpt-4-turbo, gpt-4
    - gpt-3.5-turbo
    - gpt-5 (when available)
    - o1, o1-mini, o1-pro (reasoning models)
    
    Example:
        provider = OpenAIProvider()
        response = await provider.generate([
            Message.user("What is the capital of France?")
        ])
        print(response.content)
    """
    
    provider_type = ProviderType.OPENAI
    
    # Available models (Updated December 2025)
    MODELS = [
        # GPT-4o Series (Flagship)
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "chatgpt-4o-latest",
        # GPT-4o Mini Series
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        # GPT-4o Audio/Realtime
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-12-17",
        # GPT-4.1 Series (Latest)
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
        # GPT-4 Turbo Series
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        # GPT-4 Series
        "gpt-4",
        "gpt-4-0613",
        # Reasoning Models (o-series)
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-pro",
        "o1-pro-2025-03-19",
        # o3 Series (Advanced Reasoning)
        "o3",
        "o3-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o3-mini-high",
        "o4-mini",
        "o4-mini-2025-04-16",
        # Legacy
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
    ]
    
    # Model pricing per million tokens (USD) - Updated December 2025
    MODEL_PRICING = {
        # GPT-4o Series
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
        "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
        # GPT-4o Mini
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
        # GPT-4o Audio/Realtime
        "gpt-4o-audio-preview": {"input": 100.00, "output": 200.00},
        "gpt-4o-audio-preview-2024-12-17": {"input": 100.00, "output": 200.00},
        "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
        "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "output": 20.00},
        # GPT-4.1 Series
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
        # GPT-4 Turbo
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        # GPT-4
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-0613": {"input": 30.00, "output": 60.00},
        # o1 Series
        "o1": {"input": 15.00, "output": 60.00},
        "o1-2024-12-17": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
        "o1-pro": {"input": 150.00, "output": 600.00},
        "o1-pro-2025-03-19": {"input": 150.00, "output": 600.00},
        # o3 Series
        "o3": {"input": 10.00, "output": 40.00},
        "o3-2025-04-16": {"input": 10.00, "output": 40.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
        "o3-mini-high": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
        # Legacy
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    }
    
    # Model context lengths
    MODEL_CONTEXT_LENGTHS = {
        # GPT-4o Series
        "gpt-4o": 128000,
        "gpt-4o-2024-11-20": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o-2024-05-13": 128000,
        "chatgpt-4o-latest": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4o-audio-preview": 128000,
        "gpt-4o-audio-preview-2024-12-17": 128000,
        "gpt-4o-realtime-preview": 128000,
        "gpt-4o-realtime-preview-2024-12-17": 128000,
        # GPT-4.1 Series
        "gpt-4.1": 1047576,
        "gpt-4.1-2025-04-14": 1047576,
        "gpt-4.1-mini": 1047576,
        "gpt-4.1-mini-2025-04-14": 1047576,
        "gpt-4.1-nano": 1047576,
        "gpt-4.1-nano-2025-04-14": 1047576,
        # GPT-4 Turbo
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-2024-04-09": 128000,
        "gpt-4-turbo-preview": 128000,
        # GPT-4
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        # o1 Series
        "o1": 200000,
        "o1-2024-12-17": 200000,
        "o1-mini": 128000,
        "o1-mini-2024-09-12": 128000,
        "o1-preview": 128000,
        "o1-preview-2024-09-12": 128000,
        "o1-pro": 200000,
        "o1-pro-2025-03-19": 200000,
        # o3 Series
        "o3": 200000,
        "o3-2025-04-16": 200000,
        "o3-mini": 200000,
        "o3-mini-2025-01-31": 200000,
        "o3-mini-high": 200000,
        "o4-mini": 200000,
        "o4-mini-2025-04-16": 200000,
        # Legacy
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-1106": 16385,
    }
    
    def _get_api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"
    
    def _get_default_model(self) -> str:
        return "gpt-4o"
    
    def _init_client(self) -> Any:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI, AsyncOpenAI
            
            self._sync_client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            return self._sync_client
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    @property
    def async_client(self) -> Any:
        """Get the async OpenAI client."""
        if self._client is None:
            self._init_client()
        return self._async_client
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI's API.
        
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
            # Build request parameters
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            if config.stop:
                params["stop"] = config.stop
            
            # Add extra parameters
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
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response using OpenAI's API.
        
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
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from OpenAI.
        
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
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from OpenAI.
        
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
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
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
