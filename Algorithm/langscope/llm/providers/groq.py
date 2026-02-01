"""
Groq LLM Provider.

Provides integration with Groq's API for fast LLM inference.
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


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM Provider.
    
    Supports Groq's fast inference models including:
    - llama-3.3-70b-versatile
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma2-9b-it
    
    Example:
        provider = GroqProvider()
        response = await provider.generate([
            Message.user("What is machine learning?")
        ])
        print(response.content)
    """
    
    provider_type = ProviderType.GROQ
    
    # Available models (Updated December 2025)
    MODELS = [
        # Llama 3.3 Series
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec",
        # Llama 3.2 Series (Vision)
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        # Llama 3.1 Series
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        # Llama Guard
        "llama-guard-3-8b",
        # Mixtral
        "mixtral-8x7b-32768",
        # Gemma
        "gemma2-9b-it",
        # Qwen (QwQ Reasoning)
        "qwen-qwq-32b",
        # DeepSeek R1
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        # Whisper (for audio transcription)
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        # Distil-Whisper
        "distil-whisper-large-v3-en",
    ]
    
    # Model pricing per million tokens (USD) - Updated December 2025
    MODEL_PRICING = {
        # Llama 3.3 Series
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.3-70b-specdec": {"input": 0.59, "output": 0.99},
        # Llama 3.2 Series (Vision)
        "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
        "llama-3.2-11b-vision-preview": {"input": 0.18, "output": 0.18},
        "llama-3.2-3b-preview": {"input": 0.06, "output": 0.06},
        "llama-3.2-1b-preview": {"input": 0.04, "output": 0.04},
        # Llama 3.1 Series
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        # Llama Guard
        "llama-guard-3-8b": {"input": 0.20, "output": 0.20},
        # Mixtral
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        # Gemma
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
        # Qwen (QwQ Reasoning)
        "qwen-qwq-32b": {"input": 0.29, "output": 0.39},
        # DeepSeek R1
        "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
        "deepseek-r1-distill-qwen-32b": {"input": 0.69, "output": 0.69},
        # Whisper (per hour of audio, converted to approximate per million tokens)
        "whisper-large-v3": {"input": 0.111, "output": 0.0},
        "whisper-large-v3-turbo": {"input": 0.04, "output": 0.0},
        "distil-whisper-large-v3-en": {"input": 0.02, "output": 0.0},
    }
    
    # Model context lengths
    MODEL_CONTEXT_LENGTHS = {
        "llama-3.3-70b-versatile": 128000,
        "llama-3.3-70b-specdec": 8192,
        "llama-3.2-90b-vision-preview": 128000,
        "llama-3.2-11b-vision-preview": 128000,
        "llama-3.2-3b-preview": 128000,
        "llama-3.2-1b-preview": 128000,
        "llama-3.1-70b-versatile": 128000,
        "llama-3.1-8b-instant": 128000,
        "llama-guard-3-8b": 8192,
        "mixtral-8x7b-32768": 32768,
        "gemma2-9b-it": 8192,
        "qwen-qwq-32b": 32768,
        "deepseek-r1-distill-llama-70b": 128000,
        "deepseek-r1-distill-qwen-32b": 128000,
    }
    
    def _get_api_key_env_var(self) -> str:
        return "GROQ_API_KEY"
    
    def _get_default_model(self) -> str:
        return "llama-3.3-70b-versatile"
    
    def _init_client(self) -> Any:
        """Initialize the Groq client."""
        try:
            from groq import Groq, AsyncGroq
            
            self._sync_client = Groq(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._async_client = AsyncGroq(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            return self._sync_client
        except ImportError:
            raise ImportError(
                "Groq package not installed. Install with: pip install groq"
            )
    
    @property
    def async_client(self) -> Any:
        """Get the async Groq client."""
        if self._client is None:
            self._init_client()
        return self._async_client
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using Groq's API.
        
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
            
            # Add reasoning effort for reasoning models
            if config.reasoning_effort:
                params["reasoning_effort"] = config.reasoning_effort
            
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
            logger.error(f"Groq generation error: {e}")
            raise
    
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response using Groq's API.
        
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
            
            if config.reasoning_effort:
                params["reasoning_effort"] = config.reasoning_effort
            
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
            logger.error(f"Groq generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Groq.
        
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
            logger.error(f"Groq streaming error: {e}")
            raise
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from Groq.
        
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
            logger.error(f"Groq streaming error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available Groq models."""
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
