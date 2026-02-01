"""
Anthropic LLM Provider.

Provides integration with Anthropic's API for Claude models.
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


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM Provider.
    
    Supports Claude models including:
    - claude-opus-4-1-20250805
    - claude-sonnet-4-20250514
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
    - claude-3-opus-20240229
    
    Example:
        provider = AnthropicProvider()
        response = await provider.generate([
            Message.user("Explain quantum computing")
        ])
        print(response.content)
    """
    
    provider_type = ProviderType.ANTHROPIC
    
    # Available models (Updated December 2025)
    MODELS = [
        # Claude 4.x Series (Latest)
        "claude-sonnet-4-5-20250915",
        "claude-haiku-4-5-20251015",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        # Claude 3.5 Series
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        # Claude 3 Series
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def _get_api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"
    
    def _get_default_model(self) -> str:
        return "claude-sonnet-4-20250514"
    
    def _init_client(self) -> Any:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            
            self._sync_client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            return self._sync_client
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
    
    @property
    def async_client(self) -> Any:
        """Get the async Anthropic client."""
        if self._client is None:
            self._init_client()
        return self._async_client
    
    def _format_messages_for_anthropic(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Format messages for Anthropic's API.
        
        Anthropic requires system messages to be passed separately.
        
        Returns:
            Tuple of (system_prompt, formatted_messages)
        """
        system_prompt = None
        formatted = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return system_prompt, formatted
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using Anthropic's API.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        system_prompt, formatted_messages = self._format_messages_for_anthropic(messages)
        
        start_time = time.time()
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            if config.stop:
                params["stop_sequences"] = config.stop
            
            if config.top_k is not None:
                params["top_k"] = config.top_k
            
            params.update(config.extra_params)
            
            response = await self.async_client.messages.create(**params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content from response
            content = ""
            if response.content:
                content = response.content[0].text if response.content else ""
            
            # Extract usage info
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.stop_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response using Anthropic's API.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        system_prompt, formatted_messages = self._format_messages_for_anthropic(messages)
        
        start_time = time.time()
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            if config.stop:
                params["stop_sequences"] = config.stop
            
            if config.top_k is not None:
                params["top_k"] = config.top_k
            
            params.update(config.extra_params)
            
            response = self.client.messages.create(**params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = ""
            if response.content:
                content = response.content[0].text if response.content else ""
            
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.stop_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Anthropic.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        config = self._get_config(config)
        system_prompt, formatted_messages = self._format_messages_for_anthropic(messages)
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            if config.stop:
                params["stop_sequences"] = config.stop
            
            params.update(config.extra_params)
            
            async with self.async_client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from Anthropic.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        config = self._get_config(config)
        system_prompt, formatted_messages = self._format_messages_for_anthropic(messages)
        
        try:
            params = {
                "model": config.model,
                "messages": formatted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            if config.stop:
                params["stop_sequences"] = config.stop
            
            params.update(config.extra_params)
            
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return self.MODELS.copy()

