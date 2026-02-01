"""
HuggingFace LLM Provider.

Provides integration with HuggingFace's Inference API and local transformers.
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


class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace LLM Provider.
    
    Supports HuggingFace Inference API for various open-source models:
    - google/gemma-3-27b-it
    - meta-llama/Llama-3.3-70B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3
    - Qwen/Qwen2.5-72B-Instruct
    
    Can also use local models via the transformers library.
    
    Example:
        # Using Inference API
        provider = HuggingFaceProvider()
        response = await provider.generate([
            Message.user("Explain neural networks")
        ])
        print(response.content)
        
        # Using local model
        provider = HuggingFaceProvider(use_local=True)
        response = await provider.generate(
            [Message.user("Hello!")],
            config=LLMConfig(model="google/gemma-3-270m-it")
        )
    """
    
    provider_type = ProviderType.HUGGINGFACE
    
    # Popular models available via Inference API (Updated December 2025)
    MODELS = [
        # Google Gemma
        "google/gemma-3-27b-it",
        "google/gemma-3-4b-it",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-2b-it",
        # Meta Llama
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        # Mistral
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        # Qwen
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/QwQ-32B-Preview",
        # Microsoft Phi
        "microsoft/Phi-4",
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-MoE-instruct",
        # DeepSeek
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 300,
        use_local: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the HuggingFace provider.
        
        Args:
            api_key: HuggingFace API key for Inference API.
            default_model: Default model to use.
            timeout: Request timeout in seconds.
            use_local: If True, use local transformers instead of Inference API.
            device: Device for local inference ("auto", "cpu", "cuda", "mps").
        """
        self.use_local = use_local
        self.device = device
        self._pipeline = None
        self._current_model = None
        super().__init__(api_key, default_model, timeout)
    
    def _get_api_key_env_var(self) -> str:
        return "HUGGINGFACE_API_KEY"
    
    def _get_default_model(self) -> str:
        return "google/gemma-3-27b-it"
    
    def _init_client(self) -> Any:
        """Initialize the HuggingFace client."""
        if self.use_local:
            return self._init_local_pipeline()
        else:
            return self._init_inference_client()
    
    def _init_inference_client(self) -> Any:
        """Initialize the HuggingFace Inference Client."""
        try:
            from huggingface_hub import InferenceClient, AsyncInferenceClient
            
            self._sync_client = InferenceClient(
                token=self.api_key,
                timeout=self.timeout,
            )
            self._async_client = AsyncInferenceClient(
                token=self.api_key,
                timeout=self.timeout,
            )
            return self._sync_client
        except ImportError:
            raise ImportError(
                "huggingface_hub package not installed. "
                "Install with: pip install huggingface_hub"
            )
    
    def _init_local_pipeline(self, model: Optional[str] = None) -> Any:
        """Initialize a local transformers pipeline."""
        try:
            from transformers import pipeline
            import torch
            
            model = model or self.default_model
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.device
            
            logger.info(f"Loading model {model} on device {device}")
            
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                device=device,
                torch_dtype="auto",
            )
            self._current_model = model
            
            return self._pipeline
            
        except ImportError:
            raise ImportError(
                "transformers package not installed. "
                "Install with: pip install transformers torch"
            )
    
    def _get_pipeline(self, model: str) -> Any:
        """Get or create pipeline for the specified model."""
        if self._pipeline is None or self._current_model != model:
            self._init_local_pipeline(model)
        return self._pipeline
    
    @property
    def async_client(self) -> Any:
        """Get the async HuggingFace client."""
        if self._client is None:
            self._init_client()
        return self._async_client if not self.use_local else None
    
    def _format_chat_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for HuggingFace chat format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def generate(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response using HuggingFace.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        formatted_messages = self._format_chat_messages(messages)
        
        start_time = time.time()
        
        if self.use_local:
            # Use local pipeline (sync, wrapped as async)
            return self.generate_sync(messages, config)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=config.model,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Extract usage info
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.choices[0].finish_reason if response.choices else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    def generate_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Synchronously generate a response using HuggingFace.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Returns:
            LLMResponse with generated content.
        """
        config = self._get_config(config)
        formatted_messages = self._format_chat_messages(messages)
        
        start_time = time.time()
        
        if self.use_local:
            try:
                pipe = self._get_pipeline(config.model)
                
                outputs = pipe(
                    formatted_messages,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.temperature > 0,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract generated text
                content = outputs[0]["generated_text"]
                if isinstance(content, list):
                    # Get the last assistant message
                    content = content[-1].get("content", "") if content else ""
                
                return LLMResponse(
                    content=content,
                    model=config.model,
                    provider=self.provider_type,
                    usage=None,  # Local doesn't provide usage stats
                    finish_reason="stop",
                    latency_ms=latency_ms,
                )
                
            except Exception as e:
                logger.error(f"HuggingFace local generation error: {e}")
                raise
        
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content or ""
            
            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider=self.provider_type,
                usage=usage,
                finish_reason=response.choices[0].finish_reason if response.choices else None,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from HuggingFace Inference API.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        if self.use_local:
            # Local pipeline doesn't support true streaming
            response = self.generate_sync(messages, config)
            yield response.content
            return
        
        config = self._get_config(config)
        formatted_messages = self._format_chat_messages(messages)
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=config.model,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            raise
    
    def generate_stream_sync(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Synchronously stream a response from HuggingFace.
        
        Args:
            messages: List of conversation messages.
            config: Generation configuration.
        
        Yields:
            Chunks of generated text.
        """
        if self.use_local:
            response = self.generate_sync(messages, config)
            yield response.content
            return
        
        config = self._get_config(config)
        formatted_messages = self._format_chat_messages(messages)
        
        try:
            stream = self.client.chat.completions.create(
                model=config.model,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available HuggingFace models."""
        return self.MODELS.copy()

