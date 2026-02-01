"""
LangScope LLM Provider Module.

This module provides unified access to multiple LLM providers including:
- OpenAI (GPT-4, GPT-5, etc.)
- Anthropic (Claude)
- Groq (Fast inference)
- XAI (Grok)
- HuggingFace (Open source models)

Usage:
    from langscope.llm import LLMFactory, Message
    
    # Get a provider
    provider = LLMFactory.get_provider("openai")
    
    # Generate a response
    response = await provider.generate([
        Message(role="user", content="Hello!")
    ])
"""

from langscope.llm.models import (
    Message,
    LLMResponse,
    LLMConfig,
    UsageInfo,
    ProviderType,
)
from langscope.llm.base import BaseLLMProvider
from langscope.llm.factory import LLMFactory, get_provider, list_providers

__all__ = [
    # Models
    "Message",
    "LLMResponse",
    "LLMConfig",
    "UsageInfo",
    "ProviderType",
    # Base
    "BaseLLMProvider",
    # Factory
    "LLMFactory",
    "get_provider",
    "list_providers",
]

