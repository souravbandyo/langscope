"""
Tests for LLM provider module.

Tests cover:
- LLM data models (Message, LLMConfig, LLMResponse)
- Provider base class
- Factory pattern
- Mocked provider implementations
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass


# =============================================================================
# Data Model Tests
# =============================================================================

class TestMessage:
    """Test Message dataclass."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        from langscope.llm.models import Message
        
        msg = Message(role="user", content="Hello, world!")
        
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.name is None
    
    def test_message_with_name(self):
        """Test message with name."""
        from langscope.llm.models import Message
        
        msg = Message(role="user", content="Hello", name="TestUser")
        
        assert msg.name == "TestUser"
    
    def test_message_to_dict(self):
        """Test message to_dict conversion."""
        from langscope.llm.models import Message
        
        msg = Message(role="assistant", content="Hi there!")
        
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there!"
    
    def test_message_to_dict_with_extras(self):
        """Test message to_dict with optional fields."""
        from langscope.llm.models import Message
        
        msg = Message(
            role="assistant",
            content="Response",
            name="Bot",
            tool_calls=[{"id": "tool_1", "function": {"name": "test"}}]
        )
        
        msg_dict = msg.to_dict()
        
        assert "name" in msg_dict
        assert "tool_calls" in msg_dict
    
    def test_message_from_dict(self):
        """Test message from_dict creation."""
        from langscope.llm.models import Message
        
        data = {
            "role": "user",
            "content": "Test message",
            "name": "Alice"
        }
        
        msg = Message.from_dict(data)
        
        assert msg.role == "user"
        assert msg.content == "Test message"
        assert msg.name == "Alice"
    
    def test_message_system_factory(self):
        """Test system message factory."""
        from langscope.llm.models import Message
        
        msg = Message.system("You are a helpful assistant.")
        
        assert msg.role == "system"
        assert "helpful assistant" in msg.content
    
    def test_message_user_factory(self):
        """Test user message factory."""
        from langscope.llm.models import Message
        
        msg = Message.user("What is 2+2?")
        
        assert msg.role == "user"
        assert msg.content == "What is 2+2?"
    
    def test_message_assistant_factory(self):
        """Test assistant message factory."""
        from langscope.llm.models import Message
        
        msg = Message.assistant("The answer is 4.")
        
        assert msg.role == "assistant"
        assert msg.content == "The answer is 4."


class TestUsageInfo:
    """Test UsageInfo dataclass."""
    
    def test_usage_info_creation(self):
        """Test UsageInfo creation."""
        from langscope.llm.models import UsageInfo
        
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_usage_info_defaults(self):
        """Test UsageInfo default values."""
        from langscope.llm.models import UsageInfo
        
        usage = UsageInfo()
        
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens is None
    
    def test_usage_info_from_dict(self):
        """Test UsageInfo from_dict."""
        from langscope.llm.models import UsageInfo
        
        data = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "reasoning_tokens": 50
        }
        
        usage = UsageInfo.from_dict(data)
        
        assert usage.prompt_tokens == 200
        assert usage.reasoning_tokens == 50


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_llm_response_creation(self):
        """Test LLMResponse creation."""
        from langscope.llm.models import LLMResponse, ProviderType
        
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            provider=ProviderType.OPENAI
        )
        
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.provider == ProviderType.OPENAI
    
    def test_llm_response_with_usage(self):
        """Test LLMResponse with usage info."""
        from langscope.llm.models import LLMResponse, ProviderType, UsageInfo
        
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        
        response = LLMResponse(
            content="Test response",
            model="claude-3",
            provider=ProviderType.ANTHROPIC,
            usage=usage,
            latency_ms=150.5
        )
        
        assert response.usage.total_tokens == 30
        assert response.latency_ms == 150.5
    
    def test_llm_response_to_dict(self):
        """Test LLMResponse to_dict."""
        from langscope.llm.models import LLMResponse, ProviderType, UsageInfo
        
        response = LLMResponse(
            content="Response text",
            model="gpt-3.5-turbo",
            provider=ProviderType.OPENAI,
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            finish_reason="stop"
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Response text"
        assert response_dict["model"] == "gpt-3.5-turbo"
        assert response_dict["provider"] == "openai"
        assert response_dict["usage"]["total_tokens"] == 15


class TestLLMConfig:
    """Test LLMConfig dataclass."""
    
    def test_llm_config_creation(self):
        """Test LLMConfig creation."""
        from langscope.llm.models import LLMConfig
        
        config = LLMConfig(model="gpt-4")
        
        assert config.model == "gpt-4"
        assert config.temperature == 1.0
        assert config.max_tokens == 4096
    
    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        from langscope.llm.models import LLMConfig
        
        config = LLMConfig(
            model="claude-3-opus",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            stop=["END"]
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.stop == ["END"]
    
    def test_llm_config_with_reasoning(self):
        """Test LLMConfig with reasoning effort."""
        from langscope.llm.models import LLMConfig
        
        config = LLMConfig(
            model="o1-preview",
            reasoning_effort="high"
        )
        
        assert config.reasoning_effort == "high"
    
    def test_llm_config_to_dict(self):
        """Test LLMConfig to_dict."""
        from langscope.llm.models import LLMConfig
        
        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            stream=True
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model"] == "gpt-4"
        assert config_dict["temperature"] == 0.5
        assert config_dict["stream"] is True


class TestProviderType:
    """Test ProviderType enum."""
    
    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        from langscope.llm.models import ProviderType
        
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.GROQ.value == "groq"
        assert ProviderType.XAI.value == "xai"
        assert ProviderType.HUGGINGFACE.value == "huggingface"
    
    def test_provider_type_comparison(self):
        """Test ProviderType comparison."""
        from langscope.llm.models import ProviderType
        
        # Can compare enum to string value
        assert ProviderType.OPENAI.value == "openai"


# =============================================================================
# Base Provider Tests
# =============================================================================

class TestBaseLLMProvider:
    """Test BaseLLMProvider abstract class."""
    
    def test_base_provider_is_abstract(self):
        """Test that BaseLLMProvider cannot be instantiated."""
        from langscope.llm.base import BaseLLMProvider
        
        # Should raise TypeError because it's abstract
        with pytest.raises(TypeError):
            BaseLLMProvider()
    
    def test_concrete_provider_mock(self):
        """Test a mocked concrete provider."""
        from langscope.llm.base import BaseLLMProvider
        from langscope.llm.models import ProviderType, LLMResponse, Message, LLMConfig
        
        # Create a concrete implementation
        class MockProvider(BaseLLMProvider):
            provider_type = ProviderType.OPENAI
            
            def _get_api_key_env_var(self):
                return "MOCK_API_KEY"
            
            def _get_default_model(self):
                return "mock-model"
            
            def _init_client(self):
                return MagicMock()
            
            async def generate(self, messages, config=None):
                return LLMResponse(
                    content="Mock response",
                    model=self.default_model,
                    provider=self.provider_type
                )
            
            def generate_sync(self, messages, config=None):
                return LLMResponse(
                    content="Mock response sync",
                    model=self.default_model,
                    provider=self.provider_type
                )
        
        with patch.dict("os.environ", {"MOCK_API_KEY": "test-key"}):
            provider = MockProvider()
            
            assert provider.default_model == "mock-model"
            assert provider.is_configured() is True
    
    def test_provider_not_configured(self):
        """Test provider without API key."""
        from langscope.llm.base import BaseLLMProvider
        from langscope.llm.models import ProviderType, LLMResponse
        
        class MockProvider(BaseLLMProvider):
            provider_type = ProviderType.OPENAI
            
            def _get_api_key_env_var(self):
                return "NONEXISTENT_KEY"
            
            def _get_default_model(self):
                return "mock-model"
            
            def _init_client(self):
                return None
            
            async def generate(self, messages, config=None):
                pass
            
            def generate_sync(self, messages, config=None):
                pass
        
        # Clear the environment variable
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("NONEXISTENT_KEY", None)
            
            provider = MockProvider()
            assert provider.is_configured() is False
    
    def test_provider_get_config(self):
        """Test _get_config method."""
        from langscope.llm.base import BaseLLMProvider
        from langscope.llm.models import ProviderType, LLMResponse, LLMConfig
        
        class MockProvider(BaseLLMProvider):
            provider_type = ProviderType.OPENAI
            
            def _get_api_key_env_var(self):
                return "MOCK_KEY"
            
            def _get_default_model(self):
                return "default-model"
            
            def _init_client(self):
                return None
            
            async def generate(self, messages, config=None):
                pass
            
            def generate_sync(self, messages, config=None):
                pass
        
        with patch.dict("os.environ", {"MOCK_KEY": "key"}):
            provider = MockProvider()
            
            # Test with None config
            config = provider._get_config(None)
            assert config.model == "default-model"
            
            # Test with empty model config
            config = provider._get_config(LLMConfig(model=""))
            assert config.model == "default-model"
            
            # Test with specific model
            config = provider._get_config(LLMConfig(model="specific-model"))
            assert config.model == "specific-model"
    
    def test_provider_format_messages(self):
        """Test _format_messages method."""
        from langscope.llm.base import BaseLLMProvider
        from langscope.llm.models import ProviderType, Message
        
        class MockProvider(BaseLLMProvider):
            provider_type = ProviderType.OPENAI
            
            def _get_api_key_env_var(self):
                return "MOCK_KEY"
            
            def _get_default_model(self):
                return "model"
            
            def _init_client(self):
                return None
            
            async def generate(self, messages, config=None):
                pass
            
            def generate_sync(self, messages, config=None):
                pass
        
        with patch.dict("os.environ", {"MOCK_KEY": "key"}):
            provider = MockProvider()
            
            messages = [
                Message.system("You are helpful"),
                Message.user("Hi"),
                Message.assistant("Hello!")
            ]
            
            formatted = provider._format_messages(messages)
            
            assert len(formatted) == 3
            assert formatted[0]["role"] == "system"
            assert formatted[1]["role"] == "user"
            assert formatted[2]["role"] == "assistant"


# =============================================================================
# Factory Tests
# =============================================================================

class TestLLMFactory:
    """Test LLM factory pattern."""
    
    def test_factory_module_import(self):
        """Test factory module imports."""
        from langscope.llm import factory
        assert factory is not None
    
    def test_get_available_providers(self):
        """Test getting available provider types."""
        from langscope.llm.models import ProviderType
        
        providers = list(ProviderType)
        
        assert ProviderType.OPENAI in providers
        assert ProviderType.ANTHROPIC in providers
        assert len(providers) >= 5


# =============================================================================
# Provider Implementation Tests (Mocked)
# =============================================================================

class TestOpenAIProvider:
    """Test OpenAI provider with mocks."""
    
    def test_openai_provider_import(self):
        """Test OpenAI provider imports."""
        from langscope.llm.providers import openai
        assert openai is not None
    
    def test_openai_provider_creation(self):
        """Test OpenAI provider creation with mocked client."""
        from langscope.llm.providers.openai import OpenAIProvider
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(api_key="test-key")
            
            assert provider.is_configured() is True
            assert provider.default_model is not None


class TestAnthropicProvider:
    """Test Anthropic provider with mocks."""
    
    def test_anthropic_provider_import(self):
        """Test Anthropic provider imports."""
        from langscope.llm.providers import anthropic
        assert anthropic is not None
    
    def test_anthropic_provider_creation(self):
        """Test Anthropic provider creation with mocked client."""
        from langscope.llm.providers.anthropic import AnthropicProvider
        
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(api_key="test-key")
            
            assert provider.is_configured() is True


class TestGroqProvider:
    """Test Groq provider with mocks."""
    
    def test_groq_provider_import(self):
        """Test Groq provider imports."""
        from langscope.llm.providers import groq
        assert groq is not None
    
    def test_groq_provider_creation(self):
        """Test Groq provider creation."""
        from langscope.llm.providers.groq import GroqProvider
        
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            provider = GroqProvider(api_key="test-key")
            
            assert provider.is_configured() is True


# =============================================================================
# Integration Pattern Tests
# =============================================================================

class TestLLMIntegrationPatterns:
    """Test LLM integration patterns."""
    
    def test_conversation_building(self):
        """Test building a conversation with messages."""
        from langscope.llm.models import Message
        
        conversation = [
            Message.system("You are a helpful assistant."),
            Message.user("What is the capital of France?"),
            Message.assistant("The capital of France is Paris."),
            Message.user("And what is its population?"),
        ]
        
        assert len(conversation) == 4
        assert conversation[0].role == "system"
        assert conversation[-1].role == "user"
    
    def test_config_customization(self):
        """Test config customization for different use cases."""
        from langscope.llm.models import LLMConfig
        
        # Creative writing config
        creative_config = LLMConfig(
            model="gpt-4",
            temperature=1.2,
            max_tokens=2000
        )
        
        # Factual response config
        factual_config = LLMConfig(
            model="gpt-4",
            temperature=0.1,
            max_tokens=500
        )
        
        assert creative_config.temperature > factual_config.temperature
    
    def test_response_cost_calculation(self):
        """Test calculating cost from response."""
        from langscope.llm.models import LLMResponse, ProviderType, UsageInfo
        
        # Assume $10 per million input, $30 per million output
        response = LLMResponse(
            content="Response",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            usage=UsageInfo(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500
            )
        )
        
        input_cost = (response.usage.prompt_tokens / 1_000_000) * 10
        output_cost = (response.usage.completion_tokens / 1_000_000) * 30
        total_cost = input_cost + output_cost
        
        assert total_cost > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

