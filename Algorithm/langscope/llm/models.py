"""
Data models for LLM providers.

Defines the common data structures used across all LLM provider implementations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Literal


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    XAI = "xai"
    HUGGINGFACE = "huggingface"


@dataclass
class Message:
    """
    A message in a conversation.
    
    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: The text content of the message
        name: Optional name for the message sender
        tool_calls: Optional tool calls made by the assistant
        tool_call_id: Optional ID for tool call responses
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str, name: Optional[str] = None) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content, name=name)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)


@dataclass
class UsageInfo:
    """
    Token usage information from an LLM response.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        reasoning_tokens: Tokens used for reasoning (if applicable)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageInfo":
        """Create UsageInfo from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens"),
        )


@dataclass
class LLMResponse:
    """
    Response from an LLM provider.
    
    Attributes:
        content: The generated text content
        model: The model used for generation
        provider: The provider type
        usage: Token usage information
        finish_reason: Why the generation stopped
        raw_response: The raw response from the provider
        latency_ms: Response latency in milliseconds
    """
    content: str
    model: str
    provider: ProviderType
    usage: Optional[UsageInfo] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider.value,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else 0,
                "completion_tokens": self.usage.completion_tokens if self.usage else 0,
                "total_tokens": self.usage.total_tokens if self.usage else 0,
            } if self.usage else None,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LLMConfig:
    """
    Configuration for LLM generation.
    
    Attributes:
        model: The model to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop: Stop sequences
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        reasoning_effort: Reasoning effort level (for reasoning models)
    """
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    timeout: int = 300
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.stop:
            d["stop"] = self.stop
        if self.stream:
            d["stream"] = self.stream
        if self.reasoning_effort:
            d["reasoning_effort"] = self.reasoning_effort
        return d

