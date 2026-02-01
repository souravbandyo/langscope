"""
Model definitions with pricing information from providers.

Each model entry contains:
- name: Human-readable model name
- model_id: Provider-specific model identifier (LiteLLM format)
- provider: Provider name
- input_cost_per_million: Cost per million input tokens (USD)
- output_cost_per_million: Cost per million output tokens (USD)
- pricing_source: Source of pricing information
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# Model Definitions
# =============================================================================

MODELS: List[Dict[str, Any]] = [
    # OpenAI Models
    {
        "name": "GPT-4.1",
        "model_id": "openai/gpt-4.1-2025-04-14",
        "provider": "openai",
        "input_cost_per_million": 2.00,
        "output_cost_per_million": 8.00,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "GPT-4.1 mini",
        "model_id": "openai/gpt-4.1-mini-2025-04-14",
        "provider": "openai",
        "input_cost_per_million": 0.40,
        "output_cost_per_million": 1.60,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "GPT-4.1 nano",
        "model_id": "openai/gpt-4.1-nano-2025-04-14",
        "provider": "openai",
        "input_cost_per_million": 0.10,
        "output_cost_per_million": 0.40,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "GPT-4o",
        "model_id": "openai/gpt-4o-2024-11-20",
        "provider": "openai",
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "GPT-4o mini",
        "model_id": "openai/gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "o1",
        "model_id": "openai/o1-2024-12-17",
        "provider": "openai",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 60.00,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "o1 mini",
        "model_id": "openai/o1-mini-2024-09-12",
        "provider": "openai",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 12.00,
        "pricing_source": "OpenAI API pricing page"
    },
    {
        "name": "o3 mini",
        "model_id": "openai/o3-mini-2025-01-31",
        "provider": "openai",
        "input_cost_per_million": 1.10,
        "output_cost_per_million": 4.40,
        "pricing_source": "OpenAI API pricing page"
    },
    
    # Anthropic Models
    {
        "name": "Claude 3.7 Sonnet",
        "model_id": "anthropic/claude-3-7-sonnet-20250219",
        "provider": "anthropic",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "pricing_source": "Anthropic API pricing page"
    },
    {
        "name": "Claude 3.5 Sonnet",
        "model_id": "anthropic/claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "pricing_source": "Anthropic API pricing page"
    },
    {
        "name": "Claude 3.5 Haiku",
        "model_id": "anthropic/claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "input_cost_per_million": 0.80,
        "output_cost_per_million": 4.00,
        "pricing_source": "Anthropic API pricing page"
    },
    {
        "name": "Claude 3 Opus",
        "model_id": "anthropic/claude-3-opus-20240229",
        "provider": "anthropic",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 75.00,
        "pricing_source": "Anthropic API pricing page"
    },
    
    # Google Models
    {
        "name": "Gemini 2.5 Pro",
        "model_id": "gemini/gemini-2.5-pro-preview-05-06",
        "provider": "google",
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 10.00,
        "pricing_source": "Google AI pricing page"
    },
    {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini/gemini-2.5-flash-preview-04-17",
        "provider": "google",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "pricing_source": "Google AI pricing page"
    },
    {
        "name": "Gemini 2.0 Flash",
        "model_id": "gemini/gemini-2.0-flash",
        "provider": "google",
        "input_cost_per_million": 0.10,
        "output_cost_per_million": 0.40,
        "pricing_source": "Google AI pricing page"
    },
    {
        "name": "Gemini 1.5 Pro",
        "model_id": "gemini/gemini-1.5-pro",
        "provider": "google",
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 5.00,
        "pricing_source": "Google AI pricing page"
    },
    
    # Meta Models (via various providers)
    {
        "name": "Llama 3.3 70B",
        "model_id": "groq/llama-3.3-70b-versatile",
        "provider": "groq",
        "input_cost_per_million": 0.59,
        "output_cost_per_million": 0.79,
        "pricing_source": "Groq pricing page"
    },
    {
        "name": "Llama 3.1 8B",
        "model_id": "groq/llama-3.1-8b-instant",
        "provider": "groq",
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.08,
        "pricing_source": "Groq pricing page"
    },
    
    # Mistral Models
    {
        "name": "Mistral Large",
        "model_id": "mistral/mistral-large-latest",
        "provider": "mistral",
        "input_cost_per_million": 2.00,
        "output_cost_per_million": 6.00,
        "pricing_source": "Mistral AI pricing page"
    },
    {
        "name": "Mistral Small",
        "model_id": "mistral/mistral-small-latest",
        "provider": "mistral",
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.60,
        "pricing_source": "Mistral AI pricing page"
    },
    {
        "name": "Codestral",
        "model_id": "mistral/codestral-latest",
        "provider": "mistral",
        "input_cost_per_million": 0.30,
        "output_cost_per_million": 0.90,
        "pricing_source": "Mistral AI pricing page"
    },
    
    # DeepSeek Models
    {
        "name": "DeepSeek V3",
        "model_id": "deepseek/deepseek-chat",
        "provider": "deepseek",
        "input_cost_per_million": 0.27,
        "output_cost_per_million": 1.10,
        "pricing_source": "DeepSeek pricing page"
    },
    {
        "name": "DeepSeek R1",
        "model_id": "deepseek/deepseek-reasoner",
        "provider": "deepseek",
        "input_cost_per_million": 0.55,
        "output_cost_per_million": 2.19,
        "pricing_source": "DeepSeek pricing page"
    },
    
    # Cohere Models
    {
        "name": "Command R+",
        "model_id": "cohere/command-r-plus",
        "provider": "cohere",
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
        "pricing_source": "Cohere pricing page"
    },
    {
        "name": "Command R",
        "model_id": "cohere/command-r",
        "provider": "cohere",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "pricing_source": "Cohere pricing page"
    },
]

# =============================================================================
# Model Capabilities
# =============================================================================

MODEL_CAPS = {
    "max_matches_per_model": 50,
    "default_max_tokens": 4096,
    "default_temperature": 0.7,
}

# =============================================================================
# Match Configuration
# =============================================================================

MATCH_CONFIG = {
    "players_per_match": 6,
    "min_players": 5,
    "max_players": 6,
    "judges_per_match": 5,
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_all_models() -> List[Dict[str, Any]]:
    """Get all model definitions."""
    return MODELS.copy()


def get_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get model definition by name."""
    for model in MODELS:
        if model["name"] == name:
            return model
    return None


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model definition by model_id."""
    for model in MODELS:
        if model["model_id"] == model_id:
            return model
    return None


def get_models_by_provider(provider: str) -> List[Dict[str, Any]]:
    """Get all models from a specific provider."""
    return [m for m in MODELS if m["provider"] == provider]


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str = None,
    model_id: str = None,
    input_cost_per_million: float = None,
    output_cost_per_million: float = None,
) -> float:
    """
    Calculate the cost for a model call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Model name (optional if costs provided directly)
        model_id: Model ID (optional if costs provided directly)
        input_cost_per_million: Direct input cost (optional)
        output_cost_per_million: Direct output cost (optional)
    
    Returns:
        Total cost in USD
    """
    if input_cost_per_million is None or output_cost_per_million is None:
        model = None
        if model_name:
            model = get_model_by_name(model_name)
        elif model_id:
            model = get_model_by_id(model_id)
        
        if model:
            input_cost_per_million = model["input_cost_per_million"]
            output_cost_per_million = model["output_cost_per_million"]
        else:
            raise ValueError("Model not found and costs not provided")
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    
    return input_cost + output_cost


