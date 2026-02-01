"""
Input validation utilities for LangScope.

Provides validation functions for common inputs.
"""

import re
from typing import Dict, List, Tuple, Optional, Any


def validate_ranking(
    ranking: Dict[str, int],
    expected_items: List[str] = None,
    min_items: int = 2
) -> Tuple[bool, str]:
    """
    Validate a ranking dictionary.
    
    Args:
        ranking: Dictionary {item: rank}
        expected_items: Optional list of expected items
        min_items: Minimum number of items
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ranking:
        return False, "Ranking is empty"
    
    if len(ranking) < min_items:
        return False, f"Ranking must have at least {min_items} items"
    
    # Check for valid ranks (1 to N, no duplicates)
    n = len(ranking)
    ranks = list(ranking.values())
    expected_ranks = set(range(1, n + 1))
    actual_ranks = set(ranks)
    
    if actual_ranks != expected_ranks:
        return False, f"Invalid ranks: expected {expected_ranks}, got {actual_ranks}"
    
    if len(ranks) != len(set(ranks)):
        return False, "Duplicate ranks found"
    
    # Check expected items if provided
    if expected_items:
        expected_set = set(expected_items)
        actual_set = set(ranking.keys())
        
        if actual_set != expected_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            
            if missing:
                return False, f"Missing items: {missing}"
            if extra:
                return False, f"Unexpected items: {extra}"
    
    return True, ""


def validate_model_id(model_id: str) -> Tuple[bool, str]:
    """
    Validate a model ID.
    
    Expected format: provider/model_name
    
    Args:
        model_id: Model identifier
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_id:
        return False, "Model ID is empty"
    
    if "/" not in model_id:
        return False, "Model ID should be in format: provider/model_name"
    
    parts = model_id.split("/")
    if len(parts) < 2:
        return False, "Model ID should be in format: provider/model_name"
    
    provider = parts[0]
    model_name = "/".join(parts[1:])
    
    if not provider:
        return False, "Provider cannot be empty"
    
    if not model_name:
        return False, "Model name cannot be empty"
    
    # Check for invalid characters
    if not re.match(r'^[a-zA-Z0-9_-]+$', provider):
        return False, "Provider contains invalid characters"
    
    return True, ""


def validate_domain_name(name: str) -> Tuple[bool, str]:
    """
    Validate a domain name.
    
    Args:
        name: Domain name
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Domain name is empty"
    
    if len(name) < 2:
        return False, "Domain name must be at least 2 characters"
    
    if len(name) > 64:
        return False, "Domain name must be at most 64 characters"
    
    # Must start with letter
    if not name[0].isalpha():
        return False, "Domain name must start with a letter"
    
    # Only alphanumeric and underscore
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        return False, "Domain name can only contain letters, numbers, and underscores"
    
    return True, ""


def validate_trueskill_rating(
    mu: float,
    sigma: float,
    mu_range: Tuple[float, float] = (0, 3000),
    sigma_range: Tuple[float, float] = (1, 500)
) -> Tuple[bool, str]:
    """
    Validate TrueSkill rating values.
    
    Args:
        mu: Mean rating
        sigma: Uncertainty
        mu_range: Valid range for mu
        sigma_range: Valid range for sigma
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if mu < mu_range[0] or mu > mu_range[1]:
        return False, f"μ must be between {mu_range[0]} and {mu_range[1]}"
    
    if sigma < sigma_range[0] or sigma > sigma_range[1]:
        return False, f"σ must be between {sigma_range[0]} and {sigma_range[1]}"
    
    return True, ""


def validate_correlation(value: float) -> Tuple[bool, str]:
    """
    Validate a correlation value.
    
    Args:
        value: Correlation value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < -1 or value > 1:
        return False, "Correlation must be between -1 and 1"
    
    return True, ""


def validate_probability(value: float) -> Tuple[bool, str]:
    """
    Validate a probability value.
    
    Args:
        value: Probability value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < 0 or value > 1:
        return False, "Probability must be between 0 and 1"
    
    return True, ""


def validate_cost(value: float) -> Tuple[bool, str]:
    """
    Validate a cost value.
    
    Args:
        value: Cost in USD
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < 0:
        return False, "Cost cannot be negative"
    
    return True, ""


def sanitize_text(text: str, max_length: int = 100000) -> str:
    """
    Sanitize text input.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    return text


def validate_weights(weights: List[float]) -> Tuple[bool, str]:
    """
    Validate a list of weights.
    
    Args:
        weights: List of weights
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not weights:
        return False, "Weights list is empty"
    
    for i, w in enumerate(weights):
        if w < 0:
            return False, f"Weight at index {i} is negative"
    
    total = sum(weights)
    if total <= 0:
        return False, "Sum of weights must be positive"
    
    return True, ""


