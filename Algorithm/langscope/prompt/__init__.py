"""
LangScope Prompt Management Module.

Provides domain classification, prompt processing, and caching integration.
"""

from langscope.prompt.constants import (
    CATEGORIES,
    DOMAIN_KEYWORDS,
    LANGUAGE_PATTERNS,
    DOMAIN_NAME_MAPPING,
    GROUND_TRUTH_DOMAINS,
    DOMAIN_THRESHOLDS,
)
from langscope.prompt.classifier import (
    HierarchicalDomainClassifier,
    ClassificationResult,
)
from langscope.prompt.manager import (
    PromptManager,
    PromptProcessingResult,
)

__all__ = [
    # Constants
    "CATEGORIES",
    "DOMAIN_KEYWORDS",
    "LANGUAGE_PATTERNS",
    "DOMAIN_NAME_MAPPING",
    "GROUND_TRUTH_DOMAINS",
    "DOMAIN_THRESHOLDS",
    # Classifier
    "HierarchicalDomainClassifier",
    "ClassificationResult",
    # Manager
    "PromptManager",
    "PromptProcessingResult",
]

