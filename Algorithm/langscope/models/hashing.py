"""
Content hashing utilities for deduplication and change detection.

Provides:
- Content hashing for case/question deduplication
- Price hashing for change detection
- Response caching keys
"""

import hashlib
import json
import re
from typing import Any, Dict, Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent hashing.
    
    - Converts to lowercase
    - Removes extra whitespace
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Normalize whitespace (multiple spaces/newlines -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def content_hash(text: str, domain: str = "") -> str:
    """
    Generate a hash for content deduplication.
    
    Used to detect duplicate cases, questions, or responses.
    
    Args:
        text: The content text
        domain: Optional domain for domain-specific hashing
    
    Returns:
        SHA256 hash (first 16 characters)
    """
    normalized = normalize_text(text)
    
    # Include domain in hash if provided
    if domain:
        normalized = f"{domain}:{normalized}"
    
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


def price_hash(
    model_id: str,
    input_cost_per_million: float,
    output_cost_per_million: float
) -> str:
    """
    Generate a hash for price change detection.
    
    Args:
        model_id: The model/deployment ID
        input_cost_per_million: Input cost per million tokens
        output_cost_per_million: Output cost per million tokens
    
    Returns:
        SHA256 hash (first 16 characters)
    """
    # Round to avoid floating point precision issues
    input_rounded = round(input_cost_per_million, 6)
    output_rounded = round(output_cost_per_million, 6)
    
    data = f"{model_id}:i{input_rounded}:o{output_rounded}"
    
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]


def response_cache_key(
    prompt: str,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 0,
    system_prompt: str = ""
) -> str:
    """
    Generate a cache key for response caching.
    
    Args:
        prompt: The user prompt
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        system_prompt: Optional system prompt
    
    Returns:
        SHA256 hash (first 24 characters for lower collision)
    """
    # Only cache deterministic requests (temperature = 0)
    if temperature != 0:
        # Include timestamp to avoid caching non-deterministic responses
        import time
        data = f"{model_id}:{prompt}:{system_prompt}:{temperature}:{max_tokens}:{time.time()}"
    else:
        data = f"{model_id}:{prompt}:{system_prompt}:{max_tokens}"
    
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:24]


def data_hash(data: Any) -> str:
    """
    Generate a hash for arbitrary data structures.
    
    Useful for detecting changes in complex objects.
    
    Args:
        data: Any JSON-serializable data
    
    Returns:
        SHA256 hash (first 16 characters)
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, (list, tuple)):
        data_str = json.dumps(list(data), default=str)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:16]


def benchmark_score_hash(
    base_model_id: str,
    benchmark_id: str,
    score: float,
    variant: str = ""
) -> str:
    """
    Generate a hash for benchmark score change detection.
    
    Args:
        base_model_id: Base model ID
        benchmark_id: Benchmark identifier
        score: The score value
        variant: Score variant (e.g., "5-shot")
    
    Returns:
        SHA256 hash (first 16 characters)
    """
    score_rounded = round(score, 4)
    data = f"{base_model_id}:{benchmark_id}:{variant}:{score_rounded}"
    
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]


class ContentHasher:
    """
    Helper class for tracking content hashes.
    
    Used during match generation to detect duplicates.
    """
    
    def __init__(self, db=None):
        """
        Initialize content hasher.
        
        Args:
            db: Optional MongoDB instance for persistence
        """
        self.db = db
        self._cache: Dict[str, bool] = {}
    
    def is_duplicate(
        self,
        text: str,
        content_type: str = "case",
        domain: str = ""
    ) -> bool:
        """
        Check if content is a duplicate.
        
        Args:
            text: Content text
            content_type: Type ("case", "question", "response")
            domain: Domain name
        
        Returns:
            True if duplicate exists
        """
        hash_key = content_hash(text, domain)
        
        # Check cache first
        cache_key = f"{content_type}:{hash_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check database if available
        if self.db and hasattr(self.db, 'check_content_duplicate'):
            is_dup = self.db.check_content_duplicate(
                content_hash=hash_key,
                content_type=content_type,
                domain=domain
            )
            # Cache the result
            self._cache[cache_key] = is_dup
            return is_dup
        
        return False
    
    def register(
        self,
        text: str,
        content_type: str = "case",
        domain: str = "",
        match_id: str = "",
        generator_model_id: str = ""
    ) -> str:
        """
        Register content hash.
        
        Args:
            text: Content text
            content_type: Type ("case", "question", "response")
            domain: Domain name
            match_id: ID of the match where this was generated
            generator_model_id: Model that generated this content
        
        Returns:
            The content hash
        """
        hash_key = content_hash(text, domain)
        cache_key = f"{content_type}:{hash_key}"
        
        self._cache[cache_key] = True
        
        # Persist to database if available
        if self.db and hasattr(self.db, 'register_content'):
            # Create a preview of the content (first 200 chars)
            content_preview = text[:200] if text else ""
            
            self.db.register_content(
                content_hash=hash_key,
                content_type=content_type,
                domain=domain,
                match_id=match_id,
                generator_model_id=generator_model_id,
                content_preview=content_preview
            )
        
        return hash_key
    
    def get_info(self, text: str, domain: str = "") -> Optional[Dict[str, Any]]:
        """
        Get information about a content hash.
        
        Args:
            text: Content text
            domain: Domain name
        
        Returns:
            Content hash info or None
        """
        hash_key = content_hash(text, domain)
        
        if self.db and hasattr(self.db, 'get_content_hash_info'):
            return self.db.get_content_hash_info(hash_key)
        
        return None
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache = {}

