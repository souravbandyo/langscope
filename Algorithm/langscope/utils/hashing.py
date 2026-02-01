"""
Content hashing utilities for deduplication and change detection.

Provides:
- Content hashing for case/question deduplication
- Price hashing for change detection
- Response caching keys

DEPRECATED: This module is deprecated. Import from langscope.models.hashing instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.hashing import (
    content_hash,
    price_hash,
    response_cache_key,
    data_hash,
    benchmark_score_hash,
    normalize_text,
    ContentHasher,
)

__all__ = [
    "content_hash",
    "price_hash",
    "response_cache_key",
    "data_hash",
    "benchmark_score_hash",
    "normalize_text",
    "ContentHasher",
]
