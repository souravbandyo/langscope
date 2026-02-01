"""
Benchmark results storage and retrieval.

Stores historical benchmark scores for base models.

DEPRECATED: This module is deprecated. Import from langscope.models.benchmarks instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.benchmarks import (
    BenchmarkResult,
    BenchmarkAggregates,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkAggregates",
]

