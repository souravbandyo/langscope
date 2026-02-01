"""
Benchmark correlation with LangScope ratings.

Provides:
- Correlation computation between external benchmarks and LangScope ratings
- Prior initialization from benchmark scores for new models

DEPRECATED: This module is deprecated. Import from langscope.models.benchmarks instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.benchmarks import (
    BenchmarkCorrelation,
    compute_benchmark_correlation,
    get_prior_from_benchmarks,
)

__all__ = [
    "BenchmarkCorrelation",
    "compute_benchmark_correlation",
    "get_prior_from_benchmarks",
]

