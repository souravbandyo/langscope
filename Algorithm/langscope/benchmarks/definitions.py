"""
Benchmark definitions for external evaluations.

Defines what each benchmark measures, how it's scored, and where to get results.

DEPRECATED: This module is deprecated. Import from langscope.models.benchmarks instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.benchmarks import (
    BenchmarkDefinition,
    BenchmarkCategory,
    BenchmarkScoring,
    BenchmarkSource,
    BenchmarkAutomation,
    LangScopeCorrelation,
    PREDEFINED_BENCHMARKS,
    get_benchmark_definition,
    list_benchmarks_by_category,
)

__all__ = [
    "BenchmarkDefinition",
    "BenchmarkCategory",
    "BenchmarkScoring",
    "BenchmarkSource",
    "BenchmarkAutomation",
    "LangScopeCorrelation",
    "PREDEFINED_BENCHMARKS",
    "get_benchmark_definition",
    "list_benchmarks_by_category",
]

