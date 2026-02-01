"""
Benchmark definitions, results, and correlations.

Contains classes for external benchmark evaluations (MMLU, Arena, etc.)
and their correlation with LangScope ratings.
"""

from langscope.models.benchmarks.definitions import (
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
from langscope.models.benchmarks.results import (
    BenchmarkResult,
    BenchmarkAggregates,
)
from langscope.models.benchmarks.correlation import (
    BenchmarkCorrelation,
    compute_benchmark_correlation,
    get_prior_from_benchmarks,
)

__all__ = [
    # Definitions
    "BenchmarkDefinition",
    "BenchmarkCategory",
    "BenchmarkScoring",
    "BenchmarkSource",
    "BenchmarkAutomation",
    "LangScopeCorrelation",
    "PREDEFINED_BENCHMARKS",
    "get_benchmark_definition",
    "list_benchmarks_by_category",
    # Results
    "BenchmarkResult",
    "BenchmarkAggregates",
    # Correlation
    "BenchmarkCorrelation",
    "compute_benchmark_correlation",
    "get_prior_from_benchmarks",
]

