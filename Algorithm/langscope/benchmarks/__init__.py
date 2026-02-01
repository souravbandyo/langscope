"""
Benchmarks module for LangScope.

Provides:
- Benchmark definitions (MMLU, Chatbot Arena, HumanEval, etc.)
- Benchmark results storage and retrieval
- LangScope correlation analysis with external benchmarks
"""

from langscope.benchmarks.definitions import (
    BenchmarkCategory,
    BenchmarkScoring,
    BenchmarkDefinition,
    PREDEFINED_BENCHMARKS,
    get_benchmark_definition,
    list_benchmarks_by_category,
)
from langscope.benchmarks.results import (
    BenchmarkResult,
    BenchmarkAggregates,
)
from langscope.benchmarks.correlation import (
    BenchmarkCorrelation,
    compute_benchmark_correlation,
    get_prior_from_benchmarks,
)

__all__ = [
    # Definitions
    "BenchmarkCategory",
    "BenchmarkScoring",
    "BenchmarkDefinition",
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


