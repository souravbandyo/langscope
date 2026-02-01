"""
Ground Truth Evaluation Module.

Provides objective evaluation for multimodal (ASR, TTS, Visual QA, Document Extraction)
and long context (Needle in Haystack, Long Document QA, Code Completion) domains
where ground truth exists and evaluation is deterministic.

Key Components:
- MetricRegistry: Registry of evaluation metrics per domain
- GroundTruthJudge: Base class for ground truth evaluation
- GroundTruthManager: Sample loading and stratified sampling
- GroundTruthMatchWorkflow: Complete workflow for GT evaluation
- StratifiedSampler: Balanced sampling across dimensions
"""

from langscope.ground_truth.metrics import (
    MetricRegistry,
    word_error_rate,
    character_error_rate,
    match_error_rate,
    bleu_score,
    rouge_l,
    exact_match,
    contains,
    compute_needle_metrics,
)

from langscope.ground_truth.judge import (
    EvaluationMode,
    GroundTruthScore,
    GroundTruthJudge,
)

from langscope.ground_truth.manager import (
    GroundTruthSample,
    GroundTruthManager,
)

from langscope.ground_truth.workflow import (
    GroundTruthMatchResult,
    GroundTruthMatchWorkflow,
)

from langscope.ground_truth.sampling import (
    StratificationDimension,
    SamplingStrategy,
    StratifiedSampler,
)

__all__ = [
    # Metrics
    "MetricRegistry",
    "word_error_rate",
    "character_error_rate", 
    "match_error_rate",
    "bleu_score",
    "rouge_l",
    "exact_match",
    "contains",
    "compute_needle_metrics",
    # Judge
    "EvaluationMode",
    "GroundTruthScore",
    "GroundTruthJudge",
    # Manager
    "GroundTruthSample",
    "GroundTruthManager",
    # Workflow
    "GroundTruthMatchResult",
    "GroundTruthMatchWorkflow",
    # Sampling
    "StratificationDimension",
    "SamplingStrategy",
    "StratifiedSampler",
]


