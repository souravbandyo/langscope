"""
Metrics Collection for 10-Dimensional TrueSkill Evaluation.

Provides collection of metrics needed for all 10 dimensions:
- Latency and TTFT measurement
- Consistency evaluation
- Instruction following validation
- Hallucination detection
- Long context evaluation
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime


@dataclass
class BattleMetrics:
    """
    Metrics collected during a battle/response.
    
    Contains all measurements needed for 10-dimensional scoring.
    """
    # Timing metrics
    latency_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token
    
    # Cost metrics
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Consistency metrics (from repeated runs)
    consistency_runs: int = 0
    response_variance: float = 0.0
    response_scores: List[float] = field(default_factory=list)
    
    # Instruction following
    constraints_satisfied: int = 0
    total_constraints: int = 0
    constraint_details: Dict[str, bool] = field(default_factory=dict)
    
    # Hallucination detection
    hallucination_count: int = 0
    verifiable_claims: int = 0
    hallucination_details: List[str] = field(default_factory=list)
    
    # Long context
    context_length: int = 0
    quality_at_length: float = 0.0
    
    # Metadata
    model_id: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "consistency_runs": self.consistency_runs,
            "response_variance": self.response_variance,
            "response_scores": self.response_scores,
            "constraints_satisfied": self.constraints_satisfied,
            "total_constraints": self.total_constraints,
            "constraint_details": self.constraint_details,
            "hallucination_count": self.hallucination_count,
            "verifiable_claims": self.verifiable_claims,
            "hallucination_details": self.hallucination_details,
            "context_length": self.context_length,
            "quality_at_length": self.quality_at_length,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BattleMetrics":
        """Create from dictionary."""
        return cls(
            latency_ms=data.get("latency_ms", 0.0),
            ttft_ms=data.get("ttft_ms", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            consistency_runs=data.get("consistency_runs", 0),
            response_variance=data.get("response_variance", 0.0),
            response_scores=data.get("response_scores", []),
            constraints_satisfied=data.get("constraints_satisfied", 0),
            total_constraints=data.get("total_constraints", 0),
            constraint_details=data.get("constraint_details", {}),
            hallucination_count=data.get("hallucination_count", 0),
            verifiable_claims=data.get("verifiable_claims", 0),
            hallucination_details=data.get("hallucination_details", []),
            context_length=data.get("context_length", 0),
            quality_at_length=data.get("quality_at_length", 0.0),
            model_id=data.get("model_id", ""),
            timestamp=data.get("timestamp", ""),
        )


# =============================================================================
# Latency Measurement
# =============================================================================

class LatencyTimer:
    """Context manager for measuring latency."""
    
    def __init__(self):
        self.start_time: float = 0
        self.first_token_time: float = 0
        self.end_time: float = 0
        self._first_token_recorded = False
    
    def __enter__(self) -> "LatencyTimer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    def record_first_token(self):
        """Record time of first token."""
        if not self._first_token_recorded:
            self.first_token_time = time.perf_counter()
            self._first_token_recorded = True
    
    @property
    def latency_ms(self) -> float:
        """Get total latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def ttft_ms(self) -> float:
        """Get time to first token in milliseconds."""
        if self._first_token_recorded:
            return (self.first_token_time - self.start_time) * 1000
        return self.latency_ms  # Fallback to total if not recorded


def collect_latency(
    api_call: Callable,
    *args,
    **kwargs
) -> Tuple[Any, float, float]:
    """
    Collect latency metrics from an API call.
    
    Args:
        api_call: Function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Tuple of (result, latency_ms, ttft_ms)
    """
    timer = LatencyTimer()
    with timer:
        result = api_call(*args, **kwargs)
    
    return result, timer.latency_ms, timer.latency_ms  # TTFT requires streaming


async def collect_latency_async(
    api_call: Callable,
    *args,
    **kwargs
) -> Tuple[Any, float, float]:
    """
    Collect latency metrics from an async API call.
    
    Args:
        api_call: Async function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Tuple of (result, latency_ms, ttft_ms)
    """
    start = time.perf_counter()
    result = await api_call(*args, **kwargs)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    return result, latency_ms, latency_ms


# =============================================================================
# Consistency Evaluation
# =============================================================================

async def run_consistency_evaluation(
    model_call: Callable,
    prompt: str,
    n_runs: int = 5,
    scorer: Callable[[str], float] = None
) -> Tuple[float, List[float]]:
    """
    Run consistency evaluation with repeated calls.
    
    Args:
        model_call: Async function that returns response text
        prompt: Prompt to send
        n_runs: Number of repeated runs
        scorer: Optional function to score responses (returns 0-1)
    
    Returns:
        Tuple of (variance, list of scores)
    """
    scores = []
    
    for _ in range(n_runs):
        response = await model_call(prompt)
        
        if scorer:
            score = scorer(response)
        else:
            # Default: use response length as proxy for consistency
            score = len(response)
        
        scores.append(score)
    
    if len(scores) < 2:
        return 0.0, scores
    
    variance = statistics.variance(scores)
    return variance, scores


def compute_response_variance(scores: List[float]) -> float:
    """Compute variance from a list of scores."""
    if len(scores) < 2:
        return 0.0
    return statistics.variance(scores)


# =============================================================================
# Instruction Following Validation
# =============================================================================

@dataclass
class Constraint:
    """A constraint to validate."""
    name: str
    validator: Callable[[str], bool]
    description: str = ""


def validate_instruction_following(
    response: str,
    constraints: List[Constraint]
) -> Tuple[int, int, Dict[str, bool]]:
    """
    Validate instruction following against constraints.
    
    Args:
        response: Model response text
        constraints: List of constraints to check
    
    Returns:
        Tuple of (satisfied_count, total_count, details)
    """
    details = {}
    satisfied = 0
    
    for constraint in constraints:
        try:
            passed = constraint.validator(response)
            details[constraint.name] = passed
            if passed:
                satisfied += 1
        except Exception:
            details[constraint.name] = False
    
    return satisfied, len(constraints), details


# Common constraint validators
def has_json_format(response: str) -> bool:
    """Check if response contains valid JSON."""
    import json
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json.loads(response[start:end])
            return True
        return False
    except json.JSONDecodeError:
        return False


def has_max_length(max_words: int) -> Callable[[str], bool]:
    """Create validator for maximum word count."""
    def validator(response: str) -> bool:
        return len(response.split()) <= max_words
    return validator


def has_min_length(min_words: int) -> Callable[[str], bool]:
    """Create validator for minimum word count."""
    def validator(response: str) -> bool:
        return len(response.split()) >= min_words
    return validator


def contains_keyword(keyword: str) -> Callable[[str], bool]:
    """Create validator for required keyword."""
    def validator(response: str) -> bool:
        return keyword.lower() in response.lower()
    return validator


def has_bullet_points(min_count: int = 3) -> Callable[[str], bool]:
    """Create validator for bullet point format."""
    def validator(response: str) -> bool:
        bullets = response.count("- ") + response.count("• ") + response.count("* ")
        return bullets >= min_count
    return validator


def has_numbered_list(min_count: int = 3) -> Callable[[str], bool]:
    """Create validator for numbered list format."""
    def validator(response: str) -> bool:
        import re
        pattern = r"^\d+\.|^\d+\)"
        matches = re.findall(pattern, response, re.MULTILINE)
        return len(matches) >= min_count
    return validator


# =============================================================================
# Hallucination Detection
# =============================================================================

def check_hallucinations(
    response: str,
    ground_truth: str = None,
    fact_checker: Callable[[str, str], List[str]] = None
) -> Tuple[int, int, List[str]]:
    """
    Check for hallucinations in response.
    
    Args:
        response: Model response text
        ground_truth: Optional ground truth for comparison
        fact_checker: Optional function to check facts
    
    Returns:
        Tuple of (hallucination_count, verifiable_claims, details)
    """
    if fact_checker and ground_truth:
        hallucinations = fact_checker(response, ground_truth)
        # Estimate verifiable claims as sentence count
        verifiable = len(response.split(". "))
        return len(hallucinations), verifiable, hallucinations
    
    # Without a fact checker, return zeros
    return 0, 0, []


def simple_fact_checker(response: str, ground_truth: str) -> List[str]:
    """
    Simple fact checker based on keyword presence.
    
    This is a placeholder - real implementations should use
    more sophisticated NLI or retrieval-based methods.
    
    Args:
        response: Model response
        ground_truth: Reference text
    
    Returns:
        List of potential hallucinations
    """
    hallucinations = []
    
    # Extract key phrases from ground truth
    gt_words = set(ground_truth.lower().split())
    
    # Check each sentence in response
    sentences = response.split(". ")
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        # If sentence has very low overlap, flag as potential hallucination
        overlap = len(sentence_words & gt_words) / max(len(sentence_words), 1)
        if overlap < 0.1 and len(sentence_words) > 5:
            hallucinations.append(sentence)
    
    return hallucinations


# =============================================================================
# Long Context Evaluation
# =============================================================================

@dataclass
class LongContextResult:
    """Result of long context evaluation."""
    context_lengths: List[int]
    quality_scores: Dict[int, float]
    degradation_ratio: float  # quality@max / quality@baseline
    baseline_quality: float
    max_quality: float


async def run_long_context_evaluation(
    model_call: Callable,
    base_prompt: str,
    context_generator: Callable[[int], str],
    quality_scorer: Callable[[str], float],
    test_lengths: List[int] = None,
    baseline_length: int = 4096
) -> LongContextResult:
    """
    Evaluate model performance at different context lengths.
    
    Args:
        model_call: Async function that returns response
        base_prompt: Base prompt to append context to
        context_generator: Function to generate context of given length
        quality_scorer: Function to score response quality
        test_lengths: List of context lengths to test
        baseline_length: Baseline length for degradation ratio
    
    Returns:
        LongContextResult with quality at each length
    """
    if test_lengths is None:
        test_lengths = [4096, 16384, 32768, 65536, 131072]
    
    quality_scores = {}
    
    for length in test_lengths:
        try:
            context = context_generator(length)
            full_prompt = f"{context}\n\n{base_prompt}"
            response = await model_call(full_prompt)
            score = quality_scorer(response)
            quality_scores[length] = score
        except Exception:
            # Model may not support this context length
            quality_scores[length] = 0.0
    
    baseline_quality = quality_scores.get(baseline_length, 1.0)
    max_length = max(test_lengths)
    max_quality = quality_scores.get(max_length, 0.0)
    
    degradation_ratio = max_quality / baseline_quality if baseline_quality > 0 else 0.0
    
    return LongContextResult(
        context_lengths=test_lengths,
        quality_scores=quality_scores,
        degradation_ratio=degradation_ratio,
        baseline_quality=baseline_quality,
        max_quality=max_quality,
    )


# =============================================================================
# Metrics Aggregator
# =============================================================================

class MetricsCollector:
    """Collects and aggregates metrics across multiple battles."""
    
    def __init__(self):
        self.metrics: List[BattleMetrics] = []
    
    def add(self, metrics: BattleMetrics):
        """Add metrics from a battle."""
        self.metrics.append(metrics)
    
    def get_average_latency(self) -> float:
        """Get average latency across all battles."""
        if not self.metrics:
            return 0.0
        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)
    
    def get_average_ttft(self) -> float:
        """Get average TTFT across all battles."""
        if not self.metrics:
            return 0.0
        return sum(m.ttft_ms for m in self.metrics) / len(self.metrics)
    
    def get_average_consistency_variance(self) -> float:
        """Get average response variance."""
        variances = [m.response_variance for m in self.metrics if m.response_variance > 0]
        if not variances:
            return 0.0
        return sum(variances) / len(variances)
    
    def get_instruction_following_rate(self) -> float:
        """Get overall instruction following rate."""
        total_satisfied = sum(m.constraints_satisfied for m in self.metrics)
        total_constraints = sum(m.total_constraints for m in self.metrics)
        if total_constraints == 0:
            return 1.0
        return total_satisfied / total_constraints
    
    def get_hallucination_rate(self) -> float:
        """Get overall hallucination rate."""
        total_hallucinations = sum(m.hallucination_count for m in self.metrics)
        total_claims = sum(m.verifiable_claims for m in self.metrics)
        if total_claims == 0:
            return 0.0
        return total_hallucinations / total_claims
    
    def get_total_cost(self) -> float:
        """Get total cost across all battles."""
        return sum(m.cost_usd for m in self.metrics)
    
    def get_total_tokens(self) -> Tuple[int, int]:
        """Get total input and output tokens."""
        input_tokens = sum(m.input_tokens for m in self.metrics)
        output_tokens = sum(m.output_tokens for m in self.metrics)
        return input_tokens, output_tokens
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            "total_battles": len(self.metrics),
            "avg_latency_ms": self.get_average_latency(),
            "avg_ttft_ms": self.get_average_ttft(),
            "avg_consistency_variance": self.get_average_consistency_variance(),
            "instruction_following_rate": self.get_instruction_following_rate(),
            "hallucination_rate": self.get_hallucination_rate(),
            "total_cost_usd": self.get_total_cost(),
            "total_tokens": self.get_total_tokens(),
        }


