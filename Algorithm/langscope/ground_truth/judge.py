"""
Ground Truth Judge.

Evaluates model responses against ground truth using objective metrics.
Supports three evaluation modes:
1. METRICS_ONLY: Pure automated metrics (WER, BLEU, etc.)
2. HYBRID: Metrics + LLM semantic verification
3. LLM_JUDGE: Full LLM evaluation (for TTS, image quality)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum

from langscope.ground_truth.metrics import (
    MetricRegistry,
    compute_metric,
    word_error_rate,
    character_error_rate,
    exact_match,
    contains,
    compute_needle_metrics,
)

if TYPE_CHECKING:
    pass


class EvaluationMode(Enum):
    """Evaluation modes for ground truth judging."""
    METRICS_ONLY = "metrics_only"
    HYBRID = "hybrid"
    LLM_JUDGE = "llm_judge"


@dataclass
class GroundTruthScore:
    """Score from ground truth evaluation."""
    model_id: str
    sample_id: str
    
    # Primary metrics computed
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Overall score (for ranking)
    overall: float = 0.0
    
    # Semantic match score (if hybrid mode)
    semantic_match: Optional[float] = None
    
    # Metadata
    evaluation_mode: str = "metrics_only"
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "sample_id": self.sample_id,
            "metrics": self.metrics,
            "overall": self.overall,
            "semantic_match": self.semantic_match,
            "evaluation_mode": self.evaluation_mode,
            "error": self.error,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthScore":
        """Create from dictionary."""
        return cls(
            model_id=data.get("model_id", ""),
            sample_id=data.get("sample_id", ""),
            metrics=data.get("metrics", {}),
            overall=data.get("overall", 0.0),
            semantic_match=data.get("semantic_match"),
            evaluation_mode=data.get("evaluation_mode", "metrics_only"),
            error=data.get("error"),
            timestamp=data.get("timestamp", ""),
        )


class GroundTruthJudge:
    """
    Evaluates model responses against ground truth.
    
    Supports three modes:
    1. METRICS_ONLY: Pure automated metrics (WER, BLEU, etc.)
    2. HYBRID: Metrics + LLM semantic verification
    3. LLM_JUDGE: Full LLM evaluation (for TTS, image quality)
    """
    
    def __init__(
        self,
        domain: str,
        llm_caller: Any = None,
        config: Dict = None
    ):
        """
        Initialize judge.
        
        Args:
            domain: Domain being evaluated
            llm_caller: LLM calling interface for hybrid/LLM modes
            config: Additional configuration
        """
        self.domain = domain
        self.llm_caller = llm_caller
        self.config = config or {}
        self.metric_registry = MetricRegistry()
    
    async def evaluate(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict,
        mode: EvaluationMode = None
    ) -> GroundTruthScore:
        """
        Evaluate a single response against ground truth.
        
        Args:
            response: Model's response
            ground_truth: The correct answer
            sample: Sample metadata
            mode: Override evaluation mode
        
        Returns:
            GroundTruthScore with metrics and overall score
        """
        mode = mode or self._get_default_mode()
        model_id = sample.get("model_id", "")
        sample_id = sample.get("sample_id", "")
        
        try:
            # Compute automated metrics
            metrics = self._compute_metrics(response, ground_truth, sample)
            
            # Add semantic evaluation if hybrid
            semantic_match = None
            if mode == EvaluationMode.HYBRID and self.llm_caller:
                semantic_match = await self._semantic_evaluate(
                    response, ground_truth, sample
                )
                metrics["semantic_match"] = semantic_match
            
            # Compute overall score
            overall = self._compute_overall(metrics, mode)
            
            return GroundTruthScore(
                model_id=model_id,
                sample_id=sample_id,
                metrics=metrics,
                overall=overall,
                semantic_match=semantic_match,
                evaluation_mode=mode.value,
            )
            
        except Exception as e:
            return GroundTruthScore(
                model_id=model_id,
                sample_id=sample_id,
                metrics={},
                overall=0.0,
                evaluation_mode=mode.value,
                error=str(e),
            )
    
    async def evaluate_batch(
        self,
        responses: Dict[str, str],
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, GroundTruthScore]:
        """
        Evaluate multiple responses, return scores keyed by model_id.
        
        Args:
            responses: {model_id: response_text}
            ground_truth: The correct answer
            sample: Sample metadata
        
        Returns:
            {model_id: GroundTruthScore}
        """
        scores = {}
        for model_id, response in responses.items():
            sample_with_model = {**sample, "model_id": model_id}
            scores[model_id] = await self.evaluate(
                response, ground_truth, sample_with_model
            )
        return scores
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute domain-specific metrics."""
        metrics = {}
        domain_metrics = self.metric_registry.DOMAIN_METRICS.get(self.domain, [])
        
        for metric_name in domain_metrics:
            try:
                # Build kwargs for specific metrics
                kwargs = {}
                if metric_name == "retrieval_accuracy":
                    kwargs["answer_variants"] = sample.get("answer_variants", [])
                elif metric_name == "syntax_valid":
                    kwargs["language"] = sample.get("language", "python")
                elif metric_name == "bleu":
                    refs = sample.get("reference_captions") or sample.get("reference_summaries")
                    if refs:
                        kwargs["references"] = refs
                
                value = compute_metric(
                    metric_name, response, ground_truth, **kwargs
                )
                metrics[metric_name] = value
            except Exception as e:
                metrics[metric_name] = 0.0
                metrics[f"{metric_name}_error"] = str(e)
        
        return metrics
    
    async def _semantic_evaluate(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> float:
        """Use LLM to evaluate semantic equivalence."""
        if not self.llm_caller:
            return 0.5
        
        prompt = f"""You are evaluating whether a response is semantically equivalent to the ground truth.

Ground Truth: {ground_truth}

Response: {response}

Question: Is the response semantically equivalent to the ground truth? Consider:
1. Does it contain the same key information?
2. Is the meaning preserved even if wording differs?
3. Are there any factual contradictions?

Rate the semantic equivalence from 0.0 to 1.0:
- 1.0: Perfectly equivalent
- 0.8: Mostly equivalent, minor differences
- 0.5: Partially equivalent
- 0.2: Mostly different
- 0.0: Completely different or contradictory

Respond with only a number between 0.0 and 1.0."""
        
        try:
            result = await self.llm_caller.acompletion(
                model=self.config.get("semantic_model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            return float(result.choices[0].message.content.strip())
        except Exception:
            return 0.5  # Default to uncertain
    
    def _compute_overall(
        self,
        metrics: Dict[str, float],
        mode: EvaluationMode
    ) -> float:
        """Compute overall score for ranking."""
        primary = self.metric_registry.PRIMARY_METRIC.get(self.domain)
        
        if primary and primary in metrics:
            value = metrics[primary]
            # Invert if lower is better
            if not self.metric_registry.HIGHER_IS_BETTER.get(primary, True):
                value = 1.0 - min(1.0, value)  # Cap at 1.0 for WER > 100%
            return value
        
        # Fallback: average of normalized metrics
        values = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not name.endswith("_error"):
                if not self.metric_registry.HIGHER_IS_BETTER.get(name, True):
                    value = 1.0 - min(1.0, value)
                values.append(value)
        
        return sum(values) / len(values) if values else 0.0
    
    def _get_default_mode(self) -> EvaluationMode:
        """Get default evaluation mode for domain."""
        domain_config = self.config.get("domains", {}).get(self.domain, {})
        mode_str = domain_config.get("evaluation_mode", "metrics_only")
        
        try:
            return EvaluationMode(mode_str)
        except ValueError:
            return EvaluationMode.METRICS_ONLY
    
    def get_ranking_from_scores(
        self,
        scores: Dict[str, GroundTruthScore]
    ) -> Dict[str, int]:
        """
        Convert scores to rankings.
        
        Args:
            scores: {model_id: GroundTruthScore}
        
        Returns:
            {model_id: rank} where 1 is best
        """
        # Sort by overall score (descending)
        sorted_models = sorted(
            scores.keys(),
            key=lambda m: scores[m].overall,
            reverse=True
        )
        
        return {model_id: rank + 1 for rank, model_id in enumerate(sorted_models)}


class ASRGroundTruthJudge(GroundTruthJudge):
    """Specialized judge for ASR evaluation."""
    
    def __init__(self, **kwargs):
        super().__init__(domain="asr", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute ASR-specific metrics."""
        transcript = str(ground_truth)
        hypothesis = str(response)
        
        return {
            "wer": word_error_rate(hypothesis, transcript),
            "cer": character_error_rate(hypothesis, transcript),
            "accuracy": 1.0 - min(1.0, word_error_rate(hypothesis, transcript)),
        }


class NeedleGroundTruthJudge(GroundTruthJudge):
    """Specialized judge for Needle in Haystack evaluation."""
    
    def __init__(self, **kwargs):
        super().__init__(domain="needle_in_haystack", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute needle retrieval metrics."""
        expected = str(ground_truth)
        variants = sample.get("answer_variants", [])
        
        return compute_needle_metrics(response, expected, variants)


class CodeGroundTruthJudge(GroundTruthJudge):
    """
    Specialized judge for code completion evaluation.
    
    Uses sandboxed Docker-based execution for running test cases securely.
    See langscope.ground_truth.judges.code_judge.CodeCompletionJudge for
    the full implementation with advanced features.
    """
    
    def __init__(
        self,
        execution_timeout: int = 10,
        allow_execution: bool = True,
        use_docker: bool = True,
        **kwargs
    ):
        super().__init__(domain="code_completion", **kwargs)
        self.execution_timeout = execution_timeout
        self.allow_execution = allow_execution
        self.use_docker = use_docker
        self._sandbox = None
    
    @property
    def sandbox(self):
        """Lazy-load sandbox to avoid import issues."""
        if self._sandbox is None:
            from langscope.ground_truth.judges.code_judge import CodeExecutionSandbox
            self._sandbox = CodeExecutionSandbox(
                timeout=self.execution_timeout,
                use_docker=self.use_docker
            )
        return self._sandbox
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute code-specific metrics."""
        from langscope.ground_truth.metrics import validate_syntax, bleu_score
        
        language = sample.get("language", "python")
        test_cases = sample.get("test_cases", [])
        
        # Get expected code
        if isinstance(ground_truth, dict):
            expected = ground_truth.get("expected_code", str(ground_truth))
        else:
            expected = str(ground_truth)
        
        metrics = {
            "syntax_valid": validate_syntax(response, language),
            "exact_match": exact_match(response, expected),
            "bleu": bleu_score(response, [expected]),
        }
        
        # Run test cases if allowed and syntax is valid
        if self.allow_execution and metrics["syntax_valid"] == 1.0 and test_cases:
            passed = self._run_test_cases(response, test_cases, language)
            metrics["tests_pass"] = passed
        else:
            # Estimate based on other metrics
            metrics["tests_pass"] = (
                metrics["exact_match"] + metrics["syntax_valid"] + metrics["bleu"]
            ) / 3
        
        return metrics
    
    def _run_test_cases(
        self,
        code: str,
        test_cases: List[Dict],
        language: str
    ) -> float:
        """
        Run test cases against generated code using sandboxed execution.
        
        Uses Docker-based sandboxing for security with:
        - Memory limits (128MB default)
        - CPU limits (0.5 cores default)
        - Network isolation
        - Execution timeouts (10s default)
        
        Args:
            code: Generated code
            test_cases: List of {input, expected} or {assert: ...} dicts
            language: Programming language
        
        Returns:
            Percentage of tests passing (0.0-1.0)
        """
        if not self.allow_execution:
            return 0.5  # Unknown
        
        if language.lower() not in ("python", "py", "python3"):
            return 0.5  # Only Python supported currently
        
        if not test_cases:
            return 0.0
        
        passed = 0
        total = len(test_cases)
        
        for test in test_cases:
            try:
                if self._run_single_test(code, test):
                    passed += 1
            except Exception:
                continue
        
        return passed / total if total > 0 else 0.0
    
    def _run_single_test(self, code: str, test: Dict) -> bool:
        """Run a single test case in sandbox."""
        # Build test script based on test format
        if "assert" in test:
            # Assert-style test
            test_script = f"""
{code}

try:
    assert {test["assert"]}
    print("PASS")
except AssertionError:
    print("FAIL")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            expected = "PASS"
            stdin = ""
        elif test.get("stdin"):
            # Stdin-style test
            test_script = code
            stdin = test.get("input", "")
            expected = test.get("expected", "")
        else:
            # Function call style
            test_input = test.get("input", test.get("function_call", ""))
            expected = str(test.get("expected", ""))
            test_script = f"""
{code}

try:
    result = {test_input}
    print(result)
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            stdin = ""
        
        # Execute in sandbox
        result = self.sandbox.run_python(test_script, stdin)
        
        if result["timed_out"] or not result["success"]:
            return False
        
        return result["output"].strip() == expected.strip()

