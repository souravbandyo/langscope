"""
Needle in Haystack Ground Truth Judge.

Evaluates retrieval accuracy at various context lengths and needle positions.
"""

from typing import Dict, List, Optional, Any
from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore, EvaluationMode
from langscope.ground_truth.metrics import compute_needle_metrics


class NeedleInHaystackJudge(GroundTruthJudge):
    """
    Specialized judge for Needle in Haystack evaluation.
    
    Tests information retrieval accuracy at various:
    - Context lengths (4K, 8K, 16K, 32K, 64K, 128K)
    - Needle positions (0.0, 0.25, 0.5, 0.75, 1.0)
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="needle_in_haystack", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute needle retrieval metrics."""
        # Get expected answer
        if isinstance(ground_truth, dict):
            expected = ground_truth.get("expected_answer", ground_truth.get("needle", ""))
            variants = ground_truth.get("answer_variants", [])
        else:
            expected = str(ground_truth)
            variants = sample.get("answer_variants", [])
        
        metrics = compute_needle_metrics(response, expected, variants)
        
        # Add context-aware metrics
        context_length = sample.get("context_length", 0)
        needle_position = sample.get("needle_position", 0.5)
        
        # Store metadata for analytics
        metrics["context_length"] = float(context_length)
        metrics["needle_position"] = needle_position
        
        return metrics
    
    async def compute_accuracy_heatmap(
        self,
        model_id: str,
        match_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy heatmap for a model.
        
        Args:
            model_id: Model identifier
            match_results: List of match results for this model
        
        Returns:
            Heatmap: {context_length: {needle_position: accuracy}}
        """
        heatmap: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, Dict[str, int]] = {}
        
        for result in match_results:
            scores = result.get("scores", {}).get(model_id, {})
            metadata = result.get("sample_metadata", {})
            
            ctx_len = str(metadata.get("context_length", 0))
            needle_pos = str(metadata.get("needle_position", 0.5))
            
            accuracy = scores.get("metrics", {}).get("retrieval_accuracy", 0.0)
            
            if ctx_len not in heatmap:
                heatmap[ctx_len] = {}
                counts[ctx_len] = {}
            
            if needle_pos not in heatmap[ctx_len]:
                heatmap[ctx_len][needle_pos] = 0.0
                counts[ctx_len][needle_pos] = 0
            
            heatmap[ctx_len][needle_pos] += accuracy
            counts[ctx_len][needle_pos] += 1
        
        # Compute averages
        for ctx_len in heatmap:
            for needle_pos in heatmap[ctx_len]:
                count = counts[ctx_len][needle_pos]
                if count > 0:
                    heatmap[ctx_len][needle_pos] /= count
        
        return heatmap
    
    def get_failure_patterns(
        self,
        match_results: List[Dict],
        model_id: str
    ) -> Dict[str, Any]:
        """
        Analyze failure patterns for a model.
        
        Args:
            match_results: List of match results
            model_id: Model identifier
        
        Returns:
            Analysis of where the model fails
        """
        failures_by_length: Dict[int, int] = {}
        failures_by_position: Dict[float, int] = {}
        total_by_length: Dict[int, int] = {}
        total_by_position: Dict[float, int] = {}
        
        for result in match_results:
            scores = result.get("scores", {}).get(model_id, {})
            metadata = result.get("sample_metadata", {})
            
            ctx_len = metadata.get("context_length", 0)
            needle_pos = metadata.get("needle_position", 0.5)
            accuracy = scores.get("metrics", {}).get("retrieval_accuracy", 0.0)
            
            # Track totals
            total_by_length[ctx_len] = total_by_length.get(ctx_len, 0) + 1
            total_by_position[needle_pos] = total_by_position.get(needle_pos, 0) + 1
            
            # Track failures
            if accuracy < 0.5:
                failures_by_length[ctx_len] = failures_by_length.get(ctx_len, 0) + 1
                failures_by_position[needle_pos] = failures_by_position.get(needle_pos, 0) + 1
        
        # Compute failure rates
        failure_rate_by_length = {
            k: failures_by_length.get(k, 0) / v if v > 0 else 0.0
            for k, v in total_by_length.items()
        }
        
        failure_rate_by_position = {
            k: failures_by_position.get(k, 0) / v if v > 0 else 0.0
            for k, v in total_by_position.items()
        }
        
        # Find worst cases
        worst_length = max(failure_rate_by_length.keys(), 
                          key=lambda k: failure_rate_by_length[k],
                          default=0)
        worst_position = max(failure_rate_by_position.keys(),
                            key=lambda k: failure_rate_by_position[k],
                            default=0.5)
        
        return {
            "failure_rate_by_length": failure_rate_by_length,
            "failure_rate_by_position": failure_rate_by_position,
            "worst_context_length": worst_length,
            "worst_needle_position": worst_position,
            "total_samples": sum(total_by_length.values()),
            "total_failures": sum(failures_by_length.values()),
        }


