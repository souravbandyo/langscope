"""
Long Document QA Ground Truth Judge.

Evaluates answers to questions about long documents using:
- Exact match
- Token F1
- Evidence recall
- Semantic matching (hybrid mode)
"""

from typing import Dict, List, Optional, Any
import logging

from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore, EvaluationMode
from langscope.ground_truth.metrics import (
    exact_match,
    contains,
    f1_score,
    rouge_l,
)

logger = logging.getLogger(__name__)


class LongDocQAJudge(GroundTruthJudge):
    """
    Specialized judge for Long Document QA evaluation.
    
    Evaluates whether the model correctly answered questions
    about long documents, including evidence retrieval.
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="long_document_qa", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """
        Compute long document QA metrics.
        
        Args:
            response: Model's answer
            ground_truth: Expected answer(s)
            sample: Sample metadata including evidence spans
        
        Returns:
            Dict of metric values
        """
        metrics = {}
        
        # Handle multiple acceptable answers
        if isinstance(ground_truth, list):
            # Take best score across all acceptable answers
            best_exact = 0.0
            best_f1 = 0.0
            for gt in ground_truth:
                best_exact = max(best_exact, exact_match(response, str(gt)))
                best_f1 = max(best_f1, f1_score(response, str(gt)))
            metrics["exact_match"] = best_exact
            metrics["f1"] = best_f1
            metrics["contains"] = max(contains(response, str(gt)) for gt in ground_truth)
        else:
            metrics["exact_match"] = exact_match(response, str(ground_truth))
            metrics["f1"] = f1_score(response, str(ground_truth))
            metrics["contains"] = contains(response, str(ground_truth))
        
        # Evidence recall - check if response mentions evidence spans
        evidence_spans = sample.get("evidence_spans", [])
        if evidence_spans:
            metrics["evidence_recall"] = self._compute_evidence_recall(
                response, evidence_spans
            )
        else:
            metrics["evidence_recall"] = 1.0  # No evidence to check
        
        # Accuracy is 1.0 if either exact match or F1 > threshold
        metrics["accuracy"] = 1.0 if (
            metrics["exact_match"] == 1.0 or 
            metrics["f1"] >= 0.7 or
            metrics["contains"] == 1.0
        ) else 0.0
        
        return metrics
    
    def _compute_evidence_recall(
        self,
        response: str,
        evidence_spans: List[str]
    ) -> float:
        """
        Compute evidence recall.
        
        Checks how many evidence spans are referenced in the response.
        
        Args:
            response: Model's response
            evidence_spans: List of evidence text spans
        
        Returns:
            Proportion of evidence spans mentioned
        """
        if not evidence_spans:
            return 1.0
        
        response_lower = response.lower()
        mentioned = 0
        
        for span in evidence_spans:
            # Check for substantial overlap
            span_words = set(span.lower().split())
            response_words = set(response_lower.split())
            
            overlap = len(span_words & response_words) / len(span_words) if span_words else 0
            
            if overlap >= 0.5:  # At least 50% of evidence words present
                mentioned += 1
        
        return mentioned / len(evidence_spans)
    
    async def evaluate_with_context_analysis(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict,
        model_id: str = ""
    ) -> GroundTruthScore:
        """
        Evaluate with additional context length analysis.
        
        Args:
            response: Model's answer
            ground_truth: Expected answer
            sample: Sample metadata
            model_id: Model identifier
        
        Returns:
            GroundTruthScore with context-aware metrics
        """
        metrics = self._compute_metrics(response, ground_truth, sample)
        
        # Add context-aware metrics
        context_length = sample.get("context_length", 0)
        question_type = sample.get("question_type", "factual")
        
        metrics["context_length"] = float(context_length)
        metrics["question_type"] = question_type
        
        # Compute overall
        overall = self._compute_overall(metrics, self._get_default_mode())
        
        return GroundTruthScore(
            model_id=model_id,
            sample_id=sample.get("sample_id", ""),
            metrics=metrics,
            overall=overall,
            evaluation_mode=self._get_default_mode().value,
        )


class MultiDocReasoningJudge(GroundTruthJudge):
    """
    Specialized judge for Multi-Document Reasoning evaluation.
    
    Evaluates reasoning across multiple documents:
    - Answer accuracy
    - Evidence from each document
    - Reasoning chain correctness
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="multi_document_reasoning", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute multi-document reasoning metrics."""
        metrics = {}
        
        # Basic answer metrics
        expected_answer = ground_truth if isinstance(ground_truth, str) else ground_truth.get("answer", "")
        metrics["exact_match"] = exact_match(response, expected_answer)
        metrics["f1"] = f1_score(response, expected_answer)
        metrics["answer_accuracy"] = 1.0 if metrics["f1"] >= 0.6 else 0.0
        
        # Document-level evidence accuracy
        document_evidence = sample.get("document_evidence", {})
        if document_evidence:
            metrics["evidence_accuracy"] = self._compute_doc_evidence_accuracy(
                response, document_evidence
            )
        else:
            metrics["evidence_accuracy"] = 1.0
        
        # Reasoning accuracy (via semantic match if available)
        reasoning_steps = sample.get("reasoning_steps", [])
        if reasoning_steps:
            metrics["reasoning_accuracy"] = self._compute_reasoning_accuracy(
                response, reasoning_steps
            )
        else:
            metrics["reasoning_accuracy"] = metrics["answer_accuracy"]
        
        return metrics
    
    def _compute_doc_evidence_accuracy(
        self,
        response: str,
        document_evidence: Dict[str, List[str]]
    ) -> float:
        """Check evidence from each document."""
        if not document_evidence:
            return 1.0
        
        docs_with_evidence = 0
        response_lower = response.lower()
        
        for doc_id, evidence_spans in document_evidence.items():
            for span in evidence_spans:
                if span.lower() in response_lower:
                    docs_with_evidence += 1
                    break
        
        return docs_with_evidence / len(document_evidence)
    
    def _compute_reasoning_accuracy(
        self,
        response: str,
        reasoning_steps: List[str]
    ) -> float:
        """Check if key reasoning steps are present."""
        if not reasoning_steps:
            return 1.0
        
        steps_present = 0
        response_lower = response.lower()
        
        for step in reasoning_steps:
            # Check for semantic presence of step
            step_words = set(step.lower().split())
            response_words = set(response_lower.split())
            
            overlap = len(step_words & response_words) / len(step_words) if step_words else 0
            if overlap >= 0.4:
                steps_present += 1
        
        return steps_present / len(reasoning_steps)


class LongSummarizationJudge(GroundTruthJudge):
    """
    Specialized judge for Long Document Summarization.
    
    Evaluates summaries using:
    - ROUGE scores
    - Factual consistency
    - Key point coverage
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="long_summarization", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute summarization metrics."""
        from langscope.ground_truth.metrics import bleu_score
        
        metrics = {}
        
        # Handle reference summaries
        if isinstance(ground_truth, list):
            references = ground_truth
        else:
            references = [str(ground_truth)]
        
        # ROUGE-L (vs first reference)
        metrics["rouge_l"] = rouge_l(response, references[0])
        
        # ROUGE-1 and ROUGE-2 (simplified - just check n-gram overlap)
        metrics["rouge_1"] = self._compute_rouge_n(response, references[0], 1)
        metrics["rouge_2"] = self._compute_rouge_n(response, references[0], 2)
        
        # Key points coverage
        key_points = sample.get("key_points", [])
        if key_points:
            metrics["coverage"] = self._compute_key_point_coverage(response, key_points)
        else:
            metrics["coverage"] = 0.5  # Unknown
        
        # Factual consistency placeholder (would need LLM evaluation)
        metrics["factual_consistency"] = 0.5
        
        return metrics
    
    def _compute_rouge_n(
        self,
        hypothesis: str,
        reference: str,
        n: int
    ) -> float:
        """Simplified ROUGE-N computation."""
        def get_ngrams(text: str, n: int) -> List:
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        hyp_ngrams = get_ngrams(hypothesis, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not ref_ngrams:
            return 1.0 if not hyp_ngrams else 0.0
        
        # Count overlapping n-grams
        hyp_set = set(hyp_ngrams)
        ref_set = set(ref_ngrams)
        
        overlap = len(hyp_set & ref_set)
        precision = overlap / len(hyp_set) if hyp_set else 0
        recall = overlap / len(ref_set) if ref_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_key_point_coverage(
        self,
        summary: str,
        key_points: List[str]
    ) -> float:
        """Check what fraction of key points are covered."""
        if not key_points:
            return 1.0
        
        covered = 0
        summary_lower = summary.lower()
        
        for point in key_points:
            point_words = set(point.lower().split())
            summary_words = set(summary_lower.split())
            
            overlap = len(point_words & summary_words) / len(point_words) if point_words else 0
            if overlap >= 0.5:
                covered += 1
        
        return covered / len(key_points)


