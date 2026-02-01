"""
Visual QA and Document Extraction Ground Truth Judges.

Evaluates:
- Visual Question Answering (VQA)
- Document Field Extraction
- Image Captioning
- OCR
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging

from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore, EvaluationMode
from langscope.ground_truth.metrics import (
    exact_match,
    contains,
    f1_score,
    bleu_score,
    rouge_l,
    compute_field_accuracy,
    word_error_rate,
    character_error_rate,
)

logger = logging.getLogger(__name__)


class VisualQAJudge(GroundTruthJudge):
    """
    Specialized judge for Visual Question Answering.
    
    Evaluates answers to questions about images:
    - Exact match for factual questions
    - Semantic matching for descriptive questions
    - Multiple choice accuracy
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="visual_qa", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute VQA metrics."""
        metrics = {}
        
        # Handle multiple acceptable answers
        if isinstance(ground_truth, list):
            answers = ground_truth
        else:
            answers = [str(ground_truth)]
        
        # Exact match (any acceptable answer)
        metrics["exact_match"] = max(
            exact_match(response, ans) for ans in answers
        )
        
        # Contains (any acceptable answer)
        metrics["contains"] = max(
            contains(response, ans) for ans in answers
        )
        
        # Question-type specific evaluation
        question_type = sample.get("question_type", "open")
        
        if question_type == "yes_no":
            metrics["accuracy"] = self._evaluate_yes_no(response, answers)
        elif question_type == "counting":
            metrics["accuracy"] = self._evaluate_counting(response, answers)
        elif question_type == "multiple_choice":
            metrics["accuracy"] = self._evaluate_multiple_choice(
                response, answers, sample.get("choices", [])
            )
        else:
            # Open-ended: use contains or exact match
            metrics["accuracy"] = max(
                metrics["exact_match"],
                metrics["contains"]
            )
        
        # Semantic match placeholder (for hybrid mode)
        metrics["semantic_match"] = metrics["accuracy"]
        
        return metrics
    
    def _evaluate_yes_no(
        self,
        response: str,
        expected: List[str]
    ) -> float:
        """Evaluate yes/no question."""
        response_lower = response.lower().strip()
        
        # Normalize response
        is_yes = any(w in response_lower for w in ["yes", "true", "correct", "affirmative"])
        is_no = any(w in response_lower for w in ["no", "false", "incorrect", "negative"])
        
        for exp in expected:
            exp_lower = exp.lower().strip()
            exp_is_yes = any(w in exp_lower for w in ["yes", "true"])
            exp_is_no = any(w in exp_lower for w in ["no", "false"])
            
            if (is_yes and exp_is_yes) or (is_no and exp_is_no):
                return 1.0
        
        return 0.0
    
    def _evaluate_counting(
        self,
        response: str,
        expected: List[str]
    ) -> float:
        """Evaluate counting question."""
        import re
        
        # Extract numbers from response
        response_numbers = re.findall(r'\d+', response)
        
        for exp in expected:
            exp_numbers = re.findall(r'\d+', str(exp))
            if exp_numbers and response_numbers:
                if exp_numbers[0] == response_numbers[0]:
                    return 1.0
        
        return 0.0
    
    def _evaluate_multiple_choice(
        self,
        response: str,
        expected: List[str],
        choices: List[str]
    ) -> float:
        """Evaluate multiple choice question."""
        response_lower = response.lower().strip()
        
        # Check for choice letter (A, B, C, D)
        import re
        choice_match = re.match(r'^[a-d][\).\s]', response_lower)
        if choice_match:
            chosen_idx = ord(choice_match.group()[0]) - ord('a')
            if chosen_idx < len(choices):
                chosen = choices[chosen_idx]
                if any(exp.lower() in chosen.lower() or chosen.lower() in exp.lower() 
                       for exp in expected):
                    return 1.0
        
        # Check if response contains expected answer
        for exp in expected:
            if exp.lower() in response_lower:
                return 1.0
        
        return 0.0


class DocumentExtractionJudge(GroundTruthJudge):
    """
    Specialized judge for Document Field Extraction.
    
    Evaluates extraction of structured fields from documents:
    - Field-level accuracy
    - Critical field accuracy
    - Schema validation
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="document_extraction", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute document extraction metrics."""
        # Parse response as JSON
        try:
            if isinstance(response, str):
                extracted = json.loads(response)
            elif isinstance(response, dict):
                extracted = response
            else:
                extracted = {}
        except json.JSONDecodeError:
            # Try to extract key-value pairs from text
            extracted = self._extract_fields_from_text(response)
        
        # Get expected fields
        if isinstance(ground_truth, dict):
            expected = ground_truth
        else:
            expected = sample.get("expected_fields", {})
        
        # Get critical fields
        critical_fields = sample.get("critical_fields", [])
        
        # Compute field accuracy metrics
        metrics = compute_field_accuracy(extracted, expected, critical_fields)
        
        # Schema validation
        expected_schema = sample.get("schema", {})
        if expected_schema:
            metrics["schema_valid"] = self._validate_schema(extracted, expected_schema)
        else:
            metrics["schema_valid"] = 1.0 if extracted else 0.0
        
        return metrics
    
    def _extract_fields_from_text(
        self,
        text: str
    ) -> Dict[str, str]:
        """Attempt to extract key-value pairs from text."""
        import re
        
        fields = {}
        
        # Match patterns like "Field: Value" or "Field = Value"
        patterns = [
            r'([A-Za-z_]+)\s*[:=]\s*(.+?)(?:\n|$)',
            r'"([^"]+)"\s*:\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                fields[key.strip()] = value.strip()
        
        return fields
    
    def _validate_schema(
        self,
        extracted: Dict,
        schema: Dict
    ) -> float:
        """Validate extracted data against schema."""
        if not schema:
            return 1.0
        
        required_fields = schema.get("required", [])
        if not required_fields:
            return 1.0
        
        present = sum(1 for f in required_fields if f in extracted)
        return present / len(required_fields)


class ImageCaptioningJudge(GroundTruthJudge):
    """
    Specialized judge for Image Captioning.
    
    Evaluates generated captions using:
    - BLEU score
    - ROUGE-L
    - CIDEr (simplified)
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="image_captioning", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute image captioning metrics."""
        # Get reference captions
        if isinstance(ground_truth, list):
            references = ground_truth
        else:
            references = [str(ground_truth)]
        
        metrics = {}
        
        # BLEU
        metrics["bleu"] = bleu_score(response, references)
        
        # ROUGE-L (against first reference)
        metrics["rouge_l"] = rouge_l(response, references[0])
        
        # CIDEr simplified (weighted n-gram overlap with TF-IDF)
        metrics["cider"] = self._compute_cider_simplified(response, references)
        
        # METEOR simplified (unigram overlap with stemming approximation)
        metrics["meteor"] = self._compute_meteor_simplified(response, references)
        
        return metrics
    
    def _compute_cider_simplified(
        self,
        hypothesis: str,
        references: List[str]
    ) -> float:
        """Simplified CIDEr computation."""
        # This is a very simplified version
        # Real CIDEr uses TF-IDF weighting and n-gram matching
        
        hyp_words = set(hypothesis.lower().split())
        
        total_overlap = 0
        for ref in references:
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(hyp_words & ref_words) / len(ref_words)
                total_overlap += overlap
        
        return total_overlap / len(references) if references else 0.0
    
    def _compute_meteor_simplified(
        self,
        hypothesis: str,
        references: List[str]
    ) -> float:
        """Simplified METEOR computation."""
        hyp_words = set(hypothesis.lower().split())
        
        best_score = 0.0
        for ref in references:
            ref_words = set(ref.lower().split())
            
            if not hyp_words or not ref_words:
                continue
            
            overlap = len(hyp_words & ref_words)
            precision = overlap / len(hyp_words)
            recall = overlap / len(ref_words)
            
            if precision + recall > 0:
                # METEOR uses weighted F-score
                f_score = (10 * precision * recall) / (recall + 9 * precision)
                best_score = max(best_score, f_score)
        
        return best_score


class OCRJudge(GroundTruthJudge):
    """
    Specialized judge for OCR evaluation.
    
    Evaluates text extraction accuracy using:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Line accuracy
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="ocr", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute OCR metrics."""
        expected = str(ground_truth)
        
        metrics = {
            "cer": character_error_rate(response, expected),
            "wer": word_error_rate(response, expected),
        }
        
        # Line accuracy
        if "\n" in expected:
            metrics["line_accuracy"] = self._compute_line_accuracy(response, expected)
        else:
            metrics["line_accuracy"] = 1.0 if metrics["cer"] < 0.1 else 0.0
        
        return metrics
    
    def _compute_line_accuracy(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """Compute line-by-line accuracy."""
        hyp_lines = hypothesis.strip().split("\n")
        ref_lines = reference.strip().split("\n")
        
        if not ref_lines:
            return 1.0 if not hyp_lines else 0.0
        
        correct = 0
        for i, ref_line in enumerate(ref_lines):
            if i < len(hyp_lines):
                # Consider correct if CER < 10%
                line_cer = character_error_rate(hyp_lines[i], ref_line)
                if line_cer < 0.1:
                    correct += 1
        
        return correct / len(ref_lines)


