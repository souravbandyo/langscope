"""
Ground Truth Evaluation Metrics.

Provides metric implementations and registry for ground truth evaluation:
- ASR metrics: WER, CER, MER
- Text metrics: BLEU, ROUGE, exact match, contains
- Code metrics: syntax_valid, tests_pass
- Needle metrics: retrieval_accuracy

Each metric returns a float value in the range specified by the metric definition.

Supports multiple metric library backends with fallbacks:
- jiwer for WER/CER (with native fallback)
- rouge-score for ROUGE (with native fallback)
- sacrebleu for BLEU (with native fallback)
"""

import re
import math
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check for optional libraries
_HAS_JIWER = False
_HAS_ROUGE = False
_HAS_SACREBLEU = False

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    logger.debug("jiwer not installed, using native WER implementation")

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    logger.debug("rouge-score not installed, using native ROUGE implementation")

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    logger.debug("sacrebleu not installed, using native BLEU implementation")


# =============================================================================
# Metric Registry
# =============================================================================

class MetricRegistry:
    """
    Registry of evaluation metrics per domain.
    
    Maps domains to their applicable metrics and provides
    configuration for ranking direction and primary metrics.
    """
    
    # Domain to metrics mapping
    DOMAIN_METRICS: Dict[str, List[str]] = {
        # Multimodal domains
        "asr": ["wer", "cer", "mer", "wil"],
        "tts": ["round_trip_wer", "utmos", "dnsmos", "snr", "speaker_similarity", "composite_tts"],
        "visual_qa": ["exact_match", "contains", "semantic_match", "accuracy"],
        "document_extraction": ["field_accuracy", "field_precision", "field_recall", "critical_accuracy", "schema_valid"],
        "image_captioning": ["bleu", "rouge_l", "cider", "meteor", "spice"],
        "ocr": ["cer", "wer", "line_accuracy"],
        
        # Long context domains
        "needle_in_haystack": ["retrieval_accuracy", "contains", "exact_match", "semantic_match"],
        "long_document_qa": ["exact_match", "f1", "accuracy", "evidence_recall"],
        "multi_document_reasoning": ["answer_accuracy", "evidence_accuracy", "reasoning_accuracy"],
        "code_completion": ["syntax_valid", "tests_pass", "exact_match", "bleu", "functional_equivalence"],
        "long_summarization": ["rouge_1", "rouge_2", "rouge_l", "factual_consistency", "coverage"],
    }
    
    # Primary metric per domain (for ranking)
    PRIMARY_METRIC: Dict[str, str] = {
        "asr": "wer",
        "tts": "composite_tts",
        "visual_qa": "accuracy",
        "document_extraction": "field_accuracy",
        "image_captioning": "cider",
        "ocr": "cer",
        "needle_in_haystack": "retrieval_accuracy",
        "long_document_qa": "accuracy",
        "multi_document_reasoning": "answer_accuracy",
        "code_completion": "tests_pass",
        "long_summarization": "rouge_l",
    }
    
    # Higher is better? (for ranking direction)
    HIGHER_IS_BETTER: Dict[str, bool] = {
        # ASR/TTS metrics - lower error is better
        "wer": False,
        "cer": False,
        "mer": False,
        "wil": False,
        "round_trip_wer": False,
        
        # Quality scores - higher is better
        "utmos": True,
        "dnsmos": True,
        "snr": True,
        "speaker_similarity": True,
        "composite_tts": True,
        
        # Accuracy metrics - higher is better
        "accuracy": True,
        "exact_match": True,
        "contains": True,
        "semantic_match": True,
        "retrieval_accuracy": True,
        
        # Field metrics
        "field_accuracy": True,
        "field_precision": True,
        "field_recall": True,
        "critical_accuracy": True,
        "schema_valid": True,
        
        # Text generation metrics
        "bleu": True,
        "rouge_l": True,
        "rouge_1": True,
        "rouge_2": True,
        "cider": True,
        "meteor": True,
        "spice": True,
        "factual_consistency": True,
        "coverage": True,
        
        # QA metrics
        "f1": True,
        "evidence_recall": True,
        "answer_accuracy": True,
        "evidence_accuracy": True,
        "reasoning_accuracy": True,
        
        # Code metrics
        "syntax_valid": True,
        "tests_pass": True,
        "functional_equivalence": True,
        
        # OCR
        "line_accuracy": True,
    }
    
    # Metric descriptions
    METRIC_DESCRIPTIONS: Dict[str, str] = {
        "wer": "Word Error Rate - (S + D + I) / N",
        "cer": "Character Error Rate - (S + D + I) / C",
        "mer": "Match Error Rate - (S + D + I) / (S + D + I + H)",
        "wil": "Word Information Lost - 1 - (H² / (N × P))",
        "round_trip_wer": "WER from TTS→ASR round trip",
        "utmos": "Neural MOS prediction (1-5)",
        "composite_tts": "Weighted TTS quality score",
        "exact_match": "Response exactly matches ground truth",
        "contains": "Ground truth appears in response",
        "retrieval_accuracy": "Needle successfully retrieved (binary)",
        "bleu": "N-gram precision with brevity penalty",
        "rouge_l": "Longest common subsequence F1",
        "tests_pass": "Percentage of test cases passing",
        "accuracy": "Correct answers / total",
    }
    
    @classmethod
    def get_metrics_for_domain(cls, domain: str) -> List[str]:
        """Get all metrics for a domain."""
        return cls.DOMAIN_METRICS.get(domain, [])
    
    @classmethod
    def get_primary_metric(cls, domain: str) -> Optional[str]:
        """Get primary metric for a domain."""
        return cls.PRIMARY_METRIC.get(domain)
    
    @classmethod
    def is_higher_better(cls, metric: str) -> bool:
        """Check if higher values are better for a metric."""
        return cls.HIGHER_IS_BETTER.get(metric, True)
    
    @classmethod
    def get_metric_description(cls, metric: str) -> str:
        """Get description for a metric."""
        return cls.METRIC_DESCRIPTIONS.get(metric, "")


# =============================================================================
# ASR Metrics
# =============================================================================

def _levenshtein_distance(s1: List[str], s2: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance with operations count.
    
    Returns:
        Tuple of (substitutions, deletions, insertions, hits)
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return (0, len(s1), 0, 0)
    
    # Create distance matrix
    distances = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    operations = [[None] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    
    for i in range(len(s1) + 1):
        distances[i][0] = i
        if i > 0:
            operations[i][0] = 'D'  # Deletion
    
    for j in range(len(s2) + 1):
        distances[0][j] = j
        if j > 0:
            operations[0][j] = 'I'  # Insertion
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                distances[i][j] = distances[i-1][j-1]
                operations[i][j] = 'H'  # Hit
            else:
                substitution = distances[i-1][j-1] + 1
                deletion = distances[i-1][j] + 1
                insertion = distances[i][j-1] + 1
                
                min_dist = min(substitution, deletion, insertion)
                distances[i][j] = min_dist
                
                if min_dist == substitution:
                    operations[i][j] = 'S'  # Substitution
                elif min_dist == deletion:
                    operations[i][j] = 'D'
                else:
                    operations[i][j] = 'I'
    
    # Count operations by backtracking
    subs, dels, ins, hits = 0, 0, 0, 0
    i, j = len(s1), len(s2)
    
    while i > 0 or j > 0:
        op = operations[i][j]
        if op == 'H':
            hits += 1
            i -= 1
            j -= 1
        elif op == 'S':
            subs += 1
            i -= 1
            j -= 1
        elif op == 'D':
            dels += 1
            i -= 1
        elif op == 'I':
            ins += 1
            j -= 1
        else:
            break
    
    return (subs, dels, ins, hits)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def word_error_rate(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = words in reference
    
    Uses jiwer library if available, otherwise uses native implementation.
    
    Args:
        hypothesis: Model's transcription
        reference: Ground truth transcription
        normalize: Whether to normalize text before comparison
    
    Returns:
        WER value (0 = perfect, higher = worse)
    """
    # Use jiwer if available (more accurate and feature-rich)
    if _HAS_JIWER:
        try:
            if normalize:
                transformation = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip(),
                ])
                return jiwer.wer(
                    reference, hypothesis,
                    truth_transform=transformation,
                    hypothesis_transform=transformation
                )
            else:
                return jiwer.wer(reference, hypothesis)
        except Exception as e:
            logger.debug(f"jiwer WER failed, using native: {e}")
    
    # Native implementation fallback
    if normalize:
        hypothesis = _normalize_text(hypothesis)
        reference = _normalize_text(reference)
    
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')
    
    subs, dels, ins, _ = _levenshtein_distance(ref_words, hyp_words)
    
    return (subs + dels + ins) / len(ref_words)


def character_error_rate(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """
    Compute Character Error Rate (CER).
    
    CER = (S + D + I) / C
    where C = characters in reference
    
    Uses jiwer library if available, otherwise uses native implementation.
    
    Args:
        hypothesis: Model's transcription
        reference: Ground truth transcription
        normalize: Whether to normalize text before comparison
    
    Returns:
        CER value (0 = perfect, higher = worse)
    """
    # Use jiwer if available
    if _HAS_JIWER:
        try:
            if normalize:
                transformation = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip(),
                ])
                return jiwer.cer(
                    reference, hypothesis,
                    truth_transform=transformation,
                    hypothesis_transform=transformation
                )
            else:
                return jiwer.cer(reference, hypothesis)
        except Exception as e:
            logger.debug(f"jiwer CER failed, using native: {e}")
    
    # Native implementation fallback
    if normalize:
        hypothesis = _normalize_text(hypothesis)
        reference = _normalize_text(reference)
    
    hyp_chars = list(hypothesis.replace(" ", ""))
    ref_chars = list(reference.replace(" ", ""))
    
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float('inf')
    
    subs, dels, ins, _ = _levenshtein_distance(ref_chars, hyp_chars)
    
    return (subs + dels + ins) / len(ref_chars)


def match_error_rate(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """
    Compute Match Error Rate (MER).
    
    MER = (S + D + I) / (S + D + I + H)
    
    Args:
        hypothesis: Model's transcription
        reference: Ground truth transcription
        normalize: Whether to normalize text before comparison
    
    Returns:
        MER value (0 = perfect, 1 = worst)
    """
    if normalize:
        hypothesis = _normalize_text(hypothesis)
        reference = _normalize_text(reference)
    
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    
    if len(ref_words) == 0 and len(hyp_words) == 0:
        return 0.0
    
    subs, dels, ins, hits = _levenshtein_distance(ref_words, hyp_words)
    
    total = subs + dels + ins + hits
    if total == 0:
        return 0.0
    
    return (subs + dels + ins) / total


# =============================================================================
# Text Generation Metrics
# =============================================================================

def _get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Get n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return []
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def bleu_score(
    hypothesis: str,
    references: List[str],
    max_n: int = 4,
    weights: List[float] = None
) -> float:
    """
    Compute BLEU score (Bilingual Evaluation Understudy).
    
    Uses sacrebleu library if available, otherwise uses native implementation.
    
    Args:
        hypothesis: Generated text
        references: List of reference texts
        max_n: Maximum n-gram order (default 4 for BLEU-4)
        weights: Weights for each n-gram order
    
    Returns:
        BLEU score (0-1)
    """
    if not hypothesis or not references:
        return 0.0
    
    # Use sacrebleu if available (more accurate)
    if _HAS_SACREBLEU and weights is None:
        try:
            # sacrebleu expects list of hypotheses and list of list of references
            result = sacrebleu.corpus_bleu(
                [hypothesis],
                [[ref] for ref in references]
            )
            return result.score / 100.0  # sacrebleu returns 0-100
        except Exception as e:
            logger.debug(f"sacrebleu failed, using native: {e}")
    
    # Native implementation fallback
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    hyp_words = hypothesis.lower().split()
    ref_lengths = [len(ref.lower().split()) for ref in references]
    
    # Brevity penalty
    closest_ref_len = min(ref_lengths, key=lambda x: (abs(x - len(hyp_words)), x))
    if len(hyp_words) <= closest_ref_len:
        bp = math.exp(1 - closest_ref_len / len(hyp_words)) if len(hyp_words) > 0 else 0
    else:
        bp = 1.0
    
    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hypothesis, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        
        # Count n-grams in hypothesis
        hyp_counts: Dict[Tuple, int] = {}
        for ng in hyp_ngrams:
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
        
        # Get max counts from references
        max_ref_counts: Dict[Tuple, int] = {}
        for ref in references:
            ref_ngrams = _get_ngrams(ref, n)
            ref_counts: Dict[Tuple, int] = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
            for ng, count in ref_counts.items():
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), count)
        
        # Clipped counts
        clipped = sum(min(hyp_counts.get(ng, 0), max_ref_counts.get(ng, 0)) for ng in hyp_counts)
        
        precisions.append(clipped / len(hyp_ngrams) if len(hyp_ngrams) > 0 else 0.0)
    
    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
    
    return bp * math.exp(log_precision)


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence).
    
    Uses rouge-score library if available, otherwise uses native implementation.
    
    Args:
        hypothesis: Generated text
        reference: Reference text
    
    Returns:
        ROUGE-L F1 score (0-1)
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Use rouge-score library if available
    if _HAS_ROUGE:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logger.debug(f"rouge-score failed, using native: {e}")
    
    # Native implementation fallback
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    
    # LCS length using dynamic programming
    m, n = len(hyp_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    # Precision and recall
    precision = lcs_length / m if m > 0 else 0.0
    recall = lcs_length / n if n > 0 else 0.0
    
    # F1
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """
    Compute ROUGE-N score (N-gram overlap).
    
    Uses rouge-score library if available, otherwise uses native implementation.
    
    Args:
        hypothesis: Generated text
        reference: Reference text
        n: N-gram size (1 for ROUGE-1, 2 for ROUGE-2)
    
    Returns:
        ROUGE-N F1 score (0-1)
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Use rouge-score library if available
    if _HAS_ROUGE:
        try:
            metric_name = f'rouge{n}'
            scorer = rouge_scorer.RougeScorer([metric_name], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores[metric_name].fmeasure
        except Exception as e:
            logger.debug(f"rouge-score failed, using native: {e}")
    
    # Native implementation fallback
    hyp_ngrams = _get_ngrams(hypothesis, n)
    ref_ngrams = _get_ngrams(reference, n)
    
    if not hyp_ngrams or not ref_ngrams:
        return 0.0
    
    # Count n-grams
    hyp_counts = {}
    for ng in hyp_ngrams:
        hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
    
    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1
    
    # Compute overlap
    overlap = sum(min(hyp_counts.get(ng, 0), count) for ng, count in ref_counts.items())
    
    precision = overlap / len(hyp_ngrams)
    recall = overlap / len(ref_ngrams)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Exact Match Metrics
# =============================================================================

def exact_match(response: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Check if response exactly matches ground truth.
    
    Args:
        response: Model's response
        ground_truth: Expected answer
        normalize: Whether to normalize text before comparison
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        response = _normalize_text(response)
        ground_truth = _normalize_text(ground_truth)
    
    return 1.0 if response.strip() == ground_truth.strip() else 0.0


def contains(response: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Check if ground truth appears in response.
    
    Args:
        response: Model's response
        ground_truth: Expected answer
        normalize: Whether to normalize text before comparison
    
    Returns:
        1.0 if contains, 0.0 otherwise
    """
    if normalize:
        response = _normalize_text(response)
        ground_truth = _normalize_text(ground_truth)
    
    return 1.0 if ground_truth.strip() in response else 0.0


def f1_score(hypothesis: str, reference: str) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        hypothesis: Model's response
        reference: Ground truth
    
    Returns:
        F1 score (0-1)
    """
    hyp_tokens = set(_normalize_text(hypothesis).split())
    ref_tokens = set(_normalize_text(reference).split())
    
    if len(hyp_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    common = hyp_tokens & ref_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Needle in Haystack Metrics
# =============================================================================

def compute_needle_metrics(
    response: str,
    expected_answer: str,
    answer_variants: List[str] = None
) -> Dict[str, float]:
    """
    Compute needle retrieval metrics.
    
    Args:
        response: Model's response
        expected_answer: The needle/expected answer
        answer_variants: List of acceptable answer variants
    
    Returns:
        Dict with retrieval_accuracy, contains, exact_match
    """
    # Normalize for comparison
    response_lower = response.lower().strip()
    expected_lower = expected_answer.lower().strip()
    
    # Exact match
    exact = 1.0 if expected_lower in response_lower else 0.0
    
    # Check variants
    contains_answer = exact
    if not contains_answer and answer_variants:
        for variant in answer_variants:
            if variant.lower().strip() in response_lower:
                contains_answer = 1.0
                break
    
    return {
        "exact_match": exact,
        "contains": contains_answer,
        "retrieval_accuracy": contains_answer  # Primary metric
    }


# =============================================================================
# Code Metrics
# =============================================================================

def validate_python_syntax(code: str) -> float:
    """
    Check if Python code has valid syntax.
    
    Args:
        code: Python code string
    
    Returns:
        1.0 if valid, 0.0 if invalid
    """
    try:
        compile(code, "<string>", "exec")
        return 1.0
    except SyntaxError:
        return 0.0


def validate_syntax(code: str, language: str) -> float:
    """
    Check if code has valid syntax for the given language.
    
    Args:
        code: Code string
        language: Programming language
    
    Returns:
        1.0 if valid, 0.0 if invalid
    """
    if language.lower() in ("python", "py"):
        return validate_python_syntax(code)
    
    # For other languages, we'd need external validators
    # For now, assume valid if non-empty
    return 1.0 if code.strip() else 0.0


# =============================================================================
# Document Extraction Metrics
# =============================================================================

def compute_field_accuracy(
    extracted: Dict[str, Any],
    expected: Dict[str, Any],
    critical_fields: List[str] = None
) -> Dict[str, float]:
    """
    Compute field-level accuracy metrics for document extraction.
    
    Args:
        extracted: Extracted field values
        expected: Expected field values
        critical_fields: List of critical fields (weighted higher)
    
    Returns:
        Dict with field_accuracy, field_precision, field_recall, critical_accuracy
    """
    if not expected:
        return {
            "field_accuracy": 1.0 if not extracted else 0.0,
            "field_precision": 1.0 if not extracted else 0.0,
            "field_recall": 1.0,
            "critical_accuracy": 1.0,
        }
    
    all_fields = set(expected.keys()) | set(extracted.keys())
    correct = 0
    
    for field in expected.keys():
        if field in extracted:
            # Compare values (normalize strings)
            exp_val = expected[field]
            ext_val = extracted[field]
            
            if isinstance(exp_val, str) and isinstance(ext_val, str):
                if exp_val.lower().strip() == ext_val.lower().strip():
                    correct += 1
            elif exp_val == ext_val:
                correct += 1
    
    precision = correct / len(extracted) if extracted else 0.0
    recall = correct / len(expected) if expected else 0.0
    
    # Critical fields accuracy
    critical_correct = 0
    critical_total = 0
    if critical_fields:
        for field in critical_fields:
            if field in expected:
                critical_total += 1
                if field in extracted:
                    exp_val = expected[field]
                    ext_val = extracted[field]
                    if isinstance(exp_val, str) and isinstance(ext_val, str):
                        if exp_val.lower().strip() == ext_val.lower().strip():
                            critical_correct += 1
                    elif exp_val == ext_val:
                        critical_correct += 1
    
    critical_accuracy = critical_correct / critical_total if critical_total > 0 else 1.0
    
    return {
        "field_accuracy": correct / len(all_fields) if all_fields else 1.0,
        "field_precision": precision,
        "field_recall": recall,
        "critical_accuracy": critical_accuracy,
    }


# =============================================================================
# TTS Composite Score
# =============================================================================

def compute_tts_composite_score(
    round_trip_wer: float,
    utmos: float = 3.0,
    snr: float = 20.0,
    speaker_sim: float = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute composite TTS quality score.
    
    S_tts = w1*(1-WER) + w2*(UTMOS/5) + w3*sigmoid(SNR) + w4*SpeakerSim
    
    Args:
        round_trip_wer: WER from TTS→ASR round trip
        utmos: Neural MOS prediction (1-5)
        snr: Signal-to-noise ratio in dB
        speaker_sim: Speaker similarity (0-1), optional
        weights: Custom weights
    
    Returns:
        Composite score (0-1)
    """
    if weights is None:
        weights = {
            "intelligibility": 0.35,
            "naturalness": 0.35,
            "audio_quality": 0.15,
            "speaker_match": 0.15,
        }
    
    score = 0.0
    
    # Intelligibility (1 - WER, capped at 0)
    score += weights["intelligibility"] * max(0, 1 - round_trip_wer)
    
    # Naturalness (UTMOS / 5)
    score += weights["naturalness"] * (utmos / 5.0)
    
    # Audio quality (sigmoid of SNR)
    snr_score = 1 / (1 + math.exp(-0.1 * (snr - 20)))
    score += weights["audio_quality"] * snr_score
    
    # Speaker similarity
    if speaker_sim is not None:
        score += weights["speaker_match"] * speaker_sim
    else:
        # Redistribute weight to other factors
        remaining_weight = 1.0 - weights["speaker_match"]
        score = score / remaining_weight if remaining_weight > 0 else score
    
    return min(1.0, max(0.0, score))


# =============================================================================
# Metric Computation Dispatcher
# =============================================================================

def compute_metric(
    metric_name: str,
    response: Any,
    ground_truth: Any,
    **kwargs
) -> float:
    """
    Compute a metric by name.
    
    Args:
        metric_name: Name of the metric
        response: Model's response
        ground_truth: Ground truth value
        **kwargs: Additional arguments for specific metrics
    
    Returns:
        Metric value
    """
    metric_name = metric_name.lower()
    
    if metric_name == "wer":
        return word_error_rate(str(response), str(ground_truth))
    elif metric_name == "cer":
        return character_error_rate(str(response), str(ground_truth))
    elif metric_name == "mer":
        return match_error_rate(str(response), str(ground_truth))
    elif metric_name == "exact_match":
        return exact_match(str(response), str(ground_truth))
    elif metric_name == "contains":
        return contains(str(response), str(ground_truth))
    elif metric_name == "f1":
        return f1_score(str(response), str(ground_truth))
    elif metric_name == "bleu":
        refs = kwargs.get("references", [str(ground_truth)])
        return bleu_score(str(response), refs)
    elif metric_name == "rouge_l":
        return rouge_l(str(response), str(ground_truth))
    elif metric_name == "retrieval_accuracy":
        variants = kwargs.get("answer_variants", [])
        return compute_needle_metrics(str(response), str(ground_truth), variants)["retrieval_accuracy"]
    elif metric_name == "syntax_valid":
        language = kwargs.get("language", "python")
        return validate_syntax(str(response), language)
    elif metric_name in ("accuracy", "semantic_match"):
        # These require LLM evaluation, return placeholder
        return kwargs.get("semantic_score", 0.5)
    
    # Unknown metric
    return 0.0

