"""
ASR (Automatic Speech Recognition) Ground Truth Judge.

Evaluates transcription accuracy using WER, CER, and MER metrics.
"""

from typing import Dict, List, Optional, Any
from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore
from langscope.ground_truth.metrics import (
    word_error_rate,
    character_error_rate,
    match_error_rate,
)


class ASRJudge(GroundTruthJudge):
    """
    Specialized judge for ASR evaluation.
    
    Primary metric: Word Error Rate (WER)
    Additional metrics: CER, MER
    """
    
    def __init__(self, **kwargs):
        super().__init__(domain="asr", **kwargs)
    
    def _compute_metrics(
        self,
        response: str,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """Compute ASR-specific metrics."""
        # Get reference transcript
        if isinstance(ground_truth, dict):
            reference = ground_truth.get("transcript", str(ground_truth))
        else:
            reference = str(ground_truth)
        
        hypothesis = str(response)
        
        # Compute metrics
        wer = word_error_rate(hypothesis, reference)
        cer = character_error_rate(hypothesis, reference)
        mer = match_error_rate(hypothesis, reference)
        
        # Accuracy is 1 - WER (capped at 0)
        accuracy = max(0.0, 1.0 - wer)
        
        return {
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "accuracy": accuracy,
        }
    
    def get_language_breakdown(
        self,
        match_results: List[Dict],
        model_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get WER breakdown by language.
        
        Args:
            match_results: List of match results
            model_id: Model identifier
        
        Returns:
            {language: {avg_wer, avg_cer, sample_count}}
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        for result in match_results:
            scores = result.get("scores", {}).get(model_id, {})
            metadata = result.get("sample_metadata", {})
            
            language = metadata.get("language", "unknown")
            metrics = scores.get("metrics", {})
            
            if language not in stats:
                stats[language] = {
                    "wer_sum": 0.0,
                    "cer_sum": 0.0,
                    "count": 0,
                }
            
            stats[language]["wer_sum"] += metrics.get("wer", 0.0)
            stats[language]["cer_sum"] += metrics.get("cer", 0.0)
            stats[language]["count"] += 1
        
        # Compute averages
        breakdown = {}
        for lang, data in stats.items():
            count = data["count"]
            breakdown[lang] = {
                "avg_wer": data["wer_sum"] / count if count > 0 else 0.0,
                "avg_cer": data["cer_sum"] / count if count > 0 else 0.0,
                "sample_count": count,
            }
        
        return breakdown
    
    def get_difficulty_breakdown(
        self,
        match_results: List[Dict],
        model_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get WER breakdown by difficulty.
        
        Args:
            match_results: List of match results
            model_id: Model identifier
        
        Returns:
            {difficulty: {avg_wer, avg_cer, sample_count}}
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        for result in match_results:
            scores = result.get("scores", {}).get(model_id, {})
            metadata = result.get("sample_metadata", {})
            
            difficulty = metadata.get("difficulty", "medium")
            metrics = scores.get("metrics", {})
            
            if difficulty not in stats:
                stats[difficulty] = {
                    "wer_sum": 0.0,
                    "cer_sum": 0.0,
                    "count": 0,
                }
            
            stats[difficulty]["wer_sum"] += metrics.get("wer", 0.0)
            stats[difficulty]["cer_sum"] += metrics.get("cer", 0.0)
            stats[difficulty]["count"] += 1
        
        # Compute averages
        breakdown = {}
        for diff, data in stats.items():
            count = data["count"]
            breakdown[diff] = {
                "avg_wer": data["wer_sum"] / count if count > 0 else 0.0,
                "avg_cer": data["cer_sum"] / count if count > 0 else 0.0,
                "sample_count": count,
            }
        
        return breakdown


