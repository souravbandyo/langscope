"""
Ground Truth Judges.

Specialized evaluation judges for different ground truth domains:
- ASR: Word/Character Error Rate evaluation
- TTS: Composite quality scoring with round-trip ASR
- Needle in Haystack: Information retrieval from long context
- Visual QA: Question answering about images
- Document Extraction: Structured field extraction
- Code Completion: Syntax + test execution
- Long Document QA: Multi-evidence reasoning
- Long Summarization: ROUGE + key point coverage
"""

from langscope.ground_truth.judge import (
    GroundTruthJudge,
    GroundTruthScore,
    EvaluationMode,
    ASRGroundTruthJudge,
    NeedleGroundTruthJudge,
    CodeGroundTruthJudge,
)

from langscope.ground_truth.judges.needle_judge import (
    NeedleInHaystackJudge,
)

from langscope.ground_truth.judges.tts_judge import (
    TTSGroundTruthJudge,
)

from langscope.ground_truth.judges.long_doc_judge import (
    LongDocQAJudge,
    MultiDocReasoningJudge,
    LongSummarizationJudge,
)

from langscope.ground_truth.judges.visual_judge import (
    VisualQAJudge,
    DocumentExtractionJudge,
    ImageCaptioningJudge,
    OCRJudge,
)

__all__ = [
    # Base classes
    "GroundTruthJudge",
    "GroundTruthScore",
    "EvaluationMode",
    # Core judges from judge.py
    "ASRGroundTruthJudge",
    "NeedleGroundTruthJudge",
    "CodeGroundTruthJudge",
    # Needle in Haystack
    "NeedleInHaystackJudge",
    # TTS
    "TTSGroundTruthJudge",
    # Long Document
    "LongDocQAJudge",
    "MultiDocReasoningJudge",
    "LongSummarizationJudge",
    # Visual
    "VisualQAJudge",
    "DocumentExtractionJudge",
    "ImageCaptioningJudge",
    "OCRJudge",
]


# Factory function to get appropriate judge
def get_judge_for_domain(domain: str, **kwargs) -> GroundTruthJudge:
    """
    Get the appropriate judge for a ground truth domain.
    
    Args:
        domain: Domain name
        **kwargs: Additional arguments for judge initialization
    
    Returns:
        Appropriate GroundTruthJudge subclass instance
    """
    domain_judge_map = {
        "asr": ASRGroundTruthJudge,
        "tts": TTSGroundTruthJudge,
        "needle_in_haystack": NeedleInHaystackJudge,
        "visual_qa": VisualQAJudge,
        "document_extraction": DocumentExtractionJudge,
        "image_captioning": ImageCaptioningJudge,
        "ocr": OCRJudge,
        "long_document_qa": LongDocQAJudge,
        "multi_document_reasoning": MultiDocReasoningJudge,
        "long_summarization": LongSummarizationJudge,
        "code_completion": CodeGroundTruthJudge,
    }
    
    judge_class = domain_judge_map.get(domain, GroundTruthJudge)
    
    if judge_class == GroundTruthJudge:
        return GroundTruthJudge(domain=domain, **kwargs)
    
    return judge_class(**kwargs)
