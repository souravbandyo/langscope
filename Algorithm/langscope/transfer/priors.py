"""
Default expert priors for faceted transfer learning.

Defines initial similarity estimates between facet values based on
expert knowledge. These priors are used as cold-start estimates
and are gradually overridden by observed data.

See docs/transfer_learning.md Section 5.4 for rationale.
"""

from typing import Dict, Tuple

from langscope.transfer.faceted import (
    FACET_LANGUAGE,
    FACET_FIELD,
    FACET_MODALITY,
    FACET_TASK,
    FACET_SPECIALTY,
    FacetSimilarityLearner,
    CompositeDomainSimilarity,
)


# =============================================================================
# Language Facet Priors
# =============================================================================

LANGUAGE_PRIORS: Dict[Tuple[str, str], float] = {
    # Indo-Aryan Eastern Branch
    ("bengali", "odia"): 0.75,
    ("bengali", "assamese"): 0.70,
    ("odia", "assamese"): 0.65,
    
    # Indo-Aryan Central Branch
    ("hindi", "urdu"): 0.85,
    ("hindi", "punjabi"): 0.60,
    ("hindi", "gujarati"): 0.55,
    ("hindi", "marathi"): 0.50,
    
    # Indo-Aryan Cross-Branch
    ("bengali", "hindi"): 0.55,
    ("odia", "hindi"): 0.50,
    ("assamese", "hindi"): 0.45,
    
    # Dravidian Languages
    ("tamil", "telugu"): 0.45,
    ("tamil", "kannada"): 0.50,
    ("tamil", "malayalam"): 0.55,
    ("telugu", "kannada"): 0.50,
    ("telugu", "malayalam"): 0.40,
    ("kannada", "malayalam"): 0.45,
    
    # Indo-Aryan ↔ Dravidian (Cross-Family)
    ("hindi", "tamil"): 0.20,
    ("hindi", "telugu"): 0.20,
    ("bengali", "tamil"): 0.15,
    
    # English with Indian Languages
    ("english", "hindi"): 0.30,
    ("english", "bengali"): 0.25,
    ("english", "tamil"): 0.20,
    ("english", "telugu"): 0.20,
    
    # European Languages
    ("english", "german"): 0.55,
    ("english", "dutch"): 0.50,
    ("english", "french"): 0.45,
    ("english", "spanish"): 0.40,
    ("english", "italian"): 0.35,
    ("english", "portuguese"): 0.35,
    ("french", "spanish"): 0.70,
    ("french", "italian"): 0.75,
    ("spanish", "italian"): 0.75,
    ("spanish", "portuguese"): 0.80,
    ("german", "dutch"): 0.70,
    
    # East Asian Languages
    ("chinese", "japanese"): 0.35,
    ("chinese", "korean"): 0.25,
    ("japanese", "korean"): 0.30,
    
    # Other Pairs
    ("arabic", "persian"): 0.40,
    ("arabic", "urdu"): 0.35,
    ("russian", "polish"): 0.45,
    ("turkish", "arabic"): 0.20,
    ("vietnamese", "thai"): 0.25,
    ("indonesian", "malay"): 0.85,
}


# =============================================================================
# Field Facet Priors
# =============================================================================

FIELD_PRIORS: Dict[Tuple[str, str], float] = {
    # Medical Specialties
    ("medical", "clinical"): 0.85,
    ("medical", "radiology"): 0.70,
    ("medical", "pathology"): 0.70,
    ("medical", "cardiology"): 0.65,
    ("medical", "oncology"): 0.65,
    ("medical", "neurology"): 0.65,
    ("medical", "pediatrics"): 0.60,
    ("medical", "psychiatry"): 0.55,
    ("medical", "surgery"): 0.60,
    ("medical", "pharmacy"): 0.55,
    ("clinical", "radiology"): 0.65,
    ("clinical", "pathology"): 0.65,
    
    # Healthcare Adjacent
    ("medical", "healthcare"): 0.80,
    ("medical", "nursing"): 0.60,
    ("medical", "dentistry"): 0.45,
    ("medical", "veterinary"): 0.40,
    
    # Technical/Coding
    ("coding", "algorithms"): 0.75,
    ("coding", "programming"): 0.90,
    ("coding", "software"): 0.85,
    ("coding", "data"): 0.60,
    ("algorithms", "math"): 0.65,
    ("algorithms", "mathematics"): 0.65,
    ("programming", "software"): 0.85,
    
    # Math/Science
    ("math", "mathematics"): 0.95,
    ("math", "scientific"): 0.55,
    ("math", "physics"): 0.60,
    ("math", "engineering"): 0.50,
    ("scientific", "research"): 0.70,
    ("scientific", "academic"): 0.65,
    
    # Business/Finance
    ("financial", "economics"): 0.70,
    ("financial", "accounting"): 0.75,
    ("financial", "banking"): 0.70,
    ("financial", "investment"): 0.75,
    ("financial", "insurance"): 0.60,
    ("business", "marketing"): 0.55,
    ("business", "financial"): 0.60,
    
    # Legal
    ("legal", "financial"): 0.40,
    ("legal", "business"): 0.45,
    ("legal", "medical"): 0.30,
    
    # General Cross-Field
    ("general", "medical"): 0.40,
    ("general", "coding"): 0.45,
    ("general", "math"): 0.50,
    ("general", "legal"): 0.40,
    ("general", "financial"): 0.45,
    ("general", "scientific"): 0.50,
    ("general", "education"): 0.55,
    
    # Cross-Field (Low Similarity)
    ("medical", "legal"): 0.25,
    ("medical", "coding"): 0.20,
    ("coding", "legal"): 0.20,
    ("medical", "math"): 0.30,
}


# =============================================================================
# Modality Facet Priors
# =============================================================================

MODALITY_PRIORS: Dict[Tuple[str, str], float] = {
    # Cross-Modality
    ("text", "imaging"): 0.30,
    ("text", "audio"): 0.25,
    ("text", "video"): 0.25,
    ("imaging", "video"): 0.60,
    ("audio", "video"): 0.50,
    ("imaging", "audio"): 0.20,
    
    # Multimodal Relations
    ("multimodal", "text"): 0.70,
    ("multimodal", "imaging"): 0.70,
    ("multimodal", "audio"): 0.65,
    ("multimodal", "video"): 0.75,
    
    # Document Processing
    ("text", "ocr"): 0.60,
    ("imaging", "ocr"): 0.65,
}


# =============================================================================
# Task Facet Priors
# =============================================================================

TASK_PRIORS: Dict[Tuple[str, str], float] = {
    # Question Answering Family
    ("qa", "reasoning"): 0.70,
    ("qa", "classification"): 0.50,
    ("qa", "extraction"): 0.55,
    ("reasoning", "classification"): 0.45,
    
    # Detection/Classification Family
    ("detection", "classification"): 0.65,
    ("detection", "segmentation"): 0.70,
    ("classification", "segmentation"): 0.55,
    
    # Generation Family
    ("generation", "summarization"): 0.60,
    ("generation", "translation"): 0.55,
    ("generation", "completion"): 0.70,
    ("summarization", "extraction"): 0.50,
    ("summarization", "compression"): 0.75,
    
    # Dialogue/Chat
    ("chat", "dialogue"): 0.90,
    ("chat", "qa"): 0.60,
    ("chat", "generation"): 0.55,
    ("dialogue", "conversation"): 0.90,
    
    # Search/Retrieval
    ("search", "retrieval"): 0.85,
    ("search", "qa"): 0.50,
    ("retrieval", "extraction"): 0.45,
    
    # Transcription/Speech
    ("transcription", "translation"): 0.40,
    ("transcription", "generation"): 0.35,
    
    # Analysis Tasks
    ("analysis", "extraction"): 0.60,
    ("analysis", "classification"): 0.55,
    ("analysis", "prediction"): 0.50,
    ("prediction", "forecasting"): 0.85,
    
    # Evaluation Tasks
    ("evaluation", "assessment"): 0.90,
    ("evaluation", "grading"): 0.80,
    ("grading", "scoring"): 0.85,
    ("scoring", "ranking"): 0.70,
    
    # Cross-Task (Low Similarity)
    ("qa", "detection"): 0.35,
    ("generation", "detection"): 0.25,
    ("summarization", "classification"): 0.30,
}


# =============================================================================
# Specialty Facet Priors
# =============================================================================

SPECIALTY_PRIORS: Dict[Tuple[str, str], float] = {
    # Medical Specialties
    ("thyroid", "endocrine"): 0.85,
    ("thyroid", "oncology"): 0.50,
    ("cardiology", "pulmonology"): 0.45,
    ("cardiology", "vascular"): 0.70,
    ("neurology", "psychiatry"): 0.50,
    ("oncology", "radiology"): 0.55,
    ("pediatrics", "neonatology"): 0.80,
    
    # General vs Specialized
    ("general", "thyroid"): 0.30,
    ("general", "cardiology"): 0.30,
    ("general", "algorithms"): 0.35,
    
    # Programming Specialties
    ("algorithms", "data_structures"): 0.80,
    ("web", "frontend"): 0.75,
    ("web", "backend"): 0.70,
    ("frontend", "backend"): 0.55,
    ("ml", "deep_learning"): 0.85,
    ("ml", "nlp"): 0.70,
    ("ml", "computer_vision"): 0.65,
    ("nlp", "text_processing"): 0.75,
    
    # Legal Specialties
    ("contract", "corporate"): 0.60,
    ("litigation", "arbitration"): 0.65,
    ("intellectual_property", "patent"): 0.80,
}


# =============================================================================
# All Priors Combined
# =============================================================================

ALL_PRIORS = {
    FACET_LANGUAGE: LANGUAGE_PRIORS,
    FACET_FIELD: FIELD_PRIORS,
    FACET_MODALITY: MODALITY_PRIORS,
    FACET_TASK: TASK_PRIORS,
    FACET_SPECIALTY: SPECIALTY_PRIORS,
}


# =============================================================================
# Initialization Functions
# =============================================================================

def load_priors_into_learner(
    learner: FacetSimilarityLearner,
    priors: Dict[Tuple[str, str], float] = None
) -> None:
    """
    Load priors into a FacetSimilarityLearner.
    
    Args:
        learner: The learner to load priors into
        priors: Optional specific priors; uses default for facet if None
    """
    if priors is None:
        priors = ALL_PRIORS.get(learner.facet, {})
    
    for (value_a, value_b), prior in priors.items():
        learner.set_prior(value_a, value_b, prior)


def load_all_priors(composite: CompositeDomainSimilarity) -> None:
    """
    Load all priors into a CompositeDomainSimilarity instance.
    
    Args:
        composite: The composite similarity to initialize
    """
    for facet, learner in composite.facet_learners.items():
        priors = ALL_PRIORS.get(facet, {})
        load_priors_into_learner(learner, priors)


def create_initialized_composite(
    db=None,
    combination: str = "weighted_sum"
) -> CompositeDomainSimilarity:
    """
    Create a CompositeDomainSimilarity with all priors loaded.
    
    Args:
        db: Optional database for persistence
        combination: Combination method ("weighted_sum" or "geometric")
    
    Returns:
        Initialized CompositeDomainSimilarity
    """
    composite = CompositeDomainSimilarity(db=db, combination=combination)
    load_all_priors(composite)
    return composite


# =============================================================================
# Prior Statistics
# =============================================================================

def get_prior_statistics() -> Dict[str, Dict]:
    """
    Get statistics about the priors.
    
    Returns:
        Dict with per-facet statistics
    """
    stats = {}
    
    for facet, priors in ALL_PRIORS.items():
        if not priors:
            stats[facet] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
            }
            continue
        
        values = list(priors.values())
        stats[facet] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "unique_values": len(set(
                v for pair in priors.keys() for v in pair
            )),
        }
    
    return stats


def get_all_prior_pairs() -> int:
    """Get total number of prior pairs defined."""
    return sum(len(priors) for priors in ALL_PRIORS.values())

