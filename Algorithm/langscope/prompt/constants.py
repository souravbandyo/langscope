"""
Constants for domain classification.

Defines categories, keywords, language patterns, and domain mappings.
"""

from typing import Dict, List, Set


# =============================================================================
# Domain Categories
# =============================================================================

CATEGORIES: Dict[str, List[str]] = {
    "core_language": [
        "medical",
        "legal",
        "financial",
        "customer_support",
        "education",
        "code_generation",
    ],
    "multimodal": [
        "asr",
        "tts",
        "visual_qa",
        "document_extraction",
        "image_captioning",
        "ocr",
        "image_generation",
        "video_understanding",
    ],
    "safety": [
        "bias_detection",
        "harmful_content",
        "privacy",
        "truthfulness",
    ],
    "cultural": [
        "cultural_competence",
        "regional_language",
        "religious",
        "local_context",
    ],
    "technical": [
        "scientific",
        "mathematical",
        "logical",
        "data_analysis",
    ],
    "long_context": [
        "needle_in_haystack",
        "long_document_qa",
        "multi_document",
        "code_completion",
        "long_summarization",
    ],
    "creative": [
        "creative_writing",
        "tone_adaptation",
    ],
}


# =============================================================================
# Ground Truth Domains
# =============================================================================

GROUND_TRUTH_DOMAINS: Set[str] = {
    # Multimodal (metrics-based)
    "asr",
    "tts",
    "visual_qa",
    "document_extraction",
    "image_captioning",
    "ocr",
    # Long Context (metrics-based)
    "needle_in_haystack",
    "long_document_qa",
    "code_completion",
}


# =============================================================================
# Domain Name Mapping
# =============================================================================
# Maps classifier output -> existing DOMAIN_TEMPLATES names

DOMAIN_NAME_MAPPING: Dict[str, str] = {
    # Core Language mappings
    "medical": "clinical_reasoning",
    "code_generation": "coding_python",
    "mathematical": "math_reasoning",
    
    # Direct mappings (same names)
    "asr": "asr",
    "tts": "tts",
    "visual_qa": "visual_qa",
    "needle_in_haystack": "needle_in_haystack",
    "long_document_qa": "long_document_qa",
    "code_completion": "code_execution",
    
    # Language variant mappings
    "hindi_medical": "hindi_medical",
    "hindi": "hindi",
    
    # General fallback
    "general": "general",
}


# =============================================================================
# Language Detection Patterns
# =============================================================================

LANGUAGE_PATTERNS: Dict[str, str] = {
    # Indic languages (Devanagari and related scripts)
    "hindi": r'[\u0900-\u097F]',        # Devanagari script
    "bengali": r'[\u0980-\u09FF]',      # Bengali script
    "odia": r'[\u0B00-\u0B7F]',         # Odia script
    "tamil": r'[\u0B80-\u0BFF]',        # Tamil script
    "telugu": r'[\u0C00-\u0C7F]',       # Telugu script
    "gujarati": r'[\u0A80-\u0AFF]',     # Gujarati script
    "kannada": r'[\u0C80-\u0CFF]',      # Kannada script
    "malayalam": r'[\u0D00-\u0D7F]',    # Malayalam script
    "punjabi": r'[\u0A00-\u0A7F]',      # Gurmukhi script
    
    # Note: Marathi uses Devanagari (same as Hindi)
    # Distinguished by context or explicit specification
}


# =============================================================================
# Domain-Specific Similarity Thresholds
# =============================================================================

DOMAIN_THRESHOLDS: Dict[str, float] = {
    "medical": 0.95,        # Safety-critical, avoid wrong advice
    "legal": 0.95,          # Precision required for legal guidance
    "financial": 0.93,      # Financial decisions need precision
    "code_generation": 0.90, # Structural similarity sufficient
    "creative_writing": 0.88, # More variation acceptable
    "customer_support": 0.90, # FAQ-like, moderate precision
}
DEFAULT_THRESHOLD = 0.92


# =============================================================================
# Domain Keywords
# =============================================================================

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "medical": [
        # English
        "patient", "diagnosis", "symptoms", "treatment", "doctor",
        "medicine", "disease", "hospital", "prescription", "clinical",
        "therapy", "medical", "health", "condition", "illness",
        "prognosis", "pathology", "oncology", "cardiology", "neurology",
        # Hindi
        "रोगी", "निदान", "लक्षण", "उपचार", "मधुमेह", "दवाई",
        "चिकित्सा", "अस्पताल", "बीमारी", "स्वास्थ्य",
        # Bengali
        "রোগী", "রোগ", "চিকিৎসা", "লক্ষণ",
    ],
    "legal": [
        "contract", "court", "law", "legal", "lawsuit", "attorney",
        "jurisdiction", "plaintiff", "defendant", "statute", "regulation",
        "liability", "damages", "injunction", "arbitration", "litigation",
        "patent", "copyright", "trademark", "compliance",
        # Hindi
        "अनुबंध", "कानून", "न्यायालय", "वकील", "मुकदमा",
    ],
    "financial": [
        "investment", "stock", "portfolio", "risk", "return",
        "dividend", "equity", "bond", "trading", "market",
        "interest", "loan", "credit", "asset", "liability",
        "revenue", "profit", "loss", "tax", "accounting",
        # Hindi
        "निवेश", "शेयर", "पोर्टफोलियो", "ब्याज", "कर",
    ],
    "customer_support": [
        "refund", "complaint", "ticket", "support", "help",
        "issue", "problem", "resolve", "customer", "order",
        "delivery", "return", "exchange", "warranty", "service",
    ],
    "education": [
        "student", "teacher", "course", "curriculum", "exam",
        "grade", "assignment", "lecture", "syllabus", "learning",
        "school", "university", "degree", "certificate", "training",
        # Hindi
        "छात्र", "शिक्षक", "पाठ्यक्रम", "परीक्षा", "शिक्षा",
    ],
    "code_generation": [
        "python", "javascript", "function", "code", "debug",
        "implement", "def ", "class ", "import ", "return ",
        "async", "await", "algorithm", "bug", "error",
        "variable", "loop", "array", "string", "integer",
        "api", "database", "sql", "query", "server",
    ],
    "asr": [
        "transcribe", "audio", "speech", "voice", "recording",
        "speech-to-text", "dictation", "transcription", "waveform",
        "acoustic", "speaker", "microphone", "utterance",
    ],
    "tts": [
        "synthesize", "speech synthesis", "text-to-speech", "voice",
        "narration", "vocalize", "pronunciation", "prosody",
        "intonation", "audio generation",
    ],
    "visual_qa": [
        "image", "picture", "photo", "visual", "see",
        "looking at", "in this image", "what is shown",
        "describe", "identify", "count", "color",
    ],
    "needle_in_haystack": [
        "find", "locate", "hidden", "context", "document",
        "search", "retrieve", "needle", "haystack",
        "somewhere in", "mentioned", "passage",
    ],
    "long_document_qa": [
        "document", "paper", "report", "article", "text",
        "long", "lengthy", "comprehensive", "detailed",
        "according to", "based on the text",
    ],
    "code_completion": [
        "complete", "finish", "continue", "next line",
        "missing code", "fill in", "implementation",
    ],
    "mathematical": [
        "calculate", "compute", "equation", "formula", "solve",
        "proof", "theorem", "algebra", "calculus", "geometry",
        "integral", "derivative", "matrix", "vector", "function",
        # Hindi
        "गणना", "समीकरण", "सूत्र", "प्रमाण",
    ],
    "scientific": [
        "experiment", "hypothesis", "theory", "research",
        "physics", "chemistry", "biology", "laboratory",
        "observation", "data", "analysis", "conclusion",
    ],
    "creative_writing": [
        "story", "poem", "narrative", "fiction", "creative",
        "write", "compose", "draft", "author", "literary",
        "character", "plot", "setting", "dialogue",
    ],
    "logical": [
        "logic", "reasoning", "inference", "deduction",
        "induction", "premise", "conclusion", "argument",
        "valid", "fallacy", "syllogism",
    ],
    "data_analysis": [
        "data", "analysis", "statistics", "visualization",
        "chart", "graph", "trend", "correlation", "regression",
        "dataset", "metric", "insight",
    ],
    "bias_detection": [
        "bias", "fairness", "discrimination", "stereotype",
        "prejudice", "impartial", "objective", "balanced",
    ],
    "harmful_content": [
        "harmful", "dangerous", "toxic", "offensive",
        "inappropriate", "violence", "abuse", "threat",
    ],
    "privacy": [
        "privacy", "personal", "confidential", "sensitive",
        "pii", "data protection", "gdpr", "anonymize",
    ],
    "truthfulness": [
        "fact", "verify", "accurate", "true", "false",
        "misinformation", "disinformation", "claim",
    ],
    "cultural_competence": [
        "culture", "tradition", "custom", "heritage",
        "cultural", "ethnic", "diversity", "multicultural",
    ],
    "regional_language": [
        "regional", "dialect", "local language", "vernacular",
        "native", "mother tongue",
    ],
    "religious": [
        "religion", "religious", "faith", "spiritual",
        "temple", "church", "mosque", "scripture",
    ],
    "local_context": [
        "local", "regional", "area", "community",
        "neighborhood", "city", "state", "district",
    ],
}


def map_to_template_name(classified_domain: str) -> str:
    """
    Map classifier output to existing DOMAIN_TEMPLATES name.
    
    Falls back to the original name if no mapping exists.
    
    Args:
        classified_domain: Domain name from classifier
    
    Returns:
        Template name for DomainManager
    """
    return DOMAIN_NAME_MAPPING.get(classified_domain, classified_domain)


def is_ground_truth_domain(base_domain: str) -> bool:
    """
    Check if domain uses ground truth evaluation.
    
    Args:
        base_domain: Base domain name
    
    Returns:
        True if domain uses GT evaluation
    """
    return base_domain in GROUND_TRUTH_DOMAINS


def get_domain_threshold(domain: str) -> float:
    """
    Get similarity threshold for a domain.
    
    Args:
        domain: Domain name
    
    Returns:
        Similarity threshold
    """
    return DOMAIN_THRESHOLDS.get(domain, DEFAULT_THRESHOLD)

