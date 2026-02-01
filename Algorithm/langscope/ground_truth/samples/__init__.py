"""
Ground Truth Sample Datasets.

Provides sample datasets for ground truth evaluation across domains:
- needle_in_haystack: Long context retrieval samples
- asr: Automatic Speech Recognition transcription samples
- visual_qa: Visual Question Answering samples
- code_completion: Code completion and generation samples
- long_document_qa: Long document question answering samples

Usage:
    from langscope.ground_truth.samples import load_samples, get_sample_by_id
    
    # Load all samples for a domain
    samples = load_samples("needle_in_haystack")
    
    # Get a specific sample
    sample = get_sample_by_id("needle_4k_begin_001")
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Sample directory
SAMPLES_DIR = os.path.dirname(__file__)

# Available domains
AVAILABLE_DOMAINS = [
    "needle_in_haystack",
    "asr",
    "visual_qa",
    "code_completion",
    "long_document_qa",
]

# Cache for loaded samples
_sample_cache: Dict[str, Dict] = {}


def load_samples(domain: str, force_reload: bool = False) -> Dict:
    """
    Load all samples for a domain.
    
    Args:
        domain: Domain name (e.g., "asr", "needle_in_haystack")
        force_reload: Force reload from disk
    
    Returns:
        Dict with domain metadata and samples list
    """
    global _sample_cache
    
    if not force_reload and domain in _sample_cache:
        return _sample_cache[domain]
    
    sample_file = os.path.join(SAMPLES_DIR, f"{domain}.json")
    
    if not os.path.exists(sample_file):
        logger.warning(f"Sample file not found: {sample_file}")
        return {"domain": domain, "samples": []}
    
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        _sample_cache[domain] = data
        logger.info(f"Loaded {len(data.get('samples', []))} samples for domain: {domain}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading samples for {domain}: {e}")
        return {"domain": domain, "samples": [], "error": str(e)}


def get_all_samples(domain: str) -> List[Dict]:
    """
    Get all samples for a domain as a list.
    
    Args:
        domain: Domain name
    
    Returns:
        List of sample dicts
    """
    data = load_samples(domain)
    return data.get("samples", [])


def get_sample_by_id(sample_id: str, domain: str = None) -> Optional[Dict]:
    """
    Get a specific sample by ID.
    
    Args:
        sample_id: Sample identifier
        domain: Optional domain hint for faster lookup
    
    Returns:
        Sample dict or None
    """
    domains_to_search = [domain] if domain else AVAILABLE_DOMAINS
    
    for d in domains_to_search:
        samples = get_all_samples(d)
        for sample in samples:
            if sample.get("sample_id") == sample_id:
                return sample
    
    return None


def get_samples_by_difficulty(domain: str, difficulty: str) -> List[Dict]:
    """
    Get samples filtered by difficulty.
    
    Args:
        domain: Domain name
        difficulty: Difficulty level (easy, medium, hard)
    
    Returns:
        List of matching samples
    """
    samples = get_all_samples(domain)
    return [s for s in samples if s.get("difficulty") == difficulty]


def get_samples_by_category(domain: str, category: str) -> List[Dict]:
    """
    Get samples filtered by category.
    
    Args:
        domain: Domain name
        category: Category name
    
    Returns:
        List of matching samples
    """
    samples = get_all_samples(domain)
    return [s for s in samples if s.get("category") == category]


def get_sample_statistics(domain: str) -> Dict[str, Any]:
    """
    Get statistics about samples for a domain.
    
    Args:
        domain: Domain name
    
    Returns:
        Dict with statistics
    """
    samples = get_all_samples(domain)
    
    if not samples:
        return {"total": 0}
    
    difficulties = {}
    categories = {}
    languages = {}
    
    for sample in samples:
        diff = sample.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        cat = sample.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        
        lang = sample.get("language", "en")
        languages[lang] = languages.get(lang, 0) + 1
    
    return {
        "total": len(samples),
        "by_difficulty": difficulties,
        "by_category": categories,
        "by_language": languages,
    }


def list_available_domains() -> List[str]:
    """List all available sample domains."""
    return AVAILABLE_DOMAINS.copy()


def get_random_sample(domain: str, difficulty: str = None) -> Optional[Dict]:
    """
    Get a random sample from a domain.
    
    Args:
        domain: Domain name
        difficulty: Optional difficulty filter
    
    Returns:
        Random sample or None
    """
    import random
    
    if difficulty:
        samples = get_samples_by_difficulty(domain, difficulty)
    else:
        samples = get_all_samples(domain)
    
    if not samples:
        return None
    
    return random.choice(samples)


__all__ = [
    "load_samples",
    "get_all_samples",
    "get_sample_by_id",
    "get_samples_by_difficulty",
    "get_samples_by_category",
    "get_sample_statistics",
    "list_available_domains",
    "get_random_sample",
    "AVAILABLE_DOMAINS",
]


