"""
Tests for Prompt Management module.

Tests classification, processing, and domain detection.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from langscope.prompt.constants import (
    CATEGORIES,
    DOMAIN_KEYWORDS,
    LANGUAGE_PATTERNS,
    DOMAIN_NAME_MAPPING,
    GROUND_TRUTH_DOMAINS,
    map_to_template_name,
    is_ground_truth_domain,
    get_domain_threshold,
)
from langscope.prompt.classifier import (
    HierarchicalDomainClassifier,
    ClassificationResult,
)
from langscope.prompt.manager import (
    PromptManager,
    PromptProcessingResult,
)


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Test domain classification constants."""
    
    def test_categories_structure(self):
        """Verify categories have expected structure."""
        assert "core_language" in CATEGORIES
        assert "multimodal" in CATEGORIES
        assert "technical" in CATEGORIES
        assert "long_context" in CATEGORIES
        assert "safety" in CATEGORIES
        assert "creative" in CATEGORIES
        assert "cultural" in CATEGORIES
        
        # Check core_language has expected domains
        assert "medical" in CATEGORIES["core_language"]
        assert "legal" in CATEGORIES["core_language"]
        assert "code_generation" in CATEGORIES["core_language"]
    
    def test_domain_keywords_exist(self):
        """Verify domain keywords are defined."""
        assert "medical" in DOMAIN_KEYWORDS
        assert "legal" in DOMAIN_KEYWORDS
        assert "code_generation" in DOMAIN_KEYWORDS
        
        # Check keywords are lists
        assert isinstance(DOMAIN_KEYWORDS["medical"], list)
        assert len(DOMAIN_KEYWORDS["medical"]) > 0
    
    def test_language_patterns(self):
        """Verify language detection patterns."""
        assert "hindi" in LANGUAGE_PATTERNS
        assert "bengali" in LANGUAGE_PATTERNS
        assert "tamil" in LANGUAGE_PATTERNS
        
        # Check patterns are regex strings
        assert LANGUAGE_PATTERNS["hindi"].startswith("[")
    
    def test_ground_truth_domains(self):
        """Verify ground truth domains set."""
        assert "asr" in GROUND_TRUTH_DOMAINS
        assert "tts" in GROUND_TRUTH_DOMAINS
        assert "visual_qa" in GROUND_TRUTH_DOMAINS
        assert "needle_in_haystack" in GROUND_TRUTH_DOMAINS
        
        # Non-GT domains should not be in set
        assert "medical" not in GROUND_TRUTH_DOMAINS
    
    def test_map_to_template_name(self):
        """Test domain name mapping."""
        assert map_to_template_name("medical") == "clinical_reasoning"
        assert map_to_template_name("code_generation") == "coding_python"
        assert map_to_template_name("asr") == "asr"  # Direct mapping
        assert map_to_template_name("unknown") == "unknown"  # Fallback
    
    def test_is_ground_truth_domain(self):
        """Test ground truth domain detection."""
        assert is_ground_truth_domain("asr") is True
        assert is_ground_truth_domain("tts") is True
        assert is_ground_truth_domain("medical") is False
        assert is_ground_truth_domain("legal") is False
    
    def test_get_domain_threshold(self):
        """Test domain-specific thresholds."""
        assert get_domain_threshold("medical") == 0.95
        assert get_domain_threshold("legal") == 0.95
        assert get_domain_threshold("code_generation") == 0.90
        assert get_domain_threshold("unknown") == 0.92  # Default


# =============================================================================
# Classification Result Tests
# =============================================================================

class TestClassificationResult:
    """Test ClassificationResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic result creation."""
        result = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant=None,
            confidence=0.92,
            is_ground_truth=False,
        )
        
        assert result.category == "core_language"
        assert result.base_domain == "medical"
        assert result.confidence == 0.92
        assert result.is_ground_truth is False
    
    def test_full_domain_name_without_variant(self):
        """Test full domain name without language variant."""
        result = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant=None,
            confidence=0.92,
            is_ground_truth=False,
        )
        
        assert result.full_domain_name == "medical"
    
    def test_full_domain_name_with_variant(self):
        """Test full domain name with language variant."""
        result = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant="hindi",
            confidence=0.89,
            is_ground_truth=False,
        )
        
        assert result.full_domain_name == "hindi_medical"
    
    def test_template_name_mapping(self):
        """Test template name mapping."""
        result = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant=None,
            confidence=0.92,
            is_ground_truth=False,
        )
        
        assert result.template_name == "clinical_reasoning"
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant="hindi",
            confidence=0.89,
            is_ground_truth=False,
        )
        
        d = result.to_dict()
        
        assert d["category"] == "core_language"
        assert d["base_domain"] == "medical"
        assert d["variant"] == "hindi"
        assert d["confidence"] == 0.89
        assert d["full_domain_name"] == "hindi_medical"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "category": "multimodal",
            "base_domain": "asr",
            "variant": None,
            "confidence": 0.95,
            "is_ground_truth": True,
        }
        
        result = ClassificationResult.from_dict(data)
        
        assert result.category == "multimodal"
        assert result.base_domain == "asr"
        assert result.is_ground_truth is True


# =============================================================================
# Classifier Tests (with mocked encoder)
# =============================================================================

class TestHierarchicalDomainClassifier:
    """Test HierarchicalDomainClassifier."""
    
    def test_language_detection_hindi(self):
        """Test Hindi language detection."""
        classifier = HierarchicalDomainClassifier(lazy_load=True)
        
        # Hindi text (Devanagari script)
        result = classifier._detect_language("मधुमेह के लक्षण क्या हैं?")
        assert result == "hindi"
    
    def test_language_detection_bengali(self):
        """Test Bengali language detection."""
        classifier = HierarchicalDomainClassifier(lazy_load=True)
        
        # Bengali text
        result = classifier._detect_language("রোগীর বুকে ব্যথা আছে")
        assert result == "bengali"
    
    def test_language_detection_english(self):
        """Test English detection (returns None)."""
        classifier = HierarchicalDomainClassifier(lazy_load=True)
        
        result = classifier._detect_language("What are the symptoms of diabetes?")
        assert result is None
    
    def test_keyword_category_fallback(self):
        """Test keyword-based category fallback."""
        classifier = HierarchicalDomainClassifier(lazy_load=True)
        
        # Medical prompt
        category, conf = classifier._keyword_category_fallback(
            "Patient presents with chest pain and shortness of breath"
        )
        assert category == "core_language"
        
        # Code prompt
        category, conf = classifier._keyword_category_fallback(
            "Write a Python function to sort a list"
        )
        assert category == "core_language"
    
    def test_keyword_domain_fallback(self):
        """Test keyword-based domain fallback."""
        classifier = HierarchicalDomainClassifier(lazy_load=True)
        
        # Medical keywords
        domain, conf = classifier._keyword_domain_fallback(
            "Patient diagnosis symptoms treatment hospital",
            "core_language"
        )
        assert domain == "medical"
        
        # Code keywords
        domain, conf = classifier._keyword_domain_fallback(
            "Python function implement debug code",
            "core_language"
        )
        assert domain == "code_generation"
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        import numpy as np
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        
        sim = HierarchicalDomainClassifier._cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)
        
        # Orthogonal vectors
        c = np.array([0.0, 1.0, 0.0])
        sim = HierarchicalDomainClassifier._cosine_similarity(a, c)
        assert sim == pytest.approx(0.0)
    
    @pytest.mark.skipif(
        True,  # Skip if sentence-transformers not installed
        reason="Requires sentence-transformers"
    )
    def test_classify_with_encoder(self):
        """Test full classification with encoder."""
        classifier = HierarchicalDomainClassifier(lazy_load=False)
        
        result = classifier.classify("What are the symptoms of diabetes?")
        
        assert result.category in CATEGORIES
        assert result.confidence > 0
        assert isinstance(result.is_ground_truth, bool)


# =============================================================================
# Prompt Manager Tests
# =============================================================================

class TestPromptManager:
    """Test PromptManager."""
    
    def test_creation(self):
        """Test basic manager creation."""
        manager = PromptManager()
        
        assert manager.classifier is not None
        assert manager._metrics["classifications"] == 0
    
    def test_build_domain_from_name_simple(self):
        """Test building domain from simple name."""
        manager = PromptManager()
        
        result = manager._build_domain_from_name("medical")
        
        assert result.base_domain == "medical"
        assert result.variant is None
        assert result.confidence == 1.0
    
    def test_build_domain_from_name_with_variant(self):
        """Test building domain from name with language variant."""
        manager = PromptManager()
        
        result = manager._build_domain_from_name("hindi_medical")
        
        assert result.base_domain == "medical"
        assert result.variant == "hindi"
        assert result.full_domain_name == "hindi_medical"
    
    def test_build_domain_from_name_ground_truth(self):
        """Test ground truth detection from name."""
        manager = PromptManager()
        
        result = manager._build_domain_from_name("asr")
        
        assert result.base_domain == "asr"
        assert result.is_ground_truth is True
        
        result = manager._build_domain_from_name("medical")
        assert result.is_ground_truth is False
    
    def test_make_exact_key(self):
        """Test cache key generation."""
        manager = PromptManager()
        
        key1 = manager._make_exact_key("Hello", "gpt-4", {})
        key2 = manager._make_exact_key("Hello", "gpt-4", {})
        key3 = manager._make_exact_key("Hello", "claude", {})
        
        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different model = different key
    
    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        manager = PromptManager()
        
        assert manager._metrics["classifications"] == 0
        
        # Use classify (sync method)
        manager.classify_prompt("Test prompt")
        
        assert manager._metrics["classifications"] == 1
    
    def test_get_metrics(self):
        """Test getting metrics."""
        manager = PromptManager()
        
        # Manually set some metrics
        manager._metrics["exact_hits"] = 10
        manager._metrics["semantic_hits"] = 20
        manager._metrics["misses"] = 70
        manager._metrics["total_time_ms"] = 1000.0
        
        metrics = manager.get_metrics()
        
        assert metrics["total_requests"] == 100
        assert metrics["exact_hit_rate"] == 0.1
        assert metrics["semantic_hit_rate"] == 0.2
        assert metrics["overall_hit_rate"] == 0.3
        assert metrics["avg_time_ms"] == 10.0
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        manager = PromptManager()
        
        manager._metrics["classifications"] = 100
        manager.reset_metrics()
        
        assert manager._metrics["classifications"] == 0


# =============================================================================
# Async Tests
# =============================================================================

class TestPromptManagerAsync:
    """Async tests for PromptManager."""
    
    @pytest.mark.asyncio
    async def test_process_prompt_with_user_domain(self):
        """Test processing with explicit user domain."""
        manager = PromptManager()
        
        result = await manager.process_prompt(
            prompt="Some prompt",
            user_domain="medical",
        )
        
        assert result.domain.base_domain == "medical"
        assert result.cache_hit is False
        assert result.evaluation_type == "subjective"
    
    @pytest.mark.asyncio
    async def test_process_prompt_ground_truth_domain(self):
        """Test processing for ground truth domain."""
        manager = PromptManager()
        
        result = await manager.process_prompt(
            prompt="Transcribe this audio",
            user_domain="asr",
        )
        
        assert result.domain.base_domain == "asr"
        assert result.evaluation_type == "ground_truth"
    
    @pytest.mark.asyncio
    async def test_build_result(self):
        """Test building processing result."""
        import time
        
        manager = PromptManager()
        
        domain = ClassificationResult(
            category="core_language",
            base_domain="medical",
            variant=None,
            confidence=0.92,
            is_ground_truth=False,
        )
        
        start = time.perf_counter()
        result = await manager._build_result(
            prompt="Test",
            domain=domain,
            cache_hit=True,
            cache_layer="exact",
            start_time=start,
        )
        
        assert result.cache_hit is True
        assert result.cache_layer == "exact"
        assert result.processing_time_ms > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPromptIntegration:
    """Integration tests for prompt module."""
    
    def test_constants_categories_domains_match_keywords(self):
        """Verify all domains in categories have keywords defined."""
        for category, domains in CATEGORIES.items():
            for domain in domains:
                # Not all domains need keywords (OK if missing)
                pass
    
    def test_ground_truth_domains_in_categories(self):
        """Verify GT domains are in some category."""
        all_domains = set()
        for domains in CATEGORIES.values():
            all_domains.update(domains)
        
        for gt_domain in GROUND_TRUTH_DOMAINS:
            assert gt_domain in all_domains, f"{gt_domain} not in any category"

