"""
Prompt Manager for LangScope.

Unified interface for prompt processing, classification, and caching.
"""

import time
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np

from langscope.prompt.classifier import (
    HierarchicalDomainClassifier,
    ClassificationResult,
)
from langscope.prompt.constants import (
    CATEGORIES,
    LANGUAGE_PATTERNS,
    GROUND_TRUTH_DOMAINS,
    map_to_template_name,
)

if TYPE_CHECKING:
    from langscope.domain.domain_manager import DomainManager
    from langscope.domain.domain_config import Domain
    from langscope.cache.manager import UnifiedCacheManager

logger = logging.getLogger(__name__)


@dataclass
class PromptProcessingResult:
    """Result of processing a prompt."""
    
    # Original input
    prompt: str
    
    # Classification
    domain: ClassificationResult
    domain_config: Optional['Domain'] = None
    
    # Cache status
    cache_hit: bool = False
    cache_layer: Optional[str] = None  # "exact", "semantic", None
    cache_similarity: Optional[float] = None
    cached_response: Optional[Dict[str, Any]] = None
    
    # Routing
    evaluation_type: str = "subjective"  # "subjective" or "ground_truth"
    
    # Performance
    processing_time_ms: float = 0.0
    
    # Embedding (for downstream use)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "domain": self.domain.to_dict(),
            "cache_hit": self.cache_hit,
            "cache_layer": self.cache_layer,
            "cache_similarity": self.cache_similarity,
            "evaluation_type": self.evaluation_type,
            "processing_time_ms": self.processing_time_ms,
        }


class PromptManager:
    """
    Unified prompt management.
    
    Integrates:
    - Domain classification
    - Semantic caching (when cache is provided)
    - Domain routing
    
    Connects with existing DomainManager.
    """
    
    def __init__(
        self,
        domain_manager: 'DomainManager' = None,
        classifier: HierarchicalDomainClassifier = None,
        cache_manager: 'UnifiedCacheManager' = None,
        centroids_path: str = None,
    ):
        """
        Initialize PromptManager.
        
        Args:
            domain_manager: Existing DomainManager instance
            classifier: Pre-configured classifier
            cache_manager: Cache manager for semantic caching
            centroids_path: Path to pre-trained centroids
        """
        self.domain_manager = domain_manager
        self.cache_manager = cache_manager
        
        # Initialize classifier
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = HierarchicalDomainClassifier(
                centroids_path=centroids_path,
                lazy_load=True,
            )
        
        # Metrics
        self._metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "classifications": 0,
            "total_time_ms": 0.0,
        }
    
    async def process_prompt(
        self,
        prompt: str,
        model: str = None,
        params: Dict = None,
        user_domain: str = None,
        skip_cache: bool = False,
    ) -> PromptProcessingResult:
        """
        Main entry point for prompt processing.
        
        1. Check exact match cache (Redis)
        2. Compute embedding
        3. Check semantic cache (Qdrant)
        4. Classify domain (if needed)
        5. Return result with routing info
        
        Args:
            prompt: Input prompt text
            model: Model ID (for cache key)
            params: Additional parameters
            user_domain: Optional domain override
            skip_cache: Skip cache lookup
        
        Returns:
            PromptProcessingResult with classification and cache info
        """
        start = time.perf_counter()
        
        # Step 1: If user specified domain, use it directly
        if user_domain:
            domain_result = self._build_domain_from_name(user_domain)
            return await self._build_result(
                prompt=prompt,
                domain=domain_result,
                start_time=start,
            )
        
        # Step 2: Check exact match cache
        if not skip_cache and self.cache_manager:
            exact_hit = await self._check_exact_cache(prompt, model, params)
            if exact_hit:
                self._metrics["exact_hits"] += 1
                domain_data = exact_hit.get("domain_result", {})
                domain_result = ClassificationResult.from_dict(domain_data) if domain_data else None
                
                if domain_result:
                    return await self._build_result(
                        prompt=prompt,
                        domain=domain_result,
                        cache_hit=True,
                        cache_layer="exact",
                        cached_response=exact_hit.get("response"),
                        start_time=start,
                    )
        
        # Step 3: Classify the prompt (includes embedding generation)
        domain_result = self.classifier.classify(prompt)
        self._metrics["classifications"] += 1
        
        # Step 4: Check semantic cache with the embedding
        if not skip_cache and self.cache_manager:
            semantic_hit = await self._check_semantic_cache(
                prompt,
                domain_result.embedding,
                domain_result.base_domain,
            )
            if semantic_hit:
                self._metrics["semantic_hits"] += 1
                return await self._build_result(
                    prompt=prompt,
                    domain=domain_result,
                    cache_hit=True,
                    cache_layer="semantic",
                    cache_similarity=semantic_hit.get("similarity"),
                    cached_response=semantic_hit.get("response"),
                    embedding=domain_result.embedding,
                    start_time=start,
                )
        
        # Step 5: No cache hit
        self._metrics["misses"] += 1
        
        return await self._build_result(
            prompt=prompt,
            domain=domain_result,
            cache_hit=False,
            embedding=domain_result.embedding,
            start_time=start,
        )
    
    def classify_prompt(self, prompt: str) -> ClassificationResult:
        """
        Classify prompt without caching (synchronous).
        
        Args:
            prompt: Input prompt text
        
        Returns:
            ClassificationResult
        """
        self._metrics["classifications"] += 1
        return self.classifier.classify(prompt)
    
    async def cache_response(
        self,
        prompt: str,
        embedding: np.ndarray,
        domain_result: ClassificationResult,
        response: Dict[str, Any],
        model_id: str = None,
    ) -> bool:
        """
        Cache response for future requests.
        
        Args:
            prompt: Original prompt
            embedding: Prompt embedding
            domain_result: Classification result
            response: Response to cache
            model_id: Model that generated response
        
        Returns:
            True if cached successfully
        """
        if not self.cache_manager:
            return False
        
        try:
            # Import here to avoid circular dependency
            from langscope.cache.categories import CacheCategory
            
            # Exact match cache
            cache_key = self._make_exact_key(prompt, model_id, {})
            cache_data = {
                "response": response,
                "domain_result": domain_result.to_dict(),
                "model_id": model_id,
            }
            
            await self.cache_manager.set(
                CacheCategory.PROMPT_EXACT,
                cache_key,
                cache_data,
            )
            
            # Semantic cache (if cache has semantic support)
            if hasattr(self.cache_manager, 'set_semantic'):
                await self.cache_manager.set_semantic(
                    prompt=prompt,
                    embedding=embedding,
                    domain_result=domain_result,
                    response=response,
                    model_id=model_id,
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache and classification metrics."""
        total = (
            self._metrics["exact_hits"] +
            self._metrics["semantic_hits"] +
            self._metrics["misses"]
        )
        
        return {
            **self._metrics,
            "total_requests": total,
            "exact_hit_rate": self._metrics["exact_hits"] / total if total else 0,
            "semantic_hit_rate": self._metrics["semantic_hits"] / total if total else 0,
            "overall_hit_rate": (
                self._metrics["exact_hits"] + self._metrics["semantic_hits"]
            ) / total if total else 0,
            "avg_time_ms": self._metrics["total_time_ms"] / total if total else 0,
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self._metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "classifications": 0,
            "total_time_ms": 0.0,
        }
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    async def _check_exact_cache(
        self,
        prompt: str,
        model: str,
        params: Dict,
    ) -> Optional[Dict]:
        """Check exact match cache."""
        if not self.cache_manager:
            return None
        
        try:
            from langscope.cache.categories import CacheCategory
            
            key = self._make_exact_key(prompt, model, params)
            return await self.cache_manager.get(CacheCategory.PROMPT_EXACT, key)
        except Exception as e:
            logger.debug(f"Exact cache check failed: {e}")
            return None
    
    async def _check_semantic_cache(
        self,
        prompt: str,
        embedding: np.ndarray,
        domain: str,
    ) -> Optional[Dict]:
        """Check semantic cache."""
        if not self.cache_manager:
            return None
        
        try:
            if hasattr(self.cache_manager, 'get_semantic'):
                return await self.cache_manager.get_semantic(
                    prompt=prompt,
                    embedding=embedding,
                    domain=domain,
                )
        except Exception as e:
            logger.debug(f"Semantic cache check failed: {e}")
        
        return None
    
    def _make_exact_key(
        self,
        prompt: str,
        model: str,
        params: Dict,
    ) -> str:
        """Create cache key for exact match."""
        normalized = prompt.strip().lower()
        cache_str = f"{model or ''}:{normalized}:{sorted((params or {}).items())}"
        return hashlib.sha256(cache_str.encode()).hexdigest()[:32]
    
    def _build_domain_from_name(self, domain_name: str) -> ClassificationResult:
        """Build ClassificationResult from domain name."""
        parts = domain_name.split("_", 1)
        
        # Check if first part is a language
        if len(parts) == 2 and parts[0] in LANGUAGE_PATTERNS:
            variant, base_domain = parts
        else:
            variant = None
            base_domain = domain_name
        
        # Find category
        category = "core_language"  # default
        for cat, domains in CATEGORIES.items():
            if base_domain in domains:
                category = cat
                break
        
        return ClassificationResult(
            category=category,
            base_domain=base_domain,
            variant=variant,
            confidence=1.0,  # User-specified, full confidence
            is_ground_truth=base_domain in GROUND_TRUTH_DOMAINS,
        )
    
    async def _build_result(
        self,
        prompt: str,
        domain: ClassificationResult,
        cache_hit: bool = False,
        cache_layer: str = None,
        cache_similarity: float = None,
        cached_response: Dict = None,
        embedding: np.ndarray = None,
        start_time: float = None,
    ) -> PromptProcessingResult:
        """Build processing result."""
        elapsed = (time.perf_counter() - start_time) * 1000 if start_time else 0
        self._metrics["total_time_ms"] += elapsed
        
        # Get domain config from DomainManager
        domain_config = None
        if self.domain_manager:
            # Try full domain name first
            domain_config = self.domain_manager.get_domain(domain.template_name)
            
            # Fall back to base domain
            if not domain_config:
                base_template = map_to_template_name(domain.base_domain)
                domain_config = self.domain_manager.get_domain(base_template)
        
        # Determine evaluation type
        if domain.is_ground_truth:
            evaluation_type = "ground_truth"
        elif domain_config and hasattr(domain_config.settings, 'evaluation_type'):
            evaluation_type = domain_config.settings.evaluation_type
        else:
            evaluation_type = "subjective"
        
        return PromptProcessingResult(
            prompt=prompt,
            domain=domain,
            domain_config=domain_config,
            cache_hit=cache_hit,
            cache_layer=cache_layer,
            cache_similarity=cache_similarity,
            cached_response=cached_response,
            evaluation_type=evaluation_type,
            processing_time_ms=elapsed,
            embedding=embedding.tolist() if embedding is not None else None,
        )


# =============================================================================
# Convenience Function
# =============================================================================

_default_prompt_manager: Optional[PromptManager] = None


def get_default_prompt_manager() -> PromptManager:
    """Get or create default prompt manager."""
    global _default_prompt_manager
    if _default_prompt_manager is None:
        _default_prompt_manager = PromptManager()
    return _default_prompt_manager

