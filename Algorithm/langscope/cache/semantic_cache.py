"""
Semantic Prompt Cache for LangScope.

Vector-based cache using Qdrant for semantic similarity search.
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np

from langscope.prompt.constants import DOMAIN_THRESHOLDS, DEFAULT_THRESHOLD

if TYPE_CHECKING:
    from langscope.prompt.classifier import ClassificationResult

logger = logging.getLogger(__name__)


class SemanticPromptCache:
    """
    Semantic cache for prompt responses using Qdrant.
    
    Uses vector similarity search with domain-specific thresholds.
    """
    
    COLLECTION_NAME = "prompt_cache"
    VECTOR_DIM = 384  # MiniLM dimension
    
    def __init__(
        self,
        qdrant_url: str = ":memory:",
        encoder=None,
    ):
        """
        Initialize semantic cache.
        
        Args:
            qdrant_url: Qdrant server URL or ":memory:" for in-memory
            encoder: Sentence transformer encoder (optional)
        """
        self._qdrant = None
        self._qdrant_url = qdrant_url
        self._encoder = encoder
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize Qdrant connection and collection.
        
        Returns:
            True if initialized successfully
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            
            # Connect to Qdrant
            if self._qdrant_url == ":memory:":
                self._qdrant = QdrantClient(":memory:")
            else:
                self._qdrant = QdrantClient(url=self._qdrant_url)
            
            # Check if collection exists
            collections = self._qdrant.get_collections()
            exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)
            
            if not exists:
                # Create collection
                self._qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
            
            self._initialized = True
            return True
            
        except ImportError:
            logger.warning("qdrant-client not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return False
    
    async def get(
        self,
        prompt: str,
        domain: str,
        embedding: np.ndarray = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up semantically similar cached response.
        
        Args:
            prompt: Input prompt
            domain: Base domain for filtering
            embedding: Pre-computed embedding (optional)
        
        Returns:
            Cached response or None
        """
        if not self._initialized or not self._qdrant:
            return None
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Get embedding if not provided
            if embedding is None:
                if self._encoder is None:
                    return None
                embedding = self._encoder.encode(prompt, convert_to_numpy=True)
            
            # Get domain-specific threshold
            threshold = DOMAIN_THRESHOLDS.get(domain, DEFAULT_THRESHOLD)
            
            # Build filter
            query_filter = None
            if domain:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="base_domain",
                            match=MatchValue(value=domain),
                        )
                    ]
                )
            
            # Search
            results = self._qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                query_filter=query_filter,
                limit=1,
                score_threshold=threshold,
            )
            
            if results:
                hit = results[0]
                return {
                    "response": hit.payload.get("response"),
                    "original_prompt": hit.payload.get("prompt"),
                    "similarity": hit.score,
                    "domain_result": hit.payload.get("domain_result"),
                    "cached_at": hit.payload.get("created_at"),
                    "model_id": hit.payload.get("model_id"),
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Semantic cache lookup failed: {e}")
            return None
    
    async def set(
        self,
        prompt: str,
        embedding: np.ndarray,
        domain_result: 'ClassificationResult',
        response: Dict[str, Any],
        model_id: str = None,
        metadata: Dict = None,
    ) -> Optional[str]:
        """
        Cache prompt and response.
        
        Args:
            prompt: Original prompt
            embedding: Prompt embedding vector
            domain_result: Classification result
            response: Response to cache
            model_id: Model that generated response
            metadata: Additional metadata
        
        Returns:
            Cache entry ID or None
        """
        if not self._initialized or not self._qdrant:
            return None
        
        try:
            from qdrant_client.models import PointStruct
            
            # Generate unique ID from prompt hash
            cache_id = hashlib.md5(prompt.encode()).hexdigest()
            
            # Build payload
            payload = {
                "prompt": prompt,
                "prompt_hash": cache_id,
                "category": domain_result.category,
                "base_domain": domain_result.base_domain,
                "variant": domain_result.variant,
                "domain_result": domain_result.to_dict(),
                "response": response,
                "model_id": model_id,
                "created_at": datetime.utcnow().isoformat(),
                "hit_count": 0,
            }
            
            if metadata:
                payload.update(metadata)
            
            # Upsert point
            self._qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=cache_id,
                        vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        payload=payload,
                    )
                ],
            )
            
            return cache_id
            
        except Exception as e:
            logger.error(f"Semantic cache set failed: {e}")
            return None
    
    async def invalidate_domain(self, domain: str) -> int:
        """
        Invalidate all cache entries for a domain.
        
        Args:
            domain: Domain to invalidate
        
        Returns:
            Number of entries invalidated
        """
        if not self._initialized or not self._qdrant:
            return 0
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Scroll to find all matching points
            results, _ = self._qdrant.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="base_domain",
                            match=MatchValue(value=domain),
                        )
                    ]
                ),
                limit=10000,
            )
            
            if results:
                ids = [p.id for p in results]
                self._qdrant.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=ids,
                )
                return len(ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Semantic cache invalidation failed: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized or not self._qdrant:
            return {"initialized": False}
        
        try:
            info = self._qdrant.get_collection(self.COLLECTION_NAME)
            return {
                "initialized": True,
                "total_entries": info.points_count,
                "vector_dimension": self.VECTOR_DIM,
                "collection_name": self.COLLECTION_NAME,
            }
        except Exception as e:
            return {"initialized": True, "error": str(e)}
    
    async def update_hit_count(self, cache_id: str) -> bool:
        """
        Increment hit count for a cache entry.
        
        Args:
            cache_id: Cache entry ID
        
        Returns:
            True if updated
        """
        if not self._initialized or not self._qdrant:
            return False
        
        try:
            # Get current point
            points = self._qdrant.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[cache_id],
            )
            
            if points:
                point = points[0]
                hit_count = point.payload.get("hit_count", 0) + 1
                
                # Update payload
                self._qdrant.set_payload(
                    collection_name=self.COLLECTION_NAME,
                    payload={
                        "hit_count": hit_count,
                        "last_hit": datetime.utcnow().isoformat(),
                    },
                    points=[cache_id],
                )
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Hit count update failed: {e}")
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

_default_semantic_cache: Optional[SemanticPromptCache] = None


async def get_semantic_cache() -> SemanticPromptCache:
    """Get or create default semantic cache."""
    global _default_semantic_cache
    
    if _default_semantic_cache is None:
        _default_semantic_cache = SemanticPromptCache()
        await _default_semantic_cache.initialize()
    
    return _default_semantic_cache

