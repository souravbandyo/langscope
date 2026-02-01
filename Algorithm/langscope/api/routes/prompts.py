"""
Prompt Management API Routes.

Provides endpoints for prompt classification and processing.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["prompts"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ClassifyRequest(BaseModel):
    """Request for prompt classification."""
    prompt: str = Field(..., description="Input prompt to classify")


class ClassifyResponse(BaseModel):
    """Response with classification result."""
    category: str = Field(..., description="Top-level category")
    base_domain: str = Field(..., description="Base domain within category")
    variant: Optional[str] = Field(None, description="Language variant")
    confidence: float = Field(..., description="Classification confidence")
    is_ground_truth: bool = Field(..., description="Uses ground truth evaluation")
    full_domain_name: str = Field(..., description="Full domain name with variant")
    template_name: str = Field(..., description="Domain template name")


class ProcessRequest(BaseModel):
    """Request for prompt processing."""
    prompt: str = Field(..., description="Input prompt to process")
    model: Optional[str] = Field(None, description="Model ID for cache key")
    domain: Optional[str] = Field(None, description="Override domain")
    skip_cache: bool = Field(False, description="Skip cache lookup")


class ProcessResponse(BaseModel):
    """Response with processing result."""
    prompt: str
    domain: ClassifyResponse
    cache_hit: bool = Field(..., description="Whether cache was hit")
    cache_layer: Optional[str] = Field(None, description="Cache layer that hit")
    cache_similarity: Optional[float] = Field(None, description="Semantic similarity score")
    evaluation_type: str = Field(..., description="'subjective' or 'ground_truth'")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cached_response: Optional[Dict[str, Any]] = Field(None, description="Cached response if hit")


class CacheResponseRequest(BaseModel):
    """Request to cache a response."""
    prompt: str = Field(..., description="Original prompt")
    domain: str = Field(..., description="Domain name")
    response: Dict[str, Any] = Field(..., description="Response to cache")
    model_id: Optional[str] = Field(None, description="Model that generated response")


class MetricsResponse(BaseModel):
    """Response with prompt manager metrics."""
    exact_hits: int
    semantic_hits: int
    misses: int
    classifications: int
    total_requests: int
    exact_hit_rate: float
    semantic_hit_rate: float
    overall_hit_rate: float
    avg_time_ms: float


# =============================================================================
# Dependencies
# =============================================================================

async def get_prompt_manager():
    """Get prompt manager instance."""
    from langscope.api.dependencies import get_prompt_manager as _get_pm
    return await _get_pm()


# =============================================================================
# Routes
# =============================================================================

@router.post("/classify", response_model=ClassifyResponse)
async def classify_prompt(request: ClassifyRequest):
    """
    Classify a prompt into domain hierarchy.
    
    Returns category, base domain, and language variant.
    """
    try:
        prompt_manager = await get_prompt_manager()
        result = prompt_manager.classify_prompt(request.prompt)
        
        return ClassifyResponse(
            category=result.category,
            base_domain=result.base_domain,
            variant=result.variant,
            confidence=result.confidence,
            is_ground_truth=result.is_ground_truth,
            full_domain_name=result.full_domain_name,
            template_name=result.template_name,
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=ProcessResponse)
async def process_prompt(request: ProcessRequest):
    """
    Process a prompt with classification and cache lookup.
    
    1. Check exact match cache
    2. Classify prompt
    3. Check semantic cache
    4. Return result with routing info
    """
    try:
        prompt_manager = await get_prompt_manager()
        result = await prompt_manager.process_prompt(
            prompt=request.prompt,
            model=request.model,
            user_domain=request.domain,
            skip_cache=request.skip_cache,
        )
        
        return ProcessResponse(
            prompt=result.prompt,
            domain=ClassifyResponse(
                category=result.domain.category,
                base_domain=result.domain.base_domain,
                variant=result.domain.variant,
                confidence=result.domain.confidence,
                is_ground_truth=result.domain.is_ground_truth,
                full_domain_name=result.domain.full_domain_name,
                template_name=result.domain.template_name,
            ),
            cache_hit=result.cache_hit,
            cache_layer=result.cache_layer,
            cache_similarity=result.cache_similarity,
            evaluation_type=result.evaluation_type,
            processing_time_ms=result.processing_time_ms,
            cached_response=result.cached_response,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache")
async def cache_response(request: CacheResponseRequest):
    """
    Cache a response for future requests.
    
    Stores in both exact match and semantic caches.
    """
    try:
        prompt_manager = await get_prompt_manager()
        
        # First classify to get domain result
        domain_result = prompt_manager.classify_prompt(request.prompt)
        
        # Cache with embedding
        embedding = domain_result.embedding
        success = await prompt_manager.cache_response(
            prompt=request.prompt,
            embedding=embedding,
            domain_result=domain_result,
            response=request.response,
            model_id=request.model_id,
        )
        
        return {"success": success, "domain": domain_result.full_domain_name}
    except Exception as e:
        logger.error(f"Cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get prompt processing metrics.
    
    Returns cache hit rates and timing statistics.
    """
    try:
        prompt_manager = await get_prompt_manager()
        metrics = prompt_manager.get_metrics()
        
        return MetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset prompt processing metrics."""
    try:
        prompt_manager = await get_prompt_manager()
        prompt_manager.reset_metrics()
        return {"status": "ok", "message": "Metrics reset"}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains")
async def list_domains():
    """
    List all available domain categories and domains.
    
    Returns the domain hierarchy.
    """
    from langscope.prompt.constants import CATEGORIES, GROUND_TRUTH_DOMAINS
    
    return {
        "categories": CATEGORIES,
        "ground_truth_domains": list(GROUND_TRUTH_DOMAINS),
    }


@router.get("/languages")
async def list_languages():
    """
    List supported language patterns.
    
    Returns language codes and their detection patterns.
    """
    from langscope.prompt.constants import LANGUAGE_PATTERNS
    
    return {
        "languages": list(LANGUAGE_PATTERNS.keys()),
        "patterns": LANGUAGE_PATTERNS,
    }

