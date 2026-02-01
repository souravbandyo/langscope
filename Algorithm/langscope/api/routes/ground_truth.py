"""
Ground Truth API Routes.

REST endpoints for ground truth evaluation:
- Get samples and evaluation results
- Trigger evaluations
- Get leaderboards and analytics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field

from langscope.api.dependencies import get_db, get_llm_caller
from langscope.ground_truth.metrics import MetricRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ground-truth", tags=["ground-truth"])


# =============================================================================
# Request/Response Models
# =============================================================================

class GroundTruthDomainInfo(BaseModel):
    """Domain information."""
    name: str
    category: str
    primary_metric: str
    metrics: List[str]
    evaluation_mode: str
    sample_count: int = 0


class SampleResponse(BaseModel):
    """Ground truth sample response."""
    sample_id: str
    domain: str
    category: str
    difficulty: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    usage_count: int = 0


class MatchResponse(BaseModel):
    """Ground truth match response."""
    match_id: str
    domain: str
    timestamp: str
    sample_id: str
    participants: List[str]
    rankings: Dict[str, int]
    status: str


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    rank: int
    deployment_id: str
    trueskill_mu: float
    trueskill_sigma: float
    primary_metric_avg: float
    total_evaluations: int


class EvaluationRequest(BaseModel):
    """Request to run an evaluation."""
    domain: str
    model_ids: List[str] = Field(default_factory=list)
    sample_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class HeatmapResponse(BaseModel):
    """Needle heatmap response."""
    model_id: str
    heatmap: Dict[str, Dict[str, float]]


# =============================================================================
# Domain Endpoints
# =============================================================================

@router.get("/domains", response_model=List[GroundTruthDomainInfo])
async def list_ground_truth_domains(db=Depends(get_db)):
    """List all ground truth domains with configuration."""
    registry = MetricRegistry()
    
    domains = []
    for domain_name in registry.DOMAIN_METRICS.keys():
        # Get sample count from DB
        sample_count = 0
        if db and db.connected:
            sample_count = db.get_ground_truth_sample_count(domain_name)
        
        # Determine category
        if domain_name in ("asr", "tts", "visual_qa", "document_extraction", 
                          "image_captioning", "ocr"):
            category = "multimodal"
        else:
            category = "long_context"
        
        # Determine evaluation mode
        eval_mode = "metrics_only"
        if domain_name in ("tts", "visual_qa", "long_document_qa", "long_summarization"):
            eval_mode = "hybrid"
        
        domains.append(GroundTruthDomainInfo(
            name=domain_name,
            category=category,
            primary_metric=registry.PRIMARY_METRIC.get(domain_name, ""),
            metrics=registry.DOMAIN_METRICS.get(domain_name, []),
            evaluation_mode=eval_mode,
            sample_count=sample_count,
        ))
    
    return domains


@router.get("/domains/{domain}/info")
async def get_domain_info(
    domain: str = Path(..., description="Domain name"),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed information about a ground truth domain."""
    registry = MetricRegistry()
    
    if domain not in registry.DOMAIN_METRICS:
        raise HTTPException(status_code=404, detail=f"Domain {domain} not found")
    
    sample_count = 0
    match_count = 0
    coverage = None
    
    if db and db.connected:
        sample_count = db.get_ground_truth_sample_count(domain)
        match_count = db.get_ground_truth_match_count(domain)
        coverage = db.get_ground_truth_coverage(domain)
    
    return {
        "domain": domain,
        "primary_metric": registry.PRIMARY_METRIC.get(domain),
        "metrics": registry.DOMAIN_METRICS.get(domain, []),
        "higher_is_better": {
            m: registry.HIGHER_IS_BETTER.get(m, True)
            for m in registry.DOMAIN_METRICS.get(domain, [])
        },
        "sample_count": sample_count,
        "match_count": match_count,
        "coverage": coverage,
    }


# =============================================================================
# Sample Endpoints
# =============================================================================

@router.get("/samples", response_model=List[SampleResponse])
async def list_samples(
    domain: str = Query(..., description="Domain name"),
    difficulty: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_db)
):
    """List ground truth samples."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    samples = db.get_ground_truth_samples(
        domain=domain,
        difficulty=difficulty,
        limit=limit
    )
    
    return [
        SampleResponse(
            sample_id=s.get("_id", ""),
            domain=s.get("domain", ""),
            category=s.get("category", ""),
            difficulty=s.get("difficulty", "medium"),
            metadata=s.get("metadata", {}),
            usage_count=s.get("usage_count", 0),
        )
        for s in samples
    ]


@router.get("/samples/{sample_id}")
async def get_sample(
    sample_id: str = Path(...),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get a specific sample (without ground truth for security)."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    sample = db.get_ground_truth_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    # Remove ground truth from response
    result = dict(sample)
    result.pop("ground_truth", None)
    
    return result


# =============================================================================
# Random Sample Endpoints
# =============================================================================

@router.get("/samples/random")
async def get_random_sample(
    domain: str = Query(..., description="Domain name"),
    difficulty: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get a random ground truth sample for evaluation."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    filters = {}
    if difficulty:
        filters["difficulty"] = difficulty
    if language:
        filters["metadata.language"] = language
    
    sample = db.get_random_ground_truth_sample(domain, filters)
    if not sample:
        raise HTTPException(
            status_code=404, 
            detail=f"No samples found for domain {domain} with given filters"
        )
    
    # Remove ground truth from response for security
    result = dict(sample)
    result.pop("ground_truth", None)
    
    return result


@router.post("/samples/batch")
async def get_batch_samples(
    domain: str = Query(..., description="Domain name"),
    count: int = Query(5, ge=1, le=20, description="Number of samples"),
    stratification: Dict[str, Any] = None,
    db=Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get multiple random samples with optional stratification.
    
    Stratification allows requesting samples with specific distribution,
    e.g., {"difficulty": {"easy": 2, "medium": 2, "hard": 1}}
    """
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    from langscope.ground_truth.sampling import StratifiedSampler
    
    sampler = StratifiedSampler(db=db)
    samples = sampler.get_stratified_batch(
        domain=domain,
        count=count,
        stratification=stratification or {}
    )
    
    # Remove ground truth from all samples
    results = []
    for sample in samples:
        result = dict(sample)
        result.pop("ground_truth", None)
        results.append(result)
    
    return results


# =============================================================================
# Match/Evaluation Endpoints
# =============================================================================

@router.get("/matches", response_model=List[MatchResponse])
async def list_matches(
    domain: str = Query(..., description="Domain name"),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db=Depends(get_db)
):
    """List ground truth matches."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    matches = db.get_ground_truth_matches(
        domain=domain,
        limit=limit,
        skip=skip
    )
    
    return [
        MatchResponse(
            match_id=m.get("_id", ""),
            domain=m.get("domain", ""),
            timestamp=m.get("timestamp", ""),
            sample_id=m.get("sample_id", ""),
            participants=m.get("participants", []),
            rankings=m.get("rankings", {}),
            status=m.get("status", "completed"),
        )
        for m in matches
    ]


@router.get("/matches/{match_id}")
async def get_match(
    match_id: str = Path(...),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get a specific match result."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    match = db.get_ground_truth_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    return match


@router.post("/evaluate")
async def trigger_evaluation(
    request: EvaluationRequest,
    db=Depends(get_db),
    llm_caller=Depends(get_llm_caller)
) -> Dict[str, Any]:
    """Trigger a ground truth evaluation."""
    from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
    
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get models
    models = []
    if request.model_ids:
        for model_id in request.model_ids:
            deployment = db.get_deployment(model_id)
            if deployment:
                # Convert to LLMModel (simplified)
                from langscope.core.model import LLMModel
                model = LLMModel.from_dict(deployment)
                models.append(model)
    
    if len(models) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 valid model IDs required"
        )
    
    # Run evaluation
    workflow = GroundTruthMatchWorkflow(
        domain=request.domain,
        models=models,
        db=db,
        llm_caller=llm_caller
    )
    
    result = await workflow.run_single_match(
        model_ids=request.model_ids,
        sample_id=request.sample_id,
        filters=request.filters
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Evaluation failed")
    
    return result.to_dict()


# =============================================================================
# Leaderboard Endpoints
# =============================================================================

@router.get("/leaderboards/{domain}", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    domain: str = Path(..., description="Domain name"),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_db)
):
    """Get ground truth leaderboard for a domain."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    ratings = db.get_ground_truth_leaderboard(domain, limit)
    
    registry = MetricRegistry()
    primary_metric = registry.PRIMARY_METRIC.get(domain, "accuracy")
    
    entries = []
    for i, r in enumerate(ratings):
        entries.append(LeaderboardEntry(
            rank=i + 1,
            deployment_id=r.get("deployment_id", ""),
            trueskill_mu=r.get("trueskill", {}).get("mu", 25.0),
            trueskill_sigma=r.get("trueskill", {}).get("sigma", 8.33),
            primary_metric_avg=r.get("metric_averages", {}).get(primary_metric, 0.0),
            total_evaluations=r.get("statistics", {}).get("total_evaluations", 0),
        ))
    
    return entries


@router.get("/leaderboards/{domain}/{language}", response_model=List[LeaderboardEntry])
async def get_language_specific_leaderboard(
    domain: str = Path(..., description="Domain name"),
    language: str = Path(..., description="Language code (e.g., 'en', 'hi', 'bn')"),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_db)
):
    """
    Get language-specific ground truth leaderboard.
    
    Useful for multilingual domains like ASR, TTS where models 
    may perform differently across languages.
    """
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Query ratings filtered by language
    ratings = db.get_ground_truth_leaderboard_by_language(domain, language, limit)
    
    registry = MetricRegistry()
    primary_metric = registry.PRIMARY_METRIC.get(domain, "accuracy")
    
    entries = []
    for i, r in enumerate(ratings):
        entries.append(LeaderboardEntry(
            rank=i + 1,
            deployment_id=r.get("deployment_id", ""),
            trueskill_mu=r.get("trueskill", {}).get("mu", 25.0),
            trueskill_sigma=r.get("trueskill", {}).get("sigma", 8.33),
            primary_metric_avg=r.get("metric_averages", {}).get(primary_metric, 0.0),
            total_evaluations=r.get("statistics", {}).get("total_evaluations", 0),
        ))
    
    return entries


# =============================================================================
# Analytics Endpoints
# =============================================================================

@router.get("/analytics/needle-heatmap/{model_id}", response_model=HeatmapResponse)
async def get_needle_heatmap(
    model_id: str = Path(...),
    db=Depends(get_db)
):
    """Get needle in haystack accuracy heatmap for a model."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    from langscope.ground_truth.analytics import compute_accuracy_heatmap
    
    heatmap = compute_accuracy_heatmap(db, model_id, "needle_in_haystack")
    
    return HeatmapResponse(
        model_id=model_id,
        heatmap=heatmap
    )


@router.get("/analytics/model-performance/{model_id}")
async def get_model_performance(
    model_id: str = Path(...),
    domain: str = Query(...),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get performance summary for a model in a domain."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get rating
    rating = db.get_ground_truth_rating(model_id, domain)
    if not rating:
        raise HTTPException(status_code=404, detail="No data for this model/domain")
    
    # Get recent matches
    matches = db.get_ground_truth_matches(
        domain=domain,
        deployment_id=model_id,
        limit=10
    )
    
    return {
        "model_id": model_id,
        "domain": domain,
        "trueskill": rating.get("trueskill", {}),
        "statistics": rating.get("statistics", {}),
        "metric_averages": rating.get("metric_averages", {}),
        "recent_matches": [m.get("_id") for m in matches],
    }


@router.get("/coverage/{domain}")
async def get_domain_coverage(
    domain: str = Path(...),
    db=Depends(get_db)
) -> Dict[str, Any]:
    """Get sample coverage statistics for a domain."""
    if not db or not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    coverage = db.compute_ground_truth_coverage(domain)
    
    return coverage

