"""
Leaderboard API endpoints.

Provides endpoints for retrieving model rankings.
Supports 10-dimensional rankings.
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from langscope.api.schemas import (
    LeaderboardEntry,
    LeaderboardResponse,
    MultiDimensionalLeaderboardEntry,
    MultiDimensionalLeaderboardResponse,
)
from langscope.api.dependencies import get_db, get_models
from langscope.core.dimensions import Dimension
from langscope.core.rating import DIMENSION_NAMES
from langscope.federation.router import (
    list_ground_truth_domains,
    list_subjective_domains,
    is_ground_truth_domain,
)
from langscope.domain.domain_config import DOMAIN_TEMPLATES


router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


# Valid dimension values for validation
VALID_DIMENSIONS = [d.value for d in Dimension] + ["raw", "cost_adjusted"]


@router.get(
    "",
    response_model=LeaderboardResponse,
    summary="Get global leaderboard",
    description="Get the global model leaderboard. Supports 10 dimensions."
)
async def get_global_leaderboard(
    dimension: str = Query(
        "raw_quality",
        description="Dimension to rank by (raw_quality, cost_adjusted, latency, ttft, consistency, token_efficiency, instruction_following, hallucination_resistance, long_context, combined)"
    ),
    limit: int = Query(50, ge=1, le=100)
):
    """Get global leaderboard (across all domains)."""
    # Handle legacy 'raw' and 'cost_adjusted' values
    if dimension == "raw":
        dimension = "raw_quality"
    elif dimension == "cost_adjusted" and dimension not in DIMENSION_NAMES:
        dimension = "cost_adjusted"
    
    if dimension not in DIMENSION_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension: {dimension}. Valid dimensions: {DIMENSION_NAMES}"
        )
    
    models = get_models()
    
    # Sort by specified dimension using MultiDimensionalTrueSkill
    sorted_models = sorted(
        models,
        key=lambda m: m.multi_trueskill.get_dimension(dimension).mu,
        reverse=True
    )
    
    entries = []
    for rank, model in enumerate(sorted_models[:limit], 1):
        ts = model.multi_trueskill.get_dimension(dimension)
        avg_rank = model.performance.avg_rank_by_dimension.get(dimension, 0.0)
        
        entries.append(LeaderboardEntry(
            rank=rank,
            name=model.name,
            model_id=model.model_id,
            provider=model.provider,
            mu=ts.mu,
            sigma=ts.sigma,
            conservative_estimate=ts.conservative_estimate(),
            matches_played=model.performance.total_matches_played,
            avg_rank=avg_rank,
            dimension=dimension
        ))
    
    return LeaderboardResponse(
        domain=None,
        ranking_type=dimension,
        entries=entries,
        total=len(entries),
        generated_at=datetime.utcnow().isoformat() + "Z"
    )


@router.get(
    "/domain/{domain}",
    response_model=LeaderboardResponse,
    summary="Get domain leaderboard",
    description="Get the leaderboard for a specific domain. Supports 10 dimensions."
)
async def get_domain_leaderboard(
    domain: str,
    dimension: str = Query(
        "raw_quality",
        description="Dimension to rank by"
    ),
    limit: int = Query(50, ge=1, le=100)
):
    """Get leaderboard for a specific domain."""
    # Handle legacy values
    if dimension == "raw":
        dimension = "raw_quality"
    
    if dimension not in DIMENSION_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension: {dimension}. Valid dimensions: {DIMENSION_NAMES}"
        )
    
    models = get_models(domain=domain)
    
    if not models:
        return LeaderboardResponse(
            domain=domain,
            ranking_type=dimension,
            entries=[],
            total=0,
            generated_at=datetime.utcnow().isoformat() + "Z"
        )
    
    # Sort by domain-specific dimension rating
    def get_dim_mu(m):
        multi_ts = m.multi_trueskill_by_domain.get(domain, m.multi_trueskill)
        return multi_ts.get_dimension(dimension).mu
    
    sorted_models = sorted(models, key=get_dim_mu, reverse=True)
    
    entries = []
    for rank, model in enumerate(sorted_models[:limit], 1):
        multi_ts = model.multi_trueskill_by_domain.get(domain, model.multi_trueskill)
        domain_perf = model.performance_by_domain.get(domain, model.performance)
        
        ts = multi_ts.get_dimension(dimension)
        avg_rank = domain_perf.avg_rank_by_dimension.get(dimension, 0.0)
        
        entries.append(LeaderboardEntry(
            rank=rank,
            name=model.name,
            model_id=model.model_id,
            provider=model.provider,
            mu=ts.mu,
            sigma=ts.sigma,
            conservative_estimate=ts.conservative_estimate(),
            matches_played=domain_perf.total_matches_played,
            avg_rank=avg_rank,
            dimension=dimension
        ))
    
    return LeaderboardResponse(
        domain=domain,
        ranking_type=dimension,
        entries=entries,
        total=len(entries),
        generated_at=datetime.utcnow().isoformat() + "Z"
    )


@router.get(
    "/multi-dimensional",
    response_model=MultiDimensionalLeaderboardResponse,
    summary="Get multi-dimensional leaderboard",
    description="Get leaderboard with all 10 dimension ratings for each model."
)
async def get_multi_dimensional_leaderboard(
    domain: Optional[str] = Query(None, description="Optional domain filter"),
    sort_by: str = Query("combined", description="Dimension to sort by"),
    limit: int = Query(50, ge=1, le=100)
):
    """Get leaderboard with all dimension ratings."""
    if sort_by not in DIMENSION_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by dimension: {sort_by}"
        )
    
    models = get_models(domain=domain) if domain else get_models()
    
    if not models:
        return MultiDimensionalLeaderboardResponse(
            domain=domain,
            sort_by=sort_by,
            entries=[],
            total=0,
            generated_at=datetime.utcnow().isoformat() + "Z"
        )
    
    # Sort by specified dimension
    def get_dim_mu(m):
        multi_ts = m.multi_trueskill_by_domain.get(domain, m.multi_trueskill) if domain else m.multi_trueskill
        return multi_ts.get_dimension(sort_by).mu
    
    sorted_models = sorted(models, key=get_dim_mu, reverse=True)
    
    entries = []
    for rank, model in enumerate(sorted_models[:limit], 1):
        multi_ts = model.multi_trueskill_by_domain.get(domain, model.multi_trueskill) if domain else model.multi_trueskill
        perf = model.performance_by_domain.get(domain, model.performance) if domain else model.performance
        
        dimension_ratings = {}
        for dim in DIMENSION_NAMES:
            ts = multi_ts.get_dimension(dim)
            dimension_ratings[dim] = {
                "mu": ts.mu,
                "sigma": ts.sigma,
                "conservative_estimate": ts.conservative_estimate()
            }
        
        entries.append(MultiDimensionalLeaderboardEntry(
            rank=rank,
            name=model.name,
            model_id=model.model_id,
            provider=model.provider,
            dimension_ratings=dimension_ratings,
            matches_played=perf.total_matches_played,
        ))
    
    return MultiDimensionalLeaderboardResponse(
        domain=domain,
        sort_by=sort_by,
        entries=entries,
        total=len(entries),
        generated_at=datetime.utcnow().isoformat() + "Z"
    )


@router.get(
    "/model/{model_id}",
    summary="Get model rankings across domains",
    description="Get a model's ranking position across all domains and dimensions."
)
async def get_model_rankings(
    model_id: str,
    dimension: str = Query("raw_quality", description="Dimension to rank by")
):
    """Get a model's ranking position across all domains."""
    from langscope.api.dependencies import get_model_by_id
    
    # Handle legacy values
    if dimension == "raw":
        dimension = "raw_quality"
    
    if dimension not in DIMENSION_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension: {dimension}"
        )
    
    model = get_model_by_id(model_id)
    if not model:
        return {"model_id": model_id, "rankings": {}}
    
    rankings = {}
    
    # Global ranking
    all_models = get_models()
    sorted_all = sorted(
        all_models,
        key=lambda m: -m.multi_trueskill.get_dimension(dimension).mu
    )
    
    global_rank = next(
        (i + 1 for i, m in enumerate(sorted_all) if m.model_id == model_id),
        None
    )
    rankings["global"] = {
        "rank": global_rank,
        "total": len(all_models)
    }
    
    # Per-domain rankings
    for domain in model.multi_trueskill_by_domain:
        domain_models = get_models(domain=domain)
        
        sorted_domain = sorted(
            domain_models,
            key=lambda m: -m.multi_trueskill_by_domain.get(domain, m.multi_trueskill).get_dimension(dimension).mu
        )
        
        domain_rank = next(
            (i + 1 for i, m in enumerate(sorted_domain) if m.model_id == model_id),
            None
        )
        rankings[domain] = {
            "rank": domain_rank,
            "total": len(domain_models)
        }
    
    # Also include all dimension ratings for this model
    dimension_ratings = {}
    for dim in DIMENSION_NAMES:
        ts = model.multi_trueskill.get_dimension(dim)
        dimension_ratings[dim] = {
            "mu": ts.mu,
            "sigma": ts.sigma,
            "conservative_estimate": ts.conservative_estimate()
        }
    
    return {
        "model_id": model_id,
        "model_name": model.name,
        "dimension": dimension,
        "rankings": rankings,
        "dimension_ratings": dimension_ratings
    }


@router.get(
    "/compare",
    summary="Compare models",
    description="Compare rankings of multiple models across dimensions."
)
async def compare_models(
    model_ids: str = Query(..., description="Comma-separated model IDs"),
    domain: Optional[str] = Query(None),
    dimension: str = Query("raw_quality", description="Dimension to compare")
):
    """Compare rankings of multiple models."""
    from langscope.api.dependencies import get_model_by_id
    
    # Handle legacy values
    if dimension == "raw":
        dimension = "raw_quality"
    
    if dimension not in DIMENSION_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension: {dimension}"
        )
    
    ids = [m.strip() for m in model_ids.split(",")]
    
    comparison = []
    for model_id in ids:
        model = get_model_by_id(model_id)
        if not model:
            continue
        
        if domain and domain in model.multi_trueskill_by_domain:
            multi_ts = model.multi_trueskill_by_domain[domain]
        else:
            multi_ts = model.multi_trueskill
        
        rating = multi_ts.get_dimension(dimension)
        
        # Include all dimensions in comparison
        all_ratings = {
            dim: {
                "mu": multi_ts.get_dimension(dim).mu,
                "sigma": multi_ts.get_dimension(dim).sigma,
            }
            for dim in DIMENSION_NAMES
        }
        
        comparison.append({
            "model_id": model_id,
            "name": model.name,
            "provider": model.provider,
            "mu": rating.mu,
            "sigma": rating.sigma,
            "conservative_estimate": rating.conservative_estimate(),
            "all_dimensions": all_ratings
        })
    
    # Sort by the specified dimension's mu
    comparison.sort(key=lambda x: -x["mu"])
    
    # Add ranks
    for rank, entry in enumerate(comparison, 1):
        entry["rank"] = rank
    
    return {
        "domain": domain,
        "dimension": dimension,
        "comparison": comparison
    }


# =============================================================================
# Combined Leaderboard (Subjective + Ground Truth)
# =============================================================================

@router.get(
    "/combined",
    summary="Get combined leaderboard",
    description="Get a unified view across both subjective and ground truth domains."
)
async def get_combined_leaderboard(
    limit: int = Query(50, ge=1, le=100),
    include_subjective: bool = Query(True, description="Include subjective domains"),
    include_ground_truth: bool = Query(True, description="Include ground truth domains")
):
    """
    Get combined leaderboard showing performance across all domain types.
    
    Returns:
        - Subjective domains with TrueSkill ratings
        - Ground truth domains with metric-based rankings
        - Summary statistics
    """
    from langscope.api.dependencies import get_db
    
    result = {
        "subjective_domains": [],
        "ground_truth_domains": [],
        "summary": {
            "total_domains": 0,
            "subjective_count": 0,
            "ground_truth_count": 0,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # Get subjective domain leaderboards
    if include_subjective:
        subjective_domains = list_subjective_domains()
        result["summary"]["subjective_count"] = len(subjective_domains)
        
        for domain_name in subjective_domains[:limit]:
            domain_config = DOMAIN_TEMPLATES.get(domain_name)
            if not domain_config:
                continue
            
            models = get_models(domain=domain_name)
            
            # Get top 3 models by raw quality
            sorted_models = sorted(
                models,
                key=lambda m: m.multi_trueskill_by_domain.get(
                    domain_name, m.multi_trueskill
                ).get_dimension("raw_quality").mu,
                reverse=True
            )[:3]
            
            top_models = []
            for rank, model in enumerate(sorted_models, 1):
                multi_ts = model.multi_trueskill_by_domain.get(
                    domain_name, model.multi_trueskill
                )
                ts = multi_ts.get_dimension("raw_quality")
                top_models.append({
                    "rank": rank,
                    "model_id": model.model_id,
                    "name": model.name,
                    "mu": ts.mu,
                    "sigma": ts.sigma,
                })
            
            result["subjective_domains"].append({
                "domain": domain_name,
                "display_name": domain_config.display_name,
                "evaluation_type": "subjective",
                "total_models": len(models),
                "top_models": top_models,
            })
    
    # Get ground truth domain leaderboards
    if include_ground_truth:
        gt_domains = list_ground_truth_domains()
        result["summary"]["ground_truth_count"] = len(gt_domains)
        
        try:
            db = get_db()
        except RuntimeError:
            db = None
        
        for domain_name in gt_domains[:limit]:
            domain_config = DOMAIN_TEMPLATES.get(domain_name)
            if not domain_config:
                continue
            
            gt_domain = domain_config.settings.ground_truth_domain or domain_name
            primary_metric = domain_config.settings.primary_metric or "accuracy"
            
            top_models = []
            
            if db and db.connected:
                # Get GT leaderboard
                ratings = db.get_ground_truth_leaderboard(gt_domain, limit=3)
                
                for i, r in enumerate(ratings):
                    top_models.append({
                        "rank": i + 1,
                        "model_id": r.get("deployment_id", ""),
                        "name": r.get("deployment_id", ""),
                        "mu": r.get("trueskill", {}).get("mu", 25.0),
                        "primary_metric": primary_metric,
                        "primary_metric_value": r.get("metric_averages", {}).get(
                            primary_metric, 0.0
                        ),
                    })
            
            result["ground_truth_domains"].append({
                "domain": domain_name,
                "display_name": domain_config.display_name,
                "evaluation_type": "ground_truth",
                "primary_metric": primary_metric,
                "top_models": top_models,
            })
    
    result["summary"]["total_domains"] = (
        result["summary"]["subjective_count"] + 
        result["summary"]["ground_truth_count"]
    )
    
    return result


@router.get(
    "/domains",
    summary="List all leaderboard domains",
    description="Get list of all domains with their evaluation types."
)
async def list_leaderboard_domains():
    """List all domains available for leaderboard queries."""
    domains = []
    
    for name, domain in DOMAIN_TEMPLATES.items():
        domains.append({
            "name": name,
            "display_name": domain.display_name,
            "description": domain.description,
            "evaluation_type": domain.settings.evaluation_type,
            "primary_metric": domain.settings.primary_metric,
            "parent_domain": domain.parent_domain,
        })
    
    # Sort by evaluation type, then name
    domains.sort(key=lambda d: (d["evaluation_type"], d["name"]))
    
    return {
        "domains": domains,
        "total": len(domains),
        "subjective_count": len([d for d in domains if d["evaluation_type"] == "subjective"]),
        "ground_truth_count": len([d for d in domains if d["evaluation_type"] == "ground_truth"]),
    }
