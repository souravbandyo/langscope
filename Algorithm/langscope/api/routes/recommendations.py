"""
Use-case recommendation API endpoints.

Provides personalized model recommendations based on use-case feedback patterns.
"""

from typing import Optional
from fastapi import APIRouter, Query

from langscope.api.schemas import (
    RecommendationQuery,
    RecommendationEntry,
    RecommendationResponse,
)
from langscope.api.dependencies import (
    get_models,
    get_model_by_id,
    get_use_case_manager,
)


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get(
    "/{use_case}",
    response_model=RecommendationResponse,
    summary="Get use-case recommendations",
    description="Get model recommendations adjusted for a specific use case."
)
async def get_recommendations(
    use_case: str,
    domain: Optional[str] = Query(None, description="Filter by domain"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get model recommendations for a specific use case.
    
    The recommendations are adjusted based on aggregate user feedback:
    - Models that users with similar use cases preferred get boosted
    - Models that underperformed for similar users get penalized
    
    Formula:
        μᵢ^use-case = μᵢ^global + β × Δ̄ᵢ^use-case
    
    Where β increases with more user data (β = n_users / (n_users + τ))
    """
    models = get_models(domain=domain)
    
    if not models:
        return RecommendationResponse(
            use_case=use_case,
            domain=domain,
            beta=0.0,
            n_users=0,
            recommendations=[]
        )
    
    # Get use case manager for the domain
    manager = get_use_case_manager(domain or "global")
    
    # Get adjusted rankings
    model_mus = {}
    for model in models:
        if domain and domain in model.trueskill_by_domain:
            mu = model.trueskill_by_domain[domain].raw.mu
        else:
            mu = model.trueskill.raw.mu
        model_mus[model.model_id] = mu
    
    adjusted_ranking = manager.get_adjusted_ranking(use_case, model_mus)
    
    # Build recommendations
    recommendations = []
    for model_id, adjusted_mu in adjusted_ranking[:top_k]:
        model = get_model_by_id(model_id)
        if not model:
            continue
        
        global_mu = model_mus[model_id]
        adjustment = adjusted_mu - global_mu
        
        recommendations.append(RecommendationEntry(
            model_id=model_id,
            model_name=model.name,
            adjusted_mu=adjusted_mu,
            global_mu=global_mu,
            adjustment=adjustment,
            provider=model.provider
        ))
    
    return RecommendationResponse(
        use_case=use_case,
        domain=domain,
        beta=manager.get_beta(use_case),
        n_users=manager.get_user_count(use_case),
        recommendations=recommendations
    )


@router.get(
    "/profile/{use_case}",
    summary="Get use-case profile",
    description="Get detailed profile of a use case including feedback statistics."
)
async def get_use_case_profile(
    use_case: str,
    domain: Optional[str] = Query(None)
):
    """
    Get detailed profile for a use case.
    
    Shows statistics about how users with this use case have
    rated different models.
    """
    manager = get_use_case_manager(domain or "global")
    
    profile = manager.get_profile_summary(use_case)
    
    if not profile:
        return {
            "use_case": use_case,
            "domain": domain,
            "n_users": 0,
            "beta": 0.0,
            "message": "No feedback data for this use case yet"
        }
    
    # Enrich with model names
    top_boosted = []
    for model_id, adjustment in profile.get("top_boosted", []):
        model = get_model_by_id(model_id)
        top_boosted.append({
            "model_id": model_id,
            "model_name": model.name if model else model_id,
            "adjustment": adjustment
        })
    
    top_penalized = []
    for model_id, adjustment in profile.get("top_penalized", []):
        model = get_model_by_id(model_id)
        top_penalized.append({
            "model_id": model_id,
            "model_name": model.name if model else model_id,
            "adjustment": adjustment
        })
    
    return {
        "use_case": use_case,
        "domain": domain,
        "n_users": profile["n_users"],
        "beta": profile["beta"],
        "n_models_with_data": profile["n_models"],
        "top_boosted_models": top_boosted,
        "top_penalized_models": top_penalized
    }


@router.get(
    "",
    summary="List use cases",
    description="List all known use cases with feedback data."
)
async def list_use_cases(domain: Optional[str] = Query(None)):
    """List all use cases with feedback data."""
    manager = get_use_case_manager(domain or "global")
    
    use_cases = manager.list_use_cases()
    
    summaries = []
    for use_case in use_cases:
        profile = manager.get_profile_summary(use_case)
        if profile:
            summaries.append({
                "use_case": use_case,
                "n_users": profile["n_users"],
                "beta": profile["beta"],
                "n_models": profile["n_models"]
            })
    
    return {
        "domain": domain,
        "use_cases": summaries,
        "total": len(summaries)
    }


@router.get(
    "/model/{model_id}",
    summary="Get model adjustments across use cases",
    description="See how a model is adjusted across different use cases."
)
async def get_model_use_case_adjustments(
    model_id: str,
    domain: Optional[str] = Query(None)
):
    """
    Get use-case adjustments for a specific model.
    
    Shows how this model's rating is adjusted for different use cases.
    """
    model = get_model_by_id(model_id)
    
    if not model:
        return {
            "model_id": model_id,
            "adjustments": {},
            "message": "Model not found"
        }
    
    # Get global mu
    if domain and domain in model.trueskill_by_domain:
        global_mu = model.trueskill_by_domain[domain].raw.mu
    else:
        global_mu = model.trueskill.raw.mu
    
    manager = get_use_case_manager(domain or "global")
    
    adjustments = {}
    for use_case in manager.list_use_cases():
        adjustment = manager.get_adjustment(use_case, model_id)
        if adjustment != 0:  # Only include if there's actual adjustment
            adjustments[use_case] = {
                "adjustment": adjustment,
                "adjusted_mu": global_mu + adjustment,
                "beta": manager.get_beta(use_case)
            }
    
    return {
        "model_id": model_id,
        "model_name": model.name,
        "global_mu": global_mu,
        "domain": domain,
        "adjustments": adjustments
    }


@router.get(
    "/compare",
    summary="Compare recommendations for different use cases",
    description="Compare how the same models rank for different use cases."
)
async def compare_use_case_recommendations(
    use_cases: str = Query(..., description="Comma-separated use cases"),
    domain: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=20)
):
    """
    Compare recommendations across use cases.
    
    Shows how the top models differ for different use cases.
    """
    use_case_list = [uc.strip() for uc in use_cases.split(",")]
    
    models = get_models(domain=domain)
    manager = get_use_case_manager(domain or "global")
    
    # Get global ratings
    model_mus = {}
    for model in models:
        if domain and domain in model.trueskill_by_domain:
            mu = model.trueskill_by_domain[domain].raw.mu
        else:
            mu = model.trueskill.raw.mu
        model_mus[model.model_id] = mu
    
    comparisons = {}
    for use_case in use_case_list:
        ranking = manager.get_adjusted_ranking(use_case, model_mus)
        
        top_models = []
        for model_id, adjusted_mu in ranking[:top_k]:
            model = get_model_by_id(model_id)
            if model:
                top_models.append({
                    "model_id": model_id,
                    "model_name": model.name,
                    "adjusted_mu": adjusted_mu,
                    "adjustment": adjusted_mu - model_mus[model_id]
                })
        
        comparisons[use_case] = {
            "top_models": top_models,
            "n_users": manager.get_user_count(use_case),
            "beta": manager.get_beta(use_case)
        }
    
    return {
        "domain": domain,
        "comparisons": comparisons
    }
