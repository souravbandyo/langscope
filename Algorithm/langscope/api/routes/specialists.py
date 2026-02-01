"""
Specialist detection API endpoints.

Provides endpoints for detecting specialist and weak-spot patterns.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    SpecialistQuery,
    SpecialistResultSchema,
    SpecialistProfileResponse,
    ErrorResponse,
)
from langscope.api.dependencies import (
    get_model_by_id,
    get_models,
    get_specialist_detector,
)


router = APIRouter(prefix="/specialists", tags=["specialists"])


@router.post(
    "/detect",
    response_model=List[SpecialistResultSchema],
    responses={404: {"model": ErrorResponse}},
    summary="Detect specialists for a model",
    description="Detect specialist or weak-spot patterns for a model."
)
async def detect_specialists(query: SpecialistQuery):
    """
    Detect if a model is a specialist or has weak spots.
    
    A model is a specialist if its actual performance significantly
    exceeds predictions from other domains (z-score > 2).
    
    A weak spot is when actual performance significantly underperforms
    predictions (z-score < -2).
    """
    model = get_model_by_id(query.model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {query.model_id}"
        )
    
    detector = get_specialist_detector()
    
    if query.target_domain:
        # Check specific domain
        result = detector.detect(model, query.target_domain)
        return [SpecialistResultSchema(
            model_id=result.model_id,
            domain=result.domain,
            is_specialist=result.is_specialist,
            z_score=result.z_score,
            actual_mu=result.actual_mu,
            predicted_mu=result.predicted_mu,
            p_value=result.p_value,
            category=result.category
        )]
    else:
        # Check all domains
        results = detector.detect_all_domains(model)
        return [
            SpecialistResultSchema(
                model_id=r.model_id,
                domain=r.domain,
                is_specialist=r.is_specialist,
                z_score=r.z_score,
                actual_mu=r.actual_mu,
                predicted_mu=r.predicted_mu,
                p_value=r.p_value,
                category=r.category
            )
            for r in results
        ]


@router.get(
    "/profile/{model_id}",
    response_model=SpecialistProfileResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get specialist profile",
    description="Get comprehensive specialist profile for a model."
)
async def get_specialist_profile(model_id: str):
    """Get full specialist profile for a model."""
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    from langscope.transfer.specialist import get_model_profile
    
    profile = get_model_profile(model)
    
    return SpecialistProfileResponse(
        model_id=profile["model_id"],
        model_name=profile["model_name"],
        domains_evaluated=profile["domains_evaluated"],
        specialist_domains=profile["specialist_domains"],
        weak_spot_domains=profile["weak_spot_domains"],
        specialization_score=profile["specialization_score"],
        is_generalist=profile["is_generalist"],
        detailed_results=[
            SpecialistResultSchema(
                model_id=r["model_id"],
                domain=r["domain"],
                is_specialist=r["is_specialist"],
                z_score=r["z_score"],
                actual_mu=r["actual_mu"],
                predicted_mu=r["predicted_mu"],
                p_value=r["p_value"],
                category=r["category"]
            )
            for r in profile["detailed_results"]
        ]
    )


@router.get(
    "/domain/{domain}",
    summary="Find specialists in a domain",
    description="Find all models that are specialists in a specific domain."
)
async def find_domain_specialists(
    domain: str,
    include_weak_spots: bool = Query(False, description="Also include weak spots")
):
    """Find all specialists (and optionally weak spots) for a domain."""
    models = get_models(domain=domain)
    detector = get_specialist_detector()
    
    specialists = []
    weak_spots = []
    
    for model in models:
        result = detector.detect(model, domain)
        
        if result.category == "specialist":
            specialists.append({
                "model_id": model.model_id,
                "model_name": model.name,
                "z_score": result.z_score,
                "actual_mu": result.actual_mu,
                "predicted_mu": result.predicted_mu,
                "deviation": result.deviation
            })
        elif result.category == "weak_spot" and include_weak_spots:
            weak_spots.append({
                "model_id": model.model_id,
                "model_name": model.name,
                "z_score": result.z_score,
                "actual_mu": result.actual_mu,
                "predicted_mu": result.predicted_mu,
                "deviation": result.deviation
            })
    
    # Sort by z-score
    specialists.sort(key=lambda x: -x["z_score"])
    weak_spots.sort(key=lambda x: x["z_score"])
    
    response = {
        "domain": domain,
        "specialists": specialists,
        "n_specialists": len(specialists)
    }
    
    if include_weak_spots:
        response["weak_spots"] = weak_spots
        response["n_weak_spots"] = len(weak_spots)
    
    return response


@router.get(
    "/generalists",
    summary="Find generalist models",
    description="Find models that perform consistently across domains (not specialists)."
)
async def find_generalists(
    min_domains: int = Query(3, ge=2, description="Minimum domains evaluated")
):
    """Find generalist models (consistent performance across domains)."""
    models = get_models()
    detector = get_specialist_detector()
    
    from langscope.transfer.specialist import compute_specialization_score
    
    generalists = []
    
    for model in models:
        if len(model.trueskill_by_domain) < min_domains:
            continue
        
        results = detector.detect_all_domains(model)
        
        # A generalist has no specialist or weak spot domains
        is_generalist = all(r.category == "normal" for r in results)
        
        if is_generalist:
            spec_score = compute_specialization_score(model)
            generalists.append({
                "model_id": model.model_id,
                "model_name": model.name,
                "domains_evaluated": list(model.trueskill_by_domain.keys()),
                "n_domains": len(model.trueskill_by_domain),
                "specialization_score": spec_score,
                "avg_mu": sum(
                    ts.raw.mu for ts in model.trueskill_by_domain.values()
                ) / len(model.trueskill_by_domain)
            })
    
    # Sort by specialization score (lower = more consistent)
    generalists.sort(key=lambda x: x["specialization_score"])
    
    return {
        "min_domains": min_domains,
        "generalists": generalists,
        "n_generalists": len(generalists)
    }


@router.get(
    "/summary",
    summary="Get specialists summary",
    description="Get summary of all specialist patterns in the system."
)
async def get_specialists_summary():
    """Get overview of all specialist patterns."""
    models = get_models()
    detector = get_specialist_detector()
    
    from langscope.transfer.specialist import compute_specialization_score
    
    all_specialists = []
    all_weak_spots = []
    domain_specialist_counts = {}
    domain_weak_spot_counts = {}
    
    for model in models:
        results = detector.detect_all_domains(model)
        
        for r in results:
            if r.category == "specialist":
                all_specialists.append({
                    "model_id": model.model_id,
                    "model_name": model.name,
                    "domain": r.domain,
                    "z_score": r.z_score
                })
                domain_specialist_counts[r.domain] = domain_specialist_counts.get(r.domain, 0) + 1
            elif r.category == "weak_spot":
                all_weak_spots.append({
                    "model_id": model.model_id,
                    "model_name": model.name,
                    "domain": r.domain,
                    "z_score": r.z_score
                })
                domain_weak_spot_counts[r.domain] = domain_weak_spot_counts.get(r.domain, 0) + 1
    
    # Sort specialists by z-score
    all_specialists.sort(key=lambda x: -x["z_score"])
    all_weak_spots.sort(key=lambda x: x["z_score"])
    
    return {
        "total_specialists": len(all_specialists),
        "total_weak_spots": len(all_weak_spots),
        "top_specialists": all_specialists[:10],
        "worst_weak_spots": all_weak_spots[:10],
        "specialists_by_domain": domain_specialist_counts,
        "weak_spots_by_domain": domain_weak_spot_counts
    }
