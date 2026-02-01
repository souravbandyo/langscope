"""
Transfer learning API endpoints.

Provides endpoints for transfer learning predictions and correlation management.
Includes the new Model Rank API with faceted transfer learning.
"""

import time
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    TransferPrediction,
    TransferPredictionResult,
    CorrelationUpdate,
    CorrelationResponse,
    ErrorResponse,
    # New faceted transfer schemas
    ModelRatingSchema,
    ModelRatingsSchema,
    ModelRatingRequest,
    TrueSkillRatingSchema,
    FacetContributionSchema,
    TransferDetailsSchema,
    SimilarDomainSchema,
    SimilarDomainsResponse,
    DomainFacetsSchema,
    FacetedLeaderboardEntry,
    FacetedLeaderboardResponse,
    FacetSimilaritySchema,
    FacetPriorUpdate,
    DomainIndexStats,
    RefreshIndexResponse,
)
from langscope.api.dependencies import (
    get_model_by_id,
    get_models,
    get_transfer_learner,
    get_correlation_learner,
    get_domain_index,
    get_faceted_transfer,
    refresh_domain_index,
    get_db,
)
from langscope.core.rating import TrueSkillRating
from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0


router = APIRouter(prefix="/transfer", tags=["transfer"])


@router.post(
    "/predict",
    response_model=TransferPredictionResult,
    responses={404: {"model": ErrorResponse}},
    summary="Predict model performance",
    description="Predict a model's performance in a target domain using transfer learning."
)
async def predict_performance(prediction: TransferPrediction):
    """
    Predict model performance in a new domain.
    
    Uses multi-source transfer learning with reliability weighting.
    """
    model = get_model_by_id(prediction.model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {prediction.model_id}"
        )
    
    # Gather source ratings
    source_ratings = {}
    
    if prediction.source_domains:
        # Use specified source domains
        for domain in prediction.source_domains:
            if domain in model.trueskill_by_domain:
                source_ratings[domain] = model.trueskill_by_domain[domain].raw
    else:
        # Use all available domains except target
        for domain, ts in model.trueskill_by_domain.items():
            if domain != prediction.target_domain:
                source_ratings[domain] = ts.raw
    
    if not source_ratings:
        # No source data, return prior
        from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0
        return TransferPredictionResult(
            model_id=prediction.model_id,
            target_domain=prediction.target_domain,
            predicted_mu=TRUESKILL_MU_0,
            predicted_sigma=TRUESKILL_SIGMA_0,
            source_weights={},
            confidence=0.0
        )
    
    # Perform transfer
    transfer_learner = get_transfer_learner()
    result = transfer_learner.transfer_multi_source(
        source_ratings, prediction.target_domain
    )
    
    # Calculate confidence (inverse of relative sigma)
    from langscope.core.constants import TRUESKILL_SIGMA_0
    confidence = 1.0 - (result.sigma / TRUESKILL_SIGMA_0)
    confidence = max(0.0, min(1.0, confidence))
    
    return TransferPredictionResult(
        model_id=prediction.model_id,
        target_domain=prediction.target_domain,
        predicted_mu=result.mu,
        predicted_sigma=result.sigma,
        source_weights=result.source_weights,
        confidence=confidence
    )


@router.get(
    "/correlation/{domain_a}/{domain_b}",
    response_model=CorrelationResponse,
    summary="Get domain correlation",
    description="Get the correlation between two domains."
)
async def get_correlation(domain_a: str, domain_b: str):
    """Get correlation between two domains."""
    learner = get_correlation_learner()
    
    correlation = learner.get_correlation(domain_a, domain_b)
    n_obs = learner.get_observation_count(domain_a, domain_b)
    alpha = learner.get_alpha(domain_a, domain_b)
    
    return CorrelationResponse(
        domain_a=domain_a,
        domain_b=domain_b,
        correlation=correlation,
        n_observations=n_obs,
        alpha=alpha
    )


@router.put(
    "/correlation",
    response_model=CorrelationResponse,
    summary="Set domain correlation",
    description="Set or update the correlation between two domains."
)
async def set_correlation(update: CorrelationUpdate):
    """Set correlation between two domains."""
    learner = get_correlation_learner()
    
    if update.prior:
        # Set as prior estimate
        learner.set_prior(
            update.domain_a,
            update.domain_b,
            update.correlation
        )
    else:
        # Update with observation
        learner.update_correlation(
            update.domain_a,
            update.domain_b,
            update.correlation
        )
    
    # Save to database
    try:
        db = get_db()
        correlation_data = {
            "_id": f"{update.domain_a}|{update.domain_b}",
            "domain_a": update.domain_a,
            "domain_b": update.domain_b,
            "correlation": learner.get_correlation(update.domain_a, update.domain_b),
            "n_observations": learner.get_observation_count(update.domain_a, update.domain_b),
            "alpha": learner.get_alpha(update.domain_a, update.domain_b),
        }
        db.save_correlation(correlation_data)
    except RuntimeError:
        pass
    
    return CorrelationResponse(
        domain_a=update.domain_a,
        domain_b=update.domain_b,
        correlation=learner.get_correlation(update.domain_a, update.domain_b),
        n_observations=learner.get_observation_count(update.domain_a, update.domain_b),
        alpha=learner.get_alpha(update.domain_a, update.domain_b)
    )


@router.get(
    "/correlations/{domain}",
    summary="Get all correlations for a domain",
    description="Get all correlations involving a specific domain."
)
async def get_domain_correlations(domain: str):
    """Get all correlations for a domain."""
    try:
        db = get_db()
        correlations = db.get_correlations_for_domain(domain)
        
        return {
            "domain": domain,
            "correlations": [
                {
                    "other_domain": c["domain_b"] if c["domain_a"] == domain else c["domain_a"],
                    "correlation": c.get("correlation", 0.5),
                    "n_observations": c.get("n_observations", 0)
                }
                for c in correlations
            ]
        }
    except RuntimeError:
        return {"domain": domain, "correlations": []}


@router.post(
    "/transfer-ratings",
    summary="Transfer ratings to new domain",
    description="Initialize ratings for a new domain using transfer learning."
)
async def transfer_ratings_to_domain(
    target_domain: str = Query(..., description="Target domain"),
    source_domains: Optional[List[str]] = Query(None, description="Source domains"),
    model_ids: Optional[List[str]] = Query(None, description="Specific models (all if not specified)")
):
    """
    Transfer ratings from source domains to a target domain.
    
    This initializes ratings for models in a new domain based on
    their performance in related domains.
    """
    from langscope.api.dependencies import get_models, refresh_models_cache
    
    if model_ids:
        models = [get_model_by_id(mid) for mid in model_ids if get_model_by_id(mid)]
    else:
        models = get_models()
    
    transfer_learner = get_transfer_learner()
    transferred = []
    
    for model in models:
        # Skip if already has rating in target domain
        if target_domain in model.trueskill_by_domain:
            continue
        
        # Gather source ratings
        source_ratings = {}
        for domain, ts in model.trueskill_by_domain.items():
            if source_domains is None or domain in source_domains:
                source_ratings[domain] = ts.raw
        
        if not source_ratings:
            continue
        
        # Perform transfer
        result = transfer_learner.transfer_multi_source(source_ratings, target_domain)
        
        # Set the new rating
        model.set_domain_trueskill(
            target_domain,
            raw_mu=result.mu,
            raw_sigma=result.sigma,
            cost_mu=result.mu,  # Start cost-adjusted same as raw
            cost_sigma=result.sigma
        )
        
        transferred.append({
            "model_id": model.model_id,
            "predicted_mu": result.mu,
            "predicted_sigma": result.sigma
        })
        
        # Save to database
        try:
            db = get_db()
            db.save_model(model.to_dict())
        except RuntimeError:
            pass
    
    refresh_models_cache()
    
    return {
        "target_domain": target_domain,
        "models_transferred": len(transferred),
        "results": transferred
    }


# =============================================================================
# Faceted Transfer Learning Endpoints (Model Rank API)
# =============================================================================

@router.get(
    "/models/{model_id}/rating",
    response_model=ModelRatingSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get model rating in domain",
    description="Get model rating using direct observation or transfer learning."
)
async def get_model_rating(
    model_id: str,
    domain: str = Query(..., description="Domain to get rating for"),
    dimension: str = Query("raw_quality", description="Rating dimension"),
    explain: bool = Query(False, description="Include transfer explanation")
):
    """
    Get model rating in a domain.
    
    This is the core Model Rank API endpoint. If the model has direct
    observations in the domain, returns the direct rating. Otherwise,
    uses faceted transfer learning to compute a predicted rating.
    """
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    faceted_transfer = get_faceted_transfer()
    result = faceted_transfer.get_rating_or_transfer(model, domain, dimension)
    
    # Build response
    transfer_details = None
    if explain and result.source == "transfer":
        facet_contributions = {}
        for facet, data in result.facet_contributions.items():
            facet_contributions[facet] = FacetContributionSchema(
                facet=facet,
                source_value=data.get("source_value", ""),
                target_value=data.get("target_value", ""),
                similarity=data.get("similarity", 0.0),
                weight=data.get("weight", 0.0),
                contribution=data.get("contribution", 0.0),
            )
        
        transfer_details = TransferDetailsSchema(
            source_domains=result.source_domains,
            source_weights=result.source_weights,
            correlation_used=result.correlation_used,
            facet_contributions=facet_contributions,
        )
    
    return ModelRatingSchema(
        model_id=model_id,
        domain=domain,
        dimension=dimension,
        rating=TrueSkillRatingSchema(
            mu=result.target_mu,
            sigma=result.target_sigma
        ),
        conservative_estimate=result.target_mu - 3 * result.target_sigma,
        source=result.source,
        confidence=result.confidence,
        match_count=model.performance.total_matches_played if result.source == "direct" else None,
        transfer_details=transfer_details,
        last_updated=datetime.utcnow().isoformat() + "Z",
    )


@router.get(
    "/models/{model_id}/ratings",
    response_model=ModelRatingsSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get all dimension ratings",
    description="Get ratings for all 10 dimensions in a domain."
)
async def get_model_ratings(
    model_id: str,
    domain: str = Query(..., description="Domain to get ratings for"),
    dimensions: Optional[str] = Query(None, description="Comma-separated dimensions or 'all'")
):
    """
    Get model ratings for all dimensions in a domain.
    """
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    # Parse dimensions
    all_dimensions = [
        "raw_quality", "cost_adjusted", "latency", "ttft",
        "consistency", "token_efficiency", "instruction_following",
        "hallucination_resistance", "long_context", "combined"
    ]
    
    if dimensions and dimensions != "all":
        requested_dims = [d.strip() for d in dimensions.split(",")]
    else:
        requested_dims = all_dimensions
    
    faceted_transfer = get_faceted_transfer()
    ratings = {}
    
    for dim in requested_dims:
        if dim not in all_dimensions:
            continue
        
        result = faceted_transfer.get_rating_or_transfer(model, domain, dim)
        
        ratings[dim] = ModelRatingSchema(
            model_id=model_id,
            domain=domain,
            dimension=dim,
            rating=TrueSkillRatingSchema(
                mu=result.target_mu,
                sigma=result.target_sigma
            ),
            conservative_estimate=result.target_mu - 3 * result.target_sigma,
            source=result.source,
            confidence=result.confidence,
        )
    
    return ModelRatingsSchema(
        model_id=model_id,
        domain=domain,
        ratings=ratings,
    )


@router.get(
    "/domains/{domain}/similar",
    response_model=SimilarDomainsResponse,
    summary="Get similar domains",
    description="Get domains similar to the specified domain (for transfer)."
)
async def get_similar_domains(
    domain: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum domains to return"),
    min_correlation: float = Query(0.25, ge=0.0, le=1.0, description="Minimum correlation")
):
    """
    Get similar domains for transfer learning transparency.
    
    Shows which domains would be used as sources for transfer learning.
    """
    domain_index = get_domain_index()
    
    # Get or create descriptor for target domain
    target = domain_index.get_or_create_descriptor(domain)
    
    # Get similar domains
    similar = domain_index.get_similar_domains(
        domain, k=limit, min_correlation=min_correlation
    )
    
    similar_domains = []
    for name, corr in similar:
        # Get facet breakdown
        breakdown = domain_index.get_facet_breakdown(name, domain)
        
        facet_breakdown = {}
        for facet, data in breakdown.items():
            facet_breakdown[facet] = FacetContributionSchema(
                facet=facet,
                source_value=data.get("source_value", ""),
                target_value=data.get("target_value", ""),
                similarity=data.get("similarity", 0.0),
                weight=data.get("weight", 0.0),
                contribution=data.get("contribution", 0.0),
            )
        
        similar_domains.append(SimilarDomainSchema(
            name=name,
            correlation=corr,
            facet_breakdown=facet_breakdown,
        ))
    
    return SimilarDomainsResponse(
        domain=domain,
        facets=target.get_all_facets(),
        similar_domains=similar_domains,
    )


@router.get(
    "/domains/similarity",
    summary="Explain correlation between domains",
    description="Get detailed explanation of similarity between two domains."
)
async def explain_domain_similarity(
    source: str = Query(..., description="Source domain"),
    target: str = Query(..., description="Target domain")
):
    """
    Explain the correlation between two domains.
    
    Shows how each facet contributes to the overall similarity.
    """
    domain_index = get_domain_index()
    
    source_desc = domain_index.get_or_create_descriptor(source)
    target_desc = domain_index.get_or_create_descriptor(target)
    
    correlation = domain_index.get_correlation(source, target)
    breakdown = domain_index.get_facet_breakdown(source, target)
    
    facet_contributions = []
    for facet, data in breakdown.items():
        facet_contributions.append(FacetContributionSchema(
            facet=facet,
            source_value=data.get("source_value", ""),
            target_value=data.get("target_value", ""),
            similarity=data.get("similarity", 0.0),
            weight=data.get("weight", 0.0),
            contribution=data.get("contribution", 0.0),
        ))
    
    return {
        "source": source,
        "target": target,
        "source_facets": source_desc.get_all_facets(),
        "target_facets": target_desc.get_all_facets(),
        "correlation": correlation,
        "facet_contributions": [c.model_dump() for c in facet_contributions],
    }


@router.get(
    "/domains/{domain}/facets",
    response_model=DomainFacetsSchema,
    summary="Get domain facets",
    description="Get the facet decomposition of a domain."
)
async def get_domain_facets(domain: str):
    """Get the facets for a domain."""
    domain_index = get_domain_index()
    descriptor = domain_index.get_or_create_descriptor(domain)
    
    return DomainFacetsSchema(
        name=domain,
        facets=descriptor.get_all_facets(),
    )


@router.post(
    "/domains/{domain}/facets",
    response_model=DomainFacetsSchema,
    summary="Set domain facets",
    description="Set the facet values for a domain."
)
async def set_domain_facets(domain: str, facets: Dict[str, str]):
    """Set the facets for a domain."""
    from langscope.transfer.faceted import DomainDescriptor
    
    domain_index = get_domain_index()
    
    # Create or update descriptor
    descriptor = DomainDescriptor(name=domain, facets=facets)
    domain_index.register_domain(descriptor, precompute=True)
    
    return DomainFacetsSchema(
        name=domain,
        facets=descriptor.get_all_facets(),
    )


@router.put(
    "/similarity/facets/{facet}/prior",
    summary="Set facet similarity prior",
    description="Set expert prior for similarity between two facet values."
)
async def set_facet_prior(facet: str, update: FacetPriorUpdate):
    """
    Set expert prior for facet value similarity.
    
    Used to initialize similarity estimates before observational data.
    """
    from langscope.transfer.faceted import ALL_FACETS
    
    if facet not in ALL_FACETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown facet: {facet}. Valid facets: {ALL_FACETS}"
        )
    
    domain_index = get_domain_index()
    learner = domain_index.composite.get_learner(facet)
    
    if not learner:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No learner found for facet: {facet}"
        )
    
    learner.set_prior(update.value_a, update.value_b, update.similarity)
    
    return {
        "facet": facet,
        "value_a": update.value_a,
        "value_b": update.value_b,
        "similarity": update.similarity,
        "success": True,
    }


@router.get(
    "/similarity/facets/{facet}",
    summary="Get learned facet similarities",
    description="Get all learned similarities for a facet."
)
async def get_facet_similarities(facet: str):
    """Get all learned similarities for a facet."""
    from langscope.transfer.faceted import ALL_FACETS
    
    if facet not in ALL_FACETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown facet: {facet}. Valid facets: {ALL_FACETS}"
        )
    
    domain_index = get_domain_index()
    learner = domain_index.composite.get_learner(facet)
    
    if not learner:
        return {"facet": facet, "similarities": []}
    
    similarities = learner.export_similarities()
    
    return {
        "facet": facet,
        "tau": learner.tau,
        "count": len(similarities),
        "similarities": similarities,
    }


@router.post(
    "/similarity/index/refresh",
    response_model=RefreshIndexResponse,
    summary="Refresh similarity index",
    description="Refresh the pre-computed domain similarity index."
)
async def refresh_similarity_index():
    """
    Refresh the similarity index.
    
    Re-computes Top-K similar domains for all registered domains.
    """
    start_time = time.time()
    
    refresh_domain_index()
    domain_index = get_domain_index()
    
    duration_ms = (time.time() - start_time) * 1000
    
    return RefreshIndexResponse(
        success=True,
        domains_indexed=len(domain_index.descriptors),
        similarities_computed=len(domain_index._top_k_cache),
        duration_ms=duration_ms,
    )


@router.get(
    "/similarity/index/stats",
    response_model=DomainIndexStats,
    summary="Get index statistics",
    description="Get statistics about the domain similarity index."
)
async def get_index_stats():
    """Get statistics about the similarity index."""
    domain_index = get_domain_index()
    
    return DomainIndexStats(
        total_domains=len(domain_index.descriptors),
        domains_with_facets=sum(
            1 for d in domain_index.descriptors.values() 
            if d.facets
        ),
        precomputed_similarities=len(domain_index._top_k_cache),
        last_refresh=domain_index.last_refresh.isoformat() + "Z" if domain_index.last_refresh else None,
    )


@router.get(
    "/leaderboard/{domain}",
    response_model=FacetedLeaderboardResponse,
    summary="Transfer-aware leaderboard",
    description="Get leaderboard with transfer-included entries."
)
async def get_transfer_aware_leaderboard(
    domain: str,
    dimension: str = Query("raw_quality", description="Rating dimension"),
    include_transferred: bool = Query(True, description="Include transferred ratings"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=100, description="Maximum entries")
):
    """
    Get leaderboard with transfer-included entries.
    
    Models with direct ratings are shown with source="direct".
    Models without direct ratings can have transferred ratings
    computed from similar domains (source="transfer").
    """
    models = get_models()
    faceted_transfer = get_faceted_transfer()
    
    entries = []
    direct_count = 0
    transferred_count = 0
    
    for model in models:
        result = faceted_transfer.get_rating_or_transfer(model, domain, dimension)
        
        # Filter by confidence
        if result.confidence < min_confidence:
            continue
        
        # Filter transferred if not included
        if not include_transferred and result.source == "transfer":
            continue
        
        if result.source == "direct":
            direct_count += 1
        else:
            transferred_count += 1
        
        transfer_note = None
        if result.source == "transfer" and result.source_domains:
            transfer_note = f"Transferred from {', '.join(result.source_domains[:3])}"
        
        entries.append({
            "model_id": model.model_id,
            "rating": TrueSkillRatingSchema(
                mu=result.target_mu,
                sigma=result.target_sigma
            ),
            "conservative_estimate": result.target_mu - 3 * result.target_sigma,
            "source": result.source,
            "confidence": result.confidence,
            "transfer_note": transfer_note,
        })
    
    # Sort by conservative estimate (descending)
    entries.sort(key=lambda x: -x["conservative_estimate"])
    
    # Add ranks and limit
    ranked_entries = []
    for i, entry in enumerate(entries[:limit], 1):
        ranked_entries.append(FacetedLeaderboardEntry(
            rank=i,
            model_id=entry["model_id"],
            rating=entry["rating"],
            conservative_estimate=entry["conservative_estimate"],
            source=entry["source"],
            confidence=entry["confidence"],
            transfer_note=entry["transfer_note"],
        ))
    
    return FacetedLeaderboardResponse(
        domain=domain,
        dimension=dimension,
        evaluation_type="subjective",
        entries=ranked_entries,
        total_models=len(entries),
        direct_count=direct_count,
        transferred_count=transferred_count,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


# Import Dict for type hints
from typing import Dict
