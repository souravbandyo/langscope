"""
Model management API endpoints.

Provides CRUD operations for LLM models.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelListResponse,
    DualTrueSkillSchema,
    TrueSkillRatingSchema,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import (
    get_db,
    get_models,
    get_model_by_id,
    get_model_by_name,
    refresh_models_cache,
)
from langscope.core.model import LLMModel


router = APIRouter(prefix="/models", tags=["models"])


def model_to_response(model: LLMModel) -> ModelResponse:
    """Convert LLMModel to ModelResponse schema."""
    return ModelResponse(
        name=model.name,
        model_id=model.model_id,
        provider=model.provider,
        input_cost_per_million=model.input_cost_per_million,
        output_cost_per_million=model.output_cost_per_million,
        pricing_source=model.pricing_source,
        trueskill=DualTrueSkillSchema(
            raw=TrueSkillRatingSchema(
                mu=model.trueskill.raw.mu,
                sigma=model.trueskill.raw.sigma
            ),
            cost_adjusted=TrueSkillRatingSchema(
                mu=model.trueskill.cost_adjusted.mu,
                sigma=model.trueskill.cost_adjusted.sigma
            )
        ),
        trueskill_by_domain={
            domain: DualTrueSkillSchema(
                raw=TrueSkillRatingSchema(mu=ts.raw.mu, sigma=ts.raw.sigma),
                cost_adjusted=TrueSkillRatingSchema(
                    mu=ts.cost_adjusted.mu, sigma=ts.cost_adjusted.sigma
                )
            )
            for domain, ts in model.trueskill_by_domain.items()
        },
        total_matches_played=model.performance.total_matches_played,
        domains_evaluated=model.metadata.get("domains_evaluated", []),
        avg_latency_ms=model.performance.avg_latency_ms,
        avg_ttft_ms=model.performance.avg_ttft_ms,
    )


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List all models",
    description="Get a list of all registered LLM models with their ratings."
)
async def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    domain: Optional[str] = Query(None, description="Filter by evaluated domain"),
    skip: int = Query(0, ge=0, description="Number of models to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum models to return")
):
    """List all models with optional filtering."""
    models = get_models(domain=domain)
    
    # Filter by provider if specified
    if provider:
        models = [m for m in models if m.provider == provider]
    
    # Apply pagination
    total = len(models)
    models = models[skip:skip + limit]
    
    return ModelListResponse(
        models=[model_to_response(m) for m in models],
        total=total
    )


@router.get(
    "/{model_id:path}",
    response_model=ModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get model by ID",
    description="Get detailed information about a specific model."
)
async def get_model(model_id: str):
    """Get a specific model by its ID."""
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    return model_to_response(model)


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new model",
    description="Register a new LLM model in the system."
)
async def create_model(model_data: ModelCreate):
    """Create a new model."""
    # Check if model already exists
    existing = get_model_by_id(model_data.model_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model already exists: {model_data.model_id}"
        )
    
    existing_name = get_model_by_name(model_data.name)
    if existing_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model name already in use: {model_data.name}"
        )
    
    # Create model
    model = LLMModel(
        name=model_data.name,
        model_id=model_data.model_id,
        provider=model_data.provider,
        input_cost_per_million=model_data.input_cost_per_million,
        output_cost_per_million=model_data.output_cost_per_million,
        pricing_source=model_data.pricing_source,
        max_matches=model_data.max_matches
    )
    
    # Save to database
    try:
        db = get_db()
        db.save_model(model.to_dict())
        refresh_models_cache()
    except RuntimeError:
        pass  # Database not available, model exists in memory
    
    return model_to_response(model)


@router.patch(
    "/{model_id:path}",
    response_model=ModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Update a model",
    description="Update model properties like pricing or notes."
)
async def update_model(model_id: str, updates: ModelUpdate):
    """Update an existing model."""
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    # Apply updates
    if updates.input_cost_per_million is not None:
        model.input_cost_per_million = updates.input_cost_per_million
    if updates.output_cost_per_million is not None:
        model.output_cost_per_million = updates.output_cost_per_million
    if updates.pricing_source is not None:
        model.pricing_source = updates.pricing_source
    if updates.max_matches is not None:
        model.max_matches = updates.max_matches
    if updates.notes is not None:
        model.metadata["notes"] = updates.notes
    
    # Save to database
    try:
        db = get_db()
        db.save_model(model.to_dict())
        refresh_models_cache()
    except RuntimeError:
        pass
    
    return model_to_response(model)


@router.delete(
    "/{model_id:path}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a model",
    description="Remove a model from the system."
)
async def delete_model(model_id: str):
    """Delete a model."""
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    # Delete from database
    try:
        db = get_db()
        db.delete_model(model.name)
        refresh_models_cache()
    except RuntimeError:
        pass
    
    return SuccessResponse(
        success=True,
        message=f"Model {model_id} deleted successfully"
    )


@router.get(
    "/{model_id}/domain/{domain}",
    response_model=DualTrueSkillSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get model rating for domain",
    description="Get a model's TrueSkill rating for a specific domain."
)
async def get_model_domain_rating(model_id: str, domain: str):
    """Get model rating for a specific domain."""
    model = get_model_by_id(model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    if domain not in model.trueskill_by_domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} has not been evaluated in domain {domain}"
        )
    
    ts = model.trueskill_by_domain[domain]
    return DualTrueSkillSchema(
        raw=TrueSkillRatingSchema(mu=ts.raw.mu, sigma=ts.raw.sigma),
        cost_adjusted=TrueSkillRatingSchema(
            mu=ts.cost_adjusted.mu, sigma=ts.cost_adjusted.sigma
        )
    )


@router.get(
    "/by-name/{name}",
    response_model=ModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get model by name",
    description="Get detailed information about a model by its name."
)
async def get_model_by_name_route(name: str):
    """Get a model by its name."""
    model = get_model_by_name(name)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {name}"
        )
    
    return model_to_response(model)
