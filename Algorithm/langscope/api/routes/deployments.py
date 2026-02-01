"""
Model Deployment API endpoints.

Provides CRUD operations for model deployments (cloud provider instances).
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    DeploymentCreate,
    DeploymentResponse,
    DeploymentListResponse,
    ProviderSchema,
    DeploymentConfigSchema,
    PricingSchema,
    PerformanceSchema,
    AvailabilitySchema,
    DualTrueSkillSchema,
    TrueSkillRatingSchema,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import get_db
from langscope.core.deployment import (
    ModelDeployment,
    Provider,
    ProviderType,
    DeploymentConfig,
    Pricing,
)


router = APIRouter(prefix="/deployments", tags=["deployments"])


def deployment_to_response(dep: ModelDeployment) -> DeploymentResponse:
    """Convert ModelDeployment to DeploymentResponse schema."""
    return DeploymentResponse(
        id=dep.id,
        base_model_id=dep.base_model_id,
        provider=ProviderSchema(
            id=dep.provider.id,
            name=dep.provider.name,
            type=dep.provider.type.value if hasattr(dep.provider.type, 'value') else dep.provider.type,
            api_base=dep.provider.api_base,
            api_compatible=dep.provider.api_compatible,
            website=dep.provider.website,
            docs=dep.provider.docs,
        ),
        deployment=DeploymentConfigSchema(
            model_id=dep.deployment.model_id,
            display_name=dep.deployment.display_name,
            quantization=dep.deployment.quantization,
            serving_framework=dep.deployment.serving_framework,
            max_context_length=dep.deployment.max_context_length,
            max_output_tokens=dep.deployment.max_output_tokens,
            notes=dep.deployment.notes,
        ),
        pricing=PricingSchema(
            input_cost_per_million=dep.pricing.input_cost_per_million,
            output_cost_per_million=dep.pricing.output_cost_per_million,
            currency=dep.pricing.currency,
            source_id=dep.pricing.source_id,
            source_url=dep.pricing.source_url,
            last_verified=dep.pricing.last_verified,
        ),
        performance=PerformanceSchema(
            avg_latency_ms=dep.performance.avg_latency_ms,
            p50_latency_ms=dep.performance.p50_latency_ms,
            p95_latency_ms=dep.performance.p95_latency_ms,
            p99_latency_ms=dep.performance.p99_latency_ms,
            avg_ttft_ms=dep.performance.avg_ttft_ms,
            tokens_per_second=dep.performance.tokens_per_second,
            uptime_30d=dep.performance.uptime_30d,
            error_rate_30d=dep.performance.error_rate_30d,
        ),
        availability=AvailabilitySchema(
            status=dep.availability.status.value if hasattr(dep.availability.status, 'value') else dep.availability.status,
            regions=dep.availability.regions,
            requires_waitlist=dep.availability.requires_waitlist,
            requires_enterprise=dep.availability.requires_enterprise,
        ),
        trueskill=DualTrueSkillSchema(
            raw=TrueSkillRatingSchema(
                mu=dep.trueskill.raw.mu,
                sigma=dep.trueskill.raw.sigma,
            ),
            cost_adjusted=TrueSkillRatingSchema(
                mu=dep.trueskill.cost_adjusted.mu,
                sigma=dep.trueskill.cost_adjusted.sigma,
            ),
        ),
        performance_stats=dep.performance_stats.to_dict(),
        created_at=dep.created_at,
        updated_at=dep.updated_at,
    )


@router.get(
    "",
    response_model=DeploymentListResponse,
    summary="List all deployments",
    description="Get a list of all model deployments."
)
async def list_deployments(
    base_model_id: Optional[str] = Query(None, description="Filter by base model"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    max_price: Optional[float] = Query(None, description="Maximum input price per million"),
    min_rating: Optional[float] = Query(None, description="Minimum TrueSkill mu"),
    skip: int = Query(0, ge=0, description="Number to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum to return")
):
    """List all deployments with optional filtering."""
    try:
        db = get_db()
        
        if base_model_id:
            deployments_data = db.get_deployments_by_base_model(base_model_id)
        elif provider:
            deployments_data = db.get_deployments_by_provider(provider, limit=limit + skip)
        else:
            deployments_data = db.get_all_deployments(
                max_price=max_price,
                min_rating=min_rating,
                limit=limit + skip
            )
        
        # Apply pagination
        deployments_data = deployments_data[skip:skip + limit]
        
        responses = []
        for data in deployments_data:
            dep = ModelDeployment.from_dict(data)
            responses.append(deployment_to_response(dep))
        
        return DeploymentListResponse(
            deployments=responses,
            total=len(deployments_data)
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{deployment_id:path}",
    response_model=DeploymentResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get deployment by ID",
    description="Get detailed information about a specific deployment."
)
async def get_deployment(deployment_id: str):
    """Get a specific deployment by ID."""
    try:
        db = get_db()
        data = db.get_deployment(deployment_id)
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment not found: {deployment_id}"
            )
        
        dep = ModelDeployment.from_dict(data)
        return deployment_to_response(dep)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new deployment",
    description="Register a new model deployment."
)
async def create_deployment(deployment_data: DeploymentCreate):
    """Create a new deployment."""
    try:
        db = get_db()
        
        # Check if deployment already exists
        existing = db.get_deployment(deployment_data.id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Deployment already exists: {deployment_data.id}"
            )
        
        # Verify base model exists
        base_model = db.get_base_model(deployment_data.base_model_id)
        if not base_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Base model not found: {deployment_data.base_model_id}"
            )
        
        # Create provider
        provider = Provider(
            id=deployment_data.provider.id,
            name=deployment_data.provider.name,
            type=ProviderType(deployment_data.provider.type) if deployment_data.provider.type else ProviderType.CLOUD,
            api_base=deployment_data.provider.api_base,
            api_compatible=deployment_data.provider.api_compatible,
            website=deployment_data.provider.website,
            docs=deployment_data.provider.docs,
        )
        
        # Create deployment
        dep = ModelDeployment(
            id=deployment_data.id,
            base_model_id=deployment_data.base_model_id,
            provider=provider,
        )
        
        # Set deployment config
        dep.deployment = DeploymentConfig(
            model_id=deployment_data.deployment.model_id,
            display_name=deployment_data.deployment.display_name,
            quantization=deployment_data.deployment.quantization,
            serving_framework=deployment_data.deployment.serving_framework,
            max_context_length=deployment_data.deployment.max_context_length,
            max_output_tokens=deployment_data.deployment.max_output_tokens,
            notes=deployment_data.deployment.notes,
        )
        
        # Set pricing if provided
        if deployment_data.pricing:
            dep.pricing = Pricing(
                input_cost_per_million=deployment_data.pricing.input_cost_per_million,
                output_cost_per_million=deployment_data.pricing.output_cost_per_million,
                currency=deployment_data.pricing.currency,
                source_id=deployment_data.pricing.source_id,
                source_url=deployment_data.pricing.source_url,
                last_verified=deployment_data.pricing.last_verified,
            )
        
        # Save to database
        db.save_deployment(dep.to_dict())
        
        return deployment_to_response(dep)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.delete(
    "/{deployment_id:path}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a deployment",
    description="Remove a deployment from the system."
)
async def delete_deployment(deployment_id: str):
    """Delete a deployment."""
    try:
        db = get_db()
        
        existing = db.get_deployment(deployment_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment not found: {deployment_id}"
            )
        
        db.delete_deployment(deployment_id)
        
        return SuccessResponse(
            success=True,
            message=f"Deployment {deployment_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/by-base-model/{base_model_id:path}",
    response_model=DeploymentListResponse,
    summary="Get deployments for a base model",
    description="Get all deployments for a specific base model."
)
async def get_deployments_for_base_model(
    base_model_id: str,
    include_inactive: bool = Query(False, description="Include deprecated/offline deployments")
):
    """Get all deployments for a base model."""
    try:
        db = get_db()
        deployments_data = db.get_deployments_by_base_model(
            base_model_id,
            include_inactive=include_inactive
        )
        
        responses = []
        for data in deployments_data:
            dep = ModelDeployment.from_dict(data)
            responses.append(deployment_to_response(dep))
        
        return DeploymentListResponse(
            deployments=responses,
            total=len(responses)
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/best/{base_model_id:path}",
    response_model=DeploymentResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get best deployment for a base model",
    description="Get the best-rated deployment for a base model."
)
async def get_best_deployment(
    base_model_id: str,
    domain: Optional[str] = Query(None, description="Domain for domain-specific ranking"),
    dimension: str = Query("cost_adjusted", description="Rating dimension (raw or cost_adjusted)")
):
    """Get the best deployment for a base model."""
    try:
        db = get_db()
        
        data = db.get_best_deployment(
            base_model_id=base_model_id,
            domain=domain,
            dimension=dimension
        )
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active deployments found for: {base_model_id}"
            )
        
        dep = ModelDeployment.from_dict(data)
        return deployment_to_response(dep)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


