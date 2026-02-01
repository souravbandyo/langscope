"""
Base Model API endpoints.

Provides CRUD operations for base models (the actual neural networks).
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    BaseModelCreate,
    BaseModelResponse,
    BaseModelListResponse,
    ArchitectureSchema,
    CapabilitiesSchema,
    ContextWindowSchema,
    LicenseSchema,
    QuantizationOptionSchema,
    BenchmarkScoreSchema,
    BenchmarkAggregatesSchema,
    ProviderComparisonResponse,
    ProviderComparisonEntry,
    ProviderSchema,
    PricingSchema,
    PerformanceSchema,
    DualTrueSkillSchema,
    TrueSkillRatingSchema,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import get_db
from langscope.core.base_model import BaseModel as CoreBaseModel


router = APIRouter(prefix="/base-models", tags=["base-models"])


def base_model_to_response(model: CoreBaseModel, deployment_count: int = 0) -> BaseModelResponse:
    """Convert CoreBaseModel to BaseModelResponse schema."""
    return BaseModelResponse(
        id=model.id,
        name=model.name,
        family=model.family,
        version=model.version,
        organization=model.organization,
        architecture=ArchitectureSchema(
            type=model.architecture.type.value if hasattr(model.architecture.type, 'value') else model.architecture.type,
            parameters=model.architecture.parameters,
            parameters_display=model.architecture.parameters_display,
            hidden_size=model.architecture.hidden_size,
            num_layers=model.architecture.num_layers,
            num_attention_heads=model.architecture.num_attention_heads,
            num_kv_heads=model.architecture.num_kv_heads,
            vocab_size=model.architecture.vocab_size,
            max_position_embeddings=model.architecture.max_position_embeddings,
            native_precision=model.architecture.native_precision,
            native_size_gb=model.architecture.native_size_gb,
        ),
        capabilities=CapabilitiesSchema(
            modalities=model.capabilities.modalities,
            languages=model.capabilities.languages,
            supports_function_calling=model.capabilities.supports_function_calling,
            supports_json_mode=model.capabilities.supports_json_mode,
            supports_vision=model.capabilities.supports_vision,
            supports_audio=model.capabilities.supports_audio,
            supports_system_prompt=model.capabilities.supports_system_prompt,
            supports_streaming=model.capabilities.supports_streaming,
            trained_for=model.capabilities.trained_for,
        ),
        context=ContextWindowSchema(
            max_context_length=model.context.max_context_length,
            recommended_context=model.context.recommended_context,
            max_output_tokens=model.context.max_output_tokens,
            quality_at_context={str(k): v for k, v in model.context.quality_at_context.items()},
        ),
        license=LicenseSchema(
            type=model.license.type,
            commercial_use=model.license.commercial_use,
            requires_agreement=model.license.requires_agreement,
            restrictions=model.license.restrictions,
            url=model.license.url,
        ),
        quantizations={
            name: QuantizationOptionSchema(
                bits=q.bits,
                vram_gb=q.vram_gb,
                ram_gb=q.ram_gb,
                quality_retention=q.quality_retention,
                huggingface_id=q.huggingface_id,
                supported_frameworks=q.supported_frameworks,
                notes=q.notes,
            )
            for name, q in model.quantizations.items()
        },
        benchmarks={
            name: BenchmarkScoreSchema(
                score=b.score,
                variant=b.variant,
                percentile=b.percentile,
                updated_at=b.updated_at,
            )
            for name, b in model.benchmarks.items()
        },
        benchmark_aggregates=BenchmarkAggregatesSchema(
            open_llm_average=model.benchmark_aggregates.open_llm_average,
            knowledge_average=model.benchmark_aggregates.knowledge_average,
            reasoning_average=model.benchmark_aggregates.reasoning_average,
            coding_average=model.benchmark_aggregates.coding_average,
            math_average=model.benchmark_aggregates.math_average,
            chat_average=model.benchmark_aggregates.chat_average,
            overall_rank=model.benchmark_aggregates.overall_rank,
            total_models_ranked=model.benchmark_aggregates.total_models_ranked,
        ),
        deployment_count=deployment_count,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.get(
    "",
    response_model=BaseModelListResponse,
    summary="List all base models",
    description="Get a list of all registered base models."
)
async def list_base_models(
    family: Optional[str] = Query(None, description="Filter by model family"),
    organization: Optional[str] = Query(None, description="Filter by organization"),
    skip: int = Query(0, ge=0, description="Number of models to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum models to return")
):
    """List all base models with optional filtering."""
    try:
        db = get_db()
        models_data = db.get_all_base_models(
            family=family,
            organization=organization,
            limit=limit + skip
        )
        
        # Skip for pagination
        models_data = models_data[skip:skip + limit]
        
        responses = []
        for data in models_data:
            model = CoreBaseModel.from_dict(data)
            # Count deployments
            deployments = db.get_deployments_by_base_model(model.id)
            responses.append(base_model_to_response(model, len(deployments)))
        
        # Get total count
        all_models = db.get_all_base_models(family=family, organization=organization, limit=10000)
        
        return BaseModelListResponse(
            models=responses,
            total=len(all_models)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{base_model_id:path}",
    response_model=BaseModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get base model by ID",
    description="Get detailed information about a specific base model."
)
async def get_base_model(base_model_id: str):
    """Get a specific base model by ID."""
    try:
        db = get_db()
        data = db.get_base_model(base_model_id)
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Base model not found: {base_model_id}"
            )
        
        model = CoreBaseModel.from_dict(data)
        deployments = db.get_deployments_by_base_model(model.id)
        return base_model_to_response(model, len(deployments))
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "",
    response_model=BaseModelResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new base model",
    description="Register a new base model in the system."
)
async def create_base_model(model_data: BaseModelCreate):
    """Create a new base model."""
    try:
        db = get_db()
        
        # Check if model already exists
        existing = db.get_base_model(model_data.id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Base model already exists: {model_data.id}"
            )
        
        # Create model
        model = CoreBaseModel(
            id=model_data.id,
            name=model_data.name,
            family=model_data.family,
            version=model_data.version,
            organization=model_data.organization,
        )
        
        # Apply optional fields
        if model_data.architecture:
            from langscope.core.base_model import Architecture
            model.architecture = Architecture(
                type=model_data.architecture.type,
                parameters=model_data.architecture.parameters,
                parameters_display=model_data.architecture.parameters_display,
                hidden_size=model_data.architecture.hidden_size,
                num_layers=model_data.architecture.num_layers,
                num_attention_heads=model_data.architecture.num_attention_heads,
                num_kv_heads=model_data.architecture.num_kv_heads,
                vocab_size=model_data.architecture.vocab_size,
                max_position_embeddings=model_data.architecture.max_position_embeddings,
                native_precision=model_data.architecture.native_precision,
                native_size_gb=model_data.architecture.native_size_gb,
            )
        
        if model_data.capabilities:
            from langscope.core.base_model import Capabilities
            model.capabilities = Capabilities(
                modalities=model_data.capabilities.modalities,
                languages=model_data.capabilities.languages,
                supports_function_calling=model_data.capabilities.supports_function_calling,
                supports_json_mode=model_data.capabilities.supports_json_mode,
                supports_vision=model_data.capabilities.supports_vision,
                supports_audio=model_data.capabilities.supports_audio,
                supports_system_prompt=model_data.capabilities.supports_system_prompt,
                supports_streaming=model_data.capabilities.supports_streaming,
                trained_for=model_data.capabilities.trained_for,
            )
        
        if model_data.context:
            from langscope.core.base_model import ContextWindow
            model.context = ContextWindow(
                max_context_length=model_data.context.max_context_length,
                recommended_context=model_data.context.recommended_context,
                max_output_tokens=model_data.context.max_output_tokens,
                quality_at_context={int(k): v for k, v in model_data.context.quality_at_context.items()},
            )
        
        if model_data.license:
            from langscope.core.base_model import License
            model.license = License(
                type=model_data.license.type,
                commercial_use=model_data.license.commercial_use,
                requires_agreement=model_data.license.requires_agreement,
                restrictions=model_data.license.restrictions,
                url=model_data.license.url,
            )
        
        # Save to database
        db.save_base_model(model.to_dict())
        
        return base_model_to_response(model, 0)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.delete(
    "/{base_model_id:path}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a base model",
    description="Remove a base model from the system."
)
async def delete_base_model(base_model_id: str):
    """Delete a base model."""
    try:
        db = get_db()
        
        # Check if exists
        existing = db.get_base_model(base_model_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Base model not found: {base_model_id}"
            )
        
        # Check for deployments
        deployments = db.get_deployments_by_base_model(base_model_id, include_inactive=True)
        if deployments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete base model with {len(deployments)} active deployments"
            )
        
        # Delete
        db.delete_base_model(base_model_id)
        
        return SuccessResponse(
            success=True,
            message=f"Base model {base_model_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{base_model_id:path}/compare-providers",
    response_model=ProviderComparisonResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Compare providers for a base model",
    description="Compare all provider deployments for a base model side-by-side."
)
async def compare_providers(base_model_id: str):
    """Compare all providers for a base model."""
    try:
        db = get_db()
        
        # Get base model
        base_model_data = db.get_base_model(base_model_id)
        if not base_model_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Base model not found: {base_model_id}"
            )
        
        # Get deployments
        deployments = db.get_deployments_by_base_model(base_model_id)
        
        providers = []
        for dep in deployments:
            provider = dep.get("provider", {})
            pricing = dep.get("pricing", {})
            performance = dep.get("performance", {})
            trueskill = dep.get("trueskill", {})
            
            providers.append(ProviderComparisonEntry(
                deployment_id=dep.get("_id", ""),
                provider=ProviderSchema(
                    id=provider.get("id", ""),
                    name=provider.get("name", ""),
                    type=provider.get("type", "cloud"),
                    api_base=provider.get("api_base", ""),
                    api_compatible=provider.get("api_compatible", "openai"),
                    website=provider.get("website", ""),
                    docs=provider.get("docs", ""),
                ),
                pricing=PricingSchema(
                    input_cost_per_million=pricing.get("input_cost_per_million", 0),
                    output_cost_per_million=pricing.get("output_cost_per_million", 0),
                    currency=pricing.get("currency", "USD"),
                    source_id=pricing.get("source_id", ""),
                    source_url=pricing.get("source_url", ""),
                    last_verified=pricing.get("last_verified", ""),
                ),
                performance=PerformanceSchema(
                    avg_latency_ms=performance.get("avg_latency_ms", 0),
                    p50_latency_ms=performance.get("p50_latency_ms", 0),
                    p95_latency_ms=performance.get("p95_latency_ms", 0),
                    p99_latency_ms=performance.get("p99_latency_ms", 0),
                    avg_ttft_ms=performance.get("avg_ttft_ms", 0),
                    tokens_per_second=performance.get("tokens_per_second", 0),
                    uptime_30d=performance.get("uptime_30d", 1.0),
                    error_rate_30d=performance.get("error_rate_30d", 0),
                ),
                trueskill=DualTrueSkillSchema(
                    raw=TrueSkillRatingSchema(
                        mu=trueskill.get("raw", {}).get("mu", 1500),
                        sigma=trueskill.get("raw", {}).get("sigma", 166),
                    ),
                    cost_adjusted=TrueSkillRatingSchema(
                        mu=trueskill.get("cost_adjusted", {}).get("mu", 1500),
                        sigma=trueskill.get("cost_adjusted", {}).get("sigma", 166),
                    ),
                ),
            ))
        
        return ProviderComparisonResponse(
            base_model_id=base_model_id,
            base_model_name=base_model_data.get("name", ""),
            providers=providers,
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


