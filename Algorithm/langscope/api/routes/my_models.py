"""
My Models API endpoints.

Provides CRUD operations for user's private model registrations.
Supports multiple model types (LLM, ASR, TTS, VLM, etc.) with
type-aware evaluation and comparison features.
"""

import hashlib
import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Header, status

from langscope.api.schemas import (
    UserModelCreate,
    UserModelUpdate,
    UserModelResponse,
    UserModelListResponse,
    UserModelPerformanceResponse,
    ModelComparisonResponse,
    ModelComparisonEntry,
    RunEvaluationRequest,
    RunEvaluationResponse,
    EvaluationStatusResponse,
    TestConnectionRequest,
    TestConnectionResponse,
    UpdateApiKeyRequest,
    SuccessResponse,
    ErrorResponse,
    ModelAPIConfigResponse,
    ModelCostsSchema,
    MultiDimensionalTrueSkillSchema,
    TrueSkillRatingSchema,
)
from langscope.api.dependencies import get_db
from langscope.core.user_model import (
    UserModel,
    ModelType,
    APIFormat,
    ModelAPIConfig,
    ModelTypeConfig,
    ModelCosts,
)


router = APIRouter(prefix="/my-models", tags=["my-models"])


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def get_user_id_from_header(x_user_id: str = Header(None)) -> str:
    """Extract user ID from header."""
    return x_user_id or "anonymous"


def user_model_to_response(model: UserModel) -> UserModelResponse:
    """Convert UserModel to response schema."""
    # Build TrueSkill response if present
    trueskill_response = None
    if model.trueskill:
        trueskill_response = MultiDimensionalTrueSkillSchema(
            raw_quality=TrueSkillRatingSchema(
                mu=model.trueskill.raw_quality.mu,
                sigma=model.trueskill.raw_quality.sigma
            ),
            cost_adjusted=TrueSkillRatingSchema(
                mu=model.trueskill.cost_adjusted.mu,
                sigma=model.trueskill.cost_adjusted.sigma
            ),
            latency=TrueSkillRatingSchema(
                mu=model.trueskill.latency.mu,
                sigma=model.trueskill.latency.sigma
            ),
            ttft=TrueSkillRatingSchema(
                mu=model.trueskill.ttft.mu,
                sigma=model.trueskill.ttft.sigma
            ),
            consistency=TrueSkillRatingSchema(
                mu=model.trueskill.consistency.mu,
                sigma=model.trueskill.consistency.sigma
            ),
            token_efficiency=TrueSkillRatingSchema(
                mu=model.trueskill.token_efficiency.mu,
                sigma=model.trueskill.token_efficiency.sigma
            ),
            instruction_following=TrueSkillRatingSchema(
                mu=model.trueskill.instruction_following.mu,
                sigma=model.trueskill.instruction_following.sigma
            ),
            hallucination_resistance=TrueSkillRatingSchema(
                mu=model.trueskill.hallucination_resistance.mu,
                sigma=model.trueskill.hallucination_resistance.sigma
            ),
            long_context=TrueSkillRatingSchema(
                mu=model.trueskill.long_context.mu,
                sigma=model.trueskill.long_context.sigma
            ),
            combined=TrueSkillRatingSchema(
                mu=model.trueskill.combined.mu,
                sigma=model.trueskill.combined.sigma
            ),
        )
    
    return UserModelResponse(
        id=model.id,
        user_id=model.user_id,
        name=model.name,
        description=model.description,
        model_type=model.model_type.value if isinstance(model.model_type, ModelType) else model.model_type,
        version=model.version,
        base_model_id=model.base_model_id,
        api_config=ModelAPIConfigResponse(
            endpoint=model.api_config.endpoint,
            model_id=model.api_config.model_id,
            api_format=model.api_config.api_format.value if isinstance(model.api_config.api_format, APIFormat) else model.api_config.api_format,
            has_api_key=model.api_config.has_api_key,
        ),
        type_config=model.type_config.to_dict(),
        costs=ModelCostsSchema(
            input_cost_per_million=model.costs.input_cost_per_million,
            output_cost_per_million=model.costs.output_cost_per_million,
            currency=model.costs.currency,
            is_estimate=model.costs.is_estimate,
            notes=model.costs.notes,
        ),
        is_public=model.is_public,
        is_active=model.is_active,
        trueskill=trueskill_response,
        ground_truth_metrics=model.ground_truth_metrics if model.ground_truth_metrics else None,
        total_evaluations=model.total_evaluations,
        domains_evaluated=model.domains_evaluated,
        last_evaluated_at=model.last_evaluated_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.get(
    "",
    response_model=UserModelListResponse,
    summary="List user's models",
    description="Get all models registered by the current user."
)
async def list_my_models(
    x_user_id: str = Header(None, description="User ID"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    active: Optional[bool] = Query(None, description="Filter by active status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List user's models."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get models from database
        models_data = db.get_user_models(
            user_id=user_id,
            model_type=model_type,
            active=active,
            limit=limit + skip
        )
        models_data = models_data[skip:skip + limit]
        
        # Convert to response
        responses = []
        by_type = {}
        
        for data in models_data:
            model = UserModel.from_dict(data)
            responses.append(user_model_to_response(model))
            
            # Count by type
            mtype = model.model_type.value if isinstance(model.model_type, ModelType) else model.model_type
            by_type[mtype] = by_type.get(mtype, 0) + 1
        
        return UserModelListResponse(
            models=responses,
            total=len(responses),
            by_type=by_type
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{model_id}",
    response_model=UserModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get user model by ID",
    description="Get detailed information about a specific user model."
)
async def get_my_model(
    model_id: str,
    x_user_id: str = Header(None)
):
    """Get a specific user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        data = db.get_user_model(model_id)
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check visibility
        if not model.is_visible_to(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this model"
            )
        
        return user_model_to_response(model)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "",
    response_model=UserModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model",
    description="Register a new private model for evaluation."
)
async def create_my_model(
    model_data: UserModelCreate,
    x_user_id: str = Header(None)
):
    """Create a new user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Parse model type
        try:
            model_type = ModelType(model_data.model_type)
        except ValueError:
            model_type = ModelType.LLM
        
        # Parse API format
        try:
            api_format = APIFormat(model_data.api_config.api_format)
        except ValueError:
            api_format = APIFormat.CUSTOM
        
        # Create API config
        api_config = ModelAPIConfig(
            endpoint=model_data.api_config.endpoint,
            model_id=model_data.api_config.model_id,
            api_format=api_format,
            has_api_key=bool(model_data.api_config.api_key),
            api_key_hash=hash_api_key(model_data.api_config.api_key) if model_data.api_config.api_key else None,
            headers=model_data.api_config.headers or {},
            extra_params=model_data.api_config.extra_params or {},
        )
        
        # Create type config
        type_config = ModelTypeConfig()
        if model_data.type_config:
            type_config = ModelTypeConfig(
                language=model_data.type_config.language,
                sample_rate=model_data.type_config.sample_rate,
                image_detail=model_data.type_config.image_detail,
                image_size=model_data.type_config.image_size,
                steps=model_data.type_config.steps,
                guidance_scale=model_data.type_config.guidance_scale,
                embedding_dimension=model_data.type_config.embedding_dimension,
                normalize=model_data.type_config.normalize,
                max_tokens=model_data.type_config.max_tokens,
                temperature=model_data.type_config.temperature,
            )
        
        # Create costs
        costs = ModelCosts(
            input_cost_per_million=model_data.costs.input_cost_per_million,
            output_cost_per_million=model_data.costs.output_cost_per_million,
            currency=model_data.costs.currency,
            is_estimate=model_data.costs.is_estimate,
            notes=model_data.costs.notes,
        )
        
        # Create the model
        model = UserModel.create(
            user_id=user_id,
            name=model_data.name,
            model_type=model_type,
            description=model_data.description,
            version=model_data.version,
            base_model_id=model_data.base_model_id,
            api_config=api_config,
            type_config=type_config,
            costs=costs,
            is_public=model_data.is_public,
        )
        
        # Save to database
        db.save_user_model(model.to_dict())
        
        return user_model_to_response(model)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.patch(
    "/{model_id}",
    response_model=UserModelResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Update a user model",
    description="Update an existing user model."
)
async def update_my_model(
    model_id: str,
    update_data: UserModelUpdate,
    x_user_id: str = Header(None)
):
    """Update a user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get existing model
        data = db.get_user_model(model_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check ownership
        if not model.is_owned_by(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this model"
            )
        
        # Apply updates
        if update_data.name is not None:
            model.name = update_data.name
        if update_data.description is not None:
            model.description = update_data.description
        if update_data.version is not None:
            model.version = update_data.version
        if update_data.is_public is not None:
            model.is_public = update_data.is_public
        if update_data.is_active is not None:
            model.is_active = update_data.is_active
        
        if update_data.api_config:
            if update_data.api_config.endpoint:
                model.api_config.endpoint = update_data.api_config.endpoint
            if update_data.api_config.model_id:
                model.api_config.model_id = update_data.api_config.model_id
            if update_data.api_config.api_format:
                try:
                    model.api_config.api_format = APIFormat(update_data.api_config.api_format)
                except ValueError:
                    pass
            if update_data.api_config.api_key:
                model.api_config.api_key_hash = hash_api_key(update_data.api_config.api_key)
                model.api_config.has_api_key = True
        
        if update_data.costs:
            if update_data.costs.input_cost_per_million is not None:
                model.costs.input_cost_per_million = update_data.costs.input_cost_per_million
            if update_data.costs.output_cost_per_million is not None:
                model.costs.output_cost_per_million = update_data.costs.output_cost_per_million
            if update_data.costs.currency is not None:
                model.costs.currency = update_data.costs.currency
            if update_data.costs.is_estimate is not None:
                model.costs.is_estimate = update_data.costs.is_estimate
            if update_data.costs.notes is not None:
                model.costs.notes = update_data.costs.notes
        
        model.update_timestamp()
        
        # Save
        db.save_user_model(model.to_dict())
        
        return user_model_to_response(model)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.delete(
    "/{model_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a user model",
    description="Remove a user model registration."
)
async def delete_my_model(
    model_id: str,
    x_user_id: str = Header(None)
):
    """Delete a user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get existing model
        data = db.get_user_model(model_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check ownership
        if not model.is_owned_by(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this model"
            )
        
        # Delete
        db.delete_user_model(model_id)
        
        return SuccessResponse(
            success=True,
            message=f"Model {model_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{model_id}/performance",
    response_model=UserModelPerformanceResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get model performance",
    description="Get detailed performance data for a user model."
)
async def get_model_performance(
    model_id: str,
    x_user_id: str = Header(None)
):
    """Get performance data for a user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get model
        data = db.get_user_model(model_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check visibility
        if not model.is_visible_to(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this model"
            )
        
        # Get evaluation history
        history = db.get_user_model_evaluations(model_id, limit=50)
        
        # Build response
        trueskill_response = None
        if model.trueskill:
            trueskill_response = MultiDimensionalTrueSkillSchema(
                raw_quality=TrueSkillRatingSchema(mu=model.trueskill.raw_quality.mu, sigma=model.trueskill.raw_quality.sigma),
                cost_adjusted=TrueSkillRatingSchema(mu=model.trueskill.cost_adjusted.mu, sigma=model.trueskill.cost_adjusted.sigma),
                latency=TrueSkillRatingSchema(mu=model.trueskill.latency.mu, sigma=model.trueskill.latency.sigma),
                ttft=TrueSkillRatingSchema(mu=model.trueskill.ttft.mu, sigma=model.trueskill.ttft.sigma),
                consistency=TrueSkillRatingSchema(mu=model.trueskill.consistency.mu, sigma=model.trueskill.consistency.sigma),
                token_efficiency=TrueSkillRatingSchema(mu=model.trueskill.token_efficiency.mu, sigma=model.trueskill.token_efficiency.sigma),
                instruction_following=TrueSkillRatingSchema(mu=model.trueskill.instruction_following.mu, sigma=model.trueskill.instruction_following.sigma),
                hallucination_resistance=TrueSkillRatingSchema(mu=model.trueskill.hallucination_resistance.mu, sigma=model.trueskill.hallucination_resistance.sigma),
                long_context=TrueSkillRatingSchema(mu=model.trueskill.long_context.mu, sigma=model.trueskill.long_context.sigma),
                combined=TrueSkillRatingSchema(mu=model.trueskill.combined.mu, sigma=model.trueskill.combined.sigma),
            )
        
        return UserModelPerformanceResponse(
            model_id=model.id,
            model_type=model.model_type.value if isinstance(model.model_type, ModelType) else model.model_type,
            trueskill=trueskill_response,
            trueskill_by_domain=None,  # TODO: Implement
            ground_truth_metrics=model.ground_truth_metrics if model.ground_truth_metrics else None,
            ground_truth_by_domain=model.ground_truth_by_domain if model.ground_truth_by_domain else None,
            evaluation_history=history,
            public_rank=None,  # TODO: Calculate
            public_total=None,
            percentile=None,
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare models",
    description="Compare user models against public leaderboard."
)
async def compare_models(
    domain: str = Query(..., description="Domain to compare in"),
    model_type: str = Query(..., description="Model type"),
    metric: Optional[str] = Query(None, description="Metric to compare"),
    include_public: bool = Query(True, description="Include public models"),
    limit: int = Query(20, ge=1, le=100),
    x_user_id: str = Header(None)
):
    """Compare user models against public leaderboard."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get user's models of this type
        user_models_data = db.get_user_models(
            user_id=user_id,
            model_type=model_type,
            active=True
        )
        
        entries = []
        user_model_ids = []
        
        # Add user models
        for i, data in enumerate(user_models_data):
            model = UserModel.from_dict(data)
            user_model_ids.append(model.id)
            
            # Get metric value
            metric_value = 0.0
            if metric and model.trueskill:
                rating = getattr(model.trueskill, metric, None)
                if rating:
                    metric_value = rating.mu
            elif model.ground_truth_metrics:
                metric_value = model.ground_truth_metrics.get(metric or "accuracy", 0.0)
            
            entries.append(ModelComparisonEntry(
                model_id=model.id,
                name=model.name,
                is_user_model=True,
                model_type=model_type,
                provider=None,
                metrics={metric or "score": metric_value},
                rank=i + 1,
                costs=ModelCostsSchema(
                    input_cost_per_million=model.costs.input_cost_per_million,
                    output_cost_per_million=model.costs.output_cost_per_million,
                    currency=model.costs.currency,
                    is_estimate=model.costs.is_estimate,
                    notes=model.costs.notes,
                ),
            ))
        
        # TODO: Add public models from leaderboard
        
        # Sort by metric
        entries.sort(key=lambda e: e.metrics.get(metric or "score", 0), reverse=True)
        
        # Update ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1
        
        return ModelComparisonResponse(
            domain=domain,
            model_type=model_type,
            metric=metric or "score",
            entries=entries[:limit],
            user_model_ids=user_model_ids,
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "/test-connection",
    response_model=TestConnectionResponse,
    summary="Test model connection",
    description="Test API connection to a model."
)
async def test_connection(
    request: TestConnectionRequest
):
    """Test connection to a model API."""
    import httpx
    import time
    
    try:
        start = time.time()
        
        # Build request based on API format
        if request.api_format == "openai":
            headers = {
                "Authorization": f"Bearer {request.api_key}",
                "Content-Type": "application/json",
            }
            # Try a simple models list request
            url = request.endpoint.rstrip("/") + "/models"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return TestConnectionResponse(
                    success=True,
                    latency_ms=latency,
                )
            else:
                return TestConnectionResponse(
                    success=False,
                    error=f"API returned status {response.status_code}"
                )
        else:
            # For other formats, just check if endpoint is reachable
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(request.endpoint)
            
            latency = (time.time() - start) * 1000
            
            return TestConnectionResponse(
                success=response.status_code < 500,
                latency_ms=latency,
            )
            
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False,
            error="Could not connect to endpoint"
        )
    except httpx.TimeoutException:
        return TestConnectionResponse(
            success=False,
            error="Connection timed out"
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            error=str(e)
        )


@router.post(
    "/{model_id}/api-key",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Update API key",
    description="Update the API key for a user model."
)
async def update_api_key(
    model_id: str,
    request: UpdateApiKeyRequest,
    x_user_id: str = Header(None)
):
    """Update API key for a model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get existing model
        data = db.get_user_model(model_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check ownership
        if not model.is_owned_by(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this model"
            )
        
        # Update API key
        model.api_config.api_key_hash = hash_api_key(request.api_key)
        model.api_config.has_api_key = True
        model.update_timestamp()
        
        # Save
        db.save_user_model(model.to_dict())
        
        return SuccessResponse(
            success=True,
            message="API key updated successfully"
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "/evaluate",
    response_model=RunEvaluationResponse,
    summary="Run evaluation",
    description="Start an evaluation for a user model."
)
async def run_evaluation(
    request: RunEvaluationRequest,
    x_user_id: str = Header(None)
):
    """Start an evaluation for a user model."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Get model
        data = db.get_user_model(request.model_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {request.model_id}"
            )
        
        model = UserModel.from_dict(data)
        
        # Check ownership
        if not model.is_owned_by(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to evaluate this model"
            )
        
        # Create evaluation job (in real implementation, queue this)
        evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        
        # TODO: Queue actual evaluation job
        
        return RunEvaluationResponse(
            evaluation_id=evaluation_id,
            status="queued",
            model_id=request.model_id,
            domain=request.domain,
            estimated_duration_ms=30000,  # 30 seconds estimate
            queue_position=1,
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get evaluation status",
    description="Check the status of a running evaluation."
)
async def get_evaluation_status(
    evaluation_id: str,
    x_user_id: str = Header(None)
):
    """Get evaluation status."""
    try:
        db = get_db()
        
        # Get evaluation status from database
        status_data = db.get_evaluation_status(evaluation_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation not found: {evaluation_id}"
            )
        
        return EvaluationStatusResponse(
            evaluation_id=evaluation_id,
            status=status_data.get("status", "unknown"),
            progress=status_data.get("progress"),
            current_step=status_data.get("current_step"),
            results=None,  # TODO: Parse results
            error=status_data.get("error"),
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )
