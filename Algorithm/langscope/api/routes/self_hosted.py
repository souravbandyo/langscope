"""
Self-Hosted Deployment API endpoints.

Provides CRUD operations for user-owned model deployments.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Header, status

from langscope.api.schemas import (
    SelfHostedCreate,
    SelfHostedResponse,
    SelfHostedListResponse,
    HardwareConfigSchema,
    SoftwareConfigSchema,
    SelfHostedCostsSchema,
    DualTrueSkillSchema,
    TrueSkillRatingSchema,
    CostEstimateRequest,
    CostEstimateResponse,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import get_db
from langscope.core.self_hosted import (
    SelfHostedDeployment,
    Owner,
    HardwareConfig,
    SoftwareConfig,
    SelfHostedCosts,
    CloudProvider,
    ServingFramework,
)


router = APIRouter(prefix="/self-hosted", tags=["self-hosted"])


def self_hosted_to_response(dep: SelfHostedDeployment) -> SelfHostedResponse:
    """Convert SelfHostedDeployment to SelfHostedResponse schema."""
    return SelfHostedResponse(
        id=dep.id,
        base_model_id=dep.base_model_id,
        owner_user_id=dep.owner.user_id,
        is_public=dep.owner.is_public,
        hardware=HardwareConfigSchema(
            gpu_type=dep.hardware.gpu_type,
            gpu_count=dep.hardware.gpu_count,
            gpu_memory_gb=dep.hardware.gpu_memory_gb,
            cpu_cores=dep.hardware.cpu_cores,
            ram_gb=dep.hardware.ram_gb,
            cloud_provider=dep.hardware.cloud_provider.value if hasattr(dep.hardware.cloud_provider, 'value') else dep.hardware.cloud_provider,
            instance_type=dep.hardware.instance_type,
            region=dep.hardware.region,
        ),
        software=SoftwareConfigSchema(
            serving_framework=dep.software.serving_framework.value if hasattr(dep.software.serving_framework, 'value') else dep.software.serving_framework,
            framework_version=dep.software.framework_version,
            quantization=dep.software.quantization,
            quantization_source=dep.software.quantization_source,
            tensor_parallel_size=dep.software.tensor_parallel_size,
            max_model_len=dep.software.max_model_len,
            gpu_memory_utilization=dep.software.gpu_memory_utilization,
        ),
        costs=SelfHostedCostsSchema(
            input_cost_per_million=dep.costs.input_cost_per_million,
            output_cost_per_million=dep.costs.output_cost_per_million,
            hourly_compute_cost=dep.costs.hourly_compute_cost,
            notes=dep.costs.notes,
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


def get_user_id_from_header(x_user_id: str = Header(None)) -> str:
    """Extract user ID from header (or use default for testing)."""
    return x_user_id or "anonymous"


@router.get(
    "",
    response_model=SelfHostedListResponse,
    summary="List user's self-hosted deployments",
    description="Get a list of the current user's self-hosted deployments."
)
async def list_self_hosted(
    x_user_id: str = Header(None, description="User ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List user's self-hosted deployments."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        deployments_data = db.get_self_hosted_by_user(user_id, limit=limit + skip)
        deployments_data = deployments_data[skip:skip + limit]
        
        responses = []
        for data in deployments_data:
            dep = SelfHostedDeployment.from_dict(data)
            responses.append(self_hosted_to_response(dep))
        
        return SelfHostedListResponse(
            deployments=responses,
            total=len(responses)
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/public",
    response_model=SelfHostedListResponse,
    summary="List public self-hosted deployments",
    description="Get a list of public self-hosted deployments."
)
async def list_public_self_hosted(
    base_model_id: Optional[str] = Query(None, description="Filter by base model"),
    limit: int = Query(50, ge=1, le=100)
):
    """List public self-hosted deployments."""
    try:
        db = get_db()
        
        deployments_data = db.get_public_self_hosted(
            base_model_id=base_model_id,
            limit=limit
        )
        
        responses = []
        for data in deployments_data:
            dep = SelfHostedDeployment.from_dict(data)
            responses.append(self_hosted_to_response(dep))
        
        return SelfHostedListResponse(
            deployments=responses,
            total=len(responses)
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.get(
    "/{deployment_id:path}",
    response_model=SelfHostedResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get self-hosted deployment by ID",
    description="Get detailed information about a self-hosted deployment."
)
async def get_self_hosted(
    deployment_id: str,
    x_user_id: str = Header(None)
):
    """Get a specific self-hosted deployment."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        data = db.get_self_hosted_deployment(deployment_id)
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment not found: {deployment_id}"
            )
        
        dep = SelfHostedDeployment.from_dict(data)
        
        # Check visibility
        if not dep.is_visible_to(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this deployment"
            )
        
        return self_hosted_to_response(dep)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database not available: {e}"
        )


@router.post(
    "",
    response_model=SelfHostedResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new self-hosted deployment",
    description="Register a new self-hosted deployment."
)
async def create_self_hosted(
    deployment_data: SelfHostedCreate,
    x_user_id: str = Header(None)
):
    """Create a new self-hosted deployment."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        # Generate ID
        deployment_id = f"{user_id}/{deployment_data.deployment_name}"
        
        # Check if exists
        existing = db.get_self_hosted_deployment(deployment_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Deployment already exists: {deployment_id}"
            )
        
        # Verify base model exists
        base_model = db.get_base_model(deployment_data.base_model_id)
        if not base_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Base model not found: {deployment_data.base_model_id}"
            )
        
        # Create owner
        owner = Owner(
            user_id=user_id,
            is_public=deployment_data.is_public,
        )
        
        # Create deployment
        dep = SelfHostedDeployment(
            id=deployment_id,
            base_model_id=deployment_data.base_model_id,
            owner=owner,
        )
        
        # Set hardware config
        try:
            cloud_provider = CloudProvider(deployment_data.hardware.cloud_provider)
        except ValueError:
            cloud_provider = CloudProvider.OTHER
        
        dep.hardware = HardwareConfig(
            gpu_type=deployment_data.hardware.gpu_type,
            gpu_count=deployment_data.hardware.gpu_count,
            gpu_memory_gb=deployment_data.hardware.gpu_memory_gb,
            cpu_cores=deployment_data.hardware.cpu_cores,
            ram_gb=deployment_data.hardware.ram_gb,
            cloud_provider=cloud_provider,
            instance_type=deployment_data.hardware.instance_type,
            region=deployment_data.hardware.region,
        )
        
        # Set software config
        try:
            framework = ServingFramework(deployment_data.software.serving_framework)
        except ValueError:
            framework = ServingFramework.OTHER
        
        dep.software = SoftwareConfig(
            serving_framework=framework,
            framework_version=deployment_data.software.framework_version,
            quantization=deployment_data.software.quantization,
            quantization_source=deployment_data.software.quantization_source,
            tensor_parallel_size=deployment_data.software.tensor_parallel_size,
            max_model_len=deployment_data.software.max_model_len,
            gpu_memory_utilization=deployment_data.software.gpu_memory_utilization,
        )
        
        # Set costs
        dep.costs = SelfHostedCosts(
            input_cost_per_million=deployment_data.costs.input_cost_per_million,
            output_cost_per_million=deployment_data.costs.output_cost_per_million,
            hourly_compute_cost=deployment_data.costs.hourly_compute_cost,
            notes=deployment_data.costs.notes,
        )
        
        # Save to database
        db.save_self_hosted_deployment(dep.to_dict())
        
        return self_hosted_to_response(dep)
        
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
    summary="Delete a self-hosted deployment",
    description="Remove a self-hosted deployment."
)
async def delete_self_hosted(
    deployment_id: str,
    x_user_id: str = Header(None)
):
    """Delete a self-hosted deployment."""
    try:
        db = get_db()
        user_id = get_user_id_from_header(x_user_id)
        
        existing = db.get_self_hosted_deployment(deployment_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment not found: {deployment_id}"
            )
        
        dep = SelfHostedDeployment.from_dict(existing)
        
        # Check ownership
        if not dep.is_owned_by(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this deployment"
            )
        
        db.delete_self_hosted_deployment(deployment_id)
        
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


@router.post(
    "/estimate-costs",
    response_model=CostEstimateResponse,
    summary="Estimate per-token costs from hourly compute",
    description="Calculate estimated per-token costs based on hourly compute cost and throughput."
)
async def estimate_costs(request: CostEstimateRequest):
    """Estimate per-token costs from hourly compute cost."""
    costs = SelfHostedCosts(
        hourly_compute_cost=request.hourly_compute_cost,
    )
    
    input_cost, output_cost = costs.estimate_costs_from_hourly(
        hourly_cost=request.hourly_compute_cost,
        throughput_tps=request.expected_throughput_tps,
        utilization=request.utilization,
    )
    
    return CostEstimateResponse(
        input_cost_per_million=round(input_cost, 4),
        output_cost_per_million=round(output_cost, 4),
        assumptions={
            "hourly_compute_cost": request.hourly_compute_cost,
            "expected_throughput_tps": request.expected_throughput_tps,
            "utilization": request.utilization,
            "input_output_ratio": 3.0,
            "output_cost_multiplier": 3.0,
        }
    )


