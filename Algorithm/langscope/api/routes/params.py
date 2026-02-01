"""
Parameter Management API Routes.

Provides endpoints for:
- Listing all parameter types
- Getting/setting parameters (with optional domain override)
- Resetting parameters to defaults
- Exporting/importing parameter configurations
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from langscope.config.params import (
    ParameterManager,
    get_parameter_manager,
    ParamType,
    TrueSkillParams,
    StrataParams,
    MatchParams,
    TemperatureParams,
    DimensionWeightParams,
    TransferParams,
    FeedbackParams,
    PenaltyParams,
    ConsistencyParams,
    LongContextParams,
)
from langscope.config.params.manager import DOMAIN_SPECIFIC_PARAMS


router = APIRouter(prefix="/params", tags=["parameters"])


# =============================================================================
# Response Models
# =============================================================================

class ParamTypeInfo(BaseModel):
    """Information about a parameter type."""
    name: str
    description: str
    supports_domain_override: bool


class ParamTypesResponse(BaseModel):
    """Response for listing parameter types."""
    param_types: List[ParamTypeInfo]


class ParamResponse(BaseModel):
    """Response for parameter retrieval."""
    param_type: str
    domain: Optional[str] = None
    params: Dict[str, Any]


class ParamUpdateRequest(BaseModel):
    """Request for updating parameters."""
    params: Dict[str, Any] = Field(..., description="Parameter values to set")
    domain: Optional[str] = Field(None, description="Domain for domain-specific override")


class CacheStatsResponse(BaseModel):
    """Response for cache statistics."""
    total_entries: int
    expired_entries: int
    active_entries: int
    default_ttl: int
    cleanup_interval: int


class ExportResponse(BaseModel):
    """Response for parameter export."""
    params: Dict[str, Any]


class ImportResponse(BaseModel):
    """Response for parameter import."""
    results: Dict[str, bool]


# =============================================================================
# Parameter Type Descriptions
# =============================================================================

PARAM_DESCRIPTIONS = {
    ParamType.TRUESKILL: "TrueSkill algorithm parameters (μ₀, σ₀, β, τ, k)",
    ParamType.STRATA: "Strata thresholds for role assignment (elite, high, mid)",
    ParamType.MATCH: "Match configuration (players, judges, swiss delta)",
    ParamType.TEMPERATURE: "Temperature parameters for softmax and scoring",
    ParamType.DIMENSION_WEIGHTS: "Dimension weights for Combined score calculation",
    ParamType.TRANSFER: "Transfer learning parameters (correlation tau, sigma base)",
    ParamType.FEEDBACK: "User feedback integration parameters",
    ParamType.PENALTY: "Penalty system parameters (judge penalty, outlier threshold)",
    ParamType.CONSISTENCY: "Consistency evaluation parameters (n_runs)",
    ParamType.LONG_CONTEXT: "Long context evaluation parameters (test lengths)",
}


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "",
    response_model=ParamTypesResponse,
    summary="List parameter types",
    description="Get a list of all available parameter types and their metadata."
)
async def list_param_types():
    """List all available parameter types."""
    types = []
    for param_type in ParamType:
        types.append(ParamTypeInfo(
            name=param_type.value,
            description=PARAM_DESCRIPTIONS.get(param_type, ""),
            supports_domain_override=param_type in DOMAIN_SPECIFIC_PARAMS
        ))
    return ParamTypesResponse(param_types=types)


@router.get(
    "/{param_type}",
    response_model=ParamResponse,
    summary="Get parameters",
    description="Get current parameter values for a specific type. Optionally specify a domain for domain-specific overrides."
)
async def get_params(
    param_type: str,
    domain: Optional[str] = Query(None, description="Domain for domain-specific params")
):
    """Get parameters for a type and optional domain."""
    try:
        ptype = ParamType(param_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter type: {param_type}. Valid types: {[p.value for p in ParamType]}"
        )
    
    manager = get_parameter_manager()
    params = manager.get_params(ptype, domain)
    
    return ParamResponse(
        param_type=param_type,
        domain=domain if ptype in DOMAIN_SPECIFIC_PARAMS else None,
        params=params.model_dump()
    )


@router.put(
    "/{param_type}",
    response_model=ParamResponse,
    summary="Update parameters",
    description="Update parameter values for a specific type. Optionally specify a domain for domain-specific override."
)
async def update_params(
    param_type: str,
    request: ParamUpdateRequest
):
    """Update parameters for a type."""
    try:
        ptype = ParamType(param_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter type: {param_type}"
        )
    
    manager = get_parameter_manager()
    
    try:
        success = manager.set_params(ptype, request.params, request.domain)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter values: {str(e)}"
        )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to save parameters"
        )
    
    # Return the updated params
    params = manager.get_params(ptype, request.domain)
    return ParamResponse(
        param_type=param_type,
        domain=request.domain if ptype in DOMAIN_SPECIFIC_PARAMS else None,
        params=params.model_dump()
    )


@router.delete(
    "/{param_type}/domain/{domain}",
    summary="Remove domain override",
    description="Remove a domain-specific parameter override, reverting to global defaults."
)
async def remove_domain_override(
    param_type: str,
    domain: str
):
    """Remove a domain-specific parameter override."""
    try:
        ptype = ParamType(param_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter type: {param_type}"
        )
    
    if ptype not in DOMAIN_SPECIFIC_PARAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Parameter type {param_type} does not support domain overrides"
        )
    
    manager = get_parameter_manager()
    success = manager.reset_to_defaults(ptype, domain)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to remove domain override"
        )
    
    return {"message": f"Domain override removed for {param_type}:{domain}"}


@router.post(
    "/reset/{param_type}",
    response_model=ParamResponse,
    summary="Reset to defaults",
    description="Reset parameters to their default values."
)
async def reset_to_defaults(
    param_type: str,
    domain: Optional[str] = Query(None, description="If provided, removes domain override")
):
    """Reset parameters to defaults."""
    try:
        ptype = ParamType(param_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter type: {param_type}"
        )
    
    manager = get_parameter_manager()
    success = manager.reset_to_defaults(ptype, domain)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to reset parameters"
        )
    
    # Return the reset params
    params = manager.get_params(ptype, domain)
    return ParamResponse(
        param_type=param_type,
        domain=domain if ptype in DOMAIN_SPECIFIC_PARAMS and domain else None,
        params=params.model_dump()
    )


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get cache statistics",
    description="Get statistics about the parameter cache."
)
async def get_cache_stats():
    """Get cache statistics."""
    manager = get_parameter_manager()
    stats = manager.get_cache_stats()
    return CacheStatsResponse(**stats)


@router.post(
    "/cache/invalidate",
    summary="Invalidate cache",
    description="Invalidate cached parameters. Can specify param_type and/or domain."
)
async def invalidate_cache(
    param_type: Optional[str] = Query(None, description="Parameter type to invalidate"),
    domain: Optional[str] = Query(None, description="Domain to invalidate")
):
    """Invalidate cache entries."""
    manager = get_parameter_manager()
    
    if param_type:
        try:
            ParamType(param_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameter type: {param_type}"
            )
    
    count = manager.invalidate_cache(param_type, domain)
    return {"message": f"Invalidated {count} cache entries"}


@router.get(
    "/export",
    response_model=ExportResponse,
    summary="Export all parameters",
    description="Export all parameter configurations for backup."
)
async def export_params():
    """Export all parameters."""
    manager = get_parameter_manager()
    params = manager.export_all_params()
    return ExportResponse(params=params)


@router.post(
    "/import",
    response_model=ImportResponse,
    summary="Import parameters",
    description="Import parameter configurations from a backup."
)
async def import_params(
    params: Dict[str, Any] = Body(..., description="Parameter configurations to import")
):
    """Import parameters from backup."""
    manager = get_parameter_manager()
    results = manager.import_params(params)
    return ImportResponse(results=results)


