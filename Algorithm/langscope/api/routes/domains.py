"""
Domain management API endpoints.

Provides CRUD operations for evaluation domains.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from langscope.api.schemas import (
    DomainCreate,
    DomainUpdate,
    DomainResponse,
    DomainListResponse,
    DomainStatisticsSchema,
    DomainSettingsSchema,
    DomainPromptsSchema,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import get_domain_manager, get_models
from langscope.domain.domain_config import Domain, DomainSettings, DomainPrompts


router = APIRouter(prefix="/domains", tags=["domains"])


def domain_to_response(domain: Domain) -> DomainResponse:
    """Convert Domain to DomainResponse schema."""
    return DomainResponse(
        name=domain.name,
        display_name=domain.display_name,
        description=domain.description,
        parent_domain=domain.parent_domain,
        statistics=DomainStatisticsSchema(
            total_matches=domain.statistics.total_matches,
            total_models_evaluated=domain.statistics.total_models_evaluated,
            top_model_raw=domain.statistics.top_model_raw,
            top_model_cost=domain.statistics.top_model_cost,
            last_match_timestamp=domain.statistics.last_match_timestamp
        ),
        created_at=domain.created_at,
        updated_at=domain.updated_at
    )


@router.get(
    "",
    response_model=DomainListResponse,
    summary="List all domains",
    description="Get a list of all available evaluation domains."
)
async def list_domains():
    """List all available domains."""
    manager = get_domain_manager()
    domains = manager.list_domains()
    
    return DomainListResponse(
        domains=domains,
        total=len(domains)
    )


@router.get(
    "/{domain_name}",
    response_model=DomainResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get domain by name",
    description="Get detailed information about a specific domain."
)
async def get_domain(domain_name: str):
    """Get a specific domain by name."""
    manager = get_domain_manager()
    domain = manager.get_domain(domain_name)
    
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_name}"
        )
    
    return domain_to_response(domain)


@router.post(
    "",
    response_model=DomainResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
    summary="Create a new domain",
    description="Create a new evaluation domain."
)
async def create_domain(domain_data: DomainCreate):
    """Create a new domain."""
    manager = get_domain_manager()
    
    # Check if domain already exists
    existing = manager.get_domain(domain_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Domain already exists: {domain_data.name}"
        )
    
    # Convert schema settings to domain config
    settings = None
    if domain_data.settings:
        settings = DomainSettings(
            strata_elite_threshold=domain_data.settings.strata_elite_threshold,
            strata_high_threshold=domain_data.settings.strata_high_threshold,
            strata_mid_threshold=domain_data.settings.strata_mid_threshold,
            players_per_match=domain_data.settings.players_per_match,
            min_players=domain_data.settings.min_players,
            judge_count=domain_data.settings.judge_count
        )
    
    prompts = None
    if domain_data.prompts:
        prompts = DomainPrompts(
            case_prompt=domain_data.prompts.case_prompt,
            question_prompt=domain_data.prompts.question_prompt,
            answer_prompt=domain_data.prompts.answer_prompt,
            judge_prompt=domain_data.prompts.judge_prompt
        )
    
    # Create domain
    domain = manager.create_domain(
        name=domain_data.name,
        display_name=domain_data.display_name,
        description=domain_data.description,
        parent_domain=domain_data.parent_domain,
        template=domain_data.template,
        settings=settings,
        prompts=prompts
    )
    
    return domain_to_response(domain)


@router.patch(
    "/{domain_name}",
    response_model=DomainResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Update a domain",
    description="Update domain properties."
)
async def update_domain(domain_name: str, updates: DomainUpdate):
    """Update an existing domain."""
    manager = get_domain_manager()
    
    # Convert schema settings if provided
    settings = None
    if updates.settings:
        settings = DomainSettings(
            strata_elite_threshold=updates.settings.strata_elite_threshold,
            strata_high_threshold=updates.settings.strata_high_threshold,
            strata_mid_threshold=updates.settings.strata_mid_threshold,
            players_per_match=updates.settings.players_per_match,
            min_players=updates.settings.min_players,
            judge_count=updates.settings.judge_count
        )
    
    prompts = None
    if updates.prompts:
        prompts = DomainPrompts(
            case_prompt=updates.prompts.case_prompt,
            question_prompt=updates.prompts.question_prompt,
            answer_prompt=updates.prompts.answer_prompt,
            judge_prompt=updates.prompts.judge_prompt
        )
    
    domain = manager.update_domain(
        name=domain_name,
        settings=settings,
        prompts=prompts,
        description=updates.description
    )
    
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_name}"
        )
    
    return domain_to_response(domain)


@router.delete(
    "/{domain_name}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    summary="Delete a domain",
    description="Remove a domain from the system."
)
async def delete_domain(domain_name: str):
    """Delete a domain."""
    manager = get_domain_manager()
    
    success = manager.delete_domain(domain_name)
    
    if not success:
        # Check if it's a template domain
        domain = manager.get_domain(domain_name)
        if domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete template domain: {domain_name}"
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_name}"
        )
    
    return SuccessResponse(
        success=True,
        message=f"Domain {domain_name} deleted successfully"
    )


@router.get(
    "/{domain_name}/statistics",
    response_model=DomainStatisticsSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get domain statistics",
    description="Get current statistics for a domain."
)
async def get_domain_statistics(domain_name: str, refresh: bool = Query(False)):
    """Get statistics for a domain."""
    manager = get_domain_manager()
    domain = manager.get_domain(domain_name)
    
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_name}"
        )
    
    # Optionally refresh statistics
    if refresh:
        models = get_models(domain=domain_name)
        manager.update_domain_statistics(domain_name, models)
        domain = manager.get_domain(domain_name)
    
    return DomainStatisticsSchema(
        total_matches=domain.statistics.total_matches,
        total_models_evaluated=domain.statistics.total_models_evaluated,
        top_model_raw=domain.statistics.top_model_raw,
        top_model_cost=domain.statistics.top_model_cost,
        last_match_timestamp=domain.statistics.last_match_timestamp
    )
