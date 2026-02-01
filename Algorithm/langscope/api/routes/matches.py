"""
Match execution API endpoints.

Provides endpoints for triggering matches and retrieving results.
"""

import asyncio
import logging
import uuid
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, status

logger = logging.getLogger(__name__)

from langscope.api.schemas import (
    MatchTrigger,
    MatchStatus,
    MatchResult,
    MatchListResponse,
    SuccessResponse,
    ErrorResponse,
)
from langscope.api.dependencies import (
    get_db,
    get_models,
    get_model_by_id,
    refresh_models_cache,
    get_llm_caller,
)
from langscope.federation.workflow import MultiPlayerMatchWorkflow, run_tournament
from langscope.federation.router import (
    MatchRouter,
    MatchRouterConfig,
    get_evaluation_type,
    is_ground_truth_domain,
)


router = APIRouter(prefix="/matches", tags=["matches"])


# Track running matches
_running_matches: dict[str, MatchStatus] = {}


@router.post(
    "/trigger",
    response_model=MatchStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a new match",
    description="Start a new multi-player match in the specified domain."
)
async def trigger_match(
    match_request: MatchTrigger,
    background_tasks: BackgroundTasks
):
    """
    Trigger a new multi-player match.
    
    The match runs asynchronously. Use the returned match_id to poll for status.
    """
    match_id = f"match_{uuid.uuid4().hex[:16]}"
    
    # Get models - if specific model_ids provided, get all models (not filtered by domain)
    # since new models may not have domain ratings yet
    if match_request.model_ids:
        all_models = get_models()  # Get all models without domain filter
        selected_models = [
            m for m in all_models 
            if m.model_id in match_request.model_ids
        ]
        if len(selected_models) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Need at least 2 models, got {len(selected_models)}"
            )
        models = selected_models
    else:
        # No specific models requested - use domain-filtered models
        models = get_models(domain=match_request.domain)
        if len(models) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough models for domain {match_request.domain} (need at least 2)"
            )
    
    # Create initial status
    match_status = MatchStatus(
        match_id=match_id,
        status="pending",
        domain=match_request.domain,
        started_at=datetime.utcnow().isoformat() + "Z"
    )
    _running_matches[match_id] = match_status
    
    # Run match in background
    background_tasks.add_task(
        _run_match_task,
        match_id,
        match_request.domain,
        models
    )
    
    return match_status


async def _run_match_task(match_id: str, domain: str, models):
    """Background task to run a match, routing to appropriate workflow."""
    try:
        _running_matches[match_id].status = "running"
        
        # Get database and LLM caller
        try:
            db = get_db()
        except RuntimeError:
            db = None
        
        try:
            llm_caller = get_llm_caller()
        except RuntimeError:
            llm_caller = None
        
        # Use the router to determine workflow type
        evaluation_type = get_evaluation_type(domain)
        
        if evaluation_type == "ground_truth":
            # Use ground truth workflow
            from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
            
            workflow = GroundTruthMatchWorkflow(
                domain=domain,
                models=models,
                db=db,
                llm_caller=llm_caller
            )
            result = await workflow.run_single_match(
                model_ids=[m.model_id for m in models]
            )
        else:
            # Use subjective (LLM-as-judge) workflow
            workflow = MultiPlayerMatchWorkflow(
                domain=domain,
                models=models,
                db=db,
                llm_caller=llm_caller
            )
            result = await workflow.run_single_match()
        
        if result:
            _running_matches[match_id].status = "completed"
            _running_matches[match_id].completed_at = datetime.utcnow().isoformat() + "Z"
            refresh_models_cache()
        else:
            _running_matches[match_id].status = "failed"
            _running_matches[match_id].error = "Match could not be completed"
            
    except Exception as e:
        _running_matches[match_id].status = "failed"
        _running_matches[match_id].error = str(e)


@router.get(
    "/status/{match_id}",
    response_model=MatchStatus,
    responses={404: {"model": ErrorResponse}},
    summary="Get match status",
    description="Get the current status of a running or completed match."
)
async def get_match_status(match_id: str):
    """Get status of a match."""
    if match_id in _running_matches:
        return _running_matches[match_id]
    
    # Check database for completed matches
    try:
        db = get_db()
        match_data = db.get_match(match_id)
        if match_data:
            return MatchStatus(
                match_id=match_id,
                status="completed",
                domain=match_data.get("domain", ""),
                completed_at=match_data.get("timestamp")
            )
    except RuntimeError:
        pass
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Match not found: {match_id}"
    )


@router.get(
    "/{match_id}",
    response_model=MatchResult,
    responses={404: {"model": ErrorResponse}},
    summary="Get match result",
    description="Get the full result of a completed match."
)
async def get_match_result(match_id: str):
    """Get result of a completed match."""
    try:
        db = get_db()
        match_data = db.get_match(match_id)
        
        if not match_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Match not found: {match_id}"
            )
        
        return MatchResult(
            match_id=match_data.get("_id", match_id),
            domain=match_data.get("domain", ""),
            timestamp=match_data.get("timestamp", ""),
            participants=match_data.get("participants", []),
            case_text=match_data.get("prompt", {}).get("case_text", ""),
            question_text=match_data.get("prompt", {}).get("question_text", ""),
            case_creator=match_data.get("prompt", {}).get("case_generator_id", ""),
            question_creator=match_data.get("prompt", {}).get("question_generator_id", ""),
            judges=match_data.get("judgment", {}).get("judges", []),
            raw_ranking=match_data.get("judgment", {}).get("raw_ranking", {}),
            cost_adjusted_ranking=match_data.get("judgment", {}).get("cost_adjusted_ranking", {}),
            pl_strengths=match_data.get("plackett_luce", {}).get("raw_strengths", {}),
            info_bits=match_data.get("meta", {}).get("info_bits", 0.0)
        )
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.get(
    "",
    response_model=MatchListResponse,
    summary="List matches",
    description="Get a list of matches with optional filtering."
)
async def list_matches(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    model_id: Optional[str] = Query(None, description="Filter by participant model"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """List matches with optional filtering."""
    try:
        db = get_db()
        
        if model_id:
            matches = db.get_matches_by_model(model_id, domain=domain, limit=limit)
        elif domain:
            matches = db.get_matches_by_domain(domain, limit=limit, skip=skip)
        else:
            # Get recent matches across all domains
            matches = []
            domains_checked = set()
            for model in get_models():
                for d in model.trueskill_by_domain:
                    if d not in domains_checked:
                        domains_checked.add(d)
                        matches.extend(db.get_matches_by_domain(d, limit=20))
            matches = sorted(
                matches, 
                key=lambda m: m.get("timestamp", ""), 
                reverse=True
            )[:limit]
        
        results = []
        for m in matches:
            results.append(MatchResult(
                match_id=m.get("_id", ""),
                domain=m.get("domain", ""),
                timestamp=m.get("timestamp", ""),
                participants=m.get("participants", []),
                case_text=m.get("prompt", {}).get("case_text", ""),
                question_text=m.get("prompt", {}).get("question_text", ""),
                case_creator=m.get("prompt", {}).get("case_generator_id", ""),
                question_creator=m.get("prompt", {}).get("question_generator_id", ""),
                judges=m.get("judgment", {}).get("judges", []),
                raw_ranking=m.get("judgment", {}).get("raw_ranking", {}),
                cost_adjusted_ranking=m.get("judgment", {}).get("cost_adjusted_ranking", {}),
                pl_strengths=m.get("plackett_luce", {}).get("raw_strengths", {}),
                info_bits=m.get("meta", {}).get("info_bits", 0.0)
            ))
        
        return MatchListResponse(
            matches=results,
            total=len(results)
        )
        
    except RuntimeError as e:
        logger.warning(f"Runtime error listing matches: {e}")
        return MatchListResponse(matches=[], total=0)
    except Exception as e:
        logger.error(f"Error listing matches: {e}", exc_info=True)
        return MatchListResponse(matches=[], total=0)


@router.post(
    "/tournament",
    response_model=MatchStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a tournament",
    description="Run multiple matches in a tournament format."
)
async def trigger_tournament(
    domain: str = Query(..., description="Domain for the tournament"),
    n_rounds: int = Query(10, ge=1, le=100, description="Number of rounds"),
    background_tasks: BackgroundTasks = None
):
    """
    Trigger a tournament of multiple matches.
    
    This runs n_rounds matches in sequence.
    """
    tournament_id = f"tournament_{uuid.uuid4().hex[:12]}"
    
    models = get_models(domain=domain)
    if len(models) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough models for domain {domain} (need at least 2)"
        )
    
    match_status = MatchStatus(
        match_id=tournament_id,
        status="pending",
        domain=domain,
        started_at=datetime.utcnow().isoformat() + "Z"
    )
    _running_matches[tournament_id] = match_status
    
    # Run tournament in background
    if background_tasks:
        background_tasks.add_task(
            _run_tournament_task,
            tournament_id,
            domain,
            models,
            n_rounds
        )
    
    return match_status


async def _run_tournament_task(
    tournament_id: str,
    domain: str,
    models,
    n_rounds: int
):
    """Background task to run a tournament."""
    try:
        _running_matches[tournament_id].status = "running"
        
        try:
            db = get_db()
        except RuntimeError:
            db = None
        
        results = await run_tournament(
            domain=domain,
            models=models,
            n_rounds=n_rounds,
            db=db
        )
        
        _running_matches[tournament_id].status = "completed"
        _running_matches[tournament_id].completed_at = datetime.utcnow().isoformat() + "Z"
        refresh_models_cache()
        
    except Exception as e:
        _running_matches[tournament_id].status = "failed"
        _running_matches[tournament_id].error = str(e)
