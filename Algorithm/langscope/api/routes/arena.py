"""
Arena mode API endpoints.

Provides user feedback session management for the Arena mode.

Arena mode allows users to:
1. Start a testing session
2. Run battles and provide judgments
3. Complete the session and see how their preferences differ from predictions
4. Contribute to use-case specific recommendations
"""

from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, status

from langscope.api.schemas import (
    ArenaSessionStart,
    ArenaSessionStartResponse,
    ArenaBattle,
    ArenaBattleResponse,
    ArenaSessionComplete,
    ArenaSessionResult,
    TrueSkillRatingSchema,
    ErrorResponse,
)
from langscope.api.dependencies import (
    get_models,
    get_model_by_id,
    get_feedback_workflow,
    get_use_case_manager,
    store_session,
    get_session,
    remove_session,
    get_db,
    refresh_models_cache,
)
from langscope.feedback.workflow import UserFeedbackWorkflow
from langscope.feedback.user_feedback import UserSession


router = APIRouter(prefix="/arena", tags=["arena"])


@router.post(
    "/session/start",
    response_model=ArenaSessionStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start an arena session",
    description="Start a new user feedback session for arena mode testing."
)
async def start_session(session_request: ArenaSessionStart):
    """
    Start a new arena session.
    
    This captures a snapshot of all model ratings before the user
    starts testing, so we can later compute how much ratings changed.
    
    The session tracks:
    - Pre-testing predictions for all models
    - User's use case category
    - Domain being evaluated
    """
    # Get models
    if session_request.model_ids:
        models = [get_model_by_id(mid) for mid in session_request.model_ids]
        models = [m for m in models if m is not None]
    else:
        models = get_models(domain=session_request.domain)
    
    if len(models) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 models for arena testing"
        )
    
    # Create workflow and start session
    workflow = get_feedback_workflow(session_request.domain)
    session = workflow.start_session(
        models=models,
        use_case=session_request.use_case,
        user_id=session_request.user_id
    )
    
    # Store session state
    store_session(session.session_id, {
        "session": session,
        "models": models,
        "workflow": workflow,
        "domain": session_request.domain
    })
    
    # Build predictions response
    predictions = {
        model_id: TrueSkillRatingSchema(
            mu=pred.mu_pred,
            sigma=pred.sigma_pred
        )
        for model_id, pred in session.predictions.items()
    }
    
    return ArenaSessionStartResponse(
        session_id=session.session_id,
        domain=session_request.domain,
        use_case=session_request.use_case,
        models_available=[m.model_id for m in models],
        predictions=predictions,
        started_at=session.timestamp_start
    )


@router.post(
    "/session/{session_id}/battle",
    response_model=ArenaBattleResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    summary="Submit a battle judgment",
    description="Submit user's ranking for a battle in the arena session."
)
async def submit_battle(session_id: str, battle: ArenaBattle):
    """
    Submit a battle judgment.
    
    The user provides their ranking of the participating models
    (1=best, 2=second best, etc.).
    
    Each battle updates the TrueSkill ratings with elevated user weight (2×),
    meaning user battles have more impact than automated battles.
    """
    session_data = get_session(session_id)
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    session: UserSession = session_data["session"]
    models = session_data["models"]
    workflow: UserFeedbackWorkflow = session_data["workflow"]
    
    # Validate participant IDs
    model_map = {m.model_id: m for m in models}
    players = []
    for pid in battle.participant_ids:
        if pid not in model_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model in battle: {pid}"
            )
        players.append(model_map[pid])
    
    # Validate ranking
    if set(battle.user_ranking.keys()) != set(battle.participant_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ranking must include all participants"
        )
    
    ranks = sorted(battle.user_ranking.values())
    expected_ranks = list(range(1, len(battle.participant_ids) + 1))
    if ranks != expected_ranks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ranking: must be 1 to {len(battle.participant_ids)}"
        )
    
    # Process battle
    workflow.process_battle(session, players, battle.user_ranking)
    
    # Update stored session
    store_session(session_id, session_data)
    
    return ArenaBattleResponse(
        battle_number=session.n_battles,
        participants_updated=battle.participant_ids,
        message=f"Battle {session.n_battles} processed successfully"
    )


@router.post(
    "/session/{session_id}/complete",
    response_model=ArenaSessionResult,
    responses={404: {"model": ErrorResponse}},
    summary="Complete an arena session",
    description="Complete the session, compute deltas, and save results."
)
async def complete_session(session_id: str, complete_request: ArenaSessionComplete = None):
    """
    Complete an arena session.
    
    This finalizes the session and learns from it:
    1. Compute prediction-feedback deltas (Δᵢ = μᵢ^post - μᵢ^pred)
    2. Verify zero-sum conservation (Σ Δᵢ = 0)
    3. Update use-case specific adjustments
    4. Compute prediction accuracy metrics
    5. Save session to database
    """
    session_data = get_session(session_id)
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    session: UserSession = session_data["session"]
    models = session_data["models"]
    workflow: UserFeedbackWorkflow = session_data["workflow"]
    
    if session.n_battles == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot complete session with no battles"
        )
    
    # Complete the session
    judge_rankings = None
    if complete_request and complete_request.judge_rankings:
        judge_rankings = complete_request.judge_rankings
    
    try:
        session = workflow.complete_session(
            session, models, judge_rankings
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error completing session: {str(e)}"
        )
    
    # Save models to database
    try:
        db = get_db()
        for model in models:
            db.save_model(model.to_dict())
        refresh_models_cache()
    except RuntimeError:
        pass
    
    # Get session summary
    summary = workflow.get_session_summary(session)
    
    # Build deltas response
    deltas = {
        model_id: {
            "delta": delta.delta,
            "z_score": delta.z_score,
            "mu_pred": delta.mu_pred,
            "mu_post": delta.mu_post
        }
        for model_id, delta in session.deltas.items()
    }
    
    # Clean up session from active sessions
    remove_session(session_id)
    
    return ArenaSessionResult(
        session_id=session.session_id,
        domain=session.domain,
        use_case=session.use_case,
        n_battles=session.n_battles,
        n_models=len(session.models_tested),
        prediction_accuracy=session.prediction_accuracy,
        kendall_tau=session.kendall_tau,
        conservation_satisfied=session.is_conservation_satisfied(),
        delta_sum=session.delta_sum,
        biggest_winner=summary["biggest_winner"],
        biggest_loser=summary["biggest_loser"],
        n_specialists=summary["n_specialists"],
        n_underperformers=summary["n_underperformers"],
        deltas=deltas
    )


@router.get(
    "/session/{session_id}/results",
    response_model=ArenaSessionResult,
    responses={404: {"model": ErrorResponse}},
    summary="Get session results",
    description="Get the results of a completed session."
)
async def get_session_results(session_id: str):
    """
    Get results of a completed session.
    
    Retrieves the session from the database if it has been completed.
    """
    # Check active sessions first
    session_data = get_session(session_id)
    if session_data:
        session: UserSession = session_data["session"]
        
        if not session.timestamp_end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is still active. Complete it first."
            )
        
        # Session was completed but not cleaned up
        workflow: UserFeedbackWorkflow = session_data["workflow"]
        summary = workflow.get_session_summary(session)
        
        deltas = {
            model_id: {
                "delta": delta.delta,
                "z_score": delta.z_score,
                "mu_pred": delta.mu_pred,
                "mu_post": delta.mu_post
            }
            for model_id, delta in session.deltas.items()
        }
        
        return ArenaSessionResult(
            session_id=session.session_id,
            domain=session.domain,
            use_case=session.use_case,
            n_battles=session.n_battles,
            n_models=len(session.models_tested),
            prediction_accuracy=session.prediction_accuracy,
            kendall_tau=session.kendall_tau,
            conservation_satisfied=session.is_conservation_satisfied(),
            delta_sum=session.delta_sum,
            biggest_winner=summary["biggest_winner"],
            biggest_loser=summary["biggest_loser"],
            n_specialists=summary["n_specialists"],
            n_underperformers=summary["n_underperformers"],
            deltas=deltas
        )
    
    # Check database
    try:
        db = get_db()
        session_doc = db.get_user_session(session_id)
        
        if not session_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        
        return ArenaSessionResult(
            session_id=session_doc.get("session_id", session_id),
            domain=session_doc.get("domain", ""),
            use_case=session_doc.get("use_case", ""),
            n_battles=session_doc.get("n_battles", 0),
            n_models=len(session_doc.get("models_tested", [])),
            prediction_accuracy=session_doc.get("prediction_accuracy", 0.0),
            kendall_tau=session_doc.get("kendall_tau", 0.0),
            conservation_satisfied=abs(session_doc.get("delta_sum", 0.0)) < 1e-6,
            delta_sum=session_doc.get("delta_sum", 0.0),
            biggest_winner={},
            biggest_loser={},
            n_specialists=0,
            n_underperformers=0,
            deltas=session_doc.get("deltas", {})
        )
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )


@router.get(
    "/session/{session_id}/status",
    summary="Get session status",
    description="Get the current status of an arena session."
)
async def get_session_status(session_id: str):
    """Get current status of an arena session."""
    session_data = get_session(session_id)
    
    if session_data:
        session: UserSession = session_data["session"]
        return {
            "session_id": session.session_id,
            "status": "completed" if session.timestamp_end else "active",
            "domain": session.domain,
            "use_case": session.use_case,
            "n_battles": session.n_battles,
            "n_models": len(session.models_tested),
            "started_at": session.timestamp_start,
            "completed_at": session.timestamp_end
        }
    
    # Check database
    try:
        db = get_db()
        session_doc = db.get_user_session(session_id)
        
        if session_doc:
            return {
                "session_id": session_id,
                "status": "completed",
                "domain": session_doc.get("domain", ""),
                "use_case": session_doc.get("use_case", ""),
                "n_battles": session_doc.get("n_battles", 0),
                "n_models": len(session_doc.get("models_tested", [])),
                "started_at": session_doc.get("timestamp_start"),
                "completed_at": session_doc.get("timestamp_end")
            }
    except RuntimeError:
        pass
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session not found: {session_id}"
    )


@router.get(
    "/sessions",
    summary="List arena sessions",
    description="List completed arena sessions with optional filtering."
)
async def list_sessions(
    use_case: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 50
):
    """List completed arena sessions."""
    try:
        db = get_db()
        
        if use_case:
            sessions = db.get_sessions_by_use_case(use_case, domain=domain, limit=limit)
        else:
            # Get recent sessions (would need a general query method)
            sessions = []
            
        return {
            "sessions": [
                {
                    "session_id": s.get("session_id", ""),
                    "domain": s.get("domain", ""),
                    "use_case": s.get("use_case", ""),
                    "n_battles": s.get("n_battles", 0),
                    "prediction_accuracy": s.get("prediction_accuracy", 0.0),
                    "completed_at": s.get("timestamp_end")
                }
                for s in sessions
            ],
            "total": len(sessions)
        }
        
    except RuntimeError:
        return {"sessions": [], "total": 0}
