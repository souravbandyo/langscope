"""
Complete user feedback integration workflow.

This module orchestrates the full user feedback process:
1. Start session: Record pre-testing predictions for all models
2. Process battles: Apply user-weighted TrueSkill updates
3. Complete session: Compute deltas, verify zero-sum, save to DB
4. Integration with use-case adjustments and judge calibration

The workflow enables the Arena mode where users can:
- Test models with their own judgments
- See how their preferences differ from system predictions
- Contribute to use-case-specific model recommendations
- Help calibrate LLM judges against ground truth
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime

from langscope.feedback.user_feedback import (
    PredictionState,
    FeedbackDelta,
    UserSession,
)
from langscope.feedback.weights import (
    get_user_feedback_weight,
    scale_rating_delta,
)
from langscope.feedback.delta import (
    compute_session_deltas,
    detect_user_specialists,
    detect_user_underperformers,
    summarize_session_deltas,
    ZeroSumViolationError,
)
from langscope.feedback.use_case import UseCaseAdjustmentManager
from langscope.feedback.accuracy import (
    compute_prediction_accuracy,
    compute_kendall_tau,
)
from langscope.feedback.judge_calibration import JudgeCalibrator
from langscope.ranking.trueskill import TrueSkillRating, MultiPlayerTrueSkillUpdater

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


class UserFeedbackWorkflow:
    """
    Complete workflow for integrating user feedback.
    
    This class manages the lifecycle of a user feedback session:
    
    1. **start_session()**: Record predictions before user testing
       - Snapshot current TrueSkill ratings for all models
       - Create session tracking object
    
    2. **process_battle()**: Process each user battle with elevated weight
       - Apply TrueSkill updates with user weight multiplier (2×)
       - Track which models won/lost according to user
    
    3. **complete_session()**: Finalize and learn from the session
       - Compute prediction-feedback deltas (Δᵢ = μᵢ^post - μᵢ^pred)
       - Verify zero-sum conservation (Σ Δᵢ = 0)
       - Update use-case adjustments
       - Compute prediction accuracy metrics
       - Detect specialists
       - Save session to database
    
    Example:
        >>> workflow = UserFeedbackWorkflow(
        ...     domain="medical",
        ...     use_case_manager=use_case_manager,
        ...     judge_calibrator=judge_calibrator
        ... )
        >>> 
        >>> # Start session
        >>> session = workflow.start_session(models, use_case="patient_education")
        >>> 
        >>> # User runs battles
        >>> for battle in user_battles:
        ...     workflow.process_battle(session, battle.players, battle.user_ranking)
        >>> 
        >>> # Complete session
        >>> session = workflow.complete_session(session, models)
        >>> print(f"Prediction accuracy: {session.prediction_accuracy:.1%}")
    """
    
    def __init__(
        self,
        domain: str,
        use_case_manager: Optional[UseCaseAdjustmentManager] = None,
        judge_calibrator: Optional[JudgeCalibrator] = None,
        db: Optional['MongoDB'] = None
    ):
        """
        Initialize the workflow.
        
        Args:
            domain: Domain for evaluation (e.g., "medical", "hindi")
            use_case_manager: Manager for use-case specific adjustments
            judge_calibrator: Calibrator for LLM judge weights
            db: Database instance for persistence
        """
        self.domain = domain
        self.use_case_manager = use_case_manager or UseCaseAdjustmentManager()
        self.judge_calibrator = judge_calibrator or JudgeCalibrator()
        self.db = db
        self.trueskill_updater = MultiPlayerTrueSkillUpdater()
    
    def start_session(
        self,
        models: List['LLMModel'],
        use_case: str,
        user_id: Optional[str] = None
    ) -> UserSession:
        """
        Start a user feedback session, recording initial predictions.
        
        This captures a snapshot of all model ratings before the user
        starts testing, so we can later compute how much ratings changed.
        
        Args:
            models: List of models to track
            use_case: User's use case category (e.g., "patient_education")
            user_id: Optional user identifier for personalization
        
        Returns:
            New UserSession with recorded predictions
        """
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        # Record predictions (pre-testing snapshots)
        predictions = {}
        for model in models:
            ts = model.get_domain_trueskill(self.domain)
            predictions[model.model_id] = PredictionState(
                model_id=model.model_id,
                mu_pred=ts.raw.mu,
                sigma_pred=ts.raw.sigma
            )
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            domain=self.domain,
            use_case=use_case,
            models_tested=[m.model_id for m in models],
            n_battles=0,
            predictions=predictions,
            deltas={},
            timestamp_start=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Started user session {session_id} for domain={self.domain}, use_case={use_case}")
        return session
    
    def process_battle(
        self,
        session: UserSession,
        players: List['LLMModel'],
        user_ranking: Dict[str, int]
    ):
        """
        Process a single user-judged battle.
        
        Updates TrueSkill ratings with elevated user weight (2× base),
        meaning user battles have more impact than automated battles.
        
        Args:
            session: Active user session
            players: Models participating in this battle
            user_ranking: {model_id: rank} where 1=best, from user judgment
        
        Note:
            This modifies the model ratings in place. The session tracks
            that this battle occurred for later delta computation.
        """
        session.n_battles += 1
        
        # Get user weight
        user_weight = get_user_feedback_weight()
        
        # Get current ratings
        ratings = []
        for player in players:
            ts = player.get_domain_trueskill(self.domain)
            ratings.append(TrueSkillRating(mu=ts.raw.mu, sigma=ts.raw.sigma))
        
        # Convert ranking dict to list format
        rank_list = [user_ranking[p.model_id] for p in players]
        
        # Compute base TrueSkill updates
        updated = self.trueskill_updater.update_from_ranking(ratings, rank_list)
        
        # Apply user weight scaling to the delta
        for i, player in enumerate(players):
            old_mu = ratings[i].mu
            new_mu = updated[i].mu
            
            # Scale the change by user weight
            base_delta = new_mu - old_mu
            scaled_delta = scale_rating_delta(base_delta, user_weight)
            final_mu = old_mu + scaled_delta
            
            # Update the model's rating
            player.set_domain_trueskill(
                self.domain,
                raw_mu=final_mu,
                raw_sigma=updated[i].sigma,
                cost_mu=player.get_domain_trueskill(self.domain).cost_adjusted.mu,
                cost_sigma=player.get_domain_trueskill(self.domain).cost_adjusted.sigma
            )
        
        logger.debug(f"Processed battle #{session.n_battles} in session {session.session_id}")
    
    def complete_session(
        self,
        session: UserSession,
        models: List['LLMModel'],
        judge_rankings: Optional[Dict[str, Dict[str, int]]] = None
    ) -> UserSession:
        """
        Complete a user session, computing deltas and updating system.
        
        This is where we learn from the session:
        1. Compute prediction-feedback deltas
        2. Verify zero-sum conservation (critical invariant!)
        3. Update use-case specific adjustments
        4. Compute prediction accuracy metrics
        5. Optionally calibrate judges against user feedback
        6. Save session to database
        
        Args:
            session: The session to complete
            models: Current models with post-feedback ratings
            judge_rankings: Optional {judge_id: {model_id: rank}} for calibration
        
        Returns:
            Updated session with deltas and metrics
        
        Raises:
            ZeroSumViolationError: If delta sum is not zero (indicates a bug)
        """
        session.timestamp_end = datetime.utcnow().isoformat() + "Z"
        
        # Build predictions dict
        predictions = {
            m: (session.predictions[m].mu_pred, session.predictions[m].sigma_pred)
            for m in session.predictions
        }
        
        # Get post-feedback ratings
        post_ratings = {}
        for model in models:
            if model.model_id in session.models_tested:
                ts = model.get_domain_trueskill(self.domain)
                post_ratings[model.model_id] = (ts.raw.mu, ts.raw.sigma)
        
        # Compute deltas (will raise ZeroSumViolationError if conservation violated)
        try:
            deltas, delta_sum = compute_session_deltas(predictions, post_ratings)
            session.deltas = deltas
            session.delta_sum = delta_sum
        except ZeroSumViolationError as e:
            logger.error(f"Zero-sum violation in session {session.session_id}: {e}")
            raise
        
        # Update use-case profiles
        delta_values = {m: d.delta for m, d in deltas.items()}
        self.use_case_manager.add_user_feedback(session.use_case, delta_values)
        
        # Compute accuracy metrics
        predicted_ranking = self._predictions_to_ranking(session.predictions)
        post_feedback_ranking = self._deltas_to_ranking(deltas)
        
        session.prediction_accuracy = compute_prediction_accuracy(
            predicted_ranking, post_feedback_ranking
        )
        session.kendall_tau = compute_kendall_tau(
            predicted_ranking, post_feedback_ranking
        )
        
        # Calibrate judges if rankings provided
        if judge_rankings:
            for judge_id, judge_ranking in judge_rankings.items():
                self.judge_calibrator.record_rankings(
                    judge_id=judge_id,
                    domain=self.domain,
                    judge_ranking=judge_ranking,
                    user_ranking=post_feedback_ranking
                )
        
        # Detect specialists
        specialists = detect_user_specialists(deltas)
        if specialists:
            logger.info(f"Session {session.session_id} found specialists: {specialists}")
        
        # Save to database
        if self.db:
            try:
                self.db.save_user_session(session.to_dict())
                logger.info(f"Saved session {session.session_id} to database")
            except Exception as e:
                logger.error(f"Failed to save session: {e}")
        
        return session
    
    def _predictions_to_ranking(
        self,
        predictions: Dict[str, PredictionState]
    ) -> Dict[str, int]:
        """Convert predictions to ranking (1 = best based on mu_pred)."""
        sorted_models = sorted(
            predictions.keys(),
            key=lambda m: -predictions[m].mu_pred  # Higher mu = better = lower rank
        )
        return {m: i + 1 for i, m in enumerate(sorted_models)}
    
    def _deltas_to_ranking(
        self,
        deltas: Dict[str, FeedbackDelta]
    ) -> Dict[str, int]:
        """Convert post-feedback ratings to ranking (1 = best based on mu_post)."""
        sorted_models = sorted(
            deltas.keys(),
            key=lambda m: -deltas[m].mu_post  # Higher mu = better = lower rank
        )
        return {m: i + 1 for i, m in enumerate(sorted_models)}
    
    def get_session_summary(self, session: UserSession) -> Dict[str, Any]:
        """
        Get a human-readable summary of a completed session.
        
        Args:
            session: Completed session
        
        Returns:
            Summary dictionary
        """
        delta_summary = summarize_session_deltas(session.deltas)
        
        # Find biggest winners/losers
        biggest_winner = max(session.deltas.values(), key=lambda d: d.delta) if session.deltas else None
        biggest_loser = min(session.deltas.values(), key=lambda d: d.delta) if session.deltas else None
        
        return {
            "session_id": session.session_id,
            "domain": session.domain,
            "use_case": session.use_case,
            "n_battles": session.n_battles,
            "n_models": len(session.models_tested),
            "prediction_accuracy": f"{session.prediction_accuracy:.1%}",
            "kendall_tau": f"{session.kendall_tau:.2f}",
            "conservation_satisfied": session.is_conservation_satisfied(),
            "delta_sum": session.delta_sum,
            "biggest_winner": {
                "model": biggest_winner.model_id if biggest_winner else None,
                "delta": biggest_winner.delta if biggest_winner else 0,
                "z_score": biggest_winner.z_score if biggest_winner else 0
            },
            "biggest_loser": {
                "model": biggest_loser.model_id if biggest_loser else None,
                "delta": biggest_loser.delta if biggest_loser else 0,
                "z_score": biggest_loser.z_score if biggest_loser else 0
            },
            "n_specialists": delta_summary.get("n_specialists", 0),
            "n_underperformers": delta_summary.get("n_underperformers", 0)
        }
    
    def get_use_case_recommendations(
        self,
        use_case: str,
        models: List['LLMModel'],
        top_k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Get model recommendations adjusted for a specific use case.
        
        Args:
            use_case: The use case category
            models: Available models
            top_k: Number of recommendations to return
        
        Returns:
            List of (model_id, adjusted_mu, adjustment) tuples, sorted by adjusted_mu
        """
        recommendations = []
        
        for model in models:
            ts = model.get_domain_trueskill(self.domain)
            global_mu = ts.raw.mu
            adjusted_mu = self.use_case_manager.get_adjusted_mu(
                use_case, model.model_id, global_mu
            )
            adjustment = adjusted_mu - global_mu
            
            recommendations.append((model.model_id, adjusted_mu, adjustment))
        
        recommendations.sort(key=lambda x: -x[1])  # Sort by adjusted_mu descending
        return recommendations[:top_k]
    
    def get_calibrated_judge_weights(
        self,
        judge_ids: List[str],
        base_weights: List[float]
    ) -> List[float]:
        """
        Get calibrated weights for a set of judges.
        
        Args:
            judge_ids: List of judge identifiers
            base_weights: Corresponding base weights
        
        Returns:
            List of calibrated weights
        """
        calibrated = []
        for judge_id, base_weight in zip(judge_ids, base_weights):
            adjusted = self.judge_calibrator.adjust_judge_weight(
                base_weight, judge_id, self.domain
            )
            calibrated.append(adjusted)
        return calibrated


def create_feedback_workflow(
    domain: str,
    db: Optional['MongoDB'] = None,
    tau_use_case: float = 10.0,
    calibration_gamma: float = 0.2
) -> UserFeedbackWorkflow:
    """
    Factory function to create a configured feedback workflow.
    
    Args:
        domain: Domain for evaluation
        db: Optional database instance
        tau_use_case: Smoothing parameter for use-case adjustments
        calibration_gamma: Weight adjustment factor for judge calibration
    
    Returns:
        Configured UserFeedbackWorkflow instance
    """
    use_case_manager = UseCaseAdjustmentManager(tau_use_case=tau_use_case)
    judge_calibrator = JudgeCalibrator(gamma=calibration_gamma)
    
    return UserFeedbackWorkflow(
        domain=domain,
        use_case_manager=use_case_manager,
        judge_calibrator=judge_calibrator,
        db=db
    )
