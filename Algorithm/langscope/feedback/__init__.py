"""
User feedback integration module for LangScope.

Provides prediction-feedback delta tracking, use-case adjustments,
and LLM judge calibration against user ground truth.

Key Components:
- user_feedback: Core dataclasses (PredictionState, FeedbackDelta, UserSession)
- weights: User feedback weighting for TrueSkill updates
- delta: Delta computation with zero-sum conservation
- use_case: Use-case specific adjustments
- accuracy: Prediction accuracy metrics
- judge_calibration: LLM judge calibration
- workflow: Complete user feedback workflow
"""

from langscope.feedback.user_feedback import (
    PredictionState,
    FeedbackDelta,
    UserSession,
    USER_SESSION_SCHEMA,
)
from langscope.feedback.weights import (
    USER_WEIGHT_MULTIPLIER,
    USER_WEIGHT_BASE,
    get_user_feedback_weight,
    apply_user_weighted_trueskill_update,
)
from langscope.feedback.delta import (
    compute_session_deltas,
    compute_user_surprise_score,
    detect_user_specialists,
    validate_zero_sum,
)
from langscope.feedback.use_case import UseCaseAdjustmentManager
from langscope.feedback.accuracy import (
    compute_prediction_accuracy,
    compute_kendall_tau,
)
from langscope.feedback.judge_calibration import JudgeCalibrator
from langscope.feedback.workflow import UserFeedbackWorkflow

__all__ = [
    # Core dataclasses
    "PredictionState",
    "FeedbackDelta",
    "UserSession",
    "USER_SESSION_SCHEMA",
    # Weights
    "USER_WEIGHT_MULTIPLIER",
    "USER_WEIGHT_BASE",
    "get_user_feedback_weight",
    "apply_user_weighted_trueskill_update",
    # Delta
    "compute_session_deltas",
    "compute_user_surprise_score",
    "detect_user_specialists",
    "validate_zero_sum",
    # Use case
    "UseCaseAdjustmentManager",
    # Accuracy
    "compute_prediction_accuracy",
    "compute_kendall_tau",
    # Judge calibration
    "JudgeCalibrator",
    # Workflow
    "UserFeedbackWorkflow",
]
