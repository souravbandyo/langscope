"""
LangScope - Multi-domain LLM Evaluation Framework

A mathematical framework for multi-domain LLM evaluation using TrueSkill + Plackett-Luce
for multi-player matches with cross-domain transfer learning and peer-federated evaluation.

Key Features:
- TrueSkill rating system with uncertainty quantification
- Plackett-Luce for multi-way rankings (5-6 players)
- Dual rankings: Raw performance + Cost-adjusted
- Cross-domain transfer learning
- Peer-federated evaluation with weighted judge aggregation
- User feedback integration with prediction-delta tracking
"""

__version__ = "0.1.0"
__author__ = "LangScope Team"

from langscope.core.model import LLMModel
from langscope.core.rating import TrueSkillRating, DualTrueSkill

# User Feedback Integration exports
from langscope.feedback.user_feedback import (
    PredictionState,
    FeedbackDelta,
    UserSession,
)
from langscope.feedback.workflow import UserFeedbackWorkflow
from langscope.feedback.use_case import UseCaseAdjustmentManager
from langscope.feedback.judge_calibration import JudgeCalibrator

__all__ = [
    # Core
    "LLMModel",
    "TrueSkillRating", 
    "DualTrueSkill",
    # User Feedback
    "PredictionState",
    "FeedbackDelta",
    "UserSession",
    "UserFeedbackWorkflow",
    "UseCaseAdjustmentManager",
    "JudgeCalibrator",
]


