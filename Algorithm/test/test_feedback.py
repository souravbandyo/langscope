"""
Unit tests for langscope/feedback/ modules.

Tests:
- user_feedback.py: User feedback integration
- workflow.py: Feedback workflow
- judge_calibration.py: Judge calibration
- accuracy.py: Prediction accuracy
- delta.py: Feedback delta tracking
- use_case.py: Use case recommendations
- weights.py: Weight management

Coverage target: 85%
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


# =============================================================================
# User Feedback Tests (FEED-001 to FEED-008)
# =============================================================================

class TestUserFeedback:
    """Test user feedback integration."""
    
    def test_user_feedback_module_import(self):
        """Test user feedback module imports."""
        from langscope.feedback import user_feedback
        assert user_feedback is not None
    
    def test_feedback_workflow_import(self):
        """Test workflow module imports."""
        from langscope.feedback import workflow
        assert workflow is not None
    
    def test_prediction_state_creation(self):
        """FEED-001: Create prediction state."""
        from langscope.feedback.user_feedback import PredictionState
        
        state = PredictionState(
            model_id="gpt-4",
            mu_pred=1550.0,
            sigma_pred=100.0
        )
        
        assert state.model_id == "gpt-4"
        assert state.mu_pred == 1550.0
        assert state.sigma_pred == 100.0
    
    def test_prediction_state_to_dict(self):
        """Test PredictionState serialization."""
        from langscope.feedback.user_feedback import PredictionState
        
        state = PredictionState(
            model_id="gpt-4",
            mu_pred=1550.0,
            sigma_pred=100.0
        )
        
        d = state.to_dict()
        assert "model_id" in d
        assert "mu_pred" in d
        assert "sigma_pred" in d
        assert "timestamp" in d
    
    def test_prediction_state_from_dict(self):
        """Test PredictionState deserialization."""
        from langscope.feedback.user_feedback import PredictionState
        
        d = {
            "model_id": "claude-3",
            "mu_pred": 1520.0,
            "sigma_pred": 120.0,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        state = PredictionState.from_dict(d)
        assert state.model_id == "claude-3"
        assert state.mu_pred == 1520.0
    
    def test_feedback_delta_compute(self):
        """FEED-002: Compute feedback delta."""
        from langscope.feedback.user_feedback import FeedbackDelta
        
        delta = FeedbackDelta.compute(
            model_id="gpt-4",
            mu_pred=1500.0,
            sigma_pred=100.0,
            mu_post=1520.0,
            sigma_post=95.0
        )
        
        assert delta.model_id == "gpt-4"
        assert delta.mu_pred == 1500.0
        assert delta.mu_post == 1520.0
        assert delta.delta == 20.0  # 1520 - 1500
    
    def test_feedback_delta_to_dict(self):
        """Test FeedbackDelta serialization."""
        from langscope.feedback.user_feedback import FeedbackDelta
        
        delta = FeedbackDelta.compute(
            model_id="gpt-4",
            mu_pred=1500.0,
            sigma_pred=100.0,
            mu_post=1510.0,
            sigma_post=98.0
        )
        
        d = delta.to_dict()
        assert "model_id" in d
        assert "delta" in d
        assert "z_score" in d


# =============================================================================
# Judge Calibration Tests (FEED-020 to FEED-025)
# =============================================================================

class TestJudgeCalibration:
    """Test judge calibration functions."""
    
    def test_judge_calibration_module_import(self):
        """Test judge calibration module imports."""
        from langscope.feedback import judge_calibration
        assert judge_calibration is not None
    
    def test_judge_calibrator_init(self):
        """Test JudgeCalibrator initialization."""
        from langscope.feedback.judge_calibration import JudgeCalibrator
        
        calibrator = JudgeCalibrator()
        assert calibrator is not None


# =============================================================================
# Accuracy Tests (FEED-030 to FEED-033)
# =============================================================================

class TestAccuracy:
    """Test prediction accuracy functions."""
    
    def test_accuracy_module_import(self):
        """Test accuracy module imports."""
        from langscope.feedback import accuracy
        assert accuracy is not None
    
    def test_prediction_accuracy_perfect(self):
        """FEED-030: Perfect prediction accuracy."""
        from langscope.feedback.accuracy import compute_prediction_accuracy
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 3}
        
        acc = compute_prediction_accuracy(predicted, actual)
        assert acc == 1.0
    
    def test_prediction_accuracy_reversed(self):
        """Test completely reversed prediction."""
        from langscope.feedback.accuracy import compute_prediction_accuracy
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 3, "b": 2, "c": 1}
        
        acc = compute_prediction_accuracy(predicted, actual)
        # 2 pairs reversed, 1 pair correct (b vs others)
        assert acc < 1.0
    
    def test_prediction_accuracy_partial(self):
        """Test partial prediction accuracy."""
        from langscope.feedback.accuracy import compute_prediction_accuracy
        
        predicted = {"gpt-4": 1, "claude": 2, "llama": 3}
        actual = {"gpt-4": 1, "claude": 3, "llama": 2}  # Claude/Llama swapped
        
        acc = compute_prediction_accuracy(predicted, actual)
        # 2 out of 3 pairs correct
        assert abs(acc - 0.667) < 0.01
    
    def test_prediction_accuracy_single_model(self):
        """Test accuracy with single model."""
        from langscope.feedback.accuracy import compute_prediction_accuracy
        
        predicted = {"a": 1}
        actual = {"a": 1}
        
        acc = compute_prediction_accuracy(predicted, actual)
        assert acc == 1.0  # No pairs to compare
    
    def test_kendall_tau_perfect(self):
        """Test Kendall's tau with perfect agreement."""
        from langscope.feedback.accuracy import compute_kendall_tau
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 3}
        
        tau = compute_kendall_tau(predicted, actual)
        assert tau == 1.0
    
    def test_kendall_tau_reversed(self):
        """Test Kendall's tau with reversed ranking."""
        from langscope.feedback.accuracy import compute_kendall_tau
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 3, "b": 2, "c": 1}
        
        tau = compute_kendall_tau(predicted, actual)
        assert tau == -1.0
    
    def test_kendall_tau_range(self):
        """Test Kendall's tau is in valid range."""
        from langscope.feedback.accuracy import compute_kendall_tau
        
        predicted = {"a": 1, "b": 2, "c": 3, "d": 4}
        actual = {"a": 2, "b": 1, "c": 4, "d": 3}
        
        tau = compute_kendall_tau(predicted, actual)
        assert -1 <= tau <= 1
    
    def test_spearman_rho_perfect(self):
        """Test Spearman's rho with perfect agreement."""
        from langscope.feedback.accuracy import compute_spearman_rho
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 3}
        
        rho = compute_spearman_rho(predicted, actual)
        assert abs(rho - 1.0) < 0.001
    
    def test_spearman_rho_reversed(self):
        """Test Spearman's rho with reversed ranking."""
        from langscope.feedback.accuracy import compute_spearman_rho
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 3, "b": 2, "c": 1}
        
        rho = compute_spearman_rho(predicted, actual)
        assert abs(rho - (-1.0)) < 0.001
    
    def test_top_k_accuracy_full_overlap(self):
        """Test top-k accuracy with full overlap."""
        from langscope.feedback.accuracy import compute_top_k_accuracy
        
        predicted = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        actual = {"a": 2, "b": 1, "c": 3, "d": 5, "e": 4}
        
        acc = compute_top_k_accuracy(predicted, actual, k=3)
        assert acc == 1.0  # Both have {a, b, c} in top 3
    
    def test_top_k_accuracy_no_overlap(self):
        """Test top-k accuracy with no overlap."""
        from langscope.feedback.accuracy import compute_top_k_accuracy
        
        predicted = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        actual = {"a": 4, "b": 5, "c": 6, "d": 1, "e": 2}
        
        acc = compute_top_k_accuracy(predicted, actual, k=2)
        # Predicted top-2: {a, b}, Actual top-2: {d, e}
        assert acc == 0.0
    
    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        from langscope.feedback.accuracy import compute_ndcg
        
        predicted = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 3}
        
        ndcg = compute_ndcg(predicted, actual)
        assert abs(ndcg - 1.0) < 0.001
    
    def test_ndcg_range(self):
        """Test NDCG is in valid range."""
        from langscope.feedback.accuracy import compute_ndcg
        
        predicted = {"a": 1, "b": 2, "c": 3, "d": 4}
        actual = {"a": 4, "b": 3, "c": 2, "d": 1}
        
        ndcg = compute_ndcg(predicted, actual)
        assert 0 <= ndcg <= 1
    
    def test_aggregate_accuracy_metrics(self):
        """FEED-031: Track accuracy over time."""
        from langscope.feedback.accuracy import aggregate_accuracy_metrics
        
        sessions = [
            ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
            ({"a": 1, "b": 2}, {"a": 2, "b": 1}),
            ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}),
        ]
        
        metrics = aggregate_accuracy_metrics(sessions)
        
        assert "mean_accuracy" in metrics
        assert "mean_kendall_tau" in metrics
        assert "mean_spearman_rho" in metrics
        assert "n_sessions" in metrics
        assert metrics["n_sessions"] == 3
    
    def test_aggregate_accuracy_empty(self):
        """Test aggregate metrics with empty sessions."""
        from langscope.feedback.accuracy import aggregate_accuracy_metrics
        
        metrics = aggregate_accuracy_metrics([])
        
        assert metrics["n_sessions"] == 0
        assert metrics["mean_accuracy"] == 0.0


# =============================================================================
# Delta Tests (FEED-040 to FEED-043)
# =============================================================================

class TestDelta:
    """Test feedback delta tracking."""
    
    def test_delta_module_import(self):
        """Test delta module imports."""
        from langscope.feedback import delta
        assert delta is not None
    
    def test_compute_session_deltas(self):
        """FEED-040: Compute session deltas."""
        from langscope.feedback.delta import compute_session_deltas
        
        # Two models with zero-sum delta
        predictions = {
            "gpt-4": (1500.0, 100.0),
            "claude": (1500.0, 100.0)
        }
        post_ratings = {
            "gpt-4": (1520.0, 95.0),  # +20
            "claude": (1480.0, 95.0)  # -20
        }
        
        deltas, delta_sum = compute_session_deltas(
            predictions, post_ratings, validate_conservation=False
        )
        
        assert "gpt-4" in deltas
        assert "claude" in deltas
        assert deltas["gpt-4"].delta == 20.0
        assert deltas["claude"].delta == -20.0
    
    def test_zero_sum_violation(self):
        """Test zero-sum violation detection."""
        from langscope.feedback.delta import compute_session_deltas, ZeroSumViolationError
        
        # Non-zero-sum deltas
        predictions = {"a": (1500.0, 100.0)}
        post_ratings = {"a": (1520.0, 95.0)}  # +20 with no counterbalance
        
        with pytest.raises(ZeroSumViolationError):
            compute_session_deltas(predictions, post_ratings, validate_conservation=True)
    
    def test_validate_zero_sum(self):
        """Test zero-sum validation."""
        from langscope.feedback.delta import validate_zero_sum, ZeroSumViolationError
        
        # Valid zero-sum
        valid_deltas = {"a": 20.0, "b": -15.0, "c": -5.0}
        assert validate_zero_sum(valid_deltas)
        
        # Invalid (non-zero-sum) - raises exception
        invalid_deltas = {"a": 20.0, "b": -10.0}  # Sum = 10
        with pytest.raises(ZeroSumViolationError):
            validate_zero_sum(invalid_deltas)


# =============================================================================
# Use Case Tests (FEED-050 to FEED-054)
# =============================================================================

class TestUseCase:
    """Test use case recommendations."""
    
    def test_use_case_module_import(self):
        """Test use case module imports."""
        from langscope.feedback import use_case
        assert use_case is not None
    
    def test_use_case_profile_creation(self):
        """FEED-050: Create use case profile."""
        from langscope.feedback.use_case import UseCaseProfile
        
        profile = UseCaseProfile(use_case="code_generation")
        
        assert profile.use_case == "code_generation"
        assert profile.n_users == 0
    
    def test_use_case_profile_add_feedback(self):
        """Test adding feedback to use case profile."""
        from langscope.feedback.use_case import UseCaseProfile
        
        profile = UseCaseProfile(use_case="code_generation")
        
        # Add feedback
        profile.add_feedback({"gpt-4": 20.0, "claude": -20.0})
        
        assert profile.n_users == 1
        assert profile.get_average_delta("gpt-4") == 20.0
    
    def test_use_case_profile_beta(self):
        """Test use case profile beta (smoothing factor)."""
        from langscope.feedback.use_case import UseCaseProfile
        
        profile = UseCaseProfile(use_case="code_generation")
        
        # No users: beta = 0
        assert profile.get_beta() == 0.0
        
        # Add 10 users
        for _ in range(10):
            profile.add_feedback({"gpt-4": 5.0})
        
        # beta = 10 / (10 + 10) = 0.5
        assert abs(profile.get_beta() - 0.5) < 0.01
    
    def test_use_case_profile_to_dict(self):
        """Test UseCaseProfile serialization."""
        from langscope.feedback.use_case import UseCaseProfile
        
        profile = UseCaseProfile(use_case="medical_qa")
        profile.add_feedback({"gpt-4": 15.0})
        
        d = profile.to_dict()
        assert "use_case" in d
        assert "n_users" in d
        assert "beta" in d
    
    def test_use_case_adjustment_manager(self):
        """Test UseCaseAdjustmentManager."""
        from langscope.feedback.use_case import UseCaseAdjustmentManager
        
        manager = UseCaseAdjustmentManager()
        assert manager is not None


# =============================================================================
# Weights Tests (FEED-060 to FEED-063)
# =============================================================================

class TestWeights:
    """Test weight management."""
    
    def test_weights_module_import(self):
        """Test weights module imports."""
        from langscope.feedback import weights
        assert weights is not None
    
    def test_default_weights(self):
        """FEED-060: Get default weights for all dimensions."""
        from langscope.core.dimensions import DEFAULT_COMBINED_WEIGHTS
        
        expected_dims = [
            "raw_quality", "cost_adjusted", "latency", "ttft",
            "consistency", "token_efficiency", "instruction_following",
            "hallucination_resistance", "long_context"
        ]
        
        for dim in expected_dims:
            assert dim in DEFAULT_COMBINED_WEIGHTS
    
    def test_weights_normalization(self):
        """FEED-062: Weight normalization sums to ~1.0."""
        from langscope.core.dimensions import normalize_weights
        
        weights = {
            "a": 2.0,
            "b": 3.0,
            "c": 5.0
        }
        
        normalized = normalize_weights(weights)
        
        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert abs(normalized["a"] - 0.2) < 0.001
        assert abs(normalized["b"] - 0.3) < 0.001
        assert abs(normalized["c"] - 0.5) < 0.001
    
    def test_weights_normalization_equal(self):
        """Test equal weight normalization."""
        from langscope.core.dimensions import normalize_weights
        
        weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        
        normalized = normalize_weights(weights)
        
        for v in normalized.values():
            assert abs(v - 0.25) < 0.001
    
    def test_weights_normalization_empty(self):
        """Test empty weight normalization."""
        from langscope.core.dimensions import normalize_weights
        
        normalized = normalize_weights({})
        assert normalized == {}
    
    def test_combined_weights_sum(self):
        """Test that default combined weights sum to 1.0."""
        from langscope.core.dimensions import DEFAULT_COMBINED_WEIGHTS
        
        total = sum(DEFAULT_COMBINED_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
    
    def test_get_user_feedback_weight(self):
        """Test user feedback weight multiplier."""
        from langscope.feedback.weights import get_user_feedback_weight
        
        weight = get_user_feedback_weight()
        assert weight >= 1.0  # User feedback should have at least base weight


# =============================================================================
# Workflow Tests (FEED-010 to FEED-016)
# =============================================================================

class TestFeedbackWorkflow:
    """Test feedback workflow."""
    
    def test_user_feedback_workflow_init(self):
        """FEED-010: Initialize UserFeedbackWorkflow."""
        from langscope.feedback.workflow import UserFeedbackWorkflow
        
        workflow = UserFeedbackWorkflow(domain="coding")
        assert workflow is not None
        assert workflow.domain == "coding"


# =============================================================================
# Integration Tests
# =============================================================================

class TestFeedbackIntegration:
    """Integration tests for feedback system."""
    
    def test_delta_flow(self):
        """Test complete delta calculation flow."""
        from langscope.feedback.user_feedback import FeedbackDelta
        from langscope.feedback.delta import compute_session_deltas
        
        # Simulate predictions and post-ratings
        predictions = {
            "gpt-4": (1500.0, 100.0),
            "claude": (1500.0, 100.0),
            "llama": (1500.0, 100.0)
        }
        # Zero-sum: +10 + 5 + (-15) = 0
        post_ratings = {
            "gpt-4": (1510.0, 95.0),   # +10
            "claude": (1505.0, 95.0),  # +5
            "llama": (1485.0, 95.0)    # -15
        }
        
        deltas, delta_sum = compute_session_deltas(
            predictions, post_ratings, validate_conservation=True
        )
        
        assert len(deltas) == 3
        assert abs(delta_sum) < 0.001  # Zero-sum satisfied
    
    def test_accuracy_from_session(self):
        """Test computing accuracy from session."""
        from langscope.feedback.accuracy import compute_prediction_accuracy
        
        # Simulate predicted vs actual rankings
        predicted = {"gpt-4": 1, "claude-3": 2, "llama": 3}
        actual = {"gpt-4": 1, "claude-3": 2, "llama": 3}
        
        acc = compute_prediction_accuracy(predicted, actual)
        assert acc == 1.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
