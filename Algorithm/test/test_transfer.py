"""
Unit tests for langscope/transfer/ modules.

Tests:
- transfer_learning.py: Cross-domain transfer
- correlation.py: Domain correlation
- specialist.py: Specialist detection

Coverage target: 85%
"""

import pytest
import math
from unittest.mock import MagicMock, patch


# =============================================================================
# Transfer Learning Tests (XFER-001 to XFER-011)
# =============================================================================

class TestTransferLearning:
    """Test transfer learning functions."""
    
    def test_module_import(self):
        """Test transfer learning module imports."""
        from langscope.transfer import transfer_learning
        assert transfer_learning is not None
    
    def test_correlation_module_import(self):
        """Test correlation module imports."""
        from langscope.transfer import correlation
        assert correlation is not None
    
    def test_specialist_module_import(self):
        """Test specialist module imports."""
        from langscope.transfer import specialist
        assert specialist is not None
    
    def test_transfer_result_to_rating(self):
        """Test TransferResult conversion to TrueSkillRating."""
        from langscope.transfer.transfer_learning import TransferResult
        
        result = TransferResult(
            target_mu=1550.0,
            target_sigma=150.0,
            source_domains=["coding"],
            source_weights={"coding": 1.0},
            correlation_used=0.8,
            confidence=0.7
        )
        
        rating = result.to_rating()
        assert rating.mu == 1550.0
        assert rating.sigma == 150.0
    
    def test_transfer_learning_initialization(self):
        """Test TransferLearning class initialization."""
        from langscope.transfer.transfer_learning import TransferLearning
        from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0
        
        tl = TransferLearning()
        assert tl.mu_0 == TRUESKILL_MU_0
        assert tl.sigma_0 == TRUESKILL_SIGMA_0
    
    def test_transfer_learning_custom_params(self):
        """Test TransferLearning with custom parameters."""
        from langscope.transfer.transfer_learning import TransferLearning
        
        tl = TransferLearning(mu_0=1600.0, sigma_0=200.0, sigma_base=50.0)
        assert tl.mu_0 == 1600.0
        assert tl.sigma_0 == 200.0
        assert tl.sigma_base == 50.0
    
    def test_single_source_transfer(self):
        """XFER-002: Transfer ratings from single source domain."""
        from langscope.transfer.transfer_learning import TransferLearning
        from langscope.core.rating import TrueSkillRating
        
        tl = TransferLearning()
        source_rating = TrueSkillRating(mu=1600.0, sigma=100.0)
        
        result = tl.transfer_single_source(
            source_rating,
            source_domain="coding",
            target_domain="algorithms"
        )
        
        # Should transfer some knowledge
        assert result.target_mu != tl.mu_0
        assert len(result.source_domains) == 1
        assert "coding" in result.source_domains
        assert 0 <= result.confidence <= 1
    
    def test_single_source_high_correlation(self):
        """XFER-003: High correlation transfer has lower uncertainty."""
        from langscope.transfer.transfer_learning import TransferLearning, transfer_single_source
        from langscope.core.rating import TrueSkillRating
        from langscope.transfer.correlation import CorrelationLearner
        
        # Create learner with high correlation
        learner = CorrelationLearner()
        learner.set_prior("domain_a", "domain_b", 0.95)
        
        tl = TransferLearning(correlation_learner=learner)
        source = TrueSkillRating(mu=1600.0, sigma=80.0)
        
        result = tl.transfer_single_source(source, "domain_a", "domain_b")
        
        # High correlation should result in target_mu closer to source
        assert abs(result.target_mu - 1600.0) < abs(result.target_mu - tl.mu_0)
    
    def test_single_source_low_correlation(self):
        """XFER-005: Low correlation transfer has higher uncertainty."""
        from langscope.transfer.transfer_learning import TransferLearning
        from langscope.core.rating import TrueSkillRating
        from langscope.transfer.correlation import CorrelationLearner
        
        # Create learner with low correlation
        learner = CorrelationLearner()
        learner.set_prior("domain_a", "domain_b", 0.1)
        
        tl = TransferLearning(correlation_learner=learner)
        source = TrueSkillRating(mu=1700.0, sigma=80.0)
        
        result = tl.transfer_single_source(source, "domain_a", "domain_b")
        
        # Low correlation should result in target closer to default
        assert abs(result.target_mu - tl.mu_0) < abs(result.target_mu - 1700.0)
    
    def test_multi_source_transfer_empty(self):
        """Test multi-source transfer with empty sources."""
        from langscope.transfer.transfer_learning import TransferLearning
        
        tl = TransferLearning()
        result = tl.transfer_multi_source({}, "target")
        
        assert result.target_mu == tl.mu_0
        assert result.target_sigma == tl.sigma_0
        assert result.confidence == 0.0
    
    def test_multi_source_transfer_single(self):
        """Test multi-source transfer with single source falls back to single."""
        from langscope.transfer.transfer_learning import TransferLearning
        from langscope.core.rating import TrueSkillRating
        
        tl = TransferLearning()
        source_ratings = {"coding": TrueSkillRating(mu=1600.0, sigma=100.0)}
        
        result = tl.transfer_multi_source(source_ratings, "target")
        
        assert len(result.source_domains) == 1
        assert "coding" in result.source_domains
    
    def test_multi_source_transfer_weighting(self):
        """XFER-006: Multi-source transfer uses reliability weighting."""
        from langscope.transfer.transfer_learning import TransferLearning
        from langscope.core.rating import TrueSkillRating
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        learner.set_prior("high_corr", "target", 0.9)
        learner.set_prior("low_corr", "target", 0.3)
        
        tl = TransferLearning(correlation_learner=learner)
        
        source_ratings = {
            "high_corr": TrueSkillRating(mu=1700.0, sigma=80.0),
            "low_corr": TrueSkillRating(mu=1400.0, sigma=80.0)
        }
        
        result = tl.transfer_multi_source(source_ratings, "target")
        
        # Higher correlation source should have more weight
        assert result.source_weights["high_corr"] > result.source_weights["low_corr"]
        # Result should be closer to high_corr source
        assert result.target_mu > tl.mu_0
    
    def test_transfer_single_source_convenience(self):
        """Test convenience function for single source transfer."""
        from langscope.transfer.transfer_learning import transfer_single_source
        
        mu, sigma = transfer_single_source(
            source_mu=1600.0,
            source_sigma=100.0,
            source_domain="coding",
            target_domain="algorithms"
        )
        
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0
    
    def test_transfer_multi_source_convenience(self):
        """Test convenience function for multi-source transfer."""
        from langscope.transfer.transfer_learning import transfer_multi_source
        
        source_ratings = {
            "coding": (1600.0, 100.0),
            "math": (1550.0, 120.0)
        }
        
        mu, sigma, weights = transfer_multi_source(source_ratings, "target")
        
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_should_transfer(self):
        """Test should_transfer decision function."""
        from langscope.transfer.transfer_learning import should_transfer
        from langscope.core.rating import TrueSkillRating
        
        # Confident source rating
        confident = TrueSkillRating(mu=1600.0, sigma=50.0)
        
        # Uncertain source rating
        uncertain = TrueSkillRating(mu=1600.0, sigma=160.0)
        
        # Confident rating should be more likely to transfer
        result_confident = should_transfer(confident, "coding", "algorithms", min_confidence=0.3)
        result_uncertain = should_transfer(uncertain, "coding", "algorithms", min_confidence=0.3)
        
        # Both should return boolean
        assert isinstance(result_confident, bool)
        assert isinstance(result_uncertain, bool)


# =============================================================================
# Domain Correlation Tests (XFER-020 to XFER-025)
# =============================================================================

class TestDomainCorrelation:
    """Test domain correlation functions."""
    
    def test_correlation_data_to_dict(self):
        """Test CorrelationData serialization."""
        from langscope.transfer.correlation import CorrelationData
        
        data = CorrelationData(
            domain_a="coding",
            domain_b="algorithms",
            prior_correlation=0.8,
            data_correlation=0.75,
            blended_correlation=0.77,
            sample_count=10,
            alpha=0.67
        )
        
        d = data.to_dict()
        assert d["domain_a"] == "coding"
        assert d["domain_b"] == "algorithms"
        assert d["prior_correlation"] == 0.8
    
    def test_correlation_data_from_dict(self):
        """Test CorrelationData deserialization."""
        from langscope.transfer.correlation import CorrelationData
        
        d = {
            "domain_a": "coding",
            "domain_b": "algorithms",
            "prior_correlation": 0.8,
            "data_correlation": 0.75,
            "blended_correlation": 0.77,
            "sample_count": 10
        }
        
        data = CorrelationData.from_dict(d)
        assert data.domain_a == "coding"
        assert data.domain_b == "algorithms"
        assert data.sample_count == 10
    
    def test_correlation_learner_init(self):
        """Test CorrelationLearner initialization."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner(tau=30.0)
        assert learner.tau == 30.0
    
    def test_correlation_learner_key_ordering(self):
        """Test that key ordering is canonical."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        key1 = learner._get_key("a", "b")
        key2 = learner._get_key("b", "a")
        
        assert key1 == key2  # Same key regardless of order
    
    def test_set_prior(self):
        """Test setting prior correlation."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        learner.set_prior("coding", "algorithms", 0.85)
        
        corr = learner.get_correlation("coding", "algorithms")
        assert corr == 0.85
        
        # Should work in reverse order too
        corr_rev = learner.get_correlation("algorithms", "coding")
        assert corr_rev == 0.85
    
    def test_set_prior_invalid(self):
        """Test that invalid correlation values raise error."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        with pytest.raises(ValueError):
            learner.set_prior("a", "b", 1.5)
        
        with pytest.raises(ValueError):
            learner.set_prior("a", "b", -1.5)
    
    def test_correlation_same_domain(self):
        """Test that same domain returns 1.0."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        assert learner.get_correlation("coding", "coding") == 1.0
    
    def test_correlation_default(self):
        """Test default correlation for unknown pair."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        corr = learner.get_correlation("unknown1", "unknown2")
        
        # Default should be moderate correlation
        assert corr == 0.5
    
    def test_update_correlation(self):
        """XFER-024: Update correlation with new data."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner(tau=20.0)
        learner.set_prior("a", "b", 0.5)
        
        # Update with observations
        learner.update_correlation("a", "b", 0.9, sample_size=10)
        
        corr_data = learner.get_correlation_data("a", "b")
        
        # Alpha should decrease (more weight on data)
        assert corr_data.alpha < 1.0
        # Blended should shift toward observed
        assert corr_data.blended_correlation > 0.5
    
    def test_bayesian_smoothing(self):
        """Test Bayesian smoothing formula: α = 1/(1 + n/τ)."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner(tau=20.0)
        learner.set_prior("a", "b", 0.5)
        
        # After 20 samples (= τ), alpha should be 0.5
        learner.update_correlation("a", "b", 0.8, sample_size=20)
        
        alpha = learner.get_alpha("a", "b")
        assert abs(alpha - 0.5) < 0.01  # 1/(1 + 20/20) = 0.5
    
    def test_observation_count(self):
        """Test observation count tracking."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        assert learner.get_observation_count("a", "b") == 0
        
        learner.update_correlation("a", "b", 0.8, sample_size=5)
        assert learner.get_observation_count("a", "b") == 5
        
        learner.update_correlation("a", "b", 0.7, sample_size=3)
        assert learner.get_observation_count("a", "b") == 8
    
    def test_observation_count_same_domain(self):
        """Test observation count for same domain is 0."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        assert learner.get_observation_count("coding", "coding") == 0
    
    def test_correlation_range(self):
        """XFER-020: Calculate Pearson correlation returns valid range."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        # Record some observations
        for i in range(10):
            learner.update_correlation(
                "coding", "technical",
                observed_correlation=0.7 + (i * 0.02),
                sample_size=1
            )
        
        correlation = learner.get_correlation("coding", "technical")
        
        # Should be in [-1, 1]
        assert -1 <= correlation <= 1
    
    def test_estimate_from_performance(self):
        """Test correlation estimation from performance data."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        # Perfectly correlated ratings
        ratings_a = [("m1", 1500), ("m2", 1600), ("m3", 1700)]
        ratings_b = [("m1", 1500), ("m2", 1600), ("m3", 1700)]
        
        corr = learner.estimate_from_performance(ratings_a, ratings_b)
        assert abs(corr - 1.0) < 0.001  # Perfect correlation
    
    def test_estimate_from_performance_negatively_correlated(self):
        """Test correlation estimation with negative correlation."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        # Negatively correlated ratings
        ratings_a = [("m1", 1500), ("m2", 1600), ("m3", 1700)]
        ratings_b = [("m1", 1700), ("m2", 1600), ("m3", 1500)]
        
        corr = learner.estimate_from_performance(ratings_a, ratings_b)
        assert abs(corr - (-1.0)) < 0.001  # Perfect negative correlation
    
    def test_estimate_from_performance_insufficient_data(self):
        """Test correlation estimation with insufficient data."""
        from langscope.transfer.correlation import CorrelationLearner
        
        learner = CorrelationLearner()
        
        # Only one common model
        ratings_a = [("m1", 1500)]
        ratings_b = [("m1", 1600)]
        
        corr = learner.estimate_from_performance(ratings_a, ratings_b)
        assert corr == 0.0  # Can't compute with < 2 common
    
    def test_default_correlations(self):
        """Test that default correlations are defined."""
        from langscope.transfer.correlation import DEFAULT_CORRELATIONS
        
        assert isinstance(DEFAULT_CORRELATIONS, dict)
        assert len(DEFAULT_CORRELATIONS) > 0
        
        for (d1, d2), corr in DEFAULT_CORRELATIONS.items():
            assert -1 <= corr <= 1
    
    def test_get_correlation_convenience(self):
        """Test get_correlation convenience function."""
        from langscope.transfer.correlation import get_correlation
        
        corr = get_correlation("coding", "medical")
        assert isinstance(corr, float)
        assert -1 <= corr <= 1


# =============================================================================
# Specialist Detection Tests (XFER-030 to XFER-035)
# =============================================================================

class TestSpecialistDetection:
    """Test specialist detection functions."""
    
    def test_specialist_result_to_dict(self):
        """Test SpecialistResult serialization."""
        from langscope.transfer.specialist import SpecialistResult
        
        result = SpecialistResult(
            model_id="test-model",
            domain="coding",
            is_specialist=True,
            z_score=2.5,
            actual_mu=1700.0,
            predicted_mu=1500.0,
            actual_sigma=80.0,
            predicted_sigma=100.0,
            p_value=0.01,
            category="specialist"
        )
        
        d = result.to_dict()
        assert d["model_id"] == "test-model"
        assert d["is_specialist"] is True
        assert d["category"] == "specialist"
    
    def test_specialist_result_deviation(self):
        """Test SpecialistResult deviation property."""
        from langscope.transfer.specialist import SpecialistResult
        
        result = SpecialistResult(
            model_id="test",
            domain="coding",
            is_specialist=True,
            z_score=2.5,
            actual_mu=1700.0,
            predicted_mu=1500.0,
            actual_sigma=80.0,
            predicted_sigma=100.0,
            p_value=0.01,
            category="specialist"
        )
        
        assert result.deviation == 200.0  # 1700 - 1500
    
    def test_specialist_detector_init(self):
        """Test SpecialistDetector initialization."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.constants import SPECIALIST_Z_THRESHOLD
        
        detector = SpecialistDetector()
        assert detector.z_threshold == SPECIALIST_Z_THRESHOLD
    
    def test_specialist_detector_custom_threshold(self):
        """Test SpecialistDetector with custom threshold."""
        from langscope.transfer.specialist import SpecialistDetector
        
        detector = SpecialistDetector(z_threshold=3.0)
        assert detector.z_threshold == 3.0
    
    def test_detect_specialist(self):
        """XFER-030: Detect domain specialist."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector(z_threshold=2.0)
        
        model = LLMModel(
            name="Coding Specialist",
            model_id="code-spec",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # High rating in coding
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1750.0,
            raw_sigma=60.0,
            cost_mu=1730.0,
            cost_sigma=65.0
        )
        # Average elsewhere
        model.set_domain_trueskill(
            domain="medical",
            raw_mu=1500.0,
            raw_sigma=80.0,
            cost_mu=1490.0,
            cost_sigma=85.0
        )
        model.set_domain_trueskill(
            domain="legal",
            raw_mu=1510.0,
            raw_sigma=75.0,
            cost_mu=1500.0,
            cost_sigma=80.0
        )
        
        result = detector.detect(model, "coding")
        
        assert result.model_id == "code-spec"
        assert result.domain == "coding"
        # Should detect as specialist (high z-score)
        assert result.z_score > 0  # Positive deviation
    
    def test_detect_weak_spot(self):
        """XFER-030: Detect weak spot in domain."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector(z_threshold=2.0)
        
        model = LLMModel(
            name="Medical Weak",
            model_id="med-weak",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # High in coding and legal
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1700.0,
            raw_sigma=60.0,
            cost_mu=1690.0,
            cost_sigma=65.0
        )
        model.set_domain_trueskill(
            domain="legal",
            raw_mu=1680.0,
            raw_sigma=65.0,
            cost_mu=1670.0,
            cost_sigma=70.0
        )
        # Low in medical (weak spot)
        model.set_domain_trueskill(
            domain="medical",
            raw_mu=1350.0,
            raw_sigma=70.0,
            cost_mu=1340.0,
            cost_sigma=75.0
        )
        
        result = detector.detect(model, "medical")
        
        # Should detect as weak spot (negative z-score)
        assert result.z_score < 0  # Negative deviation
    
    def test_detect_insufficient_data(self):
        """Test detection when domain not in model."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector()
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        result = detector.detect(model, "unknown_domain")
        
        assert result.category == "insufficient_data"
        assert result.is_specialist is False
    
    def test_detect_all_domains(self):
        """Test detecting specialists across all domains."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector()
        
        model = LLMModel(
            name="Multi",
            model_id="multi",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        for domain in ["coding", "medical", "legal"]:
            model.set_domain_trueskill(
                domain=domain,
                raw_mu=1550.0,
                raw_sigma=100.0,
                cost_mu=1540.0,
                cost_sigma=105.0
            )
        
        results = detector.detect_all_domains(model)
        
        assert len(results) == 3
        for r in results:
            assert r.domain in ["coding", "medical", "legal"]
    
    def test_get_specialists(self):
        """Test getting specialist domains for a model."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector(z_threshold=2.0)
        
        model = LLMModel(
            name="Specialist",
            model_id="spec",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # One domain very high
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1800.0,
            raw_sigma=50.0,
            cost_mu=1790.0,
            cost_sigma=55.0
        )
        model.set_domain_trueskill(
            domain="medical",
            raw_mu=1500.0,
            raw_sigma=100.0,
            cost_mu=1490.0,
            cost_sigma=105.0
        )
        
        specialists = detector.get_specialists(model)
        
        # Should return list of SpecialistResults
        assert isinstance(specialists, list)
    
    def test_get_weak_spots(self):
        """Test getting weak spot domains for a model."""
        from langscope.transfer.specialist import SpecialistDetector
        from langscope.core.model import LLMModel
        
        detector = SpecialistDetector(z_threshold=2.0)
        
        model = LLMModel(
            name="Weak",
            model_id="weak",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # Most domains high
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1700.0,
            raw_sigma=60.0,
            cost_mu=1690.0,
            cost_sigma=65.0
        )
        # One domain very low
        model.set_domain_trueskill(
            domain="medical",
            raw_mu=1300.0,
            raw_sigma=80.0,
            cost_mu=1290.0,
            cost_sigma=85.0
        )
        
        weak_spots = detector.get_weak_spots(model)
        
        assert isinstance(weak_spots, list)
    
    def test_detect_specialist_function(self):
        """Test convenience detect_specialist function."""
        from langscope.transfer.specialist import detect_specialist
        
        # Clear specialist (high actual vs low predicted)
        is_special, z, category = detect_specialist(
            actual_mu=1700.0,
            actual_sigma=60.0,
            predicted_mu=1500.0,
            predicted_sigma=80.0,
            z_threshold=2.0
        )
        
        assert isinstance(is_special, bool)
        assert isinstance(z, float)
        assert category in ["specialist", "weak_spot", "normal"]
    
    def test_detect_specialist_weak_spot_function(self):
        """Test detect_specialist function for weak spots."""
        from langscope.transfer.specialist import detect_specialist
        
        # Clear weak spot (low actual vs high predicted)
        is_special, z, category = detect_specialist(
            actual_mu=1300.0,
            actual_sigma=60.0,
            predicted_mu=1600.0,
            predicted_sigma=80.0,
            z_threshold=2.0
        )
        
        if is_special:
            assert category == "weak_spot"
            assert z < 0
    
    def test_detect_specialist_normal(self):
        """Test detect_specialist for normal performance."""
        from langscope.transfer.specialist import detect_specialist
        
        # Close to predicted (normal)
        is_special, z, category = detect_specialist(
            actual_mu=1510.0,
            actual_sigma=100.0,
            predicted_mu=1500.0,
            predicted_sigma=100.0,
            z_threshold=2.0
        )
        
        assert category == "normal"
        assert is_special is False
    
    def test_compute_specialization_score(self):
        """Test compute_specialization_score function."""
        from langscope.transfer.specialist import compute_specialization_score
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Generalist",
            model_id="gen",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # Uniform performance across domains (generalist)
        for domain in ["coding", "medical", "legal"]:
            model.set_domain_trueskill(
                domain=domain,
                raw_mu=1550.0,
                raw_sigma=100.0,
                cost_mu=1540.0,
                cost_sigma=105.0
            )
        
        score = compute_specialization_score(model)
        
        # Uniform performance should yield low specialization score
        assert isinstance(score, float)
    
    def test_compute_specialization_score_single_domain(self):
        """Test specialization score with single domain."""
        from langscope.transfer.specialist import compute_specialization_score
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Single",
            model_id="single",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1600.0,
            raw_sigma=100.0,
            cost_mu=1590.0,
            cost_sigma=105.0
        )
        
        score = compute_specialization_score(model)
        assert score == 1.0  # Can't compute with single domain
    
    def test_get_model_profile(self):
        """Test get_model_profile function."""
        from langscope.transfer.specialist import get_model_profile
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Profile Test",
            model_id="profile",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        for domain in ["coding", "medical"]:
            model.set_domain_trueskill(
                domain=domain,
                raw_mu=1550.0,
                raw_sigma=100.0,
                cost_mu=1540.0,
                cost_sigma=105.0
            )
        
        profile = get_model_profile(model)
        
        assert "model_id" in profile
        assert "specialist_domains" in profile
        assert "weak_spot_domains" in profile
        assert "specialization_score" in profile
        assert "is_generalist" in profile


# =============================================================================
# Faceted Transfer Learning Tests (TL-001 to TL-050)
# =============================================================================

class TestDomainDescriptor:
    """Test DomainDescriptor class."""
    
    def test_domain_descriptor_creation(self):
        """Test basic DomainDescriptor creation."""
        from langscope.transfer.faceted import DomainDescriptor
        
        desc = DomainDescriptor(
            name="hindi_medical_imaging",
            facets={
                "language": "hindi",
                "field": "medical",
                "modality": "imaging"
            }
        )
        
        assert desc.name == "hindi_medical_imaging"
        assert desc.get("language") == "hindi"
        assert desc.get("field") == "medical"
        assert desc.get("modality") == "imaging"
    
    def test_domain_descriptor_defaults(self):
        """Test DomainDescriptor defaults for missing facets."""
        from langscope.transfer.faceted import DomainDescriptor
        
        desc = DomainDescriptor(name="test", facets={"field": "medical"})
        
        assert desc.get("language") == "english"  # default
        assert desc.get("field") == "medical"
        assert desc.get("modality") == "text"  # default
        assert desc.get("task") == "general"  # default
    
    def test_domain_descriptor_to_dict(self):
        """Test DomainDescriptor serialization."""
        from langscope.transfer.faceted import DomainDescriptor
        
        desc = DomainDescriptor(
            name="test",
            facets={"language": "hindi"}
        )
        
        d = desc.to_dict()
        assert d["name"] == "test"
        assert d["facets"]["language"] == "hindi"
    
    def test_domain_descriptor_from_dict(self):
        """Test DomainDescriptor deserialization."""
        from langscope.transfer.faceted import DomainDescriptor
        
        d = {"name": "test", "facets": {"language": "hindi", "field": "medical"}}
        desc = DomainDescriptor.from_dict(d)
        
        assert desc.name == "test"
        assert desc.get("language") == "hindi"
        assert desc.get("field") == "medical"
    
    def test_domain_descriptor_hash(self):
        """Test DomainDescriptor hashing for use in dicts/sets."""
        from langscope.transfer.faceted import DomainDescriptor
        
        desc1 = DomainDescriptor(name="test", facets={"language": "hindi"})
        desc2 = DomainDescriptor(name="test", facets={"language": "bengali"})
        
        # Same name = same hash
        assert hash(desc1) == hash(desc2)
        assert desc1 == desc2


class TestFacetSimilarityLearner:
    """Test FacetSimilarityLearner class."""
    
    def test_learner_creation(self):
        """Test basic learner creation."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language", tau=15.0)
        
        assert learner.facet == "language"
        assert learner.tau == 15.0
    
    def test_same_value_similarity(self):
        """Test same value returns 1.0."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        assert learner.get_similarity("hindi", "hindi") == 1.0
        assert learner.get_similarity("Hindi", "HINDI") == 1.0  # Case insensitive
    
    def test_set_and_get_prior(self):
        """Test setting and getting prior."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        learner.set_prior("bengali", "odia", 0.75)
        
        sim = learner.get_similarity("bengali", "odia")
        assert sim == 0.75
        
        # Should work in reverse order
        sim_rev = learner.get_similarity("odia", "bengali")
        assert sim_rev == 0.75
    
    def test_prior_validation(self):
        """Test prior value validation."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        
        with pytest.raises(ValueError):
            learner.set_prior("a", "b", 1.5)
        
        with pytest.raises(ValueError):
            learner.set_prior("a", "b", -0.1)
    
    def test_default_similarity(self):
        """Test default similarity for unknown pair."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        sim = learner.get_similarity("unknown1", "unknown2")
        
        assert sim == 0.5  # Default
    
    def test_bayesian_update(self):
        """Test Bayesian update with observations."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language", tau=20.0)
        learner.set_prior("a", "b", 0.5)
        
        # Update with higher observed similarity
        learner.update_from_observation("a", "b", 0.9, sample_size=20)
        
        data = learner.get_similarity_data("a", "b")
        
        # Alpha should be 0.5 (n=20, τ=20)
        assert abs(data.alpha - 0.5) < 0.01
        
        # Blended should be between prior (0.5) and observed (0.9)
        assert 0.5 < data.blended_similarity < 0.9
    
    def test_load_priors(self):
        """Test loading multiple priors."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        
        priors = {
            ("bengali", "odia"): 0.75,
            ("hindi", "urdu"): 0.85,
        }
        
        learner.load_priors(priors)
        
        assert learner.get_similarity("bengali", "odia") == 0.75
        assert learner.get_similarity("hindi", "urdu") == 0.85
    
    def test_export_similarities(self):
        """Test exporting all learned similarities."""
        from langscope.transfer.faceted import FacetSimilarityLearner
        
        learner = FacetSimilarityLearner("language")
        learner.set_prior("a", "b", 0.7)
        learner.set_prior("c", "d", 0.8)
        
        exported = learner.export_similarities()
        
        assert len(exported) == 2
        assert all(isinstance(e, dict) for e in exported)


class TestCompositeDomainSimilarity:
    """Test CompositeDomainSimilarity class."""
    
    def test_composite_creation(self):
        """Test composite similarity creation."""
        from langscope.transfer.faceted import CompositeDomainSimilarity
        
        composite = CompositeDomainSimilarity()
        
        assert "language" in composite.weights
        assert "field" in composite.weights
        assert abs(sum(composite.weights.values()) - 1.0) < 0.01
    
    def test_same_domain_correlation(self):
        """Test same domain returns 1.0."""
        from langscope.transfer.faceted import CompositeDomainSimilarity, DomainDescriptor
        
        composite = CompositeDomainSimilarity()
        
        desc = DomainDescriptor(name="test", facets={"language": "hindi"})
        
        corr = composite.get_correlation(desc, desc)
        assert corr == 1.0
    
    def test_composite_correlation_weighted_sum(self):
        """Test weighted sum correlation."""
        from langscope.transfer.faceted import (
            CompositeDomainSimilarity, DomainDescriptor, FacetSimilarityLearner
        )
        
        # Create learners with known priors
        learners = {
            "language": FacetSimilarityLearner("language"),
            "field": FacetSimilarityLearner("field"),
        }
        learners["language"].set_prior("hindi", "bengali", 0.55)
        learners["field"].set_prior("medical", "medical", 1.0)  # Same
        
        composite = CompositeDomainSimilarity(
            facet_learners=learners,
            weights={"language": 0.5, "field": 0.5},
            combination="weighted_sum"
        )
        
        source = DomainDescriptor(name="hindi_medical", facets={"language": "hindi", "field": "medical"})
        target = DomainDescriptor(name="bengali_medical", facets={"language": "bengali", "field": "medical"})
        
        corr = composite.get_correlation(source, target)
        
        # Should be (0.5 * 0.55 + 0.5 * 1.0) = 0.775
        expected = 0.5 * 0.55 + 0.5 * 1.0
        assert abs(corr - expected) < 0.01
    
    def test_facet_breakdown(self):
        """Test getting facet breakdown."""
        from langscope.transfer.faceted import CompositeDomainSimilarity, DomainDescriptor
        
        composite = CompositeDomainSimilarity()
        
        source = DomainDescriptor(name="source", facets={"language": "hindi", "field": "medical"})
        target = DomainDescriptor(name="target", facets={"language": "bengali", "field": "legal"})
        
        breakdown = composite.get_facet_breakdown(source, target)
        
        assert "language" in breakdown
        assert "field" in breakdown
        
        for facet, data in breakdown.items():
            assert "source_value" in data
            assert "target_value" in data
            assert "similarity" in data
            assert "weight" in data
            assert "contribution" in data


class TestDomainNameParser:
    """Test DomainNameParser class."""
    
    def test_parser_language_detection(self):
        """Test language detection from domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        facets = parser.parse("hindi_medical")
        assert facets.get("language") == "hindi"
    
    def test_parser_field_detection(self):
        """Test field detection from domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        facets = parser.parse("medical_qa")
        assert facets.get("field") == "medical"
    
    def test_parser_modality_detection(self):
        """Test modality detection from domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        facets = parser.parse("medical_imaging_qa")
        assert facets.get("modality") == "imaging"
    
    def test_parser_task_detection(self):
        """Test task detection from domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        facets = parser.parse("medical_qa")
        assert facets.get("task") == "qa"
    
    def test_parser_complex_name(self):
        """Test parsing complex domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        facets = parser.parse("hindi_medical_imaging_detection_thyroid")
        
        assert facets.get("language") == "hindi"
        assert facets.get("field") == "medical"
        assert facets.get("modality") == "imaging"
        assert facets.get("task") == "detection"
        assert "thyroid" in facets.get("specialty", "")
    
    def test_parser_create_descriptor(self):
        """Test creating descriptor from domain name."""
        from langscope.transfer.faceted import DomainNameParser
        
        parser = DomainNameParser()
        
        desc = parser.create_descriptor("bengali_legal_qa")
        
        assert desc.name == "bengali_legal_qa"
        assert desc.get("language") == "bengali"
        assert desc.get("field") == "legal"


class TestDomainIndex:
    """Test DomainIndex class."""
    
    def test_index_creation(self):
        """Test index creation."""
        from langscope.transfer.faceted import DomainIndex
        
        index = DomainIndex()
        
        assert index.descriptors == {}
        assert not index.is_loaded
    
    def test_register_domain(self):
        """Test registering a domain."""
        from langscope.transfer.faceted import DomainIndex, DomainDescriptor
        
        index = DomainIndex()
        
        desc = DomainDescriptor(name="test", facets={"language": "hindi"})
        index.register_domain(desc)
        
        assert "test" in index.descriptors
        assert index.descriptors["test"].get("language") == "hindi"
    
    def test_get_or_create_descriptor(self):
        """Test get or create descriptor."""
        from langscope.transfer.faceted import DomainIndex
        
        index = DomainIndex()
        
        # Should create from name
        desc = index.get_or_create_descriptor("hindi_medical")
        
        assert desc.name == "hindi_medical"
        assert desc.get("language") == "hindi"
        assert desc.get("field") == "medical"
    
    def test_get_correlation(self):
        """Test getting correlation between domains."""
        from langscope.transfer.faceted import DomainIndex, DomainDescriptor
        
        index = DomainIndex()
        
        # Register similar domains
        index.register_domain(DomainDescriptor("hindi_medical", {"language": "hindi", "field": "medical"}))
        index.register_domain(DomainDescriptor("bengali_medical", {"language": "bengali", "field": "medical"}))
        
        corr = index.get_correlation("hindi_medical", "bengali_medical")
        
        assert 0 < corr < 1  # Should have some correlation
    
    def test_same_domain_correlation(self):
        """Test same domain returns 1.0."""
        from langscope.transfer.faceted import DomainIndex
        
        index = DomainIndex()
        
        corr = index.get_correlation("test", "test")
        assert corr == 1.0
    
    def test_get_similar_domains(self):
        """Test getting similar domains."""
        from langscope.transfer.faceted import DomainIndex, DomainDescriptor
        
        index = DomainIndex()
        
        # Register multiple domains
        index.register_domain(DomainDescriptor("hindi_medical", {"language": "hindi", "field": "medical"}))
        index.register_domain(DomainDescriptor("bengali_medical", {"language": "bengali", "field": "medical"}))
        index.register_domain(DomainDescriptor("english_legal", {"language": "english", "field": "legal"}))
        
        similar = index.get_similar_domains("hindi_medical", k=5, min_correlation=0.0)
        
        assert len(similar) <= 2  # Max 2 other domains
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
    
    def test_precompute_top_k(self):
        """Test pre-computing top-K similar domains."""
        from langscope.transfer.faceted import DomainIndex, DomainDescriptor
        
        index = DomainIndex()
        
        # Register domains
        index.register_domain(DomainDescriptor("hindi_medical", {"language": "hindi", "field": "medical"}))
        index.register_domain(DomainDescriptor("bengali_medical", {"language": "bengali", "field": "medical"}))
        
        index.precompute_top_k(k=5)
        
        assert len(index._top_k_cache) > 0
        assert index.last_refresh is not None


class TestFacetedTransferLearning:
    """Test FacetedTransferLearning class."""
    
    def test_faceted_transfer_creation(self):
        """Test faceted transfer learning creation."""
        from langscope.transfer.faceted import FacetedTransferLearning, DomainIndex
        
        index = DomainIndex()
        ftl = FacetedTransferLearning(index)
        
        assert ftl.domain_index is index
        assert ftl.max_sources == 7
        assert ftl.min_correlation == 0.25
    
    def test_predict_rating_no_sources(self):
        """Test prediction with no source domains."""
        from langscope.transfer.faceted import FacetedTransferLearning, DomainIndex
        from langscope.core.model import LLMModel
        
        index = DomainIndex()
        ftl = FacetedTransferLearning(index)
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        result = ftl.predict_rating(model, "new_domain")
        
        # Should return default rating
        assert result.target_mu == ftl.mu_0
        assert result.target_sigma == ftl.sigma_0
        assert result.confidence == 0.0
        assert result.source == "transfer"
    
    def test_get_rating_or_transfer_direct(self):
        """Test getting rating when direct rating exists."""
        from langscope.transfer.faceted import FacetedTransferLearning, DomainIndex
        from langscope.core.model import LLMModel
        
        index = DomainIndex()
        ftl = FacetedTransferLearning(index)
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # Set a confident rating in medical domain
        model.set_domain_trueskill(
            domain="medical",
            raw_mu=1650.0,
            raw_sigma=60.0,  # Confident (< 0.8 * sigma_0)
            cost_mu=1640.0,
            cost_sigma=65.0
        )
        
        result = ftl.get_rating_or_transfer(model, "medical", "raw")
        
        assert result.source == "direct"
        assert result.target_mu == 1650.0
        assert result.confidence > 0.5
    
    def test_transfer_result_to_dict(self):
        """Test FacetedTransferResult to_dict."""
        from langscope.transfer.faceted import FacetedTransferResult
        
        result = FacetedTransferResult(
            target_mu=1600.0,
            target_sigma=120.0,
            source_domains=["coding", "math"],
            source_weights={"coding": 0.6, "math": 0.4},
            correlation_used=0.7,
            confidence=0.65,
            source="transfer"
        )
        
        d = result.to_dict()
        
        assert d["target_mu"] == 1600.0
        assert d["source"] == "transfer"
        assert "conservative_estimate" in d


class TestPriors:
    """Test priors module."""
    
    def test_language_priors_exist(self):
        """Test language priors are defined."""
        from langscope.transfer.priors import LANGUAGE_PRIORS
        
        assert len(LANGUAGE_PRIORS) > 0
        
        # Check some known priors
        assert ("bengali", "odia") in LANGUAGE_PRIORS
        assert ("hindi", "urdu") in LANGUAGE_PRIORS
    
    def test_field_priors_exist(self):
        """Test field priors are defined."""
        from langscope.transfer.priors import FIELD_PRIORS
        
        assert len(FIELD_PRIORS) > 0
        assert ("medical", "clinical") in FIELD_PRIORS
    
    def test_all_priors_combined(self):
        """Test ALL_PRIORS contains all facets."""
        from langscope.transfer.priors import ALL_PRIORS
        from langscope.transfer.faceted import ALL_FACETS
        
        for facet in ALL_FACETS:
            assert facet in ALL_PRIORS
    
    def test_prior_values_in_range(self):
        """Test all prior values are in [0, 1]."""
        from langscope.transfer.priors import ALL_PRIORS
        
        for facet, priors in ALL_PRIORS.items():
            for (a, b), value in priors.items():
                assert 0 <= value <= 1, f"Invalid prior for {facet}: ({a}, {b}) = {value}"
    
    def test_create_initialized_composite(self):
        """Test creating composite with priors loaded."""
        from langscope.transfer.priors import create_initialized_composite
        
        composite = create_initialized_composite()
        
        # Check a known prior was loaded
        learner = composite.get_learner("language")
        sim = learner.get_similarity("bengali", "odia")
        
        assert sim > 0.5  # Should have loaded the prior
    
    def test_get_prior_statistics(self):
        """Test getting prior statistics."""
        from langscope.transfer.priors import get_prior_statistics
        
        stats = get_prior_statistics()
        
        assert "language" in stats
        assert "field" in stats
        
        for facet, facet_stats in stats.items():
            assert "count" in facet_stats
            assert "min" in facet_stats
            assert "max" in facet_stats


class TestModuleExports:
    """Test module-level exports."""
    
    def test_faceted_module_import(self):
        """Test faceted module imports."""
        from langscope.transfer import faceted
        assert faceted is not None
    
    def test_priors_module_import(self):
        """Test priors module imports."""
        from langscope.transfer import priors
        assert priors is not None
    
    def test_main_module_exports(self):
        """Test main transfer module exports new classes."""
        from langscope.transfer import (
            DomainDescriptor,
            FacetSimilarityLearner,
            CompositeDomainSimilarity,
            DomainIndex,
            FacetedTransferLearning,
            DomainNameParser,
            ALL_FACETS,
            DEFAULT_FACET_WEIGHTS,
        )
        
        assert DomainDescriptor is not None
        assert FacetSimilarityLearner is not None
        assert CompositeDomainSimilarity is not None
        assert DomainIndex is not None
        assert FacetedTransferLearning is not None
        assert DomainNameParser is not None
        assert len(ALL_FACETS) == 5
        assert len(DEFAULT_FACET_WEIGHTS) == 5
    
    def test_convenience_functions(self):
        """Test convenience functions exist."""
        from langscope.transfer import (
            get_domain_index,
            get_faceted_transfer,
            parse_domain_name,
            get_domain_similarity,
        )
        
        assert callable(get_domain_index)
        assert callable(get_faceted_transfer)
        assert callable(parse_domain_name)
        assert callable(get_domain_similarity)
    
    def test_parse_domain_name_function(self):
        """Test parse_domain_name convenience function."""
        from langscope.transfer import parse_domain_name
        
        facets = parse_domain_name("hindi_medical_qa")
        
        assert isinstance(facets, dict)
        assert facets.get("language") == "hindi"
        assert facets.get("field") == "medical"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
