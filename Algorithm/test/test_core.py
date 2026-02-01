"""
Unit tests for langscope/core/ modules.

Tests:
- rating.py: TrueSkillRating, DualTrueSkill, MultiDimensionalTrueSkill
- model.py: LLMModel
- dimensions.py: Dimension scoring
- constants.py: System constants
"""

import pytest
import math
from datetime import datetime


# =============================================================================
# Rating Tests (CORE-001 to CORE-009)
# =============================================================================

class TestTrueSkillRating:
    """Test TrueSkillRating class."""
    
    def test_create_default_values(self):
        """CORE-001: Create TrueSkillRating with default values."""
        from langscope.core.rating import TrueSkillRating
        from langscope.core.constants import TRUESKILL_MU_0, TRUESKILL_SIGMA_0
        
        rating = TrueSkillRating()
        assert rating.mu == TRUESKILL_MU_0
        assert rating.sigma == TRUESKILL_SIGMA_0
    
    def test_create_custom_values(self):
        """CORE-002: Create TrueSkillRating with custom values."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1600.0, sigma=150.0)
        assert rating.mu == 1600.0
        assert rating.sigma == 150.0
    
    def test_confidence_interval(self):
        """CORE-003: Calculate 95% confidence interval."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1500.0, sigma=100.0)
        lower, upper = rating.confidence_interval(z=1.96)
        
        # μ - 1.96σ = 1500 - 196 = 1304
        # μ + 1.96σ = 1500 + 196 = 1696
        assert abs(lower - 1304.0) < 0.01
        assert abs(upper - 1696.0) < 0.01
    
    def test_conservative_estimate(self):
        """CORE-004: Calculate conservative estimate (μ-3σ)."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1500.0, sigma=100.0)
        conservative = rating.conservative_estimate(k=3.0)
        
        # μ - 3σ = 1500 - 300 = 1200
        assert abs(conservative - 1200.0) < 0.01
    
    def test_to_from_dict(self):
        """CORE-005: Convert to/from dictionary preserves data."""
        from langscope.core.rating import TrueSkillRating
        
        original = TrueSkillRating(mu=1550.0, sigma=120.0)
        data = original.to_dict()
        restored = TrueSkillRating.from_dict(data)
        
        assert original == restored
        assert data == {"mu": 1550.0, "sigma": 120.0}
    
    def test_variance(self):
        """Test variance calculation (σ²)."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1500.0, sigma=100.0)
        assert rating.variance() == 10000.0
    
    def test_precision(self):
        """Test precision calculation (1/σ²)."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1500.0, sigma=100.0)
        assert rating.precision() == 0.0001
    
    def test_precision_zero_sigma(self):
        """CORE-008: Edge case - σ = 0 returns infinity precision."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=1500.0, sigma=0.0)
        assert rating.precision() == float('inf')
    
    def test_negative_mu(self):
        """CORE-009: Edge case - negative μ values handled correctly."""
        from langscope.core.rating import TrueSkillRating
        
        rating = TrueSkillRating(mu=-100.0, sigma=50.0)
        assert rating.mu == -100.0
        assert rating.conservative_estimate(k=3.0) == -250.0


class TestDualTrueSkill:
    """Test DualTrueSkill class."""
    
    def test_initialization(self):
        """CORE-006: DualTrueSkill initialization with both ratings."""
        from langscope.core.rating import DualTrueSkill, TrueSkillRating
        
        dual = DualTrueSkill()
        assert isinstance(dual.raw, TrueSkillRating)
        assert isinstance(dual.cost_adjusted, TrueSkillRating)
    
    def test_update_both_ratings(self):
        """CORE-007: DualTrueSkill update both ratings independently."""
        from langscope.core.rating import DualTrueSkill, TrueSkillRating
        
        dual = DualTrueSkill(
            raw=TrueSkillRating(mu=1500.0, sigma=166.0),
            cost_adjusted=TrueSkillRating(mu=1520.0, sigma=160.0)
        )
        
        # Modify raw
        dual.raw.mu = 1550.0
        
        # cost_adjusted should remain unchanged
        assert dual.raw.mu == 1550.0
        assert dual.cost_adjusted.mu == 1520.0
    
    def test_conservative_raw(self):
        """Test conservative estimate for raw rating."""
        from langscope.core.rating import DualTrueSkill, TrueSkillRating
        
        dual = DualTrueSkill(
            raw=TrueSkillRating(mu=1500.0, sigma=100.0),
            cost_adjusted=TrueSkillRating(mu=1500.0, sigma=100.0)
        )
        
        # μ - 3σ = 1500 - 300 = 1200
        assert dual.conservative_raw() == 1200.0
    
    def test_to_from_dict(self):
        """Test dictionary serialization."""
        from langscope.core.rating import DualTrueSkill, TrueSkillRating
        
        original = DualTrueSkill(
            raw=TrueSkillRating(mu=1550.0, sigma=160.0),
            cost_adjusted=TrueSkillRating(mu=1580.0, sigma=155.0)
        )
        
        data = original.to_dict()
        restored = DualTrueSkill.from_dict(data)
        
        assert original.raw.mu == restored.raw.mu
        assert original.cost_adjusted.sigma == restored.cost_adjusted.sigma


class TestMultiDimensionalTrueSkill:
    """Test MultiDimensionalTrueSkill class."""
    
    def test_ten_dimensions_present(self):
        """CORE-020: List all 10 scoring dimensions."""
        from langscope.core.rating import DIMENSION_NAMES
        
        expected = [
            "raw_quality", "cost_adjusted", "latency", "ttft",
            "consistency", "token_efficiency", "instruction_following",
            "hallucination_resistance", "long_context", "combined"
        ]
        assert DIMENSION_NAMES == expected
        assert len(DIMENSION_NAMES) == 10
    
    def test_get_dimension(self):
        """Test getting rating for a specific dimension."""
        from langscope.core.rating import MultiDimensionalTrueSkill, TrueSkillRating
        
        mts = MultiDimensionalTrueSkill()
        mts.latency = TrueSkillRating(mu=1600.0, sigma=100.0)
        
        latency_rating = mts.get_dimension("latency")
        assert latency_rating.mu == 1600.0
    
    def test_invalid_dimension(self):
        """Test invalid dimension raises error."""
        from langscope.core.rating import MultiDimensionalTrueSkill
        
        mts = MultiDimensionalTrueSkill()
        with pytest.raises(ValueError):
            mts.get_dimension("invalid_dimension")
    
    def test_set_dimension(self):
        """Test setting rating for a dimension."""
        from langscope.core.rating import MultiDimensionalTrueSkill
        
        mts = MultiDimensionalTrueSkill()
        mts.set_dimension("consistency", mu=1650.0, sigma=120.0)
        
        assert mts.consistency.mu == 1650.0
        assert mts.consistency.sigma == 120.0
    
    def test_update_combined(self):
        """Test combined score calculation."""
        from langscope.core.rating import MultiDimensionalTrueSkill, TrueSkillRating
        
        mts = MultiDimensionalTrueSkill()
        # Set all dimensions to 1500
        for dim in ["raw_quality", "cost_adjusted", "latency", "ttft",
                    "consistency", "token_efficiency", "instruction_following",
                    "hallucination_resistance", "long_context"]:
            mts.set_dimension(dim, mu=1500.0, sigma=166.0)
        
        mts.update_combined()
        
        # Combined should be close to 1500 (weighted average)
        assert abs(mts.combined.mu - 1500.0) < 1.0
    
    def test_from_dual(self):
        """Test creating from DualTrueSkill."""
        from langscope.core.rating import (
            MultiDimensionalTrueSkill, DualTrueSkill, TrueSkillRating
        )
        
        dual = DualTrueSkill(
            raw=TrueSkillRating(mu=1550.0, sigma=160.0),
            cost_adjusted=TrueSkillRating(mu=1580.0, sigma=155.0)
        )
        
        mts = MultiDimensionalTrueSkill.from_dual(dual)
        
        assert mts.raw_quality.mu == 1550.0
        assert mts.cost_adjusted.mu == 1580.0
    
    def test_to_dual(self):
        """Test converting to DualTrueSkill."""
        from langscope.core.rating import MultiDimensionalTrueSkill, TrueSkillRating
        
        mts = MultiDimensionalTrueSkill()
        mts.raw_quality = TrueSkillRating(mu=1600.0, sigma=150.0)
        mts.cost_adjusted = TrueSkillRating(mu=1620.0, sigma=145.0)
        
        dual = mts.to_dual()
        
        assert dual.raw.mu == 1600.0
        assert dual.cost_adjusted.mu == 1620.0


# =============================================================================
# Model Tests (CORE-010 to CORE-017)
# =============================================================================

class TestLLMModel:
    """Test LLMModel class."""
    
    def test_create_with_required_fields(self):
        """CORE-010: Create model with required fields."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="GPT-4",
            model_id="gpt-4",
            provider="openai",
            input_cost_per_million=10.0,
            output_cost_per_million=30.0
        )
        
        assert model.name == "GPT-4"
        assert model.model_id == "gpt-4"
        assert model.provider == "openai"
    
    def test_create_with_optional_fields(self):
        """CORE-011: Create model with all optional fields."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="GPT-4",
            model_id="gpt-4",
            provider="openai",
            input_cost_per_million=10.0,
            output_cost_per_million=30.0,
            pricing_source="official",
            max_matches=200,
            api_key="test-key"
        )
        
        assert model.pricing_source == "official"
        assert model.max_matches == 200
        assert model.api_key == "test-key"
    
    def test_serialization_to_dict(self):
        """CORE-012: Model serialization to dict."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Test",
            model_id="test-001",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        data = model.to_dict()
        
        assert data["name"] == "Test"
        assert data["model_id"] == "test-001"
        assert "trueskill" in data
        assert "multi_trueskill" in data
    
    def test_deserialization_from_dict(self):
        """CORE-013: Model deserialization from dict."""
        from langscope.core.model import LLMModel
        
        data = {
            "name": "Test Model",
            "model_id": "test-001",
            "provider": "test",
            "input_cost_per_million": 1.0,
            "output_cost_per_million": 2.0,
            "trueskill": {
                "raw": {"mu": 1550.0, "sigma": 160.0},
                "cost_adjusted": {"mu": 1580.0, "sigma": 155.0}
            },
            "metadata": {}
        }
        
        model = LLMModel.from_dict(data)
        
        assert model.name == "Test Model"
        assert model.trueskill.raw.mu == 1550.0
    
    def test_cost_calculation(self):
        """CORE-014: Model with cost information."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=10.0,
            output_cost_per_million=20.0
        )
        
        # 1000 input + 500 output tokens
        cost = model.calculate_response_cost(1000, 500)
        
        # (1000/1M * 10) + (500/1M * 20) = 0.01 + 0.01 = 0.02
        expected = 0.01 + 0.01
        assert abs(cost - expected) < 0.0001
    
    def test_domain_ratings_initialization(self):
        """CORE-015: Model domain ratings initialization."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        assert model.trueskill_by_domain == {}
    
    def test_add_domain_rating(self):
        """CORE-016: Add domain rating to model."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        model.set_domain_trueskill(
            domain="coding",
            raw_mu=1600.0,
            raw_sigma=150.0,
            cost_mu=1620.0,
            cost_sigma=145.0
        )
        
        assert "coding" in model.trueskill_by_domain
        assert model.trueskill_by_domain["coding"].raw.mu == 1600.0
    
    def test_get_nonexistent_domain_rating(self):
        """CORE-017: Get non-existent domain rating creates default."""
        from langscope.core.model import LLMModel
        from langscope.core.constants import TRUESKILL_MU_0
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        rating = model.get_domain_trueskill("new_domain")
        
        # Should return default rating
        assert rating.raw.mu == TRUESKILL_MU_0
        assert "new_domain" in model.trueskill_by_domain
    
    def test_get_stratum(self):
        """Test stratum calculation."""
        from langscope.core.model import LLMModel
        from langscope.core.rating import TrueSkillRating, DualTrueSkill
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        # Elite: μ >= 1520
        model.trueskill = DualTrueSkill(
            raw=TrueSkillRating(mu=1550.0, sigma=100.0),
            cost_adjusted=TrueSkillRating(mu=1550.0, sigma=100.0)
        )
        assert model.get_stratum() == 4
        
        # High: 1450 <= μ < 1520
        model.trueskill.raw.mu = 1480.0
        assert model.get_stratum() == 3
        
        # Mid: 1400 <= μ < 1450
        model.trueskill.raw.mu = 1420.0
        assert model.get_stratum() == 2
        
        # Low: μ < 1400
        model.trueskill.raw.mu = 1350.0
        assert model.get_stratum() == 1
    
    def test_can_participate(self):
        """Test match participation limit."""
        from langscope.core.model import LLMModel
        
        model = LLMModel(
            name="Test",
            model_id="test",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            max_matches=10
        )
        
        assert model.can_participate() is True
        
        # Simulate reaching limit
        model.performance.total_matches_played = 10
        assert model.can_participate() is False


# =============================================================================
# Dimensions Tests (CORE-020 to CORE-024)
# =============================================================================

class TestDimensions:
    """Test dimension scoring functions."""
    
    def test_dimension_weight_validation(self):
        """CORE-021: Default weights should be valid."""
        from langscope.core.dimensions import DEFAULT_COMBINED_WEIGHTS
        
        total = sum(DEFAULT_COMBINED_WEIGHTS.values())
        # Should sum to approximately 1.0 (might be 0.95 without combined)
        assert 0.95 <= total <= 1.0
    
    def test_latency_score(self):
        """Test latency score calculation."""
        from langscope.core.dimensions import compute_latency_score
        
        # Fast response (100ms)
        score_fast = compute_latency_score(100.0, tau_latency=1000.0)
        # S = 1 / (1 + 100/1000) = 1 / 1.1 ≈ 0.909
        assert abs(score_fast - 0.909) < 0.01
        
        # Slow response (2000ms)
        score_slow = compute_latency_score(2000.0, tau_latency=1000.0)
        # S = 1 / (1 + 2000/1000) = 1 / 3 ≈ 0.333
        assert abs(score_slow - 0.333) < 0.01
    
    def test_latency_score_negative_handled(self):
        """Test negative latency is handled."""
        from langscope.core.dimensions import compute_latency_score
        
        score = compute_latency_score(-100.0)
        assert score == 1.0  # Treated as 0ms
    
    def test_ttft_score(self):
        """Test TTFT score calculation."""
        from langscope.core.dimensions import compute_ttft_score
        
        # Fast TTFT (50ms)
        score = compute_ttft_score(50.0, tau_ttft=200.0)
        # S = 1 / (1 + 50/200) = 1 / 1.25 = 0.8
        assert abs(score - 0.8) < 0.01
    
    def test_consistency_score(self):
        """Test consistency score calculation."""
        from langscope.core.dimensions import compute_consistency_score
        
        # Low variance (consistent)
        score_consistent = compute_consistency_score(0.1)
        # S = 1 / (1 + 0.1) ≈ 0.909
        assert abs(score_consistent - 0.909) < 0.01
        
        # High variance (inconsistent)
        score_inconsistent = compute_consistency_score(2.0)
        # S = 1 / (1 + 2.0) ≈ 0.333
        assert abs(score_inconsistent - 0.333) < 0.01
    
    def test_token_efficiency_score(self):
        """Test token efficiency calculation."""
        from langscope.core.dimensions import compute_token_efficiency_score
        
        score = compute_token_efficiency_score(mu_raw=1500.0, output_tokens=1000)
        # S = 1500 / log(1001) ≈ 1500 / 6.91 ≈ 217
        expected = 1500.0 / math.log(1001)
        assert abs(score - expected) < 0.1
    
    def test_token_efficiency_zero_tokens(self):
        """Test token efficiency with zero tokens."""
        from langscope.core.dimensions import compute_token_efficiency_score
        
        score = compute_token_efficiency_score(mu_raw=1500.0, output_tokens=0)
        # Should return mu_raw with no penalty
        assert score == 1500.0
    
    def test_cost_adjusted_score(self):
        """Test cost-adjusted score calculation."""
        from langscope.core.dimensions import compute_cost_adjusted_score
        
        # Free model
        score_free = compute_cost_adjusted_score(mu_raw=1500.0, cost_per_million=0.0)
        assert score_free == 1500.0
        
        # Expensive model
        score_expensive = compute_cost_adjusted_score(mu_raw=1500.0, cost_per_million=10.0)
        expected = 1500.0 / math.log(11.0)
        assert abs(score_expensive - expected) < 0.1
    
    def test_instruction_following_score(self):
        """Test instruction following score."""
        from langscope.core.dimensions import compute_instruction_following_score
        
        # All constraints satisfied
        score_perfect = compute_instruction_following_score(10, 10)
        assert score_perfect == 1.0
        
        # Half satisfied
        score_half = compute_instruction_following_score(5, 10)
        assert score_half == 0.5
        
        # No constraints
        score_none = compute_instruction_following_score(0, 0)
        assert score_none == 1.0  # Perfect if no constraints
    
    def test_hallucination_resistance_score(self):
        """Test hallucination resistance score."""
        from langscope.core.dimensions import compute_hallucination_resistance_score
        
        # No hallucinations
        score_perfect = compute_hallucination_resistance_score(0, 10)
        assert score_perfect == 1.0
        
        # 30% hallucination rate
        score_30pct = compute_hallucination_resistance_score(3, 10)
        assert score_30pct == 0.7
    
    def test_long_context_score(self):
        """Test long context score."""
        from langscope.core.dimensions import compute_long_context_score
        
        # No degradation
        score_no_degrade = compute_long_context_score(1500.0, 1500.0)
        assert score_no_degrade == 1.0
        
        # 20% degradation
        score_degrade = compute_long_context_score(1200.0, 1500.0)
        assert abs(score_degrade - 0.8) < 0.01
    
    def test_combined_score(self):
        """CORE-024: Calculate combined score."""
        from langscope.core.dimensions import compute_combined_score
        
        mus = {
            "raw_quality": 1500.0,
            "cost_adjusted": 1500.0,
            "latency": 0.9,
            "ttft": 0.8,
            "consistency": 0.95,
            "token_efficiency": 200.0,
            "instruction_following": 0.9,
            "hallucination_resistance": 0.95,
            "long_context": 0.85,
        }
        
        # With default weights
        combined = compute_combined_score(mus)
        assert combined > 0  # Should produce a valid score


# =============================================================================
# Constants Tests (CORE-030 to CORE-031)
# =============================================================================

class TestConstants:
    """Test system constants."""
    
    def test_trueskill_constants_defined(self):
        """CORE-030: TrueSkill constants defined."""
        from langscope.core import constants
        
        assert hasattr(constants, 'TRUESKILL_MU_0')
        assert hasattr(constants, 'TRUESKILL_SIGMA_0')
        assert hasattr(constants, 'TRUESKILL_BETA')
        assert hasattr(constants, 'TRUESKILL_TAU')
        assert hasattr(constants, 'TRUESKILL_CONSERVATIVE_K')
    
    def test_constants_correct_types(self):
        """CORE-031: Constants have correct types."""
        from langscope.core.constants import (
            TRUESKILL_MU_0,
            TRUESKILL_SIGMA_0,
            TRUESKILL_BETA,
            TRUESKILL_TAU,
            PLAYERS_PER_MATCH,
            MIN_PLAYERS,
            MAX_PLAYERS,
        )
        
        assert isinstance(TRUESKILL_MU_0, float)
        assert isinstance(TRUESKILL_SIGMA_0, float)
        assert isinstance(TRUESKILL_BETA, float)
        assert isinstance(TRUESKILL_TAU, float)
        assert isinstance(PLAYERS_PER_MATCH, int)
        assert isinstance(MIN_PLAYERS, int)
        assert isinstance(MAX_PLAYERS, int)
    
    def test_strata_thresholds(self):
        """Test strata thresholds defined correctly."""
        from langscope.core.constants import STRATA_THRESHOLDS
        
        assert "elite" in STRATA_THRESHOLDS
        assert "high" in STRATA_THRESHOLDS
        assert "mid" in STRATA_THRESHOLDS
        assert "low" in STRATA_THRESHOLDS
        
        # Elite > High > Mid > Low
        assert STRATA_THRESHOLDS["elite"] > STRATA_THRESHOLDS["high"]
        assert STRATA_THRESHOLDS["high"] > STRATA_THRESHOLDS["mid"]
        assert STRATA_THRESHOLDS["mid"] > STRATA_THRESHOLDS["low"]
    
    def test_info_bits(self):
        """Test information bits defined for match sizes."""
        from langscope.core.constants import INFO_BITS
        
        assert 5 in INFO_BITS
        assert 6 in INFO_BITS
        
        # log2(5!) ≈ 6.9, log2(6!) ≈ 9.5
        assert abs(INFO_BITS[5] - math.log2(math.factorial(5))) < 0.01
        assert abs(INFO_BITS[6] - math.log2(math.factorial(6))) < 0.01


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


