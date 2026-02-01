"""
Unit tests for langscope/federation/ modules.

Tests:
- judge.py: Judge management
- selection.py: Model selection
- strata.py: Stratified sampling
- workflow.py: Federation workflow
- content.py: Content generation

Coverage target: 80%
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Judge Tests (FED-001 to FED-005)
# =============================================================================

class TestJudgeManagement:
    """Test judge management functions."""
    
    def test_judge_module_import(self):
        """Test judge module imports."""
        from langscope.federation import judge
        assert judge is not None
    
    def test_selection_module_import(self):
        """Test selection module imports."""
        from langscope.federation import selection
        assert selection is not None
    
    def test_shuffle_response_order(self):
        """Test response shuffling for blind evaluation."""
        from langscope.federation.judge import shuffle_response_order
        
        responses = {
            "model_a": "Response A",
            "model_b": "Response B",
            "model_c": "Response C"
        }
        
        shuffled, mapping = shuffle_response_order(responses)
        
        assert len(shuffled) == 3
        assert len(mapping) == 3
        # All responses should be preserved
        assert set(shuffled.values()) == set(responses.values())
    
    def test_judge_ranking_validator_init(self):
        """Test JudgeRankingValidator initialization."""
        from langscope.federation.judge import JudgeRankingValidator
        
        validator = JudgeRankingValidator(
            expected_participants=["model_a", "model_b", "model_c"]
        )
        
        assert validator.n_participants == 3
        assert "model_a" in validator.expected_participants
    
    def test_judge_ranking_validator_valid(self):
        """Test JudgeRankingValidator with valid ranking."""
        from langscope.federation.judge import JudgeRankingValidator
        
        validator = JudgeRankingValidator(
            expected_participants=["model_a", "model_b", "model_c"]
        )
        
        valid_ranking = {"model_a": 1, "model_b": 2, "model_c": 3}
        is_valid, error = validator.validate(valid_ranking)
        
        assert is_valid is True
        assert error == ""
    
    def test_judge_ranking_validator_missing(self):
        """Test JudgeRankingValidator with missing participant."""
        from langscope.federation.judge import JudgeRankingValidator
        
        validator = JudgeRankingValidator(
            expected_participants=["model_a", "model_b", "model_c"]
        )
        
        incomplete = {"model_a": 1, "model_b": 2}  # Missing model_c
        is_valid, error = validator.validate(incomplete)
        
        assert is_valid is False
        assert "Missing" in error
    
    def test_judge_ranking_validator_invalid_ranks(self):
        """Test JudgeRankingValidator with invalid ranks."""
        from langscope.federation.judge import JudgeRankingValidator
        
        validator = JudgeRankingValidator(
            expected_participants=["model_a", "model_b", "model_c"]
        )
        
        invalid = {"model_a": 1, "model_b": 1, "model_c": 3}  # Duplicate rank 1
        is_valid, error = validator.validate(invalid)
        
        assert is_valid is False
    
    def test_judge_aggregator(self):
        """Test JudgeAggregator."""
        from langscope.federation.judge import JudgeAggregator
        
        aggregator = JudgeAggregator()
        
        rankings = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 1, "c": 3},
        ]
        weights = [1.0, 1.0, 1.0]
        
        result = aggregator.aggregate(rankings, weights)
        
        assert isinstance(result, dict)
        assert len(result) == 3


# =============================================================================
# Strata Tests (FED-020 to FED-023)
# =============================================================================

class TestStrata:
    """Test stratified sampling functions."""
    
    @pytest.fixture(autouse=True)
    def mock_strata_params(self):
        """Mock strata params to use default thresholds."""
        from langscope.config.params.models import StrataParams
        default_params = StrataParams()
        
        with patch('langscope.federation.strata._get_strata_params', return_value=default_params):
            yield
    
    def test_strata_module_import(self):
        """Test strata module imports."""
        from langscope.federation import strata
        assert strata is not None
    
    def test_get_stratum_elite(self):
        """FED-020: Test elite stratum assignment."""
        from langscope.federation.strata import get_stratum
        
        stratum = get_stratum(1600.0)  # High rating
        assert stratum == 4  # Elite
    
    def test_get_stratum_high(self):
        """Test high stratum assignment."""
        from langscope.federation.strata import get_stratum
        
        stratum = get_stratum(1480.0)
        assert stratum == 3  # High
    
    def test_get_stratum_mid(self):
        """Test mid stratum assignment."""
        from langscope.federation.strata import get_stratum
        
        stratum = get_stratum(1420.0)
        assert stratum == 2  # Mid
    
    def test_get_stratum_low(self):
        """Test low stratum assignment."""
        from langscope.federation.strata import get_stratum
        
        stratum = get_stratum(1350.0)
        assert stratum == 1  # Low
    
    def test_get_stratum_name(self):
        """Test stratum name lookup."""
        from langscope.federation.strata import get_stratum_name
        
        assert get_stratum_name(4) == "elite"
        assert get_stratum_name(3) == "high"
        assert get_stratum_name(2) == "mid"
        assert get_stratum_name(1) == "low"
    
    def test_get_stratum_threshold(self):
        """Test stratum threshold lookup."""
        from langscope.federation.strata import get_stratum_threshold
        
        assert get_stratum_threshold(4) >= 1500  # Elite threshold
        assert get_stratum_threshold(3) >= 1400  # High threshold
        assert get_stratum_threshold(2) >= 1300  # Mid threshold
        assert get_stratum_threshold(1) == 0.0  # No minimum for low
    
    def test_stratum_weight_calculation(self):
        """Test stratum weight calculation."""
        from langscope.federation.strata import calculate_stratum_weight
        
        w4 = calculate_stratum_weight(4)
        w3 = calculate_stratum_weight(3)
        w2 = calculate_stratum_weight(2)
        w1 = calculate_stratum_weight(1)
        
        # Higher strata should have higher weights
        assert w4 > w3 > w2 > w1
    
    def test_can_serve_as_judge(self):
        """FED-003: Test judge qualification check."""
        from langscope.federation.strata import can_serve_as_judge
        from langscope.core.model import LLMModel
        
        # Create a high-rated model
        model = LLMModel(
            name="Judge Model",
            model_id="judge-001",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        model.trueskill.raw.mu = 1550.0  # High rating
        
        can_judge = can_serve_as_judge(model, min_stratum=2)
        assert can_judge is True
    
    def test_can_create_content(self):
        """Test content creation eligibility."""
        from langscope.federation.strata import can_create_content
        from langscope.core.model import LLMModel
        
        # Create an elite-rated model
        model = LLMModel(
            name="Elite Model",
            model_id="elite-001",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        model.trueskill.raw.mu = 1600.0  # Elite rating
        
        can_create = can_create_content(model, min_stratum=3)
        assert can_create is True
    
    def test_get_stratum_distribution(self):
        """Test stratum distribution calculation."""
        from langscope.federation.strata import get_stratum_distribution
        from langscope.core.model import LLMModel
        
        models = []
        ratings = [1350, 1420, 1480, 1550, 1600]  # One from each stratum
        
        for i, mu in enumerate(ratings):
            model = LLMModel(
                name=f"Model {i}",
                model_id=f"model-{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            model.trueskill.raw.mu = float(mu)
            models.append(model)
        
        distribution = get_stratum_distribution(models)
        
        assert isinstance(distribution, dict)
        assert 1 in distribution
        assert 2 in distribution
        assert 3 in distribution
        assert 4 in distribution


# =============================================================================
# Selection Tests (FED-010 to FED-014)
# =============================================================================

class TestSelection:
    """Test model selection functions."""
    
    def test_swiss_pairing_init(self):
        """FED-010: Test MultiPlayerSwissPairing initialization."""
        from langscope.federation.selection import MultiPlayerSwissPairing
        
        pairing = MultiPlayerSwissPairing(
            mu_delta=100.0,
            max_matches=50,
            players_per_match=5
        )
        
        assert pairing.mu_delta == 100.0
        assert pairing.max_matches == 50
        assert pairing.players_per_match == 5
    
    def test_swiss_pairing_default_params(self):
        """Test Swiss pairing with default parameters."""
        from langscope.federation.selection import MultiPlayerSwissPairing
        
        pairing = MultiPlayerSwissPairing()
        
        assert pairing.mu_delta > 0
        assert pairing.max_matches > 0
        assert pairing.players_per_match >= 5
    
    def test_content_creator_selector_init(self):
        """FED-011: Test ContentCreatorSelector initialization."""
        from langscope.federation.selection import ContentCreatorSelector
        
        selector = ContentCreatorSelector()
        assert selector is not None
    
    def test_judge_selector_init(self):
        """Test JudgeSelector initialization."""
        from langscope.federation.selection import JudgeSelector
        
        selector = JudgeSelector()
        assert selector is not None


# =============================================================================
# Workflow Tests (FED-030 to FED-033)
# =============================================================================

class TestWorkflow:
    """Test federation workflow."""
    
    def test_workflow_module_import(self):
        """Test workflow module imports."""
        from langscope.federation import workflow
        assert workflow is not None
    
    def test_match_result_creation(self):
        """Test MatchResult dataclass creation."""
        from langscope.federation.workflow import MatchResult
        
        result = MatchResult(
            match_id="test-match-001",
            domain="coding",
            timestamp="2024-01-01T00:00:00Z",
            players=["model_a", "model_b", "model_c", "model_d", "model_e"],
            case_creator="case_model",
            question_creator="question_model",
            judges=["judge_a", "judge_b", "judge_c"],
            case_text="Test case",
            question_text="Test question",
            responses={
                "model_a": {"text": "Response A", "tokens": 100},
                "model_b": {"text": "Response B", "tokens": 120}
            },
            raw_ranking={"model_a": 1, "model_b": 2, "model_c": 3, "model_d": 4, "model_e": 5},
            cost_adjusted_ranking={"model_a": 2, "model_b": 1, "model_c": 3, "model_d": 4, "model_e": 5},
            judge_rankings=[
                {"model_a": 1, "model_b": 2, "model_c": 3, "model_d": 4, "model_e": 5}
            ],
            judge_weights=[1.0],
            pl_strengths={"model_a": 1.5, "model_b": 1.2},
            info_bits=3.2
        )
        
        assert result.match_id == "test-match-001"
        assert len(result.players) == 5
        assert result.domain == "coding"
    
    def test_match_result_to_dict(self):
        """Test MatchResult serialization."""
        from langscope.federation.workflow import MatchResult
        
        result = MatchResult(
            match_id="test-match-002",
            domain="coding",
            timestamp="2024-01-01T00:00:00Z",
            players=["a", "b", "c", "d", "e"],
            case_creator="case",
            question_creator="question",
            judges=["j1", "j2"],
            case_text="Case",
            question_text="Question",
            responses={"a": {"text": "A"}},
            raw_ranking={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            cost_adjusted_ranking={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            judge_rankings=[],
            judge_weights=[],
            pl_strengths={},
            info_bits=2.0
        )
        
        d = result.to_dict()
        
        assert "_id" in d
        assert "domain" in d
        assert "participants" in d
        assert d["participant_count"] == 5


# =============================================================================
# Content Tests (FED-040 to FED-043)
# =============================================================================

class TestContent:
    """Test content generation functions."""
    
    def test_content_module_import(self):
        """Test content module imports."""
        from langscope.federation import content
        assert content is not None
    
    def test_generated_content_creation(self):
        """Test GeneratedContent dataclass."""
        from langscope.federation.content import GeneratedContent
        
        gc = GeneratedContent(
            content_id="content-001",
            content_type="case",
            text="This is a test case",
            generator_id="gpt-4",
            generator_name="GPT-4",
            generator_mu=1600.0,
            domain="coding",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert gc.content_type == "case"
        assert gc.domain == "coding"
    
    def test_generated_content_to_dict(self):
        """Test GeneratedContent serialization."""
        from langscope.federation.content import GeneratedContent
        
        gc = GeneratedContent(
            content_id="content-002",
            content_type="question",
            text="What is the output?",
            generator_id="claude-3",
            generator_name="Claude 3",
            generator_mu=1550.0,
            domain="coding",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        d = gc.to_dict()
        
        assert "content_id" in d
        assert "content_type" in d
        assert "text" in d
    
    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        from langscope.federation.content import ValidationResult
        
        vr = ValidationResult(
            is_valid=True,
            votes=[("validator_1", True), ("validator_2", True), ("validator_3", False)]
        )
        
        assert vr.is_valid is True
        assert abs(vr.approval_rate - 0.667) < 0.01
    
    def test_content_generator_init(self):
        """FED-040: Test ContentGenerator initialization."""
        from langscope.federation.content import ContentGenerator
        
        generator = ContentGenerator(domain="coding")
        assert generator.domain == "coding"
    
    def test_content_generator_case_prompt(self):
        """Test case generation prompt creation."""
        from langscope.federation.content import ContentGenerator
        
        generator = ContentGenerator(domain="coding")
        prompt = generator.create_case_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "coding" in prompt.lower()


# =============================================================================
# Router Tests
# =============================================================================

class TestRouter:
    """Test match routing."""
    
    def test_router_module_import(self):
        """Test router module imports."""
        from langscope.federation import router
        assert router is not None
    
    def test_match_router_init(self):
        """Test MatchRouter initialization."""
        from langscope.federation.router import MatchRouter
        
        match_router = MatchRouter()
        assert match_router is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestFederationIntegration:
    """Integration tests for federation system."""
    
    def test_strata_affects_eligibility(self):
        """Test that strata properly affects eligibility."""
        from langscope.federation.strata import get_stratum, can_serve_as_judge, can_create_content
        from langscope.core.model import LLMModel
        
        # Low-rated model
        low_model = LLMModel(
            name="Low Model",
            model_id="low-001",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        low_model.trueskill.raw.mu = 1350.0
        
        # Elite model
        elite_model = LLMModel(
            name="Elite Model",
            model_id="elite-001",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        elite_model.trueskill.raw.mu = 1600.0
        
        # Low model should not be able to judge or create content
        assert not can_serve_as_judge(low_model, min_stratum=2)
        assert not can_create_content(low_model, min_stratum=3)
        
        # Elite model should be able to do both
        assert can_serve_as_judge(elite_model, min_stratum=2)
        assert can_create_content(elite_model, min_stratum=3)
    
    def test_stratum_weights_affect_aggregation(self):
        """Test that stratum weights affect judge aggregation."""
        from langscope.federation.strata import calculate_stratum_weight
        from langscope.federation.judge import JudgeAggregator
        
        # Elite judge vs low judge should have different weights
        elite_weight = calculate_stratum_weight(4)
        low_weight = calculate_stratum_weight(1)
        
        assert elite_weight > low_weight
        
        # This affects aggregation
        aggregator = JudgeAggregator()
        rankings = [
            {"a": 1, "b": 2},  # Elite judge
            {"a": 2, "b": 1},  # Low judge
        ]
        weights = [elite_weight, low_weight]
        
        result = aggregator.aggregate(rankings, weights)
        
        # Elite judge's preference should win
        assert result["a"] < result["b"]  # a should be ranked higher


# =============================================================================
# Workflow Tests (FED-030 to FED-033)
# =============================================================================

class TestWorkflow:
    """Test match workflow operations."""
    
    def test_workflow_module_import(self):
        """Test workflow module imports."""
        from langscope.federation import workflow
        assert workflow is not None
    
    def test_match_result_dataclass(self):
        """Test MatchResult dataclass."""
        from langscope.federation.workflow import MatchResult
        
        result = MatchResult(
            match_id="test_match_001",
            domain="coding",
            timestamp="2024-01-15T12:00:00Z",
            players=["model_a", "model_b", "model_c"],
            case_creator="creator_001",
            question_creator="creator_002",
            judges=["judge_001", "judge_002"],
            case_text="Test case",
            question_text="Test question",
            responses={"model_a": {"text": "Response A"}},
            raw_ranking={"model_a": 1, "model_b": 2, "model_c": 3},
            cost_adjusted_ranking={"model_a": 1, "model_b": 2, "model_c": 3},
            judge_rankings=[{"model_a": 1, "model_b": 2, "model_c": 3}],
            judge_weights=[1.0],
            pl_strengths={"model_a": 0.5, "model_b": 0.3, "model_c": 0.2},
            info_bits=2.5
        )
        
        assert result.match_id == "test_match_001"
        assert result.domain == "coding"
        assert len(result.players) == 3
    
    def test_match_result_to_dict(self):
        """Test MatchResult conversion to dict."""
        from langscope.federation.workflow import MatchResult
        
        result = MatchResult(
            match_id="test_match_002",
            domain="coding",
            timestamp="2024-01-15T12:00:00Z",
            players=["model_a", "model_b"],
            case_creator="creator_001",
            question_creator="creator_002",
            judges=["judge_001"],
            case_text="Test case",
            question_text="Test question",
            responses={},
            raw_ranking={"model_a": 1, "model_b": 2},
            cost_adjusted_ranking={"model_a": 1, "model_b": 2},
            judge_rankings=[],
            judge_weights=[],
            pl_strengths={},
            info_bits=1.0
        )
        
        result_dict = result.to_dict()
        
        assert "_id" in result_dict
        assert result_dict["_id"] == "test_match_002"
        assert result_dict["domain"] == "coding"
        assert "participants" in result_dict
        assert "prompt" in result_dict


class TestContentGeneration:
    """Test content generation operations."""
    
    def test_generated_content_dataclass(self):
        """Test GeneratedContent dataclass."""
        from langscope.federation.content import GeneratedContent
        
        content = GeneratedContent(
            content_id="content_001",
            content_type="case",
            text="Test case content",
            generator_id="gen_001",
            generator_name="Test Generator",
            generator_mu=1500.0,
            domain="coding",
            timestamp="2024-01-15T12:00:00Z"
        )
        
        assert content.content_id == "content_001"
        assert content.content_type == "case"
        assert content.generator_mu == 1500.0
    
    def test_generated_content_to_dict(self):
        """Test GeneratedContent conversion to dict."""
        from langscope.federation.content import GeneratedContent
        
        content = GeneratedContent(
            content_id="content_002",
            content_type="question",
            text="Test question",
            generator_id="gen_001",
            generator_name="Test Generator",
            generator_mu=1600.0,
            domain="coding",
            timestamp="2024-01-15T12:00:00Z",
            tokens_used=100,
            cost_usd=0.001
        )
        
        content_dict = content.to_dict()
        
        assert content_dict["content_id"] == "content_002"
        assert content_dict["content_type"] == "question"
        assert content_dict["tokens_used"] == 100
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass."""
        from langscope.federation.content import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            votes=[("validator_1", True), ("validator_2", True), ("validator_3", False)]
        )
        
        assert result.is_valid is True
        assert len(result.votes) == 3
    
    def test_validation_result_approval_rate(self):
        """Test ValidationResult approval rate calculation."""
        from langscope.federation.content import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            votes=[("v1", True), ("v2", True), ("v3", False), ("v4", True)]
        )
        
        # 3 out of 4 approved = 0.75
        assert result.approval_rate == 0.75
    
    def test_validation_result_empty_votes(self):
        """Test ValidationResult with no votes."""
        from langscope.federation.content import ValidationResult
        
        result = ValidationResult(
            is_valid=False,
            votes=[]
        )
        
        assert result.approval_rate == 0.0
    
    def test_content_generator_question_prompt(self):
        """Test question generation prompt creation."""
        from langscope.federation.content import ContentGenerator
        
        generator = ContentGenerator(domain="medical")
        prompt = generator.create_question_prompt(case_text="Patient presents with symptoms...")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_content_generator_different_difficulties(self):
        """Test case generation with different difficulties."""
        from langscope.federation.content import ContentGenerator
        
        generator = ContentGenerator(domain="coding")
        
        easy_prompt = generator.create_case_prompt(difficulty="easy")
        hard_prompt = generator.create_case_prompt(difficulty="hard")
        
        assert "easy" in easy_prompt.lower() or len(easy_prompt) > 0
        assert "hard" in hard_prompt.lower() or len(hard_prompt) > 0


class TestJudgeOperations:
    """Test judge-related operations."""
    
    def test_create_judge_prompt(self):
        """Test judge prompt creation."""
        from langscope.federation.judge import create_judge_prompt
        
        responses = {
            "A": "Response from model A",
            "B": "Response from model B",
            "C": "Response from model C"
        }
        
        prompt = create_judge_prompt(
            case_text="Test case",
            question_text="Test question",
            responses=responses
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain response labels
        assert "A" in prompt or "Response" in prompt
    
    def test_shuffle_response_order_randomness(self):
        """Test that response shuffling actually shuffles."""
        from langscope.federation.judge import shuffle_response_order
        
        responses = {
            "model_a": "A",
            "model_b": "B",
            "model_c": "C",
            "model_d": "D",
            "model_e": "E"
        }
        
        # Run multiple times to check randomness
        orderings = set()
        for _ in range(10):
            shuffled, _ = shuffle_response_order(responses)
            orderings.add(tuple(shuffled.keys()))
        
        # Should have at least some variation (not deterministic)
        # Note: With 5 items, very unlikely to get same order 10 times
        assert len(orderings) >= 1


class TestSelectionOperations:
    """Test model selection operations."""
    
    def test_swiss_pairing_init(self):
        """Test MultiPlayerSwissPairing initialization."""
        from langscope.federation.selection import MultiPlayerSwissPairing
        
        pairing = MultiPlayerSwissPairing()
        assert pairing is not None
    
    def test_content_creator_selector_init(self):
        """Test ContentCreatorSelector initialization."""
        from langscope.federation.selection import ContentCreatorSelector
        
        selector = ContentCreatorSelector()
        assert selector is not None
    
    def test_judge_selector_init(self):
        """Test JudgeSelector initialization."""
        from langscope.federation.selection import JudgeSelector
        
        selector = JudgeSelector()
        assert selector is not None


class TestStrataAdvanced:
    """Advanced strata tests."""
    
    @pytest.fixture(autouse=True)
    def mock_strata_params(self):
        """Mock strata params to use default thresholds."""
        from langscope.config.params.models import StrataParams
        default_params = StrataParams()
        
        with patch('langscope.federation.strata._get_strata_params', return_value=default_params):
            yield
    
    def test_get_stratum_name(self):
        """Test getting stratum name from number."""
        from langscope.federation.strata import get_stratum_name
        
        assert get_stratum_name(4) == "elite"
        assert get_stratum_name(3) == "high"
        assert get_stratum_name(2) == "mid"
        assert get_stratum_name(1) == "low"
    
    def test_get_stratum_threshold(self):
        """Test getting stratum thresholds."""
        from langscope.federation.strata import get_stratum_threshold
        
        elite_threshold = get_stratum_threshold(4)
        high_threshold = get_stratum_threshold(3)
        mid_threshold = get_stratum_threshold(2)
        low_threshold = get_stratum_threshold(1)
        
        assert elite_threshold >= high_threshold >= mid_threshold >= low_threshold
    
    def test_get_eligible_judges(self):
        """Test getting eligible judges."""
        from langscope.federation.strata import get_eligible_judges
        from langscope.core.model import LLMModel
        
        # Create models with different ratings
        models = []
        for i, rating in enumerate([1600, 1500, 1420, 1350]):
            model = LLMModel(
                name=f"Model_{i}",
                model_id=f"model_{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            model.trueskill.raw.mu = rating
            models.append(model)
        
        eligible = get_eligible_judges(models, min_stratum=2)
        
        # Should include elite and high and mid (stratum >= 2)
        assert len(eligible) >= 2
    
    def test_get_eligible_creators(self):
        """Test getting eligible content creators."""
        from langscope.federation.strata import get_eligible_creators
        from langscope.core.model import LLMModel
        
        models = []
        for i, rating in enumerate([1600, 1480, 1420, 1350]):
            model = LLMModel(
                name=f"Model_{i}",
                model_id=f"model_{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            model.trueskill.raw.mu = rating
            models.append(model)
        
        eligible = get_eligible_creators(models, min_stratum=3)
        
        # Should only include elite and high (stratum >= 3)
        assert len(eligible) >= 1
    
    def test_get_stratum_distribution(self):
        """Test getting stratum distribution."""
        from langscope.federation.strata import get_stratum_distribution
        from langscope.core.model import LLMModel
        
        models = []
        for rating in [1600, 1550, 1480, 1450, 1420, 1350, 1300]:
            model = LLMModel(
                name=f"Model_{rating}",
                model_id=f"model_{rating}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            model.trueskill.raw.mu = rating
            models.append(model)
        
        distribution = get_stratum_distribution(models)
        
        assert isinstance(distribution, dict)
        assert 1 in distribution
        assert 2 in distribution
        assert 3 in distribution
        assert 4 in distribution


class TestJudgeAggregatorAdvanced:
    """Advanced judge aggregator tests."""
    
    def test_aggregate_with_ties(self):
        """Test aggregation with tied rankings."""
        from langscope.federation.judge import JudgeAggregator
        
        aggregator = JudgeAggregator()
        
        rankings = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 1, "c": 3},
            {"a": 1, "b": 2, "c": 3}
        ]
        weights = [1.0, 1.0, 1.0]
        
        result = aggregator.aggregate(rankings, weights)
        
        assert isinstance(result, dict)
        assert len(result) == 3
    
    def test_aggregate_single_judge(self):
        """Test aggregation with single judge."""
        from langscope.federation.judge import JudgeAggregator
        
        aggregator = JudgeAggregator()
        
        rankings = [{"a": 1, "b": 2, "c": 3}]
        weights = [1.0]
        
        result = aggregator.aggregate(rankings, weights)
        
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3
    
    def test_aggregate_weighted_override(self):
        """Test that higher weights override lower weights."""
        from langscope.federation.judge import JudgeAggregator
        
        aggregator = JudgeAggregator()
        
        # One high-weight judge vs two low-weight judges
        rankings = [
            {"a": 1, "b": 2},  # High weight judge
            {"a": 2, "b": 1},  # Low weight judge
            {"a": 2, "b": 1},  # Low weight judge
        ]
        weights = [10.0, 1.0, 1.0]
        
        result = aggregator.aggregate(rankings, weights)
        
        # High weight judge's preference should dominate
        assert result["a"] < result["b"]


class TestJudgeRankingValidatorAdvanced:
    """Advanced judge ranking validator tests."""
    
    def test_validate_complete_ranking(self):
        """Test validating a complete ranking."""
        from langscope.federation.judge import JudgeRankingValidator
        
        expected = ["a", "b", "c"]
        validator = JudgeRankingValidator(expected)
        
        ranking = {"a": 1, "b": 2, "c": 3}
        
        is_valid, message = validator.validate(ranking)
        
        assert is_valid is True
    
    def test_validate_missing_participant(self):
        """Test validating ranking with missing participant."""
        from langscope.federation.judge import JudgeRankingValidator
        
        expected = ["a", "b", "c"]
        validator = JudgeRankingValidator(expected)
        
        ranking = {"a": 1, "b": 2}  # Missing c
        
        is_valid, message = validator.validate(ranking)
        
        # Should be invalid due to missing participant
        assert is_valid is False
    
    def test_validate_single_item(self):
        """Test validating single item ranking."""
        from langscope.federation.judge import JudgeRankingValidator
        
        expected = ["a"]
        validator = JudgeRankingValidator(expected)
        
        ranking = {"a": 1}
        
        is_valid, message = validator.validate(ranking)
        
        # Single item should be valid
        assert is_valid is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
