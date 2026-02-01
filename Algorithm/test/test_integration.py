"""
Integration tests for LangScope.

Tests:
- End-to-end match flow
- Arena session flow
- Transfer learning flow
- API integration
"""

import pytest
from datetime import datetime


# =============================================================================
# End-to-End Match Flow Tests (INT-001 to INT-004)
# =============================================================================

class TestMatchFlow:
    """Test end-to-end match flow."""
    
    def test_full_match_execution(self):
        """INT-001: Full match execution flow."""
        from langscope.core.model import LLMModel
        from langscope.core.rating import TrueSkillRating, DualTrueSkill
        from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater
        from langscope.ranking.cost_adjustment import (
            create_cost_adjusted_ranking,
            aggregate_judge_rankings
        )
        from langscope.evaluation.match import Match, MatchParticipant, MatchResponse
        
        # 1. Create models
        models = []
        for i in range(5):
            model = LLMModel(
                name=f"Model {i}",
                model_id=f"model_{i}",
                provider=f"provider_{i % 3}",
                input_cost_per_million=1.0 * (i + 1),
                output_cost_per_million=2.0 * (i + 1)
            )
            models.append(model)
        
        # 2. Create match
        match = Match(
            match_id="integration_test_match",
            domain="coding",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        for model in models:
            ts = model.get_domain_trueskill("coding")
            match.competitors.append(MatchParticipant(
                model_id=model.model_id,
                model_name=model.name,
                role="competitor",
                mu_before=ts.raw.mu,
                sigma_before=ts.raw.sigma
            ))
        
        # 3. Simulate responses
        for i, model in enumerate(models):
            match.responses.append(MatchResponse(
                model_id=model.model_id,
                text=f"Response from {model.name}",
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                cost_usd=model.calculate_response_cost(100, 200),
                latency_ms=100.0 + i * 50
            ))
        
        # 4. Simulate judge rankings (3 judges)
        judge_rankings = [
            {f"model_{i}": i + 1 for i in range(5)},  # Judge 1
            {f"model_{i}": (i + 2) % 5 + 1 for i in range(5)},  # Judge 2
            {f"model_{i}": (5 - i) for i in range(5)},  # Judge 3
        ]
        
        # 5. Aggregate rankings
        raw_ranking = aggregate_judge_rankings(judge_rankings)
        
        # 6. Create cost-adjusted ranking
        costs = {model.model_id: match.responses[i].cost_usd 
                 for i, model in enumerate(models)}
        cost_ranking = create_cost_adjusted_ranking(raw_ranking, costs)
        
        match.set_rankings(raw_ranking, cost_ranking)
        
        # 7. Update TrueSkill ratings
        updater = MultiPlayerTrueSkillUpdater()
        
        current_ratings = [
            TrueSkillRating(
                mu=models[i].get_domain_trueskill("coding").raw.mu,
                sigma=models[i].get_domain_trueskill("coding").raw.sigma
            )
            for i in range(5)
        ]
        
        # Convert ranking to list form
        ranking_list = [raw_ranking[f"model_{i}"] for i in range(5)]
        
        updated_ratings = updater.update_from_ranking(current_ratings, ranking_list)
        
        # 8. Apply updates to models
        for i, model in enumerate(models):
            model.set_domain_trueskill(
                domain="coding",
                raw_mu=updated_ratings[i].mu,
                raw_sigma=updated_ratings[i].sigma,
                cost_mu=updated_ratings[i].mu,  # Simplified
                cost_sigma=updated_ratings[i].sigma
            )
            
            # Update match participant after ratings
            match.competitors[i].mu_after = updated_ratings[i].mu
            match.competitors[i].sigma_after = updated_ratings[i].sigma
        
        match.status = "completed"
        
        # Verify the flow completed successfully
        assert match.status == "completed"
        assert len(match.raw_ranking) == 5
        assert len(match.cost_ranking) == 5
        
        # At least some ratings should have changed (not all will change equally)
        mu_changes = [
            abs(match.competitors[i].mu_after - match.competitors[i].mu_before)
            for i in range(5)
        ]
        # At least one model should have changed rating
        assert sum(mu_changes) > 0
        
        # Sigma (uncertainty) should decrease for all participants
        for i in range(5):
            assert match.competitors[i].sigma_after < match.competitors[i].sigma_before
    
    def test_match_with_5_competitors(self):
        """INT-002: Match with 5 competitors."""
        from langscope.evaluation.match import Match, MatchParticipant
        
        match = Match(
            match_id="test_5_competitors",
            domain="medical",
            timestamp=""
        )
        
        for i in range(5):
            match.competitors.append(MatchParticipant(
                model_id=f"model_{i}",
                model_name=f"Model {i}",
                role="competitor"
            ))
        
        assert match.participant_count == 5
        assert len(match.get_competitor_ids()) == 5


# =============================================================================
# Rating Convergence Tests
# =============================================================================

class TestRatingConvergence:
    """Test that ratings converge correctly over multiple matches."""
    
    def test_better_model_rises(self):
        """Test that a consistently winning model's rating rises."""
        from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater, TrueSkillRating
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # Model A always wins against Model B
        model_a = TrueSkillRating(mu=1500.0, sigma=166.0)
        model_b = TrueSkillRating(mu=1500.0, sigma=166.0)
        
        for _ in range(20):
            updated = updater.update_from_ranking(
                [model_a, model_b],
                [1, 2]  # A wins
            )
            model_a = updated[0]
            model_b = updated[1]
        
        # Model A should have higher rating
        assert model_a.mu > model_b.mu
        assert model_a.mu > 1500.0
        assert model_b.mu < 1500.0
    
    def test_uncertainty_decreases(self):
        """Test that uncertainty decreases with more matches."""
        from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater, TrueSkillRating
        
        updater = MultiPlayerTrueSkillUpdater(tau=0.0)  # No dynamics
        
        initial_sigma = 166.0
        player = TrueSkillRating(mu=1500.0, sigma=initial_sigma)
        opponent = TrueSkillRating(mu=1500.0, sigma=initial_sigma)
        
        # After many matches, sigma should generally decrease
        # Note: with tau=0, this should be more consistent
        for _ in range(10):
            updated = updater.update_from_ranking(
                [player, opponent],
                [1, 2]
            )
            player = updated[0]
        
        # Sigma should be lower (player is more certain now)
        assert player.sigma < initial_sigma


# =============================================================================
# Model Serialization Tests
# =============================================================================

class TestModelSerialization:
    """Test model serialization round-trips."""
    
    def test_model_round_trip(self):
        """Test model to_dict/from_dict preserves data."""
        from langscope.core.model import LLMModel
        
        original = LLMModel(
            name="Test Model",
            model_id="test-model-001",
            provider="test-provider",
            input_cost_per_million=5.0,
            output_cost_per_million=10.0,
            pricing_source="official",
            max_matches=100
        )
        
        # Add some domain data
        original.set_domain_trueskill(
            domain="coding",
            raw_mu=1600.0,
            raw_sigma=120.0,
            cost_mu=1580.0,
            cost_sigma=125.0
        )
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = LLMModel.from_dict(data)
        
        # Verify
        assert restored.name == original.name
        assert restored.model_id == original.model_id
        assert restored.provider == original.provider
        assert restored.input_cost_per_million == original.input_cost_per_million
        assert "coding" in restored.trueskill_by_domain
        assert restored.trueskill_by_domain["coding"].raw.mu == 1600.0
    
    def test_match_round_trip(self):
        """Test match to_dict/from_dict preserves data."""
        from langscope.evaluation.match import Match, MatchParticipant, MatchResponse
        
        original = Match(
            match_id="test_match_123",
            domain="medical",
            timestamp="2024-01-01T12:00:00Z"
        )
        
        original.case_text = "Test case"
        original.question_text = "Test question"
        original.raw_ranking = {"a": 1, "b": 2}
        original.cost_ranking = {"a": 2, "b": 1}
        original.status = "completed"
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = Match.from_dict(data)
        
        # Verify
        assert restored.match_id == original.match_id
        assert restored.domain == original.domain
        assert restored.raw_ranking == original.raw_ranking
        assert restored.status == "completed"


# =============================================================================
# Dimension Score Integration Tests
# =============================================================================

class TestDimensionScoreIntegration:
    """Test dimension scoring integration."""
    
    def test_full_dimension_calculation(self):
        """Test calculating all dimension scores for a model."""
        from langscope.core.dimensions import calculate_dimension_scores
        
        scores = calculate_dimension_scores(
            mu_raw=1550.0,
            cost_per_million=5.0,
            latency_ms=500.0,
            ttft_ms=100.0,
            response_variance=0.2,
            output_tokens=500,
            constraints_satisfied=9,
            total_constraints=10,
            hallucination_count=1,
            verifiable_claims=20,
            quality_at_max=1400.0,
            quality_at_baseline=1500.0
        )
        
        # All scores should be computed
        assert scores.raw_quality == 1550.0
        assert scores.cost_adjusted > 0
        assert 0 < scores.latency < 1
        assert 0 < scores.ttft < 1
        assert 0 < scores.consistency < 1
        assert scores.token_efficiency > 0
        assert scores.instruction_following == 0.9
        assert scores.hallucination_resistance == 0.95
        assert abs(scores.long_context - 14/15) < 0.01
        assert scores.combined > 0


# =============================================================================
# Phase 24 Integration Tests
# =============================================================================

class TestMatchRouting:
    """Test match routing between subjective and ground truth workflows."""
    
    def test_subjective_domain_routing(self):
        """24.1: Subjective domain routes to MultiPlayerMatchWorkflow."""
        from langscope.federation.router import (
            get_evaluation_type,
            is_subjective_domain,
            is_ground_truth_domain,
        )
        
        # Subjective domains
        assert get_evaluation_type("general") == "subjective"
        assert get_evaluation_type("clinical_reasoning") == "subjective"
        assert get_evaluation_type("coding_python") == "subjective"
        
        assert is_subjective_domain("general") is True
        assert is_ground_truth_domain("general") is False
    
    def test_ground_truth_domain_routing(self):
        """24.1: Ground truth domain routes to GroundTruthMatchWorkflow."""
        from langscope.federation.router import (
            get_evaluation_type,
            is_subjective_domain,
            is_ground_truth_domain,
        )
        
        # Ground truth domains
        assert get_evaluation_type("asr") == "ground_truth"
        assert get_evaluation_type("visual_qa") == "ground_truth"
        assert get_evaluation_type("needle_in_haystack") == "ground_truth"
        
        assert is_ground_truth_domain("asr") is True
        assert is_subjective_domain("asr") is False
    
    def test_list_domains_by_type(self):
        """24.1: List domains by evaluation type."""
        from langscope.federation.router import (
            list_ground_truth_domains,
            list_subjective_domains,
        )
        
        gt_domains = list_ground_truth_domains()
        subjective_domains = list_subjective_domains()
        
        # Should have both types
        assert len(gt_domains) > 0
        assert len(subjective_domains) > 0
        
        # No overlap
        assert len(set(gt_domains) & set(subjective_domains)) == 0
        
        # Specific domains in correct lists
        assert "asr" in gt_domains
        assert "general" in subjective_domains


class TestDomainEvaluationType:
    """Test evaluation_type field in domains."""
    
    def test_domain_settings_evaluation_type(self):
        """24.1.2: Domain settings includes evaluation_type."""
        from langscope.domain.domain_config import DomainSettings
        
        # Default is subjective
        settings = DomainSettings()
        assert settings.evaluation_type == "subjective"
        
        # Can set to ground_truth
        gt_settings = DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="asr",
            primary_metric="wer"
        )
        assert gt_settings.evaluation_type == "ground_truth"
        assert gt_settings.ground_truth_domain == "asr"
        assert gt_settings.primary_metric == "wer"
    
    def test_domain_settings_serialization(self):
        """24.1.2: Evaluation type persists in serialization."""
        from langscope.domain.domain_config import DomainSettings
        
        settings = DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="needle_in_haystack",
            primary_metric="retrieval_accuracy"
        )
        
        # Serialize
        data = settings.to_dict()
        assert data["evaluation_type"] == "ground_truth"
        assert data["ground_truth_domain"] == "needle_in_haystack"
        assert data["primary_metric"] == "retrieval_accuracy"
        
        # Deserialize
        restored = DomainSettings.from_dict(data)
        assert restored.evaluation_type == "ground_truth"
        assert restored.ground_truth_domain == "needle_in_haystack"
    
    def test_gt_domain_templates(self):
        """24.1.2: Ground truth domain templates are configured correctly."""
        from langscope.domain.domain_config import DOMAIN_TEMPLATES
        
        # Check ASR template
        asr = DOMAIN_TEMPLATES.get("asr")
        assert asr is not None
        assert asr.settings.evaluation_type == "ground_truth"
        assert asr.settings.primary_metric == "wer"
        
        # Check Needle in Haystack template
        needle = DOMAIN_TEMPLATES.get("needle_in_haystack")
        assert needle is not None
        assert needle.settings.evaluation_type == "ground_truth"
        assert needle.settings.primary_metric == "retrieval_accuracy"


class Test10DimensionalWorkflow:
    """Test 10-dimensional rating workflow."""
    
    def test_10d_workflow_initialization(self):
        """24.2: 10D workflow initializes with enable_10d=True."""
        from langscope.federation.workflow import MultiPlayerMatchWorkflow
        from langscope.core.model import LLMModel
        
        models = [
            LLMModel(
                name=f"Model {i}",
                model_id=f"model_{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            for i in range(6)
        ]
        
        # With 10D enabled
        workflow = MultiPlayerMatchWorkflow(
            domain="general",
            models=models,
            enable_10d=True
        )
        assert workflow.enable_10d is True
        assert hasattr(workflow, 'dimension_ranker')
        
        # Without 10D
        workflow_no_10d = MultiPlayerMatchWorkflow(
            domain="general",
            models=models,
            enable_10d=False
        )
        assert workflow_no_10d.enable_10d is False
    
    def test_10d_dimension_ranker(self):
        """24.2: DimensionRanker computes all 10 dimensions."""
        from langscope.ranking.dimension_ranker import DimensionRanker
        from langscope.evaluation.metrics import BattleMetrics
        from langscope.core.dimensions import Dimension
        
        ranker = DimensionRanker()
        
        # Create test metrics
        metrics = {
            "model_a": BattleMetrics(
                model_id="model_a",
                latency_ms=200.0,
                ttft_ms=50.0,
                cost_usd=0.01,
                input_tokens=100,
                output_tokens=200,
            ),
            "model_b": BattleMetrics(
                model_id="model_b",
                latency_ms=500.0,
                ttft_ms=120.0,
                cost_usd=0.02,
                input_tokens=100,
                output_tokens=400,
            ),
        }
        
        raw_rankings = {"model_a": 1, "model_b": 2}
        mu_raws = {"model_a": 1550.0, "model_b": 1500.0}
        cost_per_millions = {"model_a": 5.0, "model_b": 10.0}
        
        result = ranker.compute_all_rankings(
            match_id="test_match",
            raw_rankings=raw_rankings,
            metrics=metrics,
            mu_raws=mu_raws,
            cost_per_millions=cost_per_millions,
        )
        
        # Should have rankings for all dimensions
        assert len(result.dimension_rankings) > 0
        
        # Model A should rank better in latency (faster)
        latency_ranking = result.dimension_rankings.get(Dimension.LATENCY)
        if latency_ranking:
            assert latency_ranking.rankings["model_a"] < latency_ranking.rankings["model_b"]


class TestGroundTruthWorkflow:
    """Test ground truth evaluation workflow."""
    
    def test_gt_match_result_structure(self):
        """24.2: GT match result has correct structure."""
        from langscope.ground_truth.workflow import GroundTruthMatchResult
        
        result = GroundTruthMatchResult(
            match_id="gt_test_001",
            domain="asr",
            sample_id="sample_001",
            rankings={"model_a": 1, "model_b": 2},
            participant_count=2
        )
        
        assert result.match_type == "ground_truth"
        assert result.status == "completed"
        
        # Serialize
        data = result.to_dict()
        assert data["_id"] == "gt_test_001"
        assert data["match_type"] == "ground_truth"
        assert data["rankings"] == {"model_a": 1, "model_b": 2}
    
    def test_gt_workflow_initialization(self):
        """24.2: GT workflow initializes correctly."""
        from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
        from langscope.ground_truth.judge import EvaluationMode
        
        workflow = GroundTruthMatchWorkflow(
            domain="asr",
            evaluation_mode=EvaluationMode.METRICS_ONLY
        )
        
        assert workflow.domain == "asr"
        assert workflow.evaluation_mode == EvaluationMode.METRICS_ONLY


class TestMultiProviderDeployments:
    """Test multi-provider deployments of same base model."""
    
    def test_same_base_different_providers(self):
        """24.2: Same base model can have multiple provider deployments."""
        from langscope.core.model import LLMModel
        
        # Same conceptual model from different providers
        gpt4_openai = LLMModel(
            name="GPT-4 (OpenAI)",
            model_id="gpt-4-openai",
            provider="openai",
            input_cost_per_million=30.0,
            output_cost_per_million=60.0,
        )
        
        gpt4_azure = LLMModel(
            name="GPT-4 (Azure)",
            model_id="gpt-4-azure",
            provider="azure",
            input_cost_per_million=28.0,
            output_cost_per_million=56.0,
        )
        
        # Should have different model_ids
        assert gpt4_openai.model_id != gpt4_azure.model_id
        
        # Can participate in same match
        assert gpt4_openai.provider != gpt4_azure.provider


class TestStratifiedSampling:
    """Test stratified sampling for ground truth."""
    
    def test_sampling_configuration(self):
        """24.2: Stratified sampling respects configuration."""
        from langscope.ground_truth.sampling import SamplingStrategy
        
        config = SamplingStrategy(
            dimensions=["difficulty", "language"],
            distributions={"difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3}},
            min_per_stratum=2
        )
        
        assert "difficulty" in config.dimensions
        assert "language" in config.dimensions
        assert config.distributions["difficulty"]["easy"] == 0.2
        assert config.min_per_stratum == 2


class TestMatchRouterIntegration:
    """Test MatchRouter integration."""
    
    def test_router_config(self):
        """24.1: Router config controls workflow behavior."""
        from langscope.federation.router import MatchRouter, MatchRouterConfig
        
        config = MatchRouterConfig(
            fallback_to_subjective=True,
            enable_10d=True,
            gt_evaluation_mode="hybrid"
        )
        
        router = MatchRouter(config=config)
        
        assert router.config.fallback_to_subjective is True
        assert router.config.enable_10d is True
        assert router.config.gt_evaluation_mode == "hybrid"
    
    def test_router_domain_detection(self):
        """24.1: Router correctly detects domain type."""
        from langscope.federation.router import MatchRouter
        
        router = MatchRouter()
        
        # Subjective domains
        assert router.get_evaluation_type("general") == "subjective"
        assert router.get_evaluation_type("clinical_reasoning") == "subjective"
        
        # GT domains
        assert router.get_evaluation_type("asr") == "ground_truth"
        assert router.get_evaluation_type("needle_in_haystack") == "ground_truth"


class TestCombinedLeaderboard:
    """Test combined leaderboard functionality."""
    
    def test_combined_leaderboard_structure(self):
        """24.1.3: Combined leaderboard includes both domain types."""
        from langscope.domain.domain_config import DOMAIN_TEMPLATES
        
        subjective_count = 0
        gt_count = 0
        
        for name, domain in DOMAIN_TEMPLATES.items():
            if domain.settings.evaluation_type == "subjective":
                subjective_count += 1
            elif domain.settings.evaluation_type == "ground_truth":
                gt_count += 1
        
        # Should have both types
        assert subjective_count > 0
        assert gt_count > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

