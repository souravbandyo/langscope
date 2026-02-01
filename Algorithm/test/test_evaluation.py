"""
Unit tests for langscope/evaluation/ modules.

Tests:
- match.py: Match management
- aggregation.py: Judge aggregation
- metrics.py: Performance metrics
- penalties.py: Penalty calculations
"""

import pytest
import math
from datetime import datetime


# =============================================================================
# Match Tests (EVAL-001 to EVAL-011)
# =============================================================================

class TestMatch:
    """Test Match class."""
    
    def test_create_new_match(self):
        """EVAL-001: Create new match."""
        from langscope.evaluation.match import Match
        
        match = Match(
            match_id="",
            domain="coding",
            timestamp=""
        )
        
        # Auto-generated ID
        assert match.match_id.startswith("match_")
        assert len(match.match_id) > 10
        
        # Timestamp generated
        assert match.timestamp != ""
        assert "Z" in match.timestamp
    
    def test_add_competitors(self):
        """EVAL-002: Add competitors to match."""
        from langscope.evaluation.match import Match, MatchParticipant
        
        match = Match(
            match_id="test_match",
            domain="coding",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Add 5 competitors
        for i in range(5):
            match.competitors.append(MatchParticipant(
                model_id=f"model_{i}",
                model_name=f"Model {i}",
                role="competitor"
            ))
        
        assert len(match.competitors) == 5
        assert match.participant_count == 5
    
    def test_add_judges(self):
        """EVAL-003: Add judges to match."""
        from langscope.evaluation.match import Match, MatchParticipant
        
        match = Match(
            match_id="test_match",
            domain="coding",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Add 3 judges
        for i in range(3):
            match.judges.append(MatchParticipant(
                model_id=f"judge_{i}",
                model_name=f"Judge {i}",
                role="judge",
                mu_before=1550.0 + i * 20
            ))
        
        assert len(match.judges) == 3
    
    def test_record_response(self):
        """EVAL-004: Record match response."""
        from langscope.evaluation.match import Match, MatchResponse
        
        match = Match(
            match_id="test_match",
            domain="coding",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        response = MatchResponse(
            model_id="model_0",
            text="This is the response text",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.001,
            latency_ms=150.0
        )
        
        match.responses.append(response)
        
        assert len(match.responses) == 1
        retrieved = match.get_response("model_0")
        assert retrieved is not None
        assert retrieved.text == "This is the response text"
    
    def test_record_rankings(self):
        """EVAL-005: Record match rankings."""
        from langscope.evaluation.match import Match, MatchResponse
        
        match = Match(
            match_id="test_match",
            domain="coding",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Add some responses
        for i in range(5):
            match.responses.append(MatchResponse(
                model_id=f"model_{i}",
                text=f"Response {i}"
            ))
        
        raw_ranking = {f"model_{i}": i + 1 for i in range(5)}
        cost_ranking = {f"model_{i}": 5 - i for i in range(5)}
        
        match.set_rankings(raw_ranking, cost_ranking)
        
        assert match.raw_ranking == raw_ranking
        assert match.cost_ranking == cost_ranking
        
        # Response ranks should be updated
        assert match.responses[0].raw_rank == 1
        assert match.responses[0].cost_rank == 5
    
    def test_serialization(self):
        """EVAL-007: Match serialization."""
        from langscope.evaluation.match import Match, MatchParticipant, MatchResponse
        
        match = Match(
            match_id="test_123",
            domain="medical",
            timestamp="2024-01-01T12:00:00Z"
        )
        
        match.case_text = "Test case"
        match.question_text = "Test question"
        match.status = "completed"
        
        data = match.to_dict()
        
        assert data["_id"] == "test_123"
        assert data["domain"] == "medical"
        assert data["prompt"]["case_text"] == "Test case"
        assert data["meta"]["status"] == "completed"
    
    def test_deserialization(self):
        """Test match deserialization."""
        from langscope.evaluation.match import Match
        
        data = {
            "_id": "match_456",
            "domain": "legal",
            "timestamp": "2024-01-15T10:00:00Z",
            "prompt": {
                "case_text": "Legal case",
                "question_text": "Legal question"
            },
            "judgment": {
                "raw_ranking": {"a": 1, "b": 2},
                "cost_adjusted_ranking": {"a": 2, "b": 1}
            },
            "meta": {
                "status": "completed",
                "info_bits": 6.9
            }
        }
        
        match = Match.from_dict(data)
        
        assert match.match_id == "match_456"
        assert match.domain == "legal"
        assert match.raw_ranking == {"a": 1, "b": 2}
    
    def test_timestamp_validation(self):
        """EVAL-008: Match timestamp validation."""
        from langscope.evaluation.match import Match
        
        match = Match(
            match_id="",
            domain="coding",
            timestamp=""
        )
        
        # Should be ISO format with Z suffix
        assert match.timestamp.endswith("Z")
        
        # Should be parseable
        try:
            datetime.fromisoformat(match.timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Timestamp is not valid ISO format")
    
    def test_info_bits_calculation(self):
        """EVAL-009: Match info bits calculation."""
        from langscope.evaluation.match import Match, MatchParticipant
        
        match = Match(
            match_id="test",
            domain="coding",
            timestamp=""
        )
        
        # Add 5 competitors
        for i in range(5):
            match.competitors.append(MatchParticipant(
                model_id=f"model_{i}",
                model_name=f"Model {i}",
                role="competitor"
            ))
        
        # Reinitialize to calculate info_bits
        match.__post_init__()
        
        # log2(5!) ≈ 6.9
        assert abs(match.info_bits - math.log2(math.factorial(5))) < 0.01
    
    def test_empty_match(self):
        """EVAL-010: Edge case - Empty match."""
        from langscope.evaluation.match import Match
        
        match = Match(
            match_id="empty",
            domain="test",
            timestamp=""
        )
        
        assert match.participant_count == 0
        assert match.get_competitor_ids() == []
        assert match.get_response("nonexistent") is None


class TestMatchParticipant:
    """Test MatchParticipant class."""
    
    def test_create_participant(self):
        """Test creating a participant."""
        from langscope.evaluation.match import MatchParticipant
        
        participant = MatchParticipant(
            model_id="test-001",
            model_name="Test Model",
            role="competitor",
            mu_before=1500.0,
            sigma_before=166.0
        )
        
        assert participant.model_id == "test-001"
        assert participant.role == "competitor"
    
    def test_to_dict(self):
        """Test participant serialization."""
        from langscope.evaluation.match import MatchParticipant
        
        participant = MatchParticipant(
            model_id="test-001",
            model_name="Test Model",
            role="judge",
            mu_before=1550.0,
            sigma_before=100.0,
            mu_after=1560.0,
            sigma_after=95.0
        )
        
        data = participant.to_dict()
        
        assert data["model_id"] == "test-001"
        assert data["role"] == "judge"
        assert data["mu_after"] == 1560.0


class TestMatchResponse:
    """Test MatchResponse class."""
    
    def test_create_response(self):
        """Test creating a response."""
        from langscope.evaluation.match import MatchResponse
        
        response = MatchResponse(
            model_id="model-001",
            text="This is the response",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.001,
            latency_ms=150.0
        )
        
        assert response.model_id == "model-001"
        assert response.total_tokens == 300
    
    def test_to_dict(self):
        """Test response serialization."""
        from langscope.evaluation.match import MatchResponse
        
        response = MatchResponse(
            model_id="model-001",
            text="Response text",
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            cost_usd=0.0005,
            latency_ms=200.0,
            raw_rank=2,
            cost_rank=1
        )
        
        data = response.to_dict()
        
        assert data["model_id"] == "model-001"
        assert data["raw_rank"] == 2
        assert data["cost_rank"] == 1


class TestCreateMatch:
    """Test create_match function."""
    
    def test_create_match_function(self):
        """Test match creation helper."""
        from langscope.evaluation.match import create_match
        from langscope.core.model import LLMModel
        
        # Create models
        competitors = []
        for i in range(5):
            model = LLMModel(
                name=f"Competitor {i}",
                model_id=f"comp_{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            competitors.append(model)
        
        judges = []
        for i in range(3):
            model = LLMModel(
                name=f"Judge {i}",
                model_id=f"judge_{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            judges.append(model)
        
        case_creator = LLMModel(
            name="Case Creator",
            model_id="case_creator",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        question_creator = LLMModel(
            name="Question Creator",
            model_id="question_creator",
            provider="test",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0
        )
        
        match = create_match(
            domain="coding",
            competitors=competitors,
            judges=judges,
            case_creator=case_creator,
            question_creator=question_creator
        )
        
        assert len(match.competitors) == 5
        assert len(match.judges) == 3
        assert match.case_creator is not None
        assert match.question_creator is not None
        assert match.status == "pending"


# =============================================================================
# Aggregation Tests (EVAL-020 to EVAL-027)
# =============================================================================

class TestAggregation:
    """Test judge aggregation functions."""
    
    def test_unanimous_votes(self):
        """EVAL-020: Aggregate unanimous judge votes."""
        from langscope.ranking.cost_adjustment import aggregate_judge_rankings
        
        # All judges agree
        rankings = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 2, "c": 3},
        ]
        
        result = aggregate_judge_rankings(rankings)
        
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3
    
    def test_split_votes(self):
        """EVAL-021: Aggregate split votes (3-2)."""
        from langscope.ranking.cost_adjustment import aggregate_judge_rankings
        
        # 3 prefer a > b, 2 prefer b > a
        rankings = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
            {"a": 2, "b": 1},
            {"a": 2, "b": 1},
        ]
        
        result = aggregate_judge_rankings(rankings)
        
        # Majority should win
        assert result["a"] == 1
    
    def test_weight_normalization(self):
        """EVAL-024: Judge weight normalization."""
        from langscope.ranking.cost_adjustment import compute_judge_weights
        
        ratings = [1600.0, 1500.0, 1550.0]
        weights = compute_judge_weights(ratings)
        
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 0.001


# =============================================================================
# Metrics Tests (EVAL-030 to EVAL-038)
# =============================================================================

class TestMetrics:
    """Test performance metrics calculations."""
    
    def test_latency_metric(self):
        """EVAL-030: Calculate latency metric."""
        from langscope.core.dimensions import compute_latency_score
        
        score = compute_latency_score(500.0, tau_latency=1000.0)
        
        # S = 1 / (1 + 500/1000) = 1/1.5 ≈ 0.667
        assert abs(score - 0.667) < 0.01
    
    def test_ttft_metric(self):
        """EVAL-031: Calculate TTFT metric."""
        from langscope.core.dimensions import compute_ttft_score
        
        score = compute_ttft_score(100.0, tau_ttft=200.0)
        
        # S = 1 / (1 + 100/200) = 1/1.5 ≈ 0.667
        assert abs(score - 0.667) < 0.01
    
    def test_token_efficiency(self):
        """EVAL-032: Calculate token efficiency."""
        from langscope.core.dimensions import compute_token_efficiency_score
        
        score = compute_token_efficiency_score(mu_raw=1500.0, output_tokens=1000)
        
        # S = 1500 / log(1001)
        expected = 1500.0 / math.log(1001)
        assert abs(score - expected) < 0.1
    
    def test_consistency_score(self):
        """EVAL-033: Calculate consistency score."""
        from langscope.core.dimensions import compute_consistency_score
        
        # Low variance = high consistency
        score = compute_consistency_score(0.5)
        
        # S = 1 / (1 + 0.5) ≈ 0.667
        assert abs(score - 0.667) < 0.01
    
    def test_instruction_following_rate(self):
        """EVAL-034: Calculate instruction following rate."""
        from langscope.core.dimensions import compute_instruction_following_score
        
        # 8 out of 10 constraints met
        score = compute_instruction_following_score(8, 10)
        
        assert score == 0.8
    
    def test_hallucination_resistance(self):
        """EVAL-035: Calculate hallucination resistance."""
        from langscope.core.dimensions import compute_hallucination_resistance_score
        
        # 2 hallucinations out of 20 claims
        score = compute_hallucination_resistance_score(2, 20)
        
        # 1 - 2/20 = 0.9
        assert score == 0.9
    
    def test_long_context_degradation(self):
        """EVAL-036: Calculate long context degradation."""
        from langscope.core.dimensions import compute_long_context_score
        
        # 10% degradation at max context
        score = compute_long_context_score(1350.0, 1500.0)
        
        # 1350/1500 = 0.9
        assert score == 0.9
    
    def test_zero_tokens(self):
        """EVAL-038: Edge case - Zero tokens."""
        from langscope.core.dimensions import compute_token_efficiency_score
        
        # Should handle zero tokens gracefully
        score = compute_token_efficiency_score(mu_raw=1500.0, output_tokens=0)
        
        # Returns raw mu when no tokens
        assert score == 1500.0


# =============================================================================
# Penalties Tests (EVAL-040 to EVAL-044)
# =============================================================================

class TestPenalties:
    """Test penalty calculations."""
    
    def test_cost_penalty(self):
        """EVAL-040: Apply cost penalty."""
        from langscope.core.dimensions import compute_cost_adjusted_score
        
        # Expensive model
        penalized = compute_cost_adjusted_score(1500.0, cost_per_million=10.0)
        
        # Should be less than raw score
        assert penalized < 1500.0
    
    def test_no_penalty_ideal_metrics(self):
        """EVAL-044: No penalty for ideal metrics."""
        from langscope.core.dimensions import (
            compute_latency_score,
            compute_ttft_score,
            compute_consistency_score
        )
        
        # Near-zero latency
        latency_score = compute_latency_score(1.0)
        assert latency_score > 0.99
        
        # Near-zero TTFT
        ttft_score = compute_ttft_score(1.0)
        assert ttft_score > 0.99
        
        # Near-zero variance (perfect consistency)
        consistency_score = compute_consistency_score(0.001)
        assert consistency_score > 0.99


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


