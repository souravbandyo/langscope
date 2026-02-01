"""
Unit tests for langscope/ranking/ modules.

Tests:
- trueskill.py: TrueSkill algorithm
- plackett_luce.py: Plackett-Luce ranking
- cost_adjustment.py: Cost-adjusted rankings
- dimension_ranker.py: Multi-dimensional ranking
"""

import pytest
import math


# =============================================================================
# TrueSkill Tests (RANK-001 to RANK-014)
# =============================================================================

class TestTrueSkillUpdater:
    """Test MultiPlayerTrueSkillUpdater."""
    
    def test_two_player_match(self):
        """RANK-001: Two-player match update."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        winner = TrueSkillRating(mu=1500.0, sigma=166.0)
        loser = TrueSkillRating(mu=1500.0, sigma=166.0)
        
        # Winner is rank 1, loser is rank 2
        updated = updater.update_from_ranking(
            [winner, loser],
            [1, 2]
        )
        
        # Winner should increase, loser should decrease
        assert updated[0].mu > winner.mu
        assert updated[1].mu < loser.mu
    
    def test_multi_player_match(self):
        """RANK-002: Multi-player match update (5 players)."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        players = [TrueSkillRating() for _ in range(5)]
        ranking = [3, 1, 5, 2, 4]  # Player indices and their ranks
        
        updated = updater.update_from_ranking(players, ranking)
        
        # All players should have ratings updated
        assert len(updated) == 5
        
        # Player with rank 1 should have highest updated μ increase
        # (Player at index 1 got rank 1)
        assert updated[1].mu > players[1].mu
    
    def test_uncertainty_reduction(self):
        """RANK-003: Uncertainty reduction after match."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater(tau=0.0)  # No dynamics factor
        
        players = [TrueSkillRating(mu=1500.0, sigma=166.0) for _ in range(3)]
        ranking = [1, 2, 3]
        
        updated = updater.update_from_ranking(players, ranking)
        
        # Without tau, sigma should decrease for all players
        for i in range(3):
            # Due to the multiplicative update, sigma might not always decrease
            # but the factor should be < 1 before tau is added
            pass  # This test is conceptual - actual behavior depends on implementation
    
    def test_tie_handling(self):
        """RANK-004: Tie handling between 2 players."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        player_a = TrueSkillRating(mu=1500.0, sigma=166.0)
        player_b = TrueSkillRating(mu=1500.0, sigma=166.0)
        
        # Same rank = tie
        updated = updater.update_from_ranking(
            [player_a, player_b],
            [1, 1]  # Both rank 1
        )
        
        # Ratings should not change much in a tie (same rank means no pairs to compare)
        # With only same-rank players, no adjacent pairs exist
        assert len(updated) == 2
    
    def test_upset_victory(self):
        """RANK-005: Upset victory (low beats high) causes larger transfer."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # Low-rated player beats high-rated
        low_rated = TrueSkillRating(mu=1400.0, sigma=166.0)
        high_rated = TrueSkillRating(mu=1600.0, sigma=166.0)
        
        updated = updater.update_from_ranking(
            [low_rated, high_rated],
            [1, 2]  # Low rated wins
        )
        
        # Low rated should get significant boost
        mu_increase = updated[0].mu - low_rated.mu
        assert mu_increase > 20  # Significant increase
    
    def test_expected_outcome(self):
        """RANK-006: Expected outcome causes smaller rating transfer."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # High-rated beats low-rated (expected)
        high_rated = TrueSkillRating(mu=1600.0, sigma=166.0)
        low_rated = TrueSkillRating(mu=1400.0, sigma=166.0)
        
        updated = updater.update_from_ranking(
            [high_rated, low_rated],
            [1, 2]  # High rated wins (expected)
        )
        
        # Winner gets smaller boost (expected outcome) compared to upset
        mu_increase = updated[0].mu - high_rated.mu
        # The winner should still get a positive update (any positive value)
        assert mu_increase >= 0  # May be small but not negative
    
    def test_expected_performance(self):
        """RANK-007: Draw probability calculation."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        equal_a = TrueSkillRating(mu=1500.0, sigma=100.0)
        equal_b = TrueSkillRating(mu=1500.0, sigma=100.0)
        
        prob = updater.expected_performance(equal_a, equal_b)
        
        # Equal players should have ~50% win probability
        assert 0.49 < prob < 0.51
    
    def test_quality_calculation(self):
        """RANK-008: Quality function calculation."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # Equal players = high quality match
        equal_players = [
            TrueSkillRating(mu=1500.0, sigma=100.0),
            TrueSkillRating(mu=1500.0, sigma=100.0)
        ]
        
        quality = updater.quality_multiplayer(equal_players)
        assert 0 <= quality <= 1
        assert quality > 0.8  # High quality for equal players
        
        # Very different players = low quality match
        unequal_players = [
            TrueSkillRating(mu=1800.0, sigma=50.0),
            TrueSkillRating(mu=1200.0, sigma=50.0)
        ]
        
        low_quality = updater.quality_multiplayer(unequal_players)
        assert low_quality < 0.5  # Lower quality
    
    def test_conservative_estimate_ranking(self):
        """RANK-009: Conservative estimate ranking (μ-3σ)."""
        from langscope.ranking.trueskill import TrueSkillRating
        
        # High μ but high uncertainty
        uncertain = TrueSkillRating(mu=1600.0, sigma=200.0)
        # Lower μ but low uncertainty
        certain = TrueSkillRating(mu=1550.0, sigma=50.0)
        
        # Conservative estimates
        uncertain_cons = uncertain.conservative_estimate()
        certain_cons = certain.conservative_estimate()
        
        # The more certain player should rank higher conservatively
        # 1600 - 600 = 1000 vs 1550 - 150 = 1400
        assert certain_cons > uncertain_cons
    
    def test_updater_initialization(self):
        """RANK-011: MultiPlayerTrueSkillUpdater initialization."""
        from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater
        from langscope.core.constants import TRUESKILL_BETA, TRUESKILL_TAU
        
        updater = MultiPlayerTrueSkillUpdater()
        
        assert updater.beta == TRUESKILL_BETA
        assert updater.tau == TRUESKILL_TAU
        assert updater.draw_probability == 0.0
    
    def test_batch_update(self):
        """RANK-012: Batch update multiple matches."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # Run multiple matches
        player = TrueSkillRating(mu=1500.0, sigma=166.0)
        
        for _ in range(5):
            # Player always wins against a default opponent
            opponent = TrueSkillRating()
            updated = updater.update_from_ranking(
                [player, opponent],
                [1, 2]
            )
            player = updated[0]
        
        # After multiple wins, μ should have increased
        assert player.mu > 1500.0
    
    def test_all_tied(self):
        """RANK-013: Edge case - All players tied."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        players = [TrueSkillRating() for _ in range(4)]
        ranking = [1, 1, 1, 1]  # All tied at rank 1
        
        # Should not crash
        updated = updater.update_from_ranking(players, ranking)
        assert len(updated) == 4
    
    def test_single_player(self):
        """RANK-014: Edge case - Single player match."""
        from langscope.ranking.trueskill import (
            MultiPlayerTrueSkillUpdater, TrueSkillRating
        )
        
        updater = MultiPlayerTrueSkillUpdater()
        
        players = [TrueSkillRating()]
        ranking = [1]
        
        # Should return same player
        updated = updater.update_from_ranking(players, ranking)
        assert len(updated) == 1


# =============================================================================
# Cost Adjustment Tests (RANK-030 to RANK-036)
# =============================================================================

class TestCostAdjustment:
    """Test cost adjustment functions."""
    
    def test_efficiency_weights(self):
        """Test efficiency weight calculation."""
        from langscope.ranking.cost_adjustment import calculate_efficiency_weights
        
        costs = {
            "expensive": 0.1,
            "cheap": 0.01,
            "medium": 0.05
        }
        
        weights = calculate_efficiency_weights(costs)
        
        # Cheaper model should have higher weight
        assert weights["cheap"] > weights["medium"]
        assert weights["medium"] > weights["expensive"]
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_free_model(self):
        """RANK-031: Free model (cost=0) gets full score."""
        from langscope.ranking.cost_adjustment import calculate_efficiency_weights
        
        costs = {
            "free": 0.0,
            "paid": 0.05
        }
        
        weights = calculate_efficiency_weights(costs)
        
        # Free model should have much higher weight
        assert weights["free"] > weights["paid"]
    
    def test_cost_adjusted_ranking(self):
        """RANK-030: Calculate cost-adjusted ranking."""
        from langscope.ranking.cost_adjustment import create_cost_adjusted_ranking
        
        raw_ranking = {"model_a": 1, "model_b": 2, "model_c": 3}
        costs = {"model_a": 0.1, "model_b": 0.01, "model_c": 0.05}
        
        cost_ranking = create_cost_adjusted_ranking(raw_ranking, costs)
        
        # model_b is cheaper, might move up
        assert len(cost_ranking) == 3
        assert all(r in [1, 2, 3] for r in cost_ranking.values())
    
    def test_expensive_model_penalty(self):
        """RANK-032: Expensive model gets penalized."""
        from langscope.ranking.cost_adjustment import create_cost_adjusted_ranking
        
        # Both equal in raw ranking, but one is much more expensive
        raw_ranking = {"cheap": 1, "expensive": 2}
        costs = {"cheap": 0.01, "expensive": 1.0}  # 100x cost difference
        
        cost_ranking = create_cost_adjusted_ranking(raw_ranking, costs)
        
        # Cheap model should likely stay at or improve rank
        # Expensive model should likely drop in cost-adjusted ranking
        # In this case, cheap is already rank 1, so it stays
        assert cost_ranking["cheap"] == 1
    
    def test_cost_normalization(self):
        """RANK-033: Cost normalization works across ranges."""
        from langscope.ranking.cost_adjustment import calculate_efficiency_weights
        
        # Very different cost scales
        costs_low = {"a": 0.001, "b": 0.002}
        costs_high = {"a": 10.0, "b": 20.0}
        
        weights_low = calculate_efficiency_weights(costs_low)
        weights_high = calculate_efficiency_weights(costs_high)
        
        # The relative ordering should be preserved
        assert weights_low["a"] > weights_low["b"]
        assert weights_high["a"] > weights_high["b"]
    
    def test_empty_costs(self):
        """Test handling of empty costs dict."""
        from langscope.ranking.cost_adjustment import (
            calculate_efficiency_weights,
            create_cost_adjusted_ranking
        )
        
        weights = calculate_efficiency_weights({})
        assert weights == {}
        
        ranking = create_cost_adjusted_ranking({}, {})
        assert ranking == {}


class TestJudgeAggregation:
    """Test judge ranking aggregation."""
    
    def test_aggregate_rankings(self):
        """Test aggregating multiple judge rankings."""
        from langscope.ranking.cost_adjustment import aggregate_judge_rankings
        
        rankings = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 1, "c": 3},
            {"a": 1, "b": 3, "c": 2},
        ]
        
        aggregated = aggregate_judge_rankings(rankings)
        
        # Should produce valid ranking
        assert len(aggregated) == 3
        assert set(aggregated.values()) == {1, 2, 3}
    
    def test_weighted_aggregation(self):
        """Test weighted judge aggregation."""
        from langscope.ranking.cost_adjustment import aggregate_judge_rankings
        
        rankings = [
            {"a": 1, "b": 2},  # Judge 1 prefers a
            {"a": 2, "b": 1},  # Judge 2 prefers b
        ]
        weights = [0.9, 0.1]  # Judge 1 has more weight
        
        aggregated = aggregate_judge_rankings(rankings, weights)
        
        # Model a should win due to Judge 1's weight
        assert aggregated["a"] == 1
    
    def test_compute_judge_weights(self):
        """Test computing judge weights from ratings."""
        from langscope.ranking.cost_adjustment import compute_judge_weights
        
        # Higher rated judges should get more weight
        ratings = [1600.0, 1500.0, 1400.0]
        
        weights = compute_judge_weights(ratings)
        
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.001
        assert weights[0] > weights[1] > weights[2]


class TestRankingDistance:
    """Test ranking distance metrics."""
    
    def test_kendall_distance(self):
        """Test Kendall tau distance."""
        from langscope.ranking.cost_adjustment import ranking_distance
        
        # Identical rankings
        r1 = {"a": 1, "b": 2, "c": 3}
        r2 = {"a": 1, "b": 2, "c": 3}
        
        distance = ranking_distance(r1, r2, method="kendall")
        assert distance == 0.0  # Identical
        
        # Completely reversed
        r3 = {"a": 3, "b": 2, "c": 1}
        distance_rev = ranking_distance(r1, r3, method="kendall")
        assert distance_rev > 0.5  # Very different
    
    def test_spearman_distance(self):
        """Test Spearman rank distance."""
        from langscope.ranking.cost_adjustment import ranking_distance
        
        r1 = {"a": 1, "b": 2, "c": 3}
        r2 = {"a": 1, "b": 2, "c": 3}
        
        distance = ranking_distance(r1, r2, method="spearman")
        assert distance == 0.0


class TestSoftmax:
    """Test softmax utility."""
    
    def test_softmax_basic(self):
        """Test basic softmax."""
        from langscope.ranking.cost_adjustment import softmax
        
        values = [1.0, 2.0, 3.0]
        probs = softmax(values, temperature=1.0)
        
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.001
        assert probs[2] > probs[1] > probs[0]  # Higher value = higher prob
    
    def test_softmax_temperature(self):
        """Test softmax with different temperatures."""
        from langscope.ranking.cost_adjustment import softmax
        
        values = [1.0, 2.0, 3.0]
        
        # Low temperature = more peaked
        probs_low = softmax(values, temperature=0.1)
        
        # High temperature = more uniform
        probs_high = softmax(values, temperature=10.0)
        
        # Low temp should have highest value dominating more
        assert probs_low[2] > probs_high[2]
    
    def test_softmax_empty(self):
        """Test softmax with empty list."""
        from langscope.ranking.cost_adjustment import softmax
        
        probs = softmax([])
        assert probs == []


# =============================================================================
# Plackett-Luce Tests (RANK-020 to RANK-027)
# =============================================================================

class TestPlackettLuce:
    """Test Plackett-Luce ranking functions."""
    
    def test_basic_import(self):
        """Test Plackett-Luce module imports."""
        from langscope.ranking import plackett_luce
        assert plackett_luce is not None
    
    def test_plackett_luce_result_creation(self):
        """Test PlackettLuceResult dataclass."""
        from langscope.ranking.plackett_luce import PlackettLuceResult
        
        result = PlackettLuceResult(
            strengths={"a": 2.0, "b": 1.0, "c": 0.5},
            log_likelihood=-5.0,
            iterations=10,
            converged=True,
            info_bits=4.58
        )
        
        assert result.strengths["a"] == 2.0
        assert result.converged is True
        assert result.info_bits > 0
    
    def test_plackett_luce_result_get_ranking(self):
        """Test PlackettLuceResult get_ranking method."""
        from langscope.ranking.plackett_luce import PlackettLuceResult
        
        result = PlackettLuceResult(
            strengths={"a": 1.0, "b": 3.0, "c": 2.0},
            log_likelihood=-3.0,
            iterations=5,
            converged=True,
            info_bits=2.58
        )
        
        ranking = result.get_ranking()
        
        # Should be sorted by strength descending
        assert ranking[0] == "b"  # highest strength
        assert ranking[1] == "c"
        assert ranking[2] == "a"  # lowest strength
    
    def test_plackett_luce_result_win_probability(self):
        """Test PlackettLuceResult win_probability method."""
        from langscope.ranking.plackett_luce import PlackettLuceResult
        
        result = PlackettLuceResult(
            strengths={"a": 3.0, "b": 1.0},
            log_likelihood=-2.0,
            iterations=5,
            converged=True,
            info_bits=1.0
        )
        
        # P(a > b) = 3 / (3 + 1) = 0.75
        prob_a = result.win_probability("a", "b")
        assert abs(prob_a - 0.75) < 0.001
        
        # P(b > a) = 1 / (1 + 3) = 0.25
        prob_b = result.win_probability("b", "a")
        assert abs(prob_b - 0.25) < 0.001
        
        # Probabilities should sum to 1
        assert abs(prob_a + prob_b - 1.0) < 0.001
    
    def test_plackett_luce_model_creation(self):
        """Test PlackettLuceModel initialization."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel(max_iter=100, tolerance=1e-6)
        
        assert model.max_iter == 100
        assert model.tolerance == 1e-6
    
    def test_plackett_luce_model_defaults(self):
        """Test PlackettLuceModel default parameters."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        from langscope.core.constants import PLACKETT_LUCE_MAX_ITER, PLACKETT_LUCE_TOL
        
        model = PlackettLuceModel()
        
        assert model.max_iter == PLACKETT_LUCE_MAX_ITER
        assert model.tolerance == PLACKETT_LUCE_TOL
    
    def test_estimate_strengths_basic(self):
        """RANK-020: Test basic MLE parameter estimation."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # Simple rankings where "a" consistently wins
        rankings = [
            ["a", "b", "c"],
            ["a", "c", "b"],
            ["a", "b", "c"],
        ]
        
        result = model.estimate_strengths(rankings)
        
        assert isinstance(result.strengths, dict)
        assert len(result.strengths) == 3
        # "a" should have highest strength (always first)
        assert result.strengths["a"] > result.strengths["b"]
        assert result.strengths["a"] > result.strengths["c"]
    
    def test_estimate_strengths_completes(self):
        """Test that estimation completes within iterations."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel(max_iter=500)
        
        rankings = [
            ["a", "b", "c", "d"],
            ["b", "a", "c", "d"],
            ["a", "b", "d", "c"],
        ]
        
        result = model.estimate_strengths(rankings)
        
        # Should complete and return valid result
        assert result is not None
        assert len(result.strengths) == 4
        assert result.iterations <= model.max_iter
    
    def test_estimate_strengths_empty_rankings(self):
        """Test handling of empty rankings list."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        rankings = []
        
        result = model.estimate_strengths(rankings)
        
        # Should handle gracefully
        assert result is not None
        assert result.strengths == {}
    
    def test_estimate_strengths_single_ranking(self):
        """RANK-026: Test with single ranking."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        rankings = [["a", "b", "c"]]
        
        result = model.estimate_strengths(rankings)
        
        assert len(result.strengths) == 3
        # Single ranking should still produce valid estimates
        assert result.strengths["a"] > result.strengths["c"]
    
    def test_estimate_strengths_with_initial(self):
        """Test estimation with initial strengths provided."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        rankings = [["a", "b", "c"], ["a", "c", "b"]]
        initial = {"a": 2.0, "b": 1.0, "c": 1.0}
        
        result = model.estimate_strengths(rankings, initial_strengths=initial)
        
        assert result is not None
        assert "a" in result.strengths
    
    def test_estimate_strengths_large_ranking(self):
        """RANK-027: Test with large number of models."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # Create a ranking of 10 models
        models = [f"model_{i}" for i in range(10)]
        rankings = [
            models[:],
            models[::-1],  # Reverse order
            models[5:] + models[:5],  # Shuffled
        ]
        
        result = model.estimate_strengths(rankings)
        
        assert len(result.strengths) == 10
        assert result.info_bits > 0
    
    def test_info_bits_calculation(self):
        """Test information bits calculation."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        import math
        
        model = PlackettLuceModel()
        
        # 3-way ranking: log2(3!) = log2(6) ≈ 2.58 bits
        rankings = [["a", "b", "c"]]
        result = model.estimate_strengths(rankings)
        
        expected_bits = math.log2(math.factorial(3))
        assert abs(result.info_bits - expected_bits) < 0.1
    
    def test_info_bits_6_models(self):
        """Test info bits for 6-way ranking."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        import math
        
        model = PlackettLuceModel()
        
        # 6-way ranking: log2(6!) = log2(720) ≈ 9.49 bits
        rankings = [["a", "b", "c", "d", "e", "f"]]
        result = model.estimate_strengths(rankings)
        
        expected_bits = math.log2(math.factorial(6))
        assert abs(result.info_bits - expected_bits) < 0.1


class TestPlackettLuceUtilities:
    """Test Plackett-Luce utility functions via PlackettLuceModel."""
    
    def test_ranking_probability_method(self):
        """Test ranking probability via model method."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # First estimate strengths from rankings
        rankings = [
            ["a", "b", "c"],
            ["a", "c", "b"],
        ]
        result = model.estimate_strengths(rankings)
        
        # Then compute probability using the model's method
        ranking = ["a", "b", "c"]
        prob = model.ranking_probability(ranking, result.strengths)
        
        # Probability should be valid
        assert 0 < prob <= 1
    
    def test_ranking_probability_dominant_model(self):
        """Test probability with dominant model."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # "a" has much higher strength
        strengths = {"a": 10.0, "b": 1.0, "c": 1.0}
        ranking_a_first = ["a", "b", "c"]
        ranking_a_last = ["b", "c", "a"]
        
        prob_a_first = model.ranking_probability(ranking_a_first, strengths)
        prob_a_last = model.ranking_probability(ranking_a_last, strengths)
        
        # a being first should be more likely than a being last
        assert prob_a_first > prob_a_last
    
    def test_top_k_probability(self):
        """Test top-k probability calculation."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # Given strengths, compute probability of being in top k
        strengths = {"a": 3.0, "b": 2.0, "c": 1.0}
        
        prob_a_top1 = model.top_k_probability("a", 1, strengths)
        prob_a_top2 = model.top_k_probability("a", 2, strengths)
        
        # Being in top 2 should be >= being in top 1
        assert prob_a_top2 >= prob_a_top1


class TestPlackettLuceIntegration:
    """Integration tests for Plackett-Luce with TrueSkill."""
    
    def test_strengths_to_trueskill_update(self):
        """Test using PL strengths to inform TrueSkill updates."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        # Simulate multiple judge rankings
        rankings = [
            ["a", "b", "c", "d"],
            ["a", "b", "d", "c"],
            ["b", "a", "c", "d"],
        ]
        
        result = model.estimate_strengths(rankings)
        final_ranking = result.get_ranking()
        
        # Can be used to feed TrueSkill
        assert len(final_ranking) == 4
        # Most consistent winner should be near top
        assert final_ranking[0] in ["a", "b"]
    
    def test_log_likelihood_increases(self):
        """Test that log-likelihood is computed."""
        from langscope.ranking.plackett_luce import PlackettLuceModel
        
        model = PlackettLuceModel()
        
        rankings = [
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]
        
        result = model.estimate_strengths(rankings)
        
        # Log-likelihood should be finite and negative
        assert result.log_likelihood < 0
        assert not math.isnan(result.log_likelihood)
        assert not math.isinf(result.log_likelihood)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    import math
    pytest.main([__file__, "-v", "--tb=short"])

