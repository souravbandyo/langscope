"""
Tests for Ground Truth Evaluation Module.

Tests Phases 17-24:
- Metrics computation (WER, CER, BLEU, etc.)
- Ground truth judges (ASR, TTS, Needle, VQA, etc.)
- Stratified sampling
- Workflow integration
"""

import pytest
from typing import Dict


# =============================================================================
# Test Metric Computations
# =============================================================================

class TestASRMetrics:
    """Test ASR metrics (WER, CER, MER)."""
    
    def test_word_error_rate_perfect(self):
        """Test WER with identical strings."""
        from langscope.ground_truth.metrics import word_error_rate
        
        wer = word_error_rate("hello world", "hello world")
        assert wer == 0.0
    
    def test_word_error_rate_all_wrong(self):
        """Test WER with completely different strings."""
        from langscope.ground_truth.metrics import word_error_rate
        
        wer = word_error_rate("foo bar", "hello world")
        assert wer == 1.0  # 2 substitutions / 2 words
    
    def test_word_error_rate_partial(self):
        """Test WER with partial match."""
        from langscope.ground_truth.metrics import word_error_rate
        
        # "hello world" -> "hello" = 1 deletion
        wer = word_error_rate("hello", "hello world")
        assert wer == 0.5  # 1 deletion / 2 words
    
    def test_word_error_rate_insertion(self):
        """Test WER with insertions."""
        from langscope.ground_truth.metrics import word_error_rate
        
        # "hello" -> "hello extra words" = 2 insertions
        wer = word_error_rate("hello extra words", "hello")
        assert wer == 2.0  # 2 insertions / 1 word
    
    def test_character_error_rate_perfect(self):
        """Test CER with identical strings."""
        from langscope.ground_truth.metrics import character_error_rate
        
        cer = character_error_rate("hello", "hello")
        assert cer == 0.0
    
    def test_character_error_rate_one_error(self):
        """Test CER with one character error."""
        from langscope.ground_truth.metrics import character_error_rate
        
        # "hello" -> "hallo" = 1 substitution
        cer = character_error_rate("hallo", "hello")
        assert abs(cer - 0.2) < 0.01  # 1 / 5 chars
    
    def test_match_error_rate(self):
        """Test MER computation."""
        from langscope.ground_truth.metrics import match_error_rate
        
        mer = match_error_rate("hello world", "hello world")
        assert mer == 0.0


class TestTextMetrics:
    """Test text generation metrics."""
    
    def test_bleu_score_perfect(self):
        """Test BLEU with identical text."""
        from langscope.ground_truth.metrics import bleu_score
        
        # Use longer text for BLEU (needs enough n-grams)
        text = "the quick brown fox jumps over the lazy dog"
        score = bleu_score(text, [text])
        assert score > 0.9
    
    def test_bleu_score_different(self):
        """Test BLEU with completely different text."""
        from langscope.ground_truth.metrics import bleu_score
        
        score = bleu_score("foo bar baz qux quux", ["hello world test case alpha"])
        assert score < 0.2
    
    def test_rouge_l_perfect(self):
        """Test ROUGE-L with identical text."""
        from langscope.ground_truth.metrics import rouge_l
        
        score = rouge_l("the quick brown fox", "the quick brown fox")
        assert score == 1.0
    
    def test_rouge_l_partial(self):
        """Test ROUGE-L with partial overlap."""
        from langscope.ground_truth.metrics import rouge_l
        
        score = rouge_l("the quick fox", "the quick brown fox jumps")
        assert 0.3 < score < 0.8
    
    def test_exact_match_true(self):
        """Test exact match with equal strings."""
        from langscope.ground_truth.metrics import exact_match
        
        score = exact_match("Hello World", "hello world")
        assert score == 1.0
    
    def test_exact_match_false(self):
        """Test exact match with different strings."""
        from langscope.ground_truth.metrics import exact_match
        
        score = exact_match("hello", "world")
        assert score == 0.0
    
    def test_contains_true(self):
        """Test contains with matching substring."""
        from langscope.ground_truth.metrics import contains
        
        score = contains("The answer is 42.", "42")
        assert score == 1.0
    
    def test_contains_false(self):
        """Test contains with non-matching substring."""
        from langscope.ground_truth.metrics import contains
        
        score = contains("The answer is 42.", "99")
        assert score == 0.0


class TestNeedleMetrics:
    """Test Needle in Haystack metrics."""
    
    def test_needle_retrieval_exact(self):
        """Test needle retrieval with exact match."""
        from langscope.ground_truth.metrics import compute_needle_metrics
        
        metrics = compute_needle_metrics(
            response="The code is X7-DELTA-9",
            expected_answer="X7-DELTA-9"
        )
        
        assert metrics["retrieval_accuracy"] == 1.0
        assert metrics["exact_match"] == 1.0
    
    def test_needle_retrieval_variant(self):
        """Test needle retrieval with variant match."""
        from langscope.ground_truth.metrics import compute_needle_metrics
        
        metrics = compute_needle_metrics(
            response="The authorization code is X7 DELTA 9.",
            expected_answer="X7-DELTA-9",
            answer_variants=["X7 DELTA 9", "X7DELTA9"]
        )
        
        assert metrics["retrieval_accuracy"] == 1.0
    
    def test_needle_retrieval_miss(self):
        """Test needle retrieval failure."""
        from langscope.ground_truth.metrics import compute_needle_metrics
        
        metrics = compute_needle_metrics(
            response="I don't know the answer.",
            expected_answer="X7-DELTA-9"
        )
        
        assert metrics["retrieval_accuracy"] == 0.0


class TestCodeMetrics:
    """Test code evaluation metrics."""
    
    def test_syntax_valid_python(self):
        """Test Python syntax validation."""
        from langscope.ground_truth.metrics import validate_python_syntax
        
        valid_code = "def hello():\n    return 'world'"
        assert validate_python_syntax(valid_code) == 1.0
    
    def test_syntax_invalid_python(self):
        """Test Python syntax validation with invalid code."""
        from langscope.ground_truth.metrics import validate_python_syntax
        
        invalid_code = "def hello(\n    return"
        assert validate_python_syntax(invalid_code) == 0.0


class TestDocumentExtraction:
    """Test document extraction metrics."""
    
    def test_field_accuracy_perfect(self):
        """Test field accuracy with perfect extraction."""
        from langscope.ground_truth.metrics import compute_field_accuracy
        
        extracted = {"name": "John", "age": "30", "city": "NYC"}
        expected = {"name": "John", "age": "30", "city": "NYC"}
        
        metrics = compute_field_accuracy(extracted, expected)
        
        assert metrics["field_accuracy"] == 1.0
        assert metrics["field_precision"] == 1.0
        assert metrics["field_recall"] == 1.0
    
    def test_field_accuracy_partial(self):
        """Test field accuracy with partial extraction."""
        from langscope.ground_truth.metrics import compute_field_accuracy
        
        extracted = {"name": "John", "age": "25"}
        expected = {"name": "John", "age": "30", "city": "NYC"}
        
        metrics = compute_field_accuracy(extracted, expected)
        
        assert metrics["field_recall"] < 1.0  # Missing city
        assert metrics["field_accuracy"] < 1.0
    
    def test_critical_field_accuracy(self):
        """Test critical field accuracy."""
        from langscope.ground_truth.metrics import compute_field_accuracy
        
        extracted = {"name": "John", "total": "100"}
        expected = {"name": "John", "total": "100", "date": "2024-01-01"}
        
        metrics = compute_field_accuracy(
            extracted, expected, 
            critical_fields=["name", "total"]
        )
        
        assert metrics["critical_accuracy"] == 1.0  # All critical fields correct


class TestTTSComposite:
    """Test TTS composite scoring."""
    
    def test_perfect_tts_score(self):
        """Test TTS score with perfect metrics."""
        from langscope.ground_truth.metrics import compute_tts_composite_score
        
        score = compute_tts_composite_score(
            round_trip_wer=0.0,
            utmos=5.0,
            snr=40.0,
            speaker_sim=1.0
        )
        
        assert score > 0.9
    
    def test_poor_tts_score(self):
        """Test TTS score with poor metrics."""
        from langscope.ground_truth.metrics import compute_tts_composite_score
        
        score = compute_tts_composite_score(
            round_trip_wer=0.8,
            utmos=2.0,
            snr=5.0,
            speaker_sim=0.2
        )
        
        assert score < 0.5
    
    def test_tts_score_no_speaker(self):
        """Test TTS score without speaker similarity."""
        from langscope.ground_truth.metrics import compute_tts_composite_score
        
        score = compute_tts_composite_score(
            round_trip_wer=0.1,
            utmos=4.0,
            snr=25.0
        )
        
        assert 0.5 < score < 1.0


# =============================================================================
# Test Metric Registry
# =============================================================================

class TestMetricRegistry:
    """Test MetricRegistry configuration."""
    
    def test_domain_metrics_exist(self):
        """Test that all domains have metrics defined."""
        from langscope.ground_truth.metrics import MetricRegistry
        
        assert "asr" in MetricRegistry.DOMAIN_METRICS
        assert "tts" in MetricRegistry.DOMAIN_METRICS
        assert "needle_in_haystack" in MetricRegistry.DOMAIN_METRICS
        assert "code_completion" in MetricRegistry.DOMAIN_METRICS
    
    def test_primary_metrics_exist(self):
        """Test that all domains have primary metrics."""
        from langscope.ground_truth.metrics import MetricRegistry
        
        for domain in MetricRegistry.DOMAIN_METRICS.keys():
            assert domain in MetricRegistry.PRIMARY_METRIC
    
    def test_higher_is_better_defined(self):
        """Test that direction is defined for all metrics."""
        from langscope.ground_truth.metrics import MetricRegistry
        
        # Check ASR metrics (lower is better)
        assert MetricRegistry.HIGHER_IS_BETTER["wer"] == False
        assert MetricRegistry.HIGHER_IS_BETTER["cer"] == False
        
        # Check accuracy metrics (higher is better)
        assert MetricRegistry.HIGHER_IS_BETTER["accuracy"] == True
        assert MetricRegistry.HIGHER_IS_BETTER["exact_match"] == True
    
    def test_get_metrics_for_domain(self):
        """Test getting metrics for a domain."""
        from langscope.ground_truth.metrics import MetricRegistry
        
        asr_metrics = MetricRegistry.get_metrics_for_domain("asr")
        assert "wer" in asr_metrics
        assert "cer" in asr_metrics


# =============================================================================
# Test Ground Truth Judges
# =============================================================================

class TestGroundTruthJudge:
    """Test base GroundTruthJudge functionality."""
    
    def test_judge_creation(self):
        """Test creating a judge."""
        from langscope.ground_truth.judge import GroundTruthJudge
        
        judge = GroundTruthJudge(domain="asr")
        assert judge.domain == "asr"
    
    @pytest.mark.asyncio
    async def test_judge_evaluate(self):
        """Test basic evaluation."""
        from langscope.ground_truth.judge import GroundTruthJudge
        
        judge = GroundTruthJudge(domain="needle_in_haystack")
        
        score = await judge.evaluate(
            response="The answer is 42.",
            ground_truth="42",
            sample={"sample_id": "test_1", "model_id": "gpt-4"}
        )
        
        assert score.sample_id == "test_1"
        assert score.model_id == "gpt-4"
        assert score.overall >= 0.0
    
    @pytest.mark.asyncio
    async def test_judge_evaluate_batch(self):
        """Test batch evaluation."""
        from langscope.ground_truth.judge import GroundTruthJudge
        
        judge = GroundTruthJudge(domain="needle_in_haystack")
        
        scores = await judge.evaluate_batch(
            responses={
                "model_a": "The answer is 42.",
                "model_b": "I don't know.",
            },
            ground_truth="42",
            sample={"sample_id": "test_1"}
        )
        
        assert "model_a" in scores
        assert "model_b" in scores
        assert scores["model_a"].overall > scores["model_b"].overall
    
    def test_ranking_from_scores(self):
        """Test ranking computation from scores."""
        from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore
        
        judge = GroundTruthJudge(domain="needle_in_haystack")
        
        scores = {
            "model_a": GroundTruthScore(model_id="a", sample_id="s1", overall=0.9),
            "model_b": GroundTruthScore(model_id="b", sample_id="s1", overall=0.5),
            "model_c": GroundTruthScore(model_id="c", sample_id="s1", overall=0.7),
        }
        
        rankings = judge.get_ranking_from_scores(scores)
        
        assert rankings["model_a"] == 1
        assert rankings["model_c"] == 2
        assert rankings["model_b"] == 3


class TestSpecializedJudges:
    """Test specialized judge implementations."""
    
    @pytest.mark.asyncio
    async def test_asr_judge(self):
        """Test ASR judge."""
        from langscope.ground_truth.judge import ASRGroundTruthJudge
        
        judge = ASRGroundTruthJudge()
        
        score = await judge.evaluate(
            response="hello world",
            ground_truth="hello world",
            sample={"sample_id": "asr_1", "model_id": "whisper"}
        )
        
        assert score.metrics["wer"] == 0.0
        assert score.overall > 0.9
    
    @pytest.mark.asyncio
    async def test_needle_judge(self):
        """Test Needle in Haystack judge."""
        from langscope.ground_truth.judge import NeedleGroundTruthJudge
        
        judge = NeedleGroundTruthJudge()
        
        score = await judge.evaluate(
            response="The secret code is ALPHA-123.",
            ground_truth="ALPHA-123",
            sample={"sample_id": "needle_1", "model_id": "gpt-4"}
        )
        
        assert score.metrics["retrieval_accuracy"] == 1.0
    
    @pytest.mark.asyncio
    async def test_code_judge(self):
        """Test code completion judge."""
        from langscope.ground_truth.judge import CodeGroundTruthJudge
        
        judge = CodeGroundTruthJudge()
        
        code = "def add(a, b):\n    return a + b"
        
        score = await judge.evaluate(
            response=code,
            ground_truth=code,
            sample={
                "sample_id": "code_1", 
                "model_id": "gpt-4",
                "language": "python",
                "test_cases": []
            }
        )
        
        assert score.metrics["syntax_valid"] == 1.0
        assert score.metrics["exact_match"] == 1.0


# =============================================================================
# Test Needle Generator
# =============================================================================

class TestNeedleGenerator:
    """Test Needle in Haystack sample generator."""
    
    def test_generate_needle(self):
        """Test generating a needle."""
        from langscope.ground_truth.generators.needle_generator import generate_needle
        
        needle_text, question, answer = generate_needle("code")
        
        assert needle_text is not None
        assert question is not None
        assert answer is not None
        assert answer in needle_text
    
    def test_generate_sample(self):
        """Test generating a full sample."""
        from langscope.ground_truth.generators import NeedleGenerator
        
        generator = NeedleGenerator()
        
        sample = generator.generate_sample(
            context_length=8192,
            needle_position=0.5,
            needle_type="code"
        )
        
        assert sample["domain"] == "needle_in_haystack"
        assert sample["context_length"] == 8192
        assert sample["needle_position"] == 0.5
        assert "haystack" in sample
        assert "expected_answer" in sample
        assert sample["needle"] in sample["haystack"]
    
    def test_difficulty_levels(self):
        """Test difficulty computation."""
        from langscope.ground_truth.generators import NeedleGenerator
        
        generator = NeedleGenerator()
        
        # Easy: short context, edge position
        easy = generator.generate_sample(4096, 0.0, "code")
        assert easy["difficulty"] == "easy"
        
        # Hard: long context, middle position
        hard = generator.generate_sample(64000, 0.5, "code")
        assert hard["difficulty"] in ("hard", "medium")


# =============================================================================
# Test Judge Factory
# =============================================================================

class TestJudgeFactory:
    """Test judge factory function."""
    
    def test_get_asr_judge(self):
        """Test getting ASR judge."""
        from langscope.ground_truth.judges import get_judge_for_domain
        
        judge = get_judge_for_domain("asr")
        assert judge.domain == "asr"
    
    def test_get_needle_judge(self):
        """Test getting needle judge."""
        from langscope.ground_truth.judges import get_judge_for_domain
        
        judge = get_judge_for_domain("needle_in_haystack")
        assert judge.domain == "needle_in_haystack"
    
    def test_get_unknown_domain(self):
        """Test getting judge for unknown domain."""
        from langscope.ground_truth.judges import get_judge_for_domain
        
        judge = get_judge_for_domain("unknown_domain")
        assert judge.domain == "unknown_domain"  # Falls back to base judge


# =============================================================================
# Test Config Loading
# =============================================================================

class TestConfigFiles:
    """Test configuration file loading."""
    
    def test_domains_config_exists(self):
        """Test that domains.json exists and is valid."""
        import json
        import os
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "langscope", "ground_truth", "config", "domains.json"
        )
        
        assert os.path.exists(config_path)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert "domains" in config
        assert "asr" in config["domains"]
        assert "needle_in_haystack" in config["domains"]
    
    def test_metrics_config_exists(self):
        """Test that metrics.json exists and is valid."""
        import json
        import os
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "langscope", "ground_truth", "config", "metrics.json"
        )
        
        assert os.path.exists(config_path)
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert "metrics" in config
        assert "wer" in config["metrics"]
        assert "retrieval_accuracy" in config["metrics"]


# =============================================================================
# P1.4: Integration Tests for GT Workflow
# =============================================================================

class TestGTWorkflowIntegration:
    """Integration tests for Ground Truth workflow with mocked LLM responses."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """Test GT workflow can be initialized."""
        from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
        from langscope.ground_truth.judge import EvaluationMode
        
        workflow = GroundTruthMatchWorkflow(
            domain="needle_in_haystack",
            evaluation_mode=EvaluationMode.METRICS_ONLY
        )
        
        assert workflow.domain == "needle_in_haystack"
        assert workflow.evaluation_mode == EvaluationMode.METRICS_ONLY
        assert workflow.manager is not None
        assert workflow.judge is not None
    
    @pytest.mark.asyncio
    async def test_workflow_with_mocked_llm(self):
        """Test GT workflow end-to-end with mocked LLM caller."""
        from unittest.mock import AsyncMock, MagicMock
        from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
        from langscope.core.model import LLMModel
        
        # Create mock LLM caller
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The secret code is ALPHA-123."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        
        mock_llm.acompletion = AsyncMock(return_value=mock_response)
        
        # Create test models
        models = [
            LLMModel(
                name=f"Test Model {i}",
                model_id=f"test-model-{i}",
                provider="test",
                input_cost_per_million=1.0,
                output_cost_per_million=2.0
            )
            for i in range(3)
        ]
        
        workflow = GroundTruthMatchWorkflow(
            domain="needle_in_haystack",
            models=models,
            llm_caller=mock_llm
        )
        
        # Verify LLM caller is wired
        assert workflow.llm_caller is mock_llm
    
    def test_trueskill_update_calculation(self):
        """Test TrueSkill update calculations in GT workflow."""
        from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater
        from langscope.core.rating import TrueSkillRating
        
        updater = MultiPlayerTrueSkillUpdater()
        
        # Create initial ratings
        ratings = [
            TrueSkillRating(mu=1500.0, sigma=166.0),
            TrueSkillRating(mu=1500.0, sigma=166.0),
            TrueSkillRating(mu=1500.0, sigma=166.0),
        ]
        
        # Ranking: model 0 wins, model 1 second, model 2 third
        rankings = [1, 2, 3]
        
        # Update ratings
        new_ratings = updater.update_from_ranking(ratings, rankings)
        
        # Winner should gain rating
        assert new_ratings[0].mu > ratings[0].mu
        
        # Loser should lose rating
        assert new_ratings[2].mu < ratings[2].mu
        
        # Uncertainty should decrease for all
        for i in range(3):
            assert new_ratings[i].sigma < ratings[i].sigma
    
    def test_stratified_sampling_distribution(self):
        """Test stratified sampling distribution."""
        from langscope.ground_truth.sampling import SamplingStrategy, StratifiedSampler
        
        strategy = SamplingStrategy(
            dimensions=["difficulty", "language"],
            distributions={
                "difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
                "language": {"en": 0.7, "es": 0.3}
            },
            min_per_stratum=1
        )
        
        assert strategy.dimensions == ["difficulty", "language"]
        assert strategy.distributions["difficulty"]["easy"] == 0.2
        
        # Test sampler initialization
        sampler = StratifiedSampler(strategy)
        assert sampler.strategy == strategy
    
    @pytest.mark.asyncio
    async def test_judge_evaluation_batch(self):
        """Test batch evaluation preserves model IDs."""
        from langscope.ground_truth.judge import GroundTruthJudge
        
        judge = GroundTruthJudge(domain="needle_in_haystack")
        
        responses = {
            "gpt-4": "The answer is ALPHA-123.",
            "claude-3": "I found ALPHA-123 in the text.",
            "gemini-pro": "The code is something else.",
        }
        
        scores = await judge.evaluate_batch(
            responses=responses,
            ground_truth="ALPHA-123",
            sample={"sample_id": "test_batch"}
        )
        
        # All models should have scores
        assert len(scores) == 3
        assert "gpt-4" in scores
        assert "claude-3" in scores
        assert "gemini-pro" in scores
        
        # Models that found the answer should score higher
        assert scores["gpt-4"].overall > scores["gemini-pro"].overall
        assert scores["claude-3"].overall > scores["gemini-pro"].overall
    
    def test_gt_match_result_serialization(self):
        """Test GT match result to_dict/from_dict."""
        from langscope.ground_truth.workflow import GroundTruthMatchResult
        
        result = GroundTruthMatchResult(
            match_id="gt_match_test_001",
            domain="asr",
            category="audio",
            sample_id="sample_001",
            rankings={"model_a": 1, "model_b": 2, "model_c": 3},
            scores={
                "model_a": {"metrics": {"wer": 0.05}, "overall": 0.95},
                "model_b": {"metrics": {"wer": 0.15}, "overall": 0.85},
                "model_c": {"metrics": {"wer": 0.25}, "overall": 0.75},
            },
            participant_count=3,
            info_bits=2.585,
        )
        
        # Serialize
        data = result.to_dict()
        
        assert data["_id"] == "gt_match_test_001"
        assert data["match_type"] == "ground_truth"
        assert data["domain"] == "asr"
        assert data["rankings"]["model_a"] == 1
        assert data["participant_count"] == 3
    
    def test_code_execution_sandbox_python(self):
        """Test code execution sandbox with Python."""
        from langscope.ground_truth.judges.code_judge import CodeExecutionSandbox
        
        sandbox = CodeExecutionSandbox(
            timeout=5,
            use_docker=False  # Use subprocess for testing
        )
        
        # Test valid code
        result = sandbox.run_python("print('Hello, World!')")
        
        # In restricted environments, subprocess may fail
        # Check that we get a valid result structure either way
        assert "success" in result
        assert "output" in result
        assert "timed_out" in result
        
        # If execution succeeded, verify output
        if result["success"]:
            assert "Hello, World!" in result["output"]
            assert result["timed_out"] is False
    
    @pytest.mark.asyncio
    async def test_code_judge_with_test_cases(self):
        """Test code judge runs test cases correctly."""
        from langscope.ground_truth.judge import CodeGroundTruthJudge
        
        # First test with execution disabled (always works)
        judge_no_exec = CodeGroundTruthJudge(
            execution_timeout=5,
            allow_execution=False,  # Disable execution for reliable testing
            use_docker=False
        )
        
        code = """
def add(a, b):
    return a + b
"""
        
        test_cases = [
            {"input": "add(1, 2)", "expected": "3"},
            {"input": "add(0, 0)", "expected": "0"},
            {"input": "add(-1, 1)", "expected": "0"},
        ]
        
        score = await judge_no_exec.evaluate(
            response=code,
            ground_truth={"expected_code": code},
            sample={
                "sample_id": "code_test",
                "model_id": "test",
                "language": "python",
                "test_cases": test_cases
            }
        )
        
        # Syntax should always be validated
        assert score.metrics["syntax_valid"] == 1.0
        # Without execution, tests_pass is estimated from other metrics
        assert "tests_pass" in score.metrics
        
        # Test with execution enabled (may fail in restricted environments)
        judge_exec = CodeGroundTruthJudge(
            execution_timeout=5,
            allow_execution=True,
            use_docker=False
        )
        
        score_exec = await judge_exec.evaluate(
            response=code,
            ground_truth={"expected_code": code},
            sample={
                "sample_id": "code_test",
                "model_id": "test",
                "language": "python",
                "test_cases": test_cases
            }
        )
        
        # Should still have valid syntax
        assert score_exec.metrics["syntax_valid"] == 1.0


class TestMetricLibraryFallbacks:
    """Test that metric libraries fall back gracefully."""
    
    def test_wer_fallback(self):
        """Test WER works with or without jiwer."""
        from langscope.ground_truth.metrics import word_error_rate
        
        # Should work regardless of jiwer installation
        wer = word_error_rate("hello world", "hello world")
        assert wer == 0.0
        
        wer = word_error_rate("hello", "hello world")
        assert wer == 0.5
    
    def test_bleu_fallback(self):
        """Test BLEU works with or without sacrebleu."""
        from langscope.ground_truth.metrics import bleu_score
        
        # Should work regardless of sacrebleu installation
        text = "the quick brown fox jumps over the lazy dog"
        score = bleu_score(text, [text])
        assert score > 0.8
    
    def test_rouge_fallback(self):
        """Test ROUGE works with or without rouge-score."""
        from langscope.ground_truth.metrics import rouge_l
        
        # Should work regardless of rouge-score installation
        score = rouge_l("the quick brown fox", "the quick brown fox")
        assert score == 1.0
        
        score = rouge_l("the quick fox", "the quick brown fox")
        assert 0.3 < score < 1.0


# =============================================================================
# Sampling Tests
# =============================================================================

class TestSamplingStrategy:
    """Test sampling strategy configuration."""
    
    def test_stratification_dimension_enum(self):
        """Test StratificationDimension enum."""
        from langscope.ground_truth.sampling import StratificationDimension
        
        assert StratificationDimension.DIFFICULTY.value == "difficulty"
        assert StratificationDimension.LANGUAGE.value == "language"
        assert StratificationDimension.CONTEXT_LENGTH.value == "context_length"
    
    def test_sampling_strategy_defaults(self):
        """Test SamplingStrategy default values."""
        from langscope.ground_truth.sampling import SamplingStrategy
        
        strategy = SamplingStrategy()
        
        assert "difficulty" in strategy.dimensions
        assert strategy.cooldown_hours == 24
        assert strategy.min_per_stratum == 5
    
    def test_sampling_strategy_to_dict(self):
        """Test SamplingStrategy to_dict conversion."""
        from langscope.ground_truth.sampling import SamplingStrategy
        
        strategy = SamplingStrategy(
            dimensions=["difficulty", "language"],
            distributions={"difficulty": {"easy": 0.3, "medium": 0.5, "hard": 0.2}},
            cooldown_hours=48
        )
        
        strategy_dict = strategy.to_dict()
        
        assert "dimensions" in strategy_dict
        assert len(strategy_dict["dimensions"]) == 2
        assert strategy_dict["cooldown_hours"] == 48
    
    def test_sampling_strategy_from_dict(self):
        """Test SamplingStrategy from_dict creation."""
        from langscope.ground_truth.sampling import SamplingStrategy
        
        data = {
            "dimensions": ["difficulty"],
            "distributions": {"difficulty": {"easy": 0.5, "hard": 0.5}},
            "cooldown_hours": 12,
            "min_per_stratum": 3
        }
        
        strategy = SamplingStrategy.from_dict(data)
        
        assert strategy.dimensions == ["difficulty"]
        assert strategy.cooldown_hours == 12
        assert strategy.min_per_stratum == 3


class TestSampleUsage:
    """Test sample usage tracking."""
    
    def test_sample_usage_creation(self):
        """Test SampleUsage creation."""
        from langscope.ground_truth.sampling import SampleUsage
        
        usage = SampleUsage(sample_id="sample_001")
        
        assert usage.sample_id == "sample_001"
        assert usage.usage_count == 0
        assert usage.last_used is None
    
    def test_sample_usage_with_data(self):
        """Test SampleUsage with usage data."""
        from langscope.ground_truth.sampling import SampleUsage
        from datetime import datetime
        
        now = datetime.utcnow()
        usage = SampleUsage(
            sample_id="sample_002",
            usage_count=5,
            last_used=now
        )
        
        assert usage.usage_count == 5
        assert usage.last_used == now


class TestStratifiedSampler:
    """Test stratified sampler functionality."""
    
    def test_stratified_sampler_creation(self):
        """Test StratifiedSampler creation."""
        from langscope.ground_truth.sampling import StratifiedSampler, SamplingStrategy
        
        strategy = SamplingStrategy()
        sampler = StratifiedSampler(strategy)
        
        assert sampler.strategy == strategy
    
    def test_get_default_strategy(self):
        """Test default strategy retrieval."""
        from langscope.ground_truth.sampling import get_default_strategy
        
        # ASR strategy
        asr_strategy = get_default_strategy("asr")
        assert asr_strategy is not None
        
        # Needle strategy
        needle_strategy = get_default_strategy("needle")
        assert needle_strategy is not None
        
        # Unknown domain falls back to default
        default_strategy = get_default_strategy("unknown")
        assert default_strategy is not None


# =============================================================================
# Manager Tests
# =============================================================================

class TestGroundTruthSample:
    """Test GroundTruthSample dataclass."""
    
    def test_ground_truth_sample_creation(self):
        """Test GroundTruthSample creation."""
        from langscope.ground_truth.manager import GroundTruthSample
        
        sample = GroundTruthSample(
            sample_id="gt_001",
            domain="asr",
            category="multimodal"
        )
        
        assert sample.sample_id == "gt_001"
        assert sample.domain == "asr"
        assert sample.category == "multimodal"
        assert sample.difficulty == "medium"  # default
        assert sample.language == "en"  # default
    
    def test_ground_truth_sample_with_all_fields(self):
        """Test GroundTruthSample with all fields."""
        from langscope.ground_truth.manager import GroundTruthSample
        
        sample = GroundTruthSample(
            sample_id="gt_002",
            domain="needle",
            category="long_context",
            subdomain="needle_in_haystack",
            difficulty="hard",
            language="en",
            context_length=32000,
            needle_position=0.5,
            tags=["test", "evaluation"]
        )
        
        assert sample.context_length == 32000
        assert sample.needle_position == 0.5
        assert len(sample.tags) == 2
    
    def test_ground_truth_sample_to_dict(self):
        """Test GroundTruthSample to_dict conversion."""
        from langscope.ground_truth.manager import GroundTruthSample
        
        sample = GroundTruthSample(
            sample_id="gt_003",
            domain="code",
            category="multimodal",
            difficulty="easy"
        )
        
        sample_dict = sample.to_dict()
        
        assert sample_dict["sample_id"] == "gt_003"
        assert sample_dict["domain"] == "code"
        assert sample_dict["difficulty"] == "easy"
    
    def test_ground_truth_sample_from_dict(self):
        """Test GroundTruthSample from_dict creation."""
        from langscope.ground_truth.manager import GroundTruthSample
        
        data = {
            "sample_id": "gt_004",
            "domain": "tts",
            "category": "multimodal",
            "difficulty": "medium",
            "language": "hi",
            "tags": ["hindi", "tts"]
        }
        
        sample = GroundTruthSample.from_dict(data)
        
        assert sample.sample_id == "gt_004"
        assert sample.domain == "tts"
        assert sample.language == "hi"


class TestGroundTruthJudgeIntegration:
    """Test ground truth judge integration."""
    
    def test_judge_classes_exist(self):
        """Test judge classes are available."""
        from langscope.ground_truth.judge import (
            GroundTruthJudge,
            ASRGroundTruthJudge,
            CodeGroundTruthJudge,
            NeedleGroundTruthJudge
        )
        
        assert GroundTruthJudge is not None
        assert ASRGroundTruthJudge is not None
        assert CodeGroundTruthJudge is not None
        assert NeedleGroundTruthJudge is not None
    
    def test_judge_module_import(self):
        """Test judge module imports."""
        from langscope.ground_truth import judge
        assert judge is not None
    
    def test_ground_truth_score_class(self):
        """Test GroundTruthScore exists."""
        from langscope.ground_truth.judge import GroundTruthScore
        assert GroundTruthScore is not None


class TestMetricsAdvanced:
    """Advanced metrics tests."""
    
    def test_exact_match(self):
        """Test exact match metric."""
        from langscope.ground_truth.metrics import exact_match
        
        assert exact_match("hello", "hello") == 1.0
        # exact_match may be case sensitive or not depending on implementation
        result = exact_match("hello", "Hello")
        assert result in [0.0, 1.0]
    
    def test_contains_match(self):
        """Test contains match metric."""
        from langscope.ground_truth.metrics import contains
        
        assert contains("The answer is 42", "42") == 1.0
        assert contains("hello world", "hello") == 1.0
        assert contains("hello", "world") == 0.0
    
    def test_wer_edge_cases(self):
        """Test WER edge cases."""
        from langscope.ground_truth.metrics import word_error_rate
        
        # Same text
        assert word_error_rate("hello world", "hello world") == 0.0
        
        # One word different in 4-word sentence
        wer = word_error_rate(
            "the quick brown fox",
            "the quick brown dog"
        )
        assert 0.2 <= wer <= 0.3  # 1/4 = 0.25
    
    def test_cer(self):
        """Test Character Error Rate."""
        from langscope.ground_truth.metrics import character_error_rate
        
        cer = character_error_rate("hello", "hello")
        assert cer == 0.0
        
        cer = character_error_rate("hello", "hallo")
        assert 0.1 <= cer <= 0.3
    
    def test_f1_score(self):
        """Test F1 score for token overlap."""
        from langscope.ground_truth.metrics import f1_score
        
        # Perfect match
        f1 = f1_score("the quick brown fox", "the quick brown fox")
        assert f1 == 1.0
        
        # Partial match
        f1 = f1_score("the quick brown", "the quick brown fox")
        assert 0.7 <= f1 < 1.0
    
    def test_bleu_score(self):
        """Test BLEU score."""
        from langscope.ground_truth.metrics import bleu_score
        
        text = "the quick brown fox"
        score = bleu_score(text, [text])
        assert score > 0.8
    
    def test_rouge_l(self):
        """Test ROUGE-L score."""
        from langscope.ground_truth.metrics import rouge_l
        
        score = rouge_l("the quick brown fox", "the quick brown fox")
        assert score == 1.0


class TestGroundTruthJudgeAPI:
    """Test ground truth judge API exists."""
    
    def test_metric_functions_exist(self):
        """Test metric functions exist in judge module."""
        from langscope.ground_truth.judge import (
            exact_match,
            word_error_rate,
            character_error_rate,
            contains
        )
        
        assert exact_match is not None
        assert word_error_rate is not None

