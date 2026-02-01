"""
Shared fixtures and configuration for LangScope tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# =============================================================================
# Rating Fixtures
# =============================================================================

@pytest.fixture
def default_rating():
    """Create a default TrueSkillRating."""
    from langscope.core.rating import TrueSkillRating
    return TrueSkillRating()


@pytest.fixture
def custom_rating():
    """Create a custom TrueSkillRating."""
    from langscope.core.rating import TrueSkillRating
    return TrueSkillRating(mu=1600.0, sigma=150.0)


@pytest.fixture
def dual_rating():
    """Create a DualTrueSkill rating."""
    from langscope.core.rating import DualTrueSkill, TrueSkillRating
    return DualTrueSkill(
        raw=TrueSkillRating(mu=1550.0, sigma=160.0),
        cost_adjusted=TrueSkillRating(mu=1580.0, sigma=155.0)
    )


@pytest.fixture
def multi_dimensional_rating():
    """Create a MultiDimensionalTrueSkill rating."""
    from langscope.core.rating import create_initial_multi_dimensional_rating
    return create_initial_multi_dimensional_rating()


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def sample_model():
    """Create a sample LLMModel."""
    from langscope.core.model import LLMModel
    return LLMModel(
        name="Test Model",
        model_id="test-model-001",
        provider="test-provider",
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
        pricing_source="test",
        max_matches=100
    )


@pytest.fixture
def model_list():
    """Create a list of sample LLMModels for testing."""
    from langscope.core.model import LLMModel
    models = []
    for i in range(6):
        model = LLMModel(
            name=f"Model {i}",
            model_id=f"model-{i:03d}",
            provider=f"provider-{i % 3}",
            input_cost_per_million=0.5 * (i + 1),
            output_cost_per_million=1.0 * (i + 1),
        )
        models.append(model)
    return models


# =============================================================================
# Match Fixtures
# =============================================================================

@pytest.fixture
def sample_match():
    """Create a sample Match."""
    from langscope.evaluation.match import Match, MatchParticipant, MatchResponse
    
    match = Match(
        match_id="match_test_001",
        domain="coding",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    
    # Add competitors
    for i in range(5):
        match.competitors.append(MatchParticipant(
            model_id=f"model_{i}",
            model_name=f"Model {i}",
            role="competitor",
            mu_before=1500.0 + i * 10,
            sigma_before=166.0
        ))
    
    # Add judges
    for i in range(3):
        match.judges.append(MatchParticipant(
            model_id=f"judge_{i}",
            model_name=f"Judge {i}",
            role="judge",
            mu_before=1550.0 + i * 20,
            sigma_before=100.0
        ))
    
    # Add responses
    for i in range(5):
        match.responses.append(MatchResponse(
            model_id=f"model_{i}",
            text=f"Response from model {i}",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.001 * (i + 1),
            latency_ms=100.0 * (i + 1)
        ))
    
    return match


@pytest.fixture
def sample_rankings():
    """Create sample raw and cost rankings."""
    return {
        "raw": {"model_0": 1, "model_1": 2, "model_2": 3, "model_3": 4, "model_4": 5},
        "cost": {"model_0": 2, "model_1": 1, "model_2": 3, "model_3": 5, "model_4": 4}
    }


# =============================================================================
# TrueSkill Fixtures
# =============================================================================

@pytest.fixture
def trueskill_updater():
    """Create a TrueSkill updater."""
    from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater
    return MultiPlayerTrueSkillUpdater()


@pytest.fixture
def player_ratings():
    """Create a list of player ratings for testing."""
    from langscope.ranking.trueskill import TrueSkillRating
    return [TrueSkillRating() for _ in range(6)]


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_collection():
    """Create a mock MongoDB collection."""
    collection = MagicMock()
    collection.find_one.return_value = None
    collection.find.return_value = MagicMock()
    collection.find.return_value.__iter__ = lambda self: iter([])
    collection.find.return_value.sort.return_value = collection.find.return_value
    collection.find.return_value.skip.return_value = collection.find.return_value
    collection.find.return_value.limit.return_value = collection.find.return_value
    collection.insert_one.return_value = MagicMock()
    collection.update_one.return_value = MagicMock()
    collection.delete_one.return_value = MagicMock(deleted_count=1)
    collection.count_documents.return_value = 0
    return collection


@pytest.fixture
def mock_db(mock_collection):
    """Create a mock database instance."""
    db = MagicMock()
    db.connected = True
    db.db_name = "test_langscope"
    db.__getitem__ = MagicMock(return_value=mock_collection)
    
    # Setup default returns
    db.get_match.return_value = None
    db.get_matches_by_domain.return_value = []
    db.get_model.return_value = None
    db.get_all_models.return_value = []
    db.save_model.return_value = True
    db.delete_model.return_value = True
    
    return db


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def api_key_headers():
    """Default API key headers."""
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def mock_api_db():
    """Mock database for API tests."""
    mock_db = MagicMock()
    mock_db.connected = True
    mock_db.db_name = "test_langscope"
    mock_db.get_match.return_value = None
    mock_db.get_matches_by_domain.return_value = []
    mock_db.get_all_base_models.return_value = []
    mock_db.get_base_model.return_value = None
    return mock_db


# =============================================================================
# Domain Fixtures
# =============================================================================

@pytest.fixture
def sample_domains():
    """Sample domain names for testing."""
    return ["coding", "medical", "legal", "creative", "technical"]


@pytest.fixture
def domain_config():
    """Sample domain configuration."""
    return {
        "name": "coding",
        "display_name": "Coding",
        "description": "Programming and software development tasks",
        "parent_domain": None,
        "evaluation_criteria": ["correctness", "efficiency", "readability"]
    }


# =============================================================================
# Transfer Learning Fixtures
# =============================================================================

@pytest.fixture
def domain_similarity_matrix():
    """Sample domain similarity matrix."""
    return {
        ("coding", "technical"): 0.8,
        ("coding", "medical"): 0.3,
        ("medical", "legal"): 0.4,
        ("creative", "coding"): 0.2,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def create_rating_dict(mu=1500.0, sigma=166.0):
    """Create a rating dictionary."""
    return {"mu": mu, "sigma": sigma}


def create_model_dict(name="Test", model_id="test-001", provider="test"):
    """Create a model dictionary."""
    return {
        "name": name,
        "model_id": model_id,
        "provider": provider,
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 2.0,
        "trueskill": {
            "raw": create_rating_dict(),
            "cost_adjusted": create_rating_dict()
        }
    }


