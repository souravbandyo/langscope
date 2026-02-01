"""
Comprehensive Database Tests for LangScope MongoDB Operations.

Tests all database operations using mocked MongoDB collections.
Uses pytest with mongomock for isolated testing.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from typing import Dict, Any, List


# =============================================================================
# Fixtures
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
    collection.create_index.return_value = None
    return collection


@pytest.fixture
def mock_db(mock_collection):
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=mock_collection)
    db.list_collection_names.return_value = []
    db.create_collection.return_value = None
    return db


@pytest.fixture
def mock_client(mock_db):
    """Create a mock MongoDB client."""
    client = MagicMock()
    client.admin.command.return_value = {"ok": 1}
    client.__getitem__ = MagicMock(return_value=mock_db)
    return client


@pytest.fixture
def mongodb_instance(mock_client, mock_db):
    """Create a MongoDB instance with mocked connection."""
    with patch('langscope.database.mongodb.MongoClient', return_value=mock_client):
        with patch('langscope.database.mongodb.PYMONGO_AVAILABLE', True):
            from langscope.database.mongodb import MongoDB
            db = MongoDB("mongodb://localhost:27017", db_name="test_db", auto_connect=False)
            db.client = mock_client
            db.db = mock_db
            db.connected = True
            return db


# =============================================================================
# Connection Tests
# =============================================================================

class TestConnection:
    """Test database connection operations."""
    
    def test_connect_success(self, mock_client):
        """Test successful connection to MongoDB."""
        with patch('langscope.database.mongodb.MongoClient', return_value=mock_client):
            with patch('langscope.database.mongodb.PYMONGO_AVAILABLE', True):
                from langscope.database.mongodb import MongoDB
                db = MongoDB("mongodb://localhost:27017", auto_connect=True)
                assert db.connected is True
    
    def test_connect_failure(self):
        """Test connection failure handling."""
        with patch('langscope.database.mongodb.PYMONGO_AVAILABLE', True):
            from langscope.database.mongodb import MongoDB
            from pymongo.errors import ConnectionFailure
            
            with patch('langscope.database.mongodb.MongoClient') as mock_client_class:
                mock_client_class.side_effect = ConnectionFailure("Connection refused")
                db = MongoDB("mongodb://localhost:27017", auto_connect=True)
                assert db.connected is False
    
    def test_disconnect(self, mongodb_instance):
        """Test disconnecting from MongoDB."""
        mongodb_instance.disconnect()
        assert mongodb_instance.connected is False


# =============================================================================
# Model Operations Tests
# =============================================================================

class TestModelOperations:
    """Test model CRUD operations."""
    
    def test_save_model(self, mongodb_instance, mock_collection):
        """Test saving a model."""
        model_data = {
            "name": "test-model",
            "model_id": "test-model-id",
            "provider": "test-provider",
            "input_cost_per_million": 1.0,
            "output_cost_per_million": 2.0,
        }
        
        result = mongodb_instance.save_model(model_data)
        assert result is True
        mock_collection.update_one.assert_called_once()
    
    def test_save_model_not_connected(self, mongodb_instance):
        """Test saving model when not connected."""
        mongodb_instance.connected = False
        result = mongodb_instance.save_model({"name": "test"})
        assert result is False
    
    def test_get_model(self, mongodb_instance, mock_collection):
        """Test getting a model by name."""
        mock_collection.find_one.return_value = {
            "name": "test-model",
            "model_id": "test-id",
            "provider": "test-provider"
        }
        
        result = mongodb_instance.get_model("test-model")
        assert result is not None
        assert result["name"] == "test-model"
    
    def test_get_model_not_found(self, mongodb_instance, mock_collection):
        """Test getting a model that doesn't exist."""
        mock_collection.find_one.return_value = None
        result = mongodb_instance.get_model("nonexistent")
        assert result is None
    
    def test_get_model_by_id(self, mongodb_instance, mock_collection):
        """Test getting a model by model_id."""
        mock_collection.find_one.return_value = {"model_id": "test-id"}
        result = mongodb_instance.get_model_by_id("test-id")
        assert result is not None
    
    def test_get_all_models(self, mongodb_instance, mock_collection):
        """Test getting all models."""
        mock_collection.find.return_value = [
            {"name": "model1"},
            {"name": "model2"}
        ]
        
        result = mongodb_instance.get_all_models()
        assert len(result) == 2
    
    def test_get_models_by_provider(self, mongodb_instance, mock_collection):
        """Test getting models by provider."""
        mock_collection.find.return_value = [{"provider": "openai"}]
        result = mongodb_instance.get_models_by_provider("openai")
        assert len(result) == 1
    
    def test_get_models_by_domain(self, mongodb_instance, mock_collection):
        """Test getting models by domain."""
        mock_collection.find.return_value = [{"name": "model1"}]
        result = mongodb_instance.get_models_by_domain("coding")
        mock_collection.find.assert_called()
    
    def test_delete_model(self, mongodb_instance, mock_collection):
        """Test deleting a model."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_model("test-model")
        assert result is True
    
    def test_delete_model_not_found(self, mongodb_instance, mock_collection):
        """Test deleting a model that doesn't exist."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=0)
        result = mongodb_instance.delete_model("nonexistent")
        assert result is False


# =============================================================================
# Match Operations Tests
# =============================================================================

class TestMatchOperations:
    """Test match CRUD operations."""
    
    def test_save_match(self, mongodb_instance, mock_collection):
        """Test saving a match."""
        match_data = {
            "_id": "match_123",
            "domain": "coding",
            "participants": ["model1", "model2", "model3", "model4", "model5"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = mongodb_instance.save_match(match_data)
        assert result is True
        mock_collection.insert_one.assert_called_once()
    
    def test_get_match(self, mongodb_instance, mock_collection):
        """Test getting a match by ID."""
        mock_collection.find_one.return_value = {"_id": "match_123"}
        result = mongodb_instance.get_match("match_123")
        assert result is not None
    
    def test_get_matches_by_domain(self, mongodb_instance, mock_collection):
        """Test getting matches by domain."""
        mock_find = MagicMock()
        mock_find.sort.return_value.skip.return_value.limit.return_value = [
            {"domain": "coding", "_id": "match_1"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_matches_by_domain("coding", limit=10)
        assert len(result) == 1
    
    def test_get_matches_by_model(self, mongodb_instance, mock_collection):
        """Test getting matches involving a specific model."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"participants": ["model1", "model2"]}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_matches_by_model("model1")
        assert len(result) == 1
    
    def test_get_match_count(self, mongodb_instance, mock_collection):
        """Test getting match count."""
        mock_collection.count_documents.return_value = 42
        result = mongodb_instance.get_match_count()
        assert result == 42
    
    def test_get_match_count_by_domain(self, mongodb_instance, mock_collection):
        """Test getting match count for a specific domain."""
        mock_collection.count_documents.return_value = 10
        result = mongodb_instance.get_match_count(domain="coding")
        assert result == 10


# =============================================================================
# Domain Operations Tests
# =============================================================================

class TestDomainOperations:
    """Test domain CRUD operations."""
    
    def test_save_domain(self, mongodb_instance, mock_collection):
        """Test saving a domain."""
        domain_data = {
            "_id": "coding",
            "name": "Coding",
            "description": "Programming tasks"
        }
        
        result = mongodb_instance.save_domain(domain_data)
        assert result is True
    
    def test_get_domain(self, mongodb_instance, mock_collection):
        """Test getting a domain by name."""
        mock_collection.find_one.return_value = {"_id": "coding", "name": "Coding"}
        result = mongodb_instance.get_domain("coding")
        assert result is not None
        assert result["_id"] == "coding"
    
    def test_get_all_domains(self, mongodb_instance, mock_collection):
        """Test getting all domains."""
        mock_collection.find.return_value = [
            {"_id": "coding"},
            {"_id": "medical"}
        ]
        result = mongodb_instance.get_all_domains()
        assert len(result) == 2
    
    def test_delete_domain(self, mongodb_instance, mock_collection):
        """Test deleting a domain."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_domain("coding")
        assert result is True


# =============================================================================
# Correlation Operations Tests
# =============================================================================

class TestCorrelationOperations:
    """Test domain correlation operations."""
    
    def test_save_correlation(self, mongodb_instance, mock_collection):
        """Test saving a correlation."""
        correlation_data = {
            "_id": "coding|medical",
            "domain_a": "coding",
            "domain_b": "medical",
            "correlation": 0.75
        }
        
        result = mongodb_instance.save_correlation(correlation_data)
        assert result is True
    
    def test_get_correlation(self, mongodb_instance, mock_collection):
        """Test getting a correlation between two domains."""
        mock_collection.find_one.return_value = {
            "_id": "coding|medical",
            "correlation": 0.75
        }
        
        result = mongodb_instance.get_correlation("coding", "medical")
        assert result is not None
        assert result["correlation"] == 0.75
    
    def test_get_correlation_reversed_order(self, mongodb_instance, mock_collection):
        """Test getting correlation with reversed domain order."""
        # First call returns None, second returns the correlation
        mock_collection.find_one.side_effect = [
            None,
            {"_id": "medical|coding", "correlation": 0.75}
        ]
        
        result = mongodb_instance.get_correlation("coding", "medical")
        assert result is not None
    
    def test_get_correlations_for_domain(self, mongodb_instance, mock_collection):
        """Test getting all correlations for a domain."""
        mock_collection.find.return_value = [
            {"domain_a": "coding", "domain_b": "medical"},
            {"domain_a": "legal", "domain_b": "coding"}
        ]
        
        result = mongodb_instance.get_correlations_for_domain("coding")
        assert len(result) == 2


# =============================================================================
# Leaderboard Operations Tests
# =============================================================================

class TestLeaderboardOperations:
    """Test leaderboard operations."""
    
    def test_get_leaderboard_global(self, mongodb_instance, mock_collection):
        """Test getting global leaderboard."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"name": "model1", "trueskill": {"raw": {"mu": 1600}}},
            {"name": "model2", "trueskill": {"raw": {"mu": 1550}}}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_leaderboard(ranking_type="raw", limit=10)
        assert len(result) == 2
    
    def test_get_leaderboard_by_domain(self, mongodb_instance, mock_collection):
        """Test getting domain-specific leaderboard."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"name": "model1"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_leaderboard(domain="coding", ranking_type="raw")
        assert len(result) == 1


# =============================================================================
# User Session Operations Tests
# =============================================================================

class TestUserSessionOperations:
    """Test user session operations."""
    
    def test_save_user_session(self, mongodb_instance, mock_collection):
        """Test saving a user session."""
        session_data = {
            "_id": "session_123",
            "session_id": "session_123",
            "domain": "coding",
            "use_case": "testing",
            "n_battles": 5
        }
        
        result = mongodb_instance.save_user_session(session_data)
        assert result is True
    
    def test_get_user_session(self, mongodb_instance, mock_collection):
        """Test getting a user session."""
        mock_collection.find_one.return_value = {
            "session_id": "session_123",
            "domain": "coding"
        }
        
        result = mongodb_instance.get_user_session("session_123")
        assert result is not None
    
    def test_get_sessions_by_use_case(self, mongodb_instance, mock_collection):
        """Test getting sessions by use case."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"use_case": "testing"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_sessions_by_use_case("testing")
        assert len(result) == 1
    
    def test_get_sessions_by_user(self, mongodb_instance, mock_collection):
        """Test getting sessions by user ID."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"user_id": "user123"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_sessions_by_user("user123")
        assert len(result) == 1
    
    def test_get_session_count(self, mongodb_instance, mock_collection):
        """Test getting session count."""
        mock_collection.count_documents.return_value = 25
        result = mongodb_instance.get_session_count()
        assert result == 25
    
    def test_get_aggregate_deltas_by_use_case(self, mongodb_instance, mock_collection):
        """Test getting aggregate deltas by use case."""
        mock_collection.find.return_value = [
            {
                "use_case": "testing",
                "deltas": {
                    "model1": {"delta": 0.5},
                    "model2": {"delta": -0.3}
                }
            }
        ]
        
        result = mongodb_instance.get_aggregate_deltas_by_use_case("testing")
        assert "model1" in result
        assert "model2" in result


# =============================================================================
# Judge Calibration Operations Tests
# =============================================================================

class TestJudgeCalibrationOperations:
    """Test judge calibration operations."""
    
    def test_save_judge_calibration(self, mongodb_instance, mock_collection):
        """Test saving judge calibration."""
        calibration_data = {
            "_id": "judge123",
            "judge_id": "judge123",
            "agreed": 80,
            "total": 100,
            "calibration": 0.8
        }
        
        result = mongodb_instance.save_judge_calibration(calibration_data)
        assert result is True
    
    def test_get_judge_calibration(self, mongodb_instance, mock_collection):
        """Test getting judge calibration."""
        mock_collection.find_one.return_value = {
            "judge_id": "judge123",
            "calibration": 0.8
        }
        
        result = mongodb_instance.get_judge_calibration("judge123")
        assert result is not None
        assert result["calibration"] == 0.8
    
    def test_get_all_judge_calibrations(self, mongodb_instance, mock_collection):
        """Test getting all judge calibrations."""
        mock_collection.find.return_value = [
            {"judge_id": "judge1"},
            {"judge_id": "judge2"}
        ]
        
        result = mongodb_instance.get_all_judge_calibrations()
        assert len(result) == 2


# =============================================================================
# Parameter Operations Tests
# =============================================================================

class TestParameterOperations:
    """Test parameter management operations."""
    
    def test_save_params(self, mongodb_instance, mock_collection):
        """Test saving parameters."""
        params = {
            "mu_0": 1500,
            "sigma_0": 166,
            "beta": 200
        }
        
        result = mongodb_instance.save_params("trueskill", params)
        assert result is True
    
    def test_save_params_with_domain(self, mongodb_instance, mock_collection):
        """Test saving domain-specific parameters."""
        params = {"judge_count": 7}
        result = mongodb_instance.save_params("match", params, domain="medical")
        assert result is True
    
    def test_get_params(self, mongodb_instance, mock_collection):
        """Test getting parameters."""
        mock_collection.find_one.return_value = {
            "param_type": "trueskill",
            "params": {"mu_0": 1500}
        }
        
        result = mongodb_instance.get_params("trueskill")
        assert result is not None
        assert result["mu_0"] == 1500
    
    def test_get_params_not_found(self, mongodb_instance, mock_collection):
        """Test getting parameters that don't exist."""
        mock_collection.find_one.return_value = None
        result = mongodb_instance.get_params("nonexistent")
        assert result is None
    
    def test_list_param_overrides(self, mongodb_instance, mock_collection):
        """Test listing parameter overrides."""
        mock_collection.find.return_value = [
            {"domain": "coding"},
            {"domain": "medical"}
        ]
        
        result = mongodb_instance.list_param_overrides()
        assert len(result) == 2
    
    def test_delete_param_override(self, mongodb_instance, mock_collection):
        """Test deleting a parameter override."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_param_override("trueskill", "coding")
        assert result is True
    
    def test_get_all_params(self, mongodb_instance, mock_collection):
        """Test getting all parameters."""
        mock_collection.find.return_value = [
            {"_id": "trueskill", "params": {}},
            {"_id": "match", "params": {}}
        ]
        
        result = mongodb_instance.get_all_params()
        assert len(result) == 2


# =============================================================================
# Base Model Operations Tests
# =============================================================================

class TestBaseModelOperations:
    """Test base model CRUD operations."""
    
    def test_save_base_model(self, mongodb_instance, mock_collection):
        """Test saving a base model."""
        base_model_data = {
            "_id": "meta-llama/llama-3.1-70b",
            "name": "Llama 3.1 70B",
            "family": "llama",
            "organization": "Meta"
        }
        
        result = mongodb_instance.save_base_model(base_model_data)
        assert result is True
    
    def test_get_base_model(self, mongodb_instance, mock_collection):
        """Test getting a base model by ID."""
        mock_collection.find_one.return_value = {
            "_id": "meta-llama/llama-3.1-70b",
            "name": "Llama 3.1 70B"
        }
        
        result = mongodb_instance.get_base_model("meta-llama/llama-3.1-70b")
        assert result is not None
        assert result["name"] == "Llama 3.1 70B"
    
    def test_get_all_base_models(self, mongodb_instance, mock_collection):
        """Test getting all base models."""
        mock_find = MagicMock()
        mock_find.limit.return_value = [
            {"_id": "model1"},
            {"_id": "model2"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_all_base_models()
        assert len(result) == 2
    
    def test_get_base_models_filtered(self, mongodb_instance, mock_collection):
        """Test getting base models with filters."""
        mock_find = MagicMock()
        mock_find.limit.return_value = [{"family": "llama"}]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_all_base_models(family="llama")
        assert len(result) == 1
    
    def test_delete_base_model(self, mongodb_instance, mock_collection):
        """Test deleting a base model."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_base_model("model-id")
        assert result is True


# =============================================================================
# Deployment Operations Tests
# =============================================================================

class TestDeploymentOperations:
    """Test model deployment CRUD operations."""
    
    def test_save_deployment(self, mongodb_instance, mock_collection):
        """Test saving a deployment."""
        deployment_data = {
            "_id": "groq/llama-3.1-70b",
            "base_model_id": "meta-llama/llama-3.1-70b",
            "provider": {"id": "groq", "name": "Groq"}
        }
        
        result = mongodb_instance.save_deployment(deployment_data)
        assert result is True
    
    def test_get_deployment(self, mongodb_instance, mock_collection):
        """Test getting a deployment by ID."""
        mock_collection.find_one.return_value = {
            "_id": "groq/llama-3.1-70b",
            "base_model_id": "meta-llama/llama-3.1-70b"
        }
        
        result = mongodb_instance.get_deployment("groq/llama-3.1-70b")
        assert result is not None
    
    def test_get_deployments_by_base_model(self, mongodb_instance, mock_collection):
        """Test getting deployments for a base model."""
        mock_collection.find.return_value = [
            {"_id": "groq/llama-3.1-70b"},
            {"_id": "together/llama-3.1-70b"}
        ]
        
        result = mongodb_instance.get_deployments_by_base_model("meta-llama/llama-3.1-70b")
        assert len(result) == 2
    
    def test_get_deployments_by_provider(self, mongodb_instance, mock_collection):
        """Test getting deployments by provider."""
        mock_find = MagicMock()
        mock_find.limit.return_value = [{"provider": {"id": "groq"}}]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_deployments_by_provider("groq")
        assert len(result) == 1
    
    def test_get_all_deployments(self, mongodb_instance, mock_collection):
        """Test getting all deployments."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"_id": "dep1"},
            {"_id": "dep2"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_all_deployments()
        assert len(result) == 2
    
    def test_get_best_deployment(self, mongodb_instance, mock_collection):
        """Test getting the best deployment for a base model."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"_id": "best-deployment"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_best_deployment("base-model-id")
        assert result is not None
        assert result["_id"] == "best-deployment"
    
    def test_delete_deployment(self, mongodb_instance, mock_collection):
        """Test deleting a deployment."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_deployment("deployment-id")
        assert result is True


# =============================================================================
# Self-Hosted Deployment Operations Tests
# =============================================================================

class TestSelfHostedOperations:
    """Test self-hosted deployment CRUD operations."""
    
    def test_save_self_hosted_deployment(self, mongodb_instance, mock_collection):
        """Test saving a self-hosted deployment."""
        deployment_data = {
            "_id": "user123/my-llama",
            "base_model_id": "meta-llama/llama-3.1-70b",
            "owner": {"user_id": "user123", "is_public": True}
        }
        
        result = mongodb_instance.save_self_hosted_deployment(deployment_data)
        assert result is True
    
    def test_get_self_hosted_deployment(self, mongodb_instance, mock_collection):
        """Test getting a self-hosted deployment."""
        mock_collection.find_one.return_value = {
            "_id": "user123/my-llama",
            "owner": {"user_id": "user123"}
        }
        
        result = mongodb_instance.get_self_hosted_deployment("user123/my-llama")
        assert result is not None
    
    def test_get_self_hosted_by_user(self, mongodb_instance, mock_collection):
        """Test getting self-hosted deployments by user."""
        mock_find = MagicMock()
        mock_find.limit.return_value = [
            {"_id": "user123/dep1"},
            {"_id": "user123/dep2"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_self_hosted_by_user("user123")
        assert len(result) == 2
    
    def test_get_public_self_hosted(self, mongodb_instance, mock_collection):
        """Test getting public self-hosted deployments."""
        mock_find = MagicMock()
        mock_find.limit.return_value = [{"owner": {"is_public": True}}]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_public_self_hosted()
        assert len(result) == 1
    
    def test_delete_self_hosted_deployment(self, mongodb_instance, mock_collection):
        """Test deleting a self-hosted deployment."""
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        result = mongodb_instance.delete_self_hosted_deployment("user123/my-llama")
        assert result is True


# =============================================================================
# Time Series Operations Tests
# =============================================================================

class TestTimeSeriesOperations:
    """Test time series operations for history tracking."""
    
    def test_save_rating_history(self, mongodb_instance, mock_collection):
        """Test saving rating history."""
        result = mongodb_instance.save_rating_history(
            deployment_id="deployment-123",
            domain="coding",
            trueskill_raw={"mu": 1550, "sigma": 150},
            trueskill_cost={"mu": 1520, "sigma": 155},
            trigger="match",
            trigger_id="match_123"
        )
        assert result is True
    
    def test_get_rating_history(self, mongodb_instance, mock_collection):
        """Test getting rating history."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"trueskill_raw": {"mu": 1550}}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_rating_history("deployment-123")
        assert len(result) == 1
    
    def test_get_rating_at_time(self, mongodb_instance, mock_collection):
        """Test getting rating at a specific time."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"trueskill_raw": {"mu": 1500}}
        ]
        mock_collection.find.return_value = mock_find
        
        timestamp = datetime.utcnow()
        result = mongodb_instance.get_rating_at_time("deployment-123", "coding", timestamp)
        assert result is not None
    
    def test_save_performance_metrics(self, mongodb_instance, mock_collection):
        """Test saving performance metrics."""
        result = mongodb_instance.save_performance_metrics(
            deployment_id="deployment-123",
            domain="coding",
            match_id="match_123",
            latency_ms=150.5,
            ttft_ms=50.2,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            raw_rank=2,
            cost_rank=3
        )
        assert result is True
    
    def test_get_performance_metrics(self, mongodb_instance, mock_collection):
        """Test getting performance metrics."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"latency_ms": 150.5}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_performance_metrics("deployment-123")
        assert len(result) == 1
    
    def test_save_price_history(self, mongodb_instance, mock_collection):
        """Test saving price history."""
        result = mongodb_instance.save_price_history(
            deployment_id="deployment-123",
            provider="groq",
            input_cost_per_million=0.5,
            output_cost_per_million=1.5,
            previous_input_cost=0.6,
            previous_output_cost=1.8
        )
        assert result is True
    
    def test_get_price_history(self, mongodb_instance, mock_collection):
        """Test getting price history."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"input_cost_per_million": 0.5}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_price_history("deployment-123")
        assert len(result) == 1


# =============================================================================
# Sync History Operations Tests
# =============================================================================

class TestSyncHistoryOperations:
    """Test sync history operations."""
    
    def test_save_sync_result(self, mongodb_instance, mock_collection):
        """Test saving sync result."""
        result_data = {
            "source_id": "openai-pricing",
            "provider": "openai",
            "status": "success",
            "duration_ms": 500,
            "models_found": 10,
            "models_updated": 5
        }
        
        result = mongodb_instance.save_sync_result(result_data)
        assert result is True
    
    def test_get_sync_history(self, mongodb_instance, mock_collection):
        """Test getting sync history."""
        mock_find = MagicMock()
        mock_find.sort.return_value = [{"status": "success"}]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_sync_history("openai-pricing", days=30)
        assert len(result) == 1


# =============================================================================
# Content Hashes Operations Tests
# =============================================================================

class TestContentHashOperations:
    """Test content hash operations for deduplication."""
    
    def test_check_content_duplicate_not_found(self, mongodb_instance, mock_collection):
        """Test checking for non-existent content hash."""
        mock_collection.count_documents.return_value = 0
        result = mongodb_instance.check_content_duplicate("hash123", "case", "coding")
        assert result is False
    
    def test_check_content_duplicate_found(self, mongodb_instance, mock_collection):
        """Test checking for existing content hash."""
        mock_collection.count_documents.return_value = 1
        result = mongodb_instance.check_content_duplicate("hash123", "case", "coding")
        assert result is True
    
    def test_register_content(self, mongodb_instance, mock_collection):
        """Test registering a content hash."""
        result = mongodb_instance.register_content(
            content_hash="hash123",
            content_type="case",
            domain="coding",
            match_id="match_123",
            generator_model_id="model1",
            content_preview="This is a test case..."
        )
        assert result is True
    
    def test_get_content_hash_info(self, mongodb_instance, mock_collection):
        """Test getting content hash info."""
        mock_collection.find_one.return_value = {
            "_id": "hash123",
            "content_type": "case",
            "usage_count": 5
        }
        
        result = mongodb_instance.get_content_hash_info("hash123")
        assert result is not None
        assert result["usage_count"] == 5


# =============================================================================
# Deployment Leaderboard Operations Tests
# =============================================================================

class TestDeploymentLeaderboardOperations:
    """Test deployment leaderboard operations."""
    
    def test_get_deployment_leaderboard(self, mongodb_instance, mock_collection):
        """Test getting deployment leaderboard."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [
            {"_id": "dep1", "trueskill": {"raw": {"mu": 1600}}},
            {"_id": "dep2", "trueskill": {"raw": {"mu": 1550}}}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_deployment_leaderboard()
        assert len(result) == 4  # Cloud + self-hosted (2 + 2)
    
    def test_get_deployment_leaderboard_by_domain(self, mongodb_instance, mock_collection):
        """Test getting deployment leaderboard for a domain."""
        mock_find = MagicMock()
        mock_find.sort.return_value.limit.return_value = [{"_id": "dep1"}]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_deployment_leaderboard(domain="coding")
        assert len(result) >= 1


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_operations_when_not_connected(self, mongodb_instance):
        """Test operations return appropriate values when not connected."""
        mongodb_instance.connected = False
        mongodb_instance.db = None
        
        assert mongodb_instance.get_model("test") is None
        assert mongodb_instance.get_all_models() == []
        assert mongodb_instance.save_model({"name": "test"}) is False
        assert mongodb_instance.delete_model("test") is False
        assert mongodb_instance.get_match_count() == 0
        assert mongodb_instance.get_leaderboard() == []
    
    def test_save_model_with_exception(self, mongodb_instance, mock_collection):
        """Test save model handles exceptions gracefully."""
        mock_collection.update_one.side_effect = Exception("DB error")
        result = mongodb_instance.save_model({"name": "test"})
        assert result is False
    
    def test_delete_model_with_exception(self, mongodb_instance, mock_collection):
        """Test delete model handles exceptions gracefully."""
        mock_collection.delete_one.side_effect = Exception("DB error")
        result = mongodb_instance.delete_model("test")
        assert result is False


# =============================================================================
# Ground Truth Operations Tests
# =============================================================================

class TestGroundTruthOperations:
    """Test ground truth database operations."""
    
    def test_get_ground_truth_sample(self, mongodb_instance, mock_collection):
        """Test getting a ground truth sample."""
        mock_collection.find_one.return_value = {
            "sample_id": "gt_sample_001",
            "domain": "asr"
        }
        result = mongodb_instance.get_ground_truth_sample("gt_sample_001")
        assert result is not None
        assert result["domain"] == "asr"
    
    def test_get_ground_truth_sample_not_found(self, mongodb_instance, mock_collection):
        """Test getting non-existent ground truth sample."""
        mock_collection.find_one.return_value = None
        result = mongodb_instance.get_ground_truth_sample("non_existent")
        assert result is None
    
    def test_get_ground_truth_sample_count(self, mongodb_instance, mock_collection):
        """Test counting ground truth samples."""
        mock_collection.count_documents.return_value = 50
        result = mongodb_instance.get_ground_truth_sample_count(domain="asr")
        assert result == 50
    
    def test_get_ground_truth_rating(self, mongodb_instance, mock_collection):
        """Test getting a ground truth rating."""
        mock_collection.find_one.return_value = {
            "deployment_id": "dep_001",
            "domain": "asr",
            "trueskill": {"mu": 1500}
        }
        result = mongodb_instance.get_ground_truth_rating("dep_001", "asr")
        assert result is not None
        assert result["trueskill"]["mu"] == 1500
    
    def test_get_ground_truth_rating_not_found(self, mongodb_instance, mock_collection):
        """Test getting non-existent ground truth rating."""
        mock_collection.find_one.return_value = None
        result = mongodb_instance.get_ground_truth_rating("non_existent", "asr")
        assert result is None
    
    def test_get_ground_truth_match(self, mongodb_instance, mock_collection):
        """Test getting a ground truth match."""
        mock_collection.find_one.return_value = {
            "match_id": "gt_match_001",
            "domain": "asr"
        }
        result = mongodb_instance.get_ground_truth_match("gt_match_001")
        assert result is not None


class TestBenchmarkOperations:
    """Test benchmark-related database operations."""
    
    def test_get_benchmark_definition(self, mongodb_instance, mock_collection):
        """Test getting a benchmark definition."""
        mock_collection.find_one.return_value = {
            "name": "test_benchmark",
            "metrics": ["accuracy"]
        }
        result = mongodb_instance.get_benchmark_definition("test_benchmark")
        assert result is not None
        assert result["name"] == "test_benchmark"
    
    def test_get_all_benchmark_definitions(self, mongodb_instance, mock_collection):
        """Test getting all benchmark definitions."""
        mock_find = MagicMock()
        mock_find.__iter__ = lambda self: iter([
            {"name": "bench1"},
            {"name": "bench2"}
        ])
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_all_benchmark_definitions()
        assert len(result) == 2


class TestIndexCreation:
    """Test database index creation."""
    
    def test_create_indexes_runs_without_error(self, mongodb_instance, mock_collection, mock_db):
        """Test that index creation doesn't throw errors."""
        # Mock the database attribute
        mongodb_instance.db = mock_db
        
        # Should not raise
        mongodb_instance._create_indexes()
        
        # Verify create_index was called multiple times
        assert mock_collection.create_index.called
    
    def test_create_indexes_when_db_none(self, mongodb_instance):
        """Test that index creation handles None db gracefully."""
        mongodb_instance.db = None
        # Should not raise
        mongodb_instance._create_indexes()


class TestTimeSeriesCollections:
    """Test time series collection operations."""
    
    def test_ensure_time_series_collections(self, mongodb_instance, mock_db):
        """Test time series collections are created."""
        mock_db.list_collection_names.return_value = []
        mongodb_instance.db = mock_db
        
        mongodb_instance._ensure_time_series_collections()
        
        # Should attempt to create collections
        # (may be called or may silently pass if collections exist)
        assert mock_db.list_collection_names.called


class TestQueryFilters:
    """Test various query filter scenarios."""
    
    def test_get_matches_with_date_range(self, mongodb_instance, mock_collection):
        """Test getting matches with date range filter."""
        mock_find = MagicMock()
        mock_find.sort.return_value.skip.return_value.limit.return_value = [
            {"match_id": "m1", "timestamp": "2024-01-15T00:00:00Z"}
        ]
        mock_collection.find.return_value = mock_find
        
        result = mongodb_instance.get_matches_by_domain(
            domain="coding",
            skip=0,
            limit=10
        )
        assert len(result) == 1


class TestSyncOperations:
    """Test sync operations - verifies methods exist and are callable."""
    
    def test_sync_methods_exist(self, mongodb_instance):
        """Test that sync-related methods exist."""
        assert hasattr(mongodb_instance, 'get_sync_history')
        assert hasattr(mongodb_instance, 'save_sync_result')
        assert callable(mongodb_instance.get_sync_history)
        assert callable(mongodb_instance.save_sync_result)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

