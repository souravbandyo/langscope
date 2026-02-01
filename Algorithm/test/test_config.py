"""
Unit tests for langscope/config/ modules.

Tests:
- settings.py: Application settings
- params/: Parameter management
- api_keys.py: API key management
"""

import pytest
import os
from unittest.mock import patch


# =============================================================================
# Settings Tests (CFG-001 to CFG-004)
# =============================================================================

class TestSettings:
    """Test application settings."""
    
    def test_settings_module_import(self):
        """Test settings module imports."""
        from langscope.config import settings
        assert settings is not None


# =============================================================================
# Parameter Tests (CFG-010 to CFG-014)
# =============================================================================

class TestParameters:
    """Test parameter management."""
    
    def test_params_module_import(self):
        """Test params module imports."""
        from langscope.config import params
        assert params is not None
    
    def test_trueskill_params_loading(self):
        """CFG-010: Load TrueSkill params."""
        try:
            from langscope.config.params import get_parameter_manager
            
            manager = get_parameter_manager()
            params = manager.get_trueskill_params()
            
            assert hasattr(params, 'mu_0')
            assert hasattr(params, 'sigma_0')
            assert hasattr(params, 'beta')
            assert hasattr(params, 'tau')
        except ImportError:
            pytest.skip("ParameterManager not implemented")
    
    def test_param_models(self):
        """Test parameter models."""
        from langscope.config.params.models import (
            TrueSkillParams,
            TemperatureParams,
            MatchParams,
        )
        
        # Create TrueSkill params
        ts_params = TrueSkillParams()
        assert ts_params.mu_0 == 1500.0
        assert ts_params.sigma_0 == 166.0
        
        # Create Temperature params
        temp_params = TemperatureParams()
        assert temp_params.cost_temp > 0
        assert temp_params.rating_temp > 0
        
        # Create Match params
        match_params = MatchParams()
        assert match_params.players_per_match >= 5


# =============================================================================
# API Keys Tests (CFG-020 to CFG-023)
# =============================================================================

class TestAPIKeys:
    """Test API key management."""
    
    def test_api_keys_module_import(self):
        """Test api_keys module imports."""
        from langscope.config import api_keys
        assert api_keys is not None
    
    def test_get_api_key_from_env(self):
        """Test getting API key from environment."""
        from langscope.config.api_keys import get_api_key, APIKeyManager
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            # Create fresh manager to pick up env var
            manager = APIKeyManager()
            key = manager.get_key("openai")
            assert key == "test-key-123"
    
    def test_get_api_key_not_set(self):
        """Test getting API key when not set."""
        from langscope.config.api_keys import APIKeyManager
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            manager = APIKeyManager()
            key = manager.get_key("nonexistent_provider")
            # Should return None for unknown provider
            assert key is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

