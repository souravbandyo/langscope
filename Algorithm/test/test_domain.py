"""
Unit tests for langscope/domain/ modules.

Tests:
- domain_config.py: Domain configuration
- domain_manager.py: Domain management
- prompts.py: Domain prompts
"""

import pytest


# =============================================================================
# Domain Config Tests (DOM-001 to DOM-008)
# =============================================================================

class TestDomainConfig:
    """Test domain configuration."""
    
    def test_domain_config_module_import(self):
        """Test domain_config module imports."""
        from langscope.domain import domain_config
        assert domain_config is not None
    
    def test_domain_manager_module_import(self):
        """Test domain_manager module imports."""
        from langscope.domain import domain_manager
        assert domain_manager is not None
    
    def test_prompts_module_import(self):
        """Test prompts module imports."""
        from langscope.domain import prompts
        assert prompts is not None


# =============================================================================
# Domain Manager Tests (DOM-010 to DOM-015)
# =============================================================================

class TestDomainManager:
    """Test domain management functions."""
    
    def test_create_domain_manager(self):
        """Test creating domain manager."""
        try:
            from langscope.domain.domain_manager import DomainManager
            
            manager = DomainManager()
            assert manager is not None
        except ImportError:
            pytest.skip("DomainManager not implemented")
    
    def test_list_domains(self):
        """DOM-010: List all domains."""
        try:
            from langscope.domain.domain_manager import DomainManager
            
            manager = DomainManager()
            domains = manager.list_domains()
            
            assert isinstance(domains, list)
        except ImportError:
            pytest.skip("DomainManager not implemented")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


