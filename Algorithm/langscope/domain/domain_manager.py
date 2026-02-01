"""
Domain management for LangScope.

Handles CRUD operations for domains, leaderboards, and statistics.
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime

from langscope.domain.domain_config import (
    Domain,
    DomainSettings,
    DomainPrompts,
    DomainStatistics,
    DOMAIN_TEMPLATES,
)

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB
    from langscope.core.model import LLMModel

logger = logging.getLogger(__name__)


class DomainManager:
    """
    Manages evaluation domains.
    
    Handles domain creation, retrieval, configuration updates,
    and leaderboard generation.
    """
    
    def __init__(self, db: 'MongoDB' = None):
        """
        Initialize domain manager.
        
        Args:
            db: Database instance for persistence
        """
        self.db = db
        self._cache: Dict[str, Domain] = {}
    
    def create_domain(
        self,
        name: str,
        display_name: str = None,
        description: str = "",
        parent_domain: str = None,
        template: str = None,
        settings: DomainSettings = None,
        prompts: DomainPrompts = None
    ) -> Domain:
        """
        Create a new domain.
        
        Args:
            name: Unique domain identifier
            display_name: Human-readable name
            description: Domain description
            parent_domain: Parent domain for correlation inheritance
            template: Template to base domain on
            settings: Custom settings
            prompts: Custom prompts
        
        Returns:
            Created Domain
        """
        # Start from template if provided
        if template and template in DOMAIN_TEMPLATES:
            base = DOMAIN_TEMPLATES[template]
            domain = Domain(
                name=name,
                display_name=display_name or base.display_name,
                description=description or base.description,
                parent_domain=parent_domain or base.parent_domain,
                prompts=prompts or base.prompts,
                settings=settings or base.settings,
            )
        else:
            domain = Domain(
                name=name,
                display_name=display_name or name.replace("_", " ").title(),
                description=description,
                parent_domain=parent_domain,
                prompts=prompts or DomainPrompts(),
                settings=settings or DomainSettings(),
            )
        
        # Cache and persist
        self._cache[name] = domain
        
        if self.db:
            self.db.save_domain(domain.to_dict())
        
        # Set up correlation with parent if exists
        if parent_domain:
            self._setup_parent_correlation(name, parent_domain)
        
        logger.info(f"Created domain: {name}")
        return domain
    
    def get_domain(self, name: str) -> Optional[Domain]:
        """
        Get domain by name.
        
        Args:
            name: Domain name
        
        Returns:
            Domain or None
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]
        
        # Check templates
        if name in DOMAIN_TEMPLATES:
            return DOMAIN_TEMPLATES[name]
        
        # Check database
        if self.db:
            data = self.db.get_domain(name)
            if data:
                domain = Domain.from_dict(data)
                self._cache[name] = domain
                return domain
        
        return None
    
    def list_domains(self) -> List[str]:
        """
        List all available domains.
        
        Returns:
            List of domain IDs (used for leaderboard lookups)
        """
        domains = set(DOMAIN_TEMPLATES.keys())
        domains.update(self._cache.keys())
        
        if self.db:
            db_domains = self.db.get_all_domains()
            for d in db_domains:
                # Use _id (domain ID) for consistency with leaderboard lookups
                domains.add(d.get("_id", d.get("name", "")))
        
        return sorted(domains)
    
    def update_domain(
        self,
        name: str,
        settings: DomainSettings = None,
        prompts: DomainPrompts = None,
        description: str = None
    ) -> Optional[Domain]:
        """
        Update domain configuration.
        
        Args:
            name: Domain name
            settings: New settings
            prompts: New prompts
            description: New description
        
        Returns:
            Updated Domain or None
        """
        domain = self.get_domain(name)
        if not domain:
            return None
        
        if settings:
            domain.settings = settings
        if prompts:
            domain.prompts = prompts
        if description is not None:
            domain.description = description
        
        domain.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Update cache and persist
        self._cache[name] = domain
        if self.db:
            self.db.save_domain(domain.to_dict())
        
        return domain
    
    def delete_domain(self, name: str) -> bool:
        """
        Delete a domain.
        
        Args:
            name: Domain name
        
        Returns:
            True if deleted
        """
        # Can't delete templates
        if name in DOMAIN_TEMPLATES:
            logger.warning(f"Cannot delete template domain: {name}")
            return False
        
        # Remove from cache
        if name in self._cache:
            del self._cache[name]
        
        # Remove from database
        if self.db:
            return self.db.delete_domain(name)
        
        return True
    
    def update_domain_statistics(
        self,
        name: str,
        models: List['LLMModel'] = None
    ):
        """
        Update domain statistics from current data.
        
        Args:
            name: Domain name
            models: List of models (if available)
        """
        domain = self.get_domain(name)
        if not domain:
            return
        
        stats = domain.statistics
        
        # Update from database if available
        if self.db:
            match_count = self.db.get_match_count(name)
            stats.total_matches = match_count
            
            domain_models = self.db.get_models_by_domain(name)
            stats.total_models_evaluated = len(domain_models)
            
            # Find top models
            if domain_models:
                # Sort by raw mu
                sorted_raw = sorted(
                    domain_models,
                    key=lambda m: m.get("trueskill_by_domain", {}).get(name, {}).get("raw", {}).get("mu", 0),
                    reverse=True
                )
                if sorted_raw:
                    stats.top_model_raw = sorted_raw[0].get("name", "")
                
                # Sort by cost-adjusted mu
                sorted_cost = sorted(
                    domain_models,
                    key=lambda m: m.get("trueskill_by_domain", {}).get(name, {}).get("cost_adjusted", {}).get("mu", 0),
                    reverse=True
                )
                if sorted_cost:
                    stats.top_model_cost = sorted_cost[0].get("name", "")
        
        # Update timestamp
        stats.last_match_timestamp = datetime.utcnow().isoformat() + "Z"
        domain.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Persist
        self._cache[name] = domain
        if self.db:
            self.db.save_domain(domain.to_dict())
    
    def get_leaderboard(
        self,
        domain: str,
        ranking_type: str = "raw",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard for a domain.
        
        Args:
            domain: Domain name
            ranking_type: "raw" or "cost_adjusted"
            limit: Maximum entries
        
        Returns:
            List of leaderboard entries
        """
        if not self.db:
            return []
        
        return self.db.get_leaderboard(domain, ranking_type, limit)
    
    def _setup_parent_correlation(self, child: str, parent: str):
        """Set up default correlation with parent domain."""
        from langscope.transfer.correlation import set_prior_correlation
        
        # Default high correlation with parent
        set_prior_correlation(child, parent, 0.8)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[DomainManager] = None


def get_default_manager() -> DomainManager:
    """Get or create default domain manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DomainManager()
    return _default_manager


def create_domain(
    name: str,
    display_name: str = None,
    description: str = "",
    parent_domain: str = None,
    template: str = None
) -> Domain:
    """Create domain using default manager."""
    return get_default_manager().create_domain(
        name, display_name, description, parent_domain, template
    )


def get_domain(name: str) -> Optional[Domain]:
    """Get domain using default manager."""
    return get_default_manager().get_domain(name)


def list_domains() -> List[str]:
    """List domains using default manager."""
    return get_default_manager().list_domains()


