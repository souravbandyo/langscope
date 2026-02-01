"""
Parameter Manager for LangScope.

Provides centralized parameter management with:
- Database-backed storage
- In-memory caching with TTL
- Domain-specific parameter overrides
- Runtime parameter updates
"""

import logging
from typing import Dict, Any, Optional, Type, Union
from pydantic import BaseModel

from langscope.config.params.models import (
    ParamType,
    TrueSkillParams,
    StrataParams,
    MatchParams,
    TemperatureParams,
    DimensionWeightParams,
    TransferParams,
    FeedbackParams,
    PenaltyParams,
    ConsistencyParams,
    LongContextParams,
    SystemParams,
    PARAM_TYPE_TO_CLASS,
    get_default_params,
)
from langscope.config.params.cache import ParamCache

logger = logging.getLogger(__name__)


# Domain-specific parameter types (allow per-domain overrides)
DOMAIN_SPECIFIC_PARAMS = {
    ParamType.STRATA,
    ParamType.MATCH,
    ParamType.DIMENSION_WEIGHTS,
    ParamType.PENALTY,
}


class ParameterManager:
    """
    Centralized parameter management with database backing and caching.
    
    Features:
    - Runtime parameter updates via set_params()
    - Per-domain parameter overrides for supported types
    - In-memory caching with configurable TTL
    - Fallback to global defaults when no domain override exists
    """
    
    def __init__(
        self,
        db: Optional[Any] = None,
        cache_ttl: int = 300,
        auto_cache: bool = True
    ):
        """
        Initialize ParameterManager.
        
        Args:
            db: MongoDB instance (optional, uses defaults if None)
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
            auto_cache: Whether to automatically cache retrieved params
        """
        self._db = db
        self._cache = ParamCache(default_ttl=cache_ttl)
        self._auto_cache = auto_cache
        
        # Default params (fallback when DB unavailable)
        self._defaults = SystemParams()
    
    def set_database(self, db: Any) -> None:
        """Set the database instance."""
        self._db = db
        # Invalidate cache when DB changes
        self._cache.invalidate_all()
    
    # =========================================================================
    # Generic Parameter Access
    # =========================================================================
    
    def get_params(
        self,
        param_type: Union[ParamType, str],
        domain: Optional[str] = None
    ) -> BaseModel:
        """
        Get parameters for a given type and optional domain.
        
        Resolution order:
        1. Cache (if available and not expired)
        2. Database (domain-specific if domain provided)
        3. Database (global if no domain override)
        4. Defaults
        
        Args:
            param_type: Parameter type
            domain: Optional domain for domain-specific params
        
        Returns:
            Parameter model instance
        """
        if isinstance(param_type, str):
            param_type = ParamType(param_type)
        
        # Check if domain overrides are supported for this type
        if domain and param_type not in DOMAIN_SPECIFIC_PARAMS:
            domain = None  # Ignore domain for global-only params
        
        # Check cache first
        cached = self._cache.get(param_type.value, domain)
        if cached is not None:
            return cached
        
        # Try to load from database
        params = self._load_from_db(param_type, domain)
        
        if params is None and domain:
            # Fallback to global params if no domain override
            params = self._load_from_db(param_type, None)
        
        if params is None:
            # Use defaults
            params = get_default_params(param_type)
        
        # Cache the result
        if self._auto_cache:
            self._cache.set(param_type.value, params, domain)
        
        return params
    
    def set_params(
        self,
        param_type: Union[ParamType, str],
        params: Union[BaseModel, Dict[str, Any]],
        domain: Optional[str] = None
    ) -> bool:
        """
        Set parameters for a given type and optional domain.
        
        Args:
            param_type: Parameter type
            params: Parameter values (model or dict)
            domain: Optional domain for domain-specific override
        
        Returns:
            True if successfully saved
        """
        if isinstance(param_type, str):
            param_type = ParamType(param_type)
        
        # Check if domain overrides are supported
        if domain and param_type not in DOMAIN_SPECIFIC_PARAMS:
            logger.warning(
                f"Parameter type {param_type.value} does not support "
                f"domain-specific overrides. Ignoring domain."
            )
            domain = None
        
        # Convert dict to model if needed
        if isinstance(params, dict):
            model_class = PARAM_TYPE_TO_CLASS[param_type]
            params = model_class(**params)
        
        # Save to database
        success = self._save_to_db(param_type, params, domain)
        
        if success:
            # Update cache
            self._cache.set(param_type.value, params, domain)
        
        return success
    
    def reset_to_defaults(
        self,
        param_type: Union[ParamType, str],
        domain: Optional[str] = None
    ) -> bool:
        """
        Reset parameters to defaults.
        
        Args:
            param_type: Parameter type to reset
            domain: If provided, removes domain override; if None, resets global
        
        Returns:
            True if successfully reset
        """
        if isinstance(param_type, str):
            param_type = ParamType(param_type)
        
        if domain:
            # Remove domain override
            success = self._delete_from_db(param_type, domain)
        else:
            # Reset to defaults
            defaults = get_default_params(param_type)
            success = self._save_to_db(param_type, defaults, None)
        
        if success:
            self._cache.invalidate(param_type.value, domain)
        
        return success
    
    # =========================================================================
    # Typed Parameter Accessors
    # =========================================================================
    
    def get_trueskill_params(self) -> TrueSkillParams:
        """Get TrueSkill parameters (global only)."""
        return self.get_params(ParamType.TRUESKILL)
    
    def get_strata_params(self, domain: Optional[str] = None) -> StrataParams:
        """Get strata parameters (supports domain override)."""
        return self.get_params(ParamType.STRATA, domain)
    
    def get_match_params(self, domain: Optional[str] = None) -> MatchParams:
        """Get match parameters (supports domain override)."""
        return self.get_params(ParamType.MATCH, domain)
    
    def get_temperature_params(self) -> TemperatureParams:
        """Get temperature parameters (global only)."""
        return self.get_params(ParamType.TEMPERATURE)
    
    def get_dimension_weights(
        self,
        domain: Optional[str] = None
    ) -> DimensionWeightParams:
        """Get dimension weights (supports domain override)."""
        return self.get_params(ParamType.DIMENSION_WEIGHTS, domain)
    
    def get_transfer_params(self) -> TransferParams:
        """Get transfer learning parameters (global only)."""
        return self.get_params(ParamType.TRANSFER)
    
    def get_feedback_params(self) -> FeedbackParams:
        """Get user feedback parameters (global only)."""
        return self.get_params(ParamType.FEEDBACK)
    
    def get_penalty_params(self, domain: Optional[str] = None) -> PenaltyParams:
        """Get penalty parameters (supports domain override)."""
        return self.get_params(ParamType.PENALTY, domain)
    
    def get_consistency_params(self) -> ConsistencyParams:
        """Get consistency evaluation parameters (global only)."""
        return self.get_params(ParamType.CONSISTENCY)
    
    def get_long_context_params(self) -> LongContextParams:
        """Get long context evaluation parameters (global only)."""
        return self.get_params(ParamType.LONG_CONTEXT)
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_all_params(self) -> Dict[str, Any]:
        """
        Export all parameters for backup.
        
        Returns:
            Dictionary with all parameter groups
        """
        result = {}
        
        for param_type in ParamType:
            params = self.get_params(param_type)
            result[param_type.value] = params.model_dump()
            
            # Export domain overrides for supported types
            if param_type in DOMAIN_SPECIFIC_PARAMS:
                overrides = self._list_domain_overrides(param_type)
                if overrides:
                    result[f"{param_type.value}_domains"] = {
                        domain: self.get_params(param_type, domain).model_dump()
                        for domain in overrides
                    }
        
        return result
    
    def import_params(self, params: Dict[str, Any]) -> Dict[str, bool]:
        """
        Import parameters from backup.
        
        Args:
            params: Dictionary with parameter groups
        
        Returns:
            Dictionary mapping param types to import success
        """
        results = {}
        
        for key, value in params.items():
            # Skip domain override dictionaries (handled separately)
            if key.endswith("_domains"):
                continue
            
            try:
                param_type = ParamType(key)
                success = self.set_params(param_type, value)
                results[key] = success
                
                # Import domain overrides if present
                domain_key = f"{key}_domains"
                if domain_key in params:
                    for domain, domain_params in params[domain_key].items():
                        domain_success = self.set_params(param_type, domain_params, domain)
                        results[f"{key}:{domain}"] = domain_success
                        
            except ValueError:
                logger.warning(f"Unknown parameter type: {key}")
                results[key] = False
        
        return results
    
    # =========================================================================
    # Database Operations
    # =========================================================================
    
    def _load_from_db(
        self,
        param_type: ParamType,
        domain: Optional[str]
    ) -> Optional[BaseModel]:
        """Load parameters from database."""
        if not self._db:
            return None
        
        try:
            data = self._db.get_params(param_type.value, domain)
            if data:
                model_class = PARAM_TYPE_TO_CLASS[param_type]
                return model_class(**data)
        except Exception as e:
            logger.error(f"Error loading params from DB: {e}")
        
        return None
    
    def _save_to_db(
        self,
        param_type: ParamType,
        params: BaseModel,
        domain: Optional[str]
    ) -> bool:
        """Save parameters to database."""
        if not self._db:
            logger.warning("No database connection, params not persisted")
            return True  # Still return True so caching works
        
        try:
            return self._db.save_params(
                param_type.value,
                params.model_dump(),
                domain
            )
        except Exception as e:
            logger.error(f"Error saving params to DB: {e}")
            return False
    
    def _delete_from_db(
        self,
        param_type: ParamType,
        domain: str
    ) -> bool:
        """Delete domain override from database."""
        if not self._db:
            return True
        
        try:
            return self._db.delete_param_override(param_type.value, domain)
        except Exception as e:
            logger.error(f"Error deleting param override: {e}")
            return False
    
    def _list_domain_overrides(self, param_type: ParamType) -> list:
        """List domains with overrides for a parameter type."""
        if not self._db:
            return []
        
        try:
            return self._db.list_param_overrides(param_type.value)
        except Exception as e:
            logger.error(f"Error listing param overrides: {e}")
            return []
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def invalidate_cache(
        self,
        param_type: Optional[Union[ParamType, str]] = None,
        domain: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            param_type: Specific type to invalidate (all if None)
            domain: Specific domain to invalidate
        
        Returns:
            Number of entries invalidated
        """
        if param_type is None:
            return self._cache.invalidate_all()
        
        if isinstance(param_type, str):
            param_type = ParamType(param_type)
        
        if domain:
            return 1 if self._cache.invalidate(param_type.value, domain) else 0
        else:
            return self._cache.invalidate_type(param_type.value)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


# =============================================================================
# Global Instance
# =============================================================================

_parameter_manager: Optional[ParameterManager] = None


def get_parameter_manager() -> ParameterManager:
    """Get or create global ParameterManager instance."""
    global _parameter_manager
    if _parameter_manager is None:
        _parameter_manager = ParameterManager()
    return _parameter_manager


def initialize_parameter_manager(
    db: Any = None,
    cache_ttl: int = 300
) -> ParameterManager:
    """
    Initialize global ParameterManager with database.
    
    Args:
        db: MongoDB instance
        cache_ttl: Cache TTL in seconds
    
    Returns:
        Initialized ParameterManager
    """
    global _parameter_manager
    _parameter_manager = ParameterManager(db=db, cache_ttl=cache_ttl)
    return _parameter_manager


