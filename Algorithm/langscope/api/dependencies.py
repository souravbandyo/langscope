"""
Dependency injection for LangScope API.

Provides database, managers, and other dependencies for API routes.

Cache cleared: 2026-02-01 05:30
"""

import os
import logging
from typing import Optional, List, TYPE_CHECKING
from functools import lru_cache

from langscope.database.mongodb import MongoDB, initialize_database, get_database
from langscope.domain.domain_manager import DomainManager
from langscope.feedback.workflow import UserFeedbackWorkflow, create_feedback_workflow
from langscope.feedback.use_case import UseCaseAdjustmentManager
from langscope.feedback.judge_calibration import JudgeCalibrator
from langscope.transfer.specialist import SpecialistDetector
from langscope.transfer.transfer_learning import TransferLearning
from langscope.transfer.correlation import CorrelationLearner
from langscope.core.model import LLMModel
from langscope.core.model_definitions import get_all_models
from langscope.config.params import get_parameter_manager
from langscope.config.params.manager import initialize_parameter_manager

if TYPE_CHECKING:
    from langscope.cache.manager import UnifiedCacheManager
    from langscope.prompt.manager import PromptManager

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@lru_cache()
def get_settings():
    """Get application settings from environment."""
    return {
        "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("DB_NAME", "langscope"),
        "api_key": os.getenv("LANGSCOPE_API_KEY", ""),
        "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        # Supabase Auth
        "supabase_url": os.getenv("SUPABASE_URL", ""),
        "supabase_jwt_secret": os.getenv("SUPABASE_JWT_SECRET", ""),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY", ""),
        "supabase_service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    }


# =============================================================================
# Database
# =============================================================================

_db_instance: Optional[MongoDB] = None


def get_db() -> MongoDB:
    """
    Get database instance.
    
    Creates a new connection if not already connected.
    Also initializes ParameterManager with database connection.
    
    Returns:
        MongoDB instance
    
    Raises:
        RuntimeError: If database connection fails
    """
    global _db_instance
    
    if _db_instance is None or not _db_instance.connected:
        settings = get_settings()
        _db_instance = initialize_database(
            settings["mongodb_uri"],
            settings["db_name"]
        )
        
        if not _db_instance.connected:
            raise RuntimeError("Failed to connect to database")
        
        # Initialize ParameterManager with database connection
        initialize_parameter_manager(db=_db_instance)
        logger.info("ParameterManager initialized with database")
    
    return _db_instance


def close_db():
    """Close database connection."""
    global _db_instance
    if _db_instance:
        _db_instance.disconnect()
        _db_instance = None


# =============================================================================
# Domain Manager
# =============================================================================

_domain_manager: Optional[DomainManager] = None


def get_domain_manager() -> DomainManager:
    """Get domain manager instance."""
    global _domain_manager
    
    if _domain_manager is None:
        try:
            db = get_db()
            _domain_manager = DomainManager(db=db)
        except RuntimeError:
            # Fall back to manager without DB
            _domain_manager = DomainManager()
    
    return _domain_manager


# =============================================================================
# Transfer Learning
# =============================================================================

_transfer_learner: Optional[TransferLearning] = None
_correlation_learner: Optional[CorrelationLearner] = None


def get_transfer_learner() -> TransferLearning:
    """Get transfer learning instance."""
    global _transfer_learner
    
    if _transfer_learner is None:
        _transfer_learner = TransferLearning()
    
    return _transfer_learner


def get_correlation_learner() -> CorrelationLearner:
    """Get correlation learner instance."""
    global _correlation_learner
    
    if _correlation_learner is None:
        _correlation_learner = CorrelationLearner()
    
    return _correlation_learner


def get_specialist_detector() -> SpecialistDetector:
    """Get specialist detector instance."""
    return SpecialistDetector(transfer_learner=get_transfer_learner())


# =============================================================================
# Faceted Transfer Learning (New)
# =============================================================================

_domain_index: Optional['DomainIndex'] = None
_faceted_transfer: Optional['FacetedTransferLearning'] = None


def get_domain_index() -> 'DomainIndex':
    """
    Get or create domain index singleton.
    
    Returns:
        DomainIndex instance with all domains loaded
    """
    global _domain_index
    
    if _domain_index is None:
        from langscope.transfer.faceted import DomainIndex
        from langscope.transfer.priors import create_initialized_composite
        
        # Create composite with priors loaded
        try:
            db = get_db()
        except RuntimeError:
            db = None
        
        composite = create_initialized_composite(db=db)
        
        _domain_index = DomainIndex(
            domain_manager=get_domain_manager(),
            composite=composite
        )
        _domain_index.load()
        
        # Pre-compute top-K for existing domains
        if len(_domain_index.descriptors) > 0:
            _domain_index.precompute_top_k(k=10)
            logger.info(f"DomainIndex initialized with {len(_domain_index.descriptors)} domains")
    
    return _domain_index


def get_faceted_transfer() -> 'FacetedTransferLearning':
    """
    Get or create faceted transfer learning singleton.
    
    Returns:
        FacetedTransferLearning instance
    """
    global _faceted_transfer
    
    if _faceted_transfer is None:
        from langscope.transfer.faceted import FacetedTransferLearning
        
        _faceted_transfer = FacetedTransferLearning(
            domain_index=get_domain_index()
        )
        logger.info("FacetedTransferLearning initialized")
    
    return _faceted_transfer


def refresh_domain_index():
    """Refresh the domain index with latest domains."""
    global _domain_index
    if _domain_index:
        _domain_index.load()
        _domain_index.precompute_top_k(k=10)
        logger.info("DomainIndex refreshed")


# =============================================================================
# User Feedback
# =============================================================================

_use_case_managers: dict[str, UseCaseAdjustmentManager] = {}
_judge_calibrators: dict[str, JudgeCalibrator] = {}
_active_sessions: dict[str, dict] = {}  # session_id -> session data (legacy, use cache manager)

# =============================================================================
# Cache Manager
# =============================================================================

_cache_manager: Optional['UnifiedCacheManager'] = None


async def get_cache_manager() -> 'UnifiedCacheManager':
    """
    Get or create cache manager singleton.
    
    Returns:
        UnifiedCacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        from langscope.cache.manager import UnifiedCacheManager
        
        settings = get_settings()
        
        # Get Async MongoDB client (Motor) for cache manager
        mongodb_client = None
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            import certifi
            
            # Use same URI as main DB
            uri = settings["mongodb_uri"]
            kwargs = {}
            
            # Check for TLS need (similar to mongodb.py)
            is_local = any(host in uri for host in ['localhost', '127.0.0.1', '0.0.0.0'])
            if not is_local:
                kwargs["tlsCAFile"] = certifi.where()
                
            mongodb_client = AsyncIOMotorClient(uri, **kwargs)
            logger.info("Created AsyncIOMotorClient for CacheManager")
        except ImportError:
            logger.warning("motor or certifi not installed, cache persistence disabled")
        except Exception as e:
            logger.warning(f"Failed to create AsyncIOMotorClient: {e}")
        
        _cache_manager = UnifiedCacheManager(
            redis_url=settings.get("redis_url", os.getenv("REDIS_URL", "redis://localhost:6379")),
            mongodb_client=mongodb_client,
            db_name=settings["db_name"],
        )
        await _cache_manager.initialize()
        logger.info("CacheManager initialized")
    
    return _cache_manager


# =============================================================================
# Prompt Manager
# =============================================================================

_prompt_manager: Optional['PromptManager'] = None


async def get_prompt_manager() -> 'PromptManager':
    """
    Get or create prompt manager singleton.
    
    Returns:
        PromptManager instance
    """
    global _prompt_manager
    
    if _prompt_manager is None:
        from langscope.prompt.manager import PromptManager
        from langscope.prompt.classifier import HierarchicalDomainClassifier
        from langscope.data import get_centroids_path
        
        # Get cache manager (may be None if not initialized yet)
        cache_mgr = None
        try:
            cache_mgr = await get_cache_manager()
        except Exception as e:
            logger.warning(f"Cache manager not available for PromptManager: {e}")
        
        _prompt_manager = PromptManager(
            domain_manager=get_domain_manager(),
            classifier=HierarchicalDomainClassifier(
                centroids_path=get_centroids_path(),
                lazy_load=True,
            ),
            cache_manager=cache_mgr,
        )
        logger.info("PromptManager initialized")
    
    return _prompt_manager


def get_use_case_manager(domain: str = "default") -> UseCaseAdjustmentManager:
    """Get use case manager for a domain."""
    if domain not in _use_case_managers:
        _use_case_managers[domain] = UseCaseAdjustmentManager()
    return _use_case_managers[domain]


def get_judge_calibrator(domain: str = "default") -> JudgeCalibrator:
    """Get judge calibrator for a domain."""
    if domain not in _judge_calibrators:
        _judge_calibrators[domain] = JudgeCalibrator()
    return _judge_calibrators[domain]


def get_feedback_workflow(domain: str) -> UserFeedbackWorkflow:
    """Get user feedback workflow for a domain."""
    try:
        db = get_db()
    except RuntimeError:
        db = None
    
    return create_feedback_workflow(
        domain=domain,
        db=db,
        tau_use_case=10.0,
        calibration_gamma=0.2
    )


def store_session(session_id: str, session_data: dict):
    """
    Store active session data.
    
    Note: For async code, use cache_manager directly.
    This is a sync fallback for backward compatibility.
    """
    _active_sessions[session_id] = session_data


def get_session(session_id: str) -> Optional[dict]:
    """
    Get active session data.
    
    Note: For async code, use cache_manager directly.
    This is a sync fallback for backward compatibility.
    """
    return _active_sessions.get(session_id)


def remove_session(session_id: str):
    """
    Remove completed session.
    
    Note: For async code, use cache_manager directly.
    """
    if session_id in _active_sessions:
        del _active_sessions[session_id]


async def store_session_async(session_id: str, session_data: dict):
    """Store session in Redis + MongoDB (async version)."""
    from langscope.cache.categories import CacheCategory
    
    cache = await get_cache_manager()
    await cache.set(CacheCategory.SESSION, session_id, session_data, ttl=1800)
    # Also keep in local dict for sync access
    _active_sessions[session_id] = session_data


async def get_session_async(session_id: str) -> Optional[dict]:
    """Get session from Redis with MongoDB fallback (async version)."""
    from langscope.cache.categories import CacheCategory
    
    cache = await get_cache_manager()
    session = await cache.get(CacheCategory.SESSION, session_id)
    
    if session:
        return session
    
    # Check local fallback
    return _active_sessions.get(session_id)


async def remove_session_async(session_id: str):
    """Remove session from all layers (async version)."""
    from langscope.cache.categories import CacheCategory
    
    cache = await get_cache_manager()
    await cache.delete(CacheCategory.SESSION, session_id)
    
    if session_id in _active_sessions:
        del _active_sessions[session_id]


# =============================================================================
# LLM Caller
# =============================================================================

_llm_caller: Optional[object] = None


def get_llm_caller():
    """
    Get LLM caller instance (e.g., LiteLLM).
    
    Returns:
        LLM caller or None if not configured
    """
    global _llm_caller
    
    if _llm_caller is None:
        try:
            import litellm
            _llm_caller = litellm
            logger.info("LiteLLM caller initialized")
        except ImportError:
            logger.warning("LiteLLM not installed, LLM calls will be mocked")
            _llm_caller = None
    
    return _llm_caller


# =============================================================================
# Models
# =============================================================================

_models_cache: Optional[List[LLMModel]] = None
_models_cache_time: float = 0


def get_models(domain: str = None, force_refresh: bool = False) -> List[LLMModel]:
    """
    Get all models, optionally filtered by domain.
    
    Uses caching with a 60-second TTL.
    
    Args:
        domain: Optional domain filter
        force_refresh: Force cache refresh
    
    Returns:
        List of LLMModel instances
    """
    global _models_cache, _models_cache_time
    import time
    
    current_time = time.time()
    
    # Check cache validity (60 second TTL)
    if (not force_refresh and 
        _models_cache is not None and 
        current_time - _models_cache_time < 60):
        models = _models_cache
    else:
        # Try to load from database first
        try:
            db = get_db()
            model_dicts = db.get_all_models()
            
            if model_dicts:
                models = [LLMModel.from_dict(m) for m in model_dicts]
            else:
                # Fall back to model definitions (convert dicts to LLMModel)
                model_defs = get_all_models()
                models = [LLMModel.from_dict(m) for m in model_defs]
        except (RuntimeError, Exception):
            # Fall back to model definitions (convert dicts to LLMModel)
            model_defs = get_all_models()
            models = [LLMModel.from_dict(m) for m in model_defs]
        
        _models_cache = models
        _models_cache_time = current_time
    
    # Filter by domain if specified
    if domain:
        models = [
            m for m in models 
            if domain in m.trueskill_by_domain
        ]
    
    return models


def get_model_by_id(model_id: str) -> Optional[LLMModel]:
    """Get a specific model by ID."""
    models = get_models()
    for model in models:
        if model.model_id == model_id:
            return model
    return None


def get_model_by_name(name: str) -> Optional[LLMModel]:
    """Get a specific model by name."""
    models = get_models()
    for model in models:
        if model.name == name:
            return model
    return None


def refresh_models_cache():
    """Force refresh of models cache."""
    global _models_cache, _models_cache_time
    _models_cache = None
    _models_cache_time = 0


# =============================================================================
# Cleanup
# =============================================================================

def cleanup():
    """Clean up all resources."""
    close_db()
    global _domain_manager, _transfer_learner, _correlation_learner
    global _use_case_managers, _judge_calibrators, _active_sessions
    global _models_cache, _cache_manager, _prompt_manager
    global _domain_index, _faceted_transfer
    
    _domain_manager = None
    _transfer_learner = None
    _correlation_learner = None
    _use_case_managers = {}
    _judge_calibrators = {}
    _active_sessions = {}
    _models_cache = None
    _cache_manager = None
    _prompt_manager = None
    _domain_index = None
    _faceted_transfer = None


async def cleanup_async():
    """Async cleanup with cache manager close."""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
    
    cleanup()
