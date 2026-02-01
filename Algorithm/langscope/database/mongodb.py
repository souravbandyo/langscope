"""
MongoDB connection and operations for LangScope.

Provides database connectivity, CRUD operations, and index management.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from pymongo.database import Database
    from pymongo.collection import Collection
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import certifi
    CERTIFI_AVAILABLE = True
except ImportError:
    CERTIFI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Collection names
MODELS_COLLECTION = "models"
MATCHES_COLLECTION = "matches"
DOMAINS_COLLECTION = "domains"
CORRELATIONS_COLLECTION = "domain_correlations"

# User feedback collections
USER_SESSIONS_COLLECTION = "user_sessions"
USE_CASE_ADJUSTMENTS_COLLECTION = "use_case_adjustments"
JUDGE_CALIBRATIONS_COLLECTION = "judge_calibrations"

# Parameter management collection
PARAMETERS_COLLECTION = "parameters"

# Phase 11: Multi-provider collections
BASE_MODELS_COLLECTION = "base_models"
MODEL_DEPLOYMENTS_COLLECTION = "model_deployments"
SELF_HOSTED_DEPLOYMENTS_COLLECTION = "self_hosted_deployments"

# Phase 12: Benchmark collections
BENCHMARK_DEFINITIONS_COLLECTION = "benchmark_definitions"
BENCHMARK_RESULTS_COLLECTION = "benchmark_results"

# Phase 13: Automation collections
DATA_SOURCES_COLLECTION = "data_sources"
SOURCE_SYNC_HISTORY_COLLECTION = "source_sync_history"
PRICE_HISTORY_COLLECTION = "price_history"

# Phase 14: Time series collections
MODEL_RATINGS_HISTORY_COLLECTION = "model_ratings_history"
PERFORMANCE_METRICS_COLLECTION = "performance_metrics"

# Phase 16: Content hashing collection
CONTENT_HASHES_COLLECTION = "content_hashes"

# Phase 17-24: Ground Truth collections
GROUND_TRUTH_SAMPLES_COLLECTION = "ground_truth_samples"
GROUND_TRUTH_MATCHES_COLLECTION = "ground_truth_matches"
GROUND_TRUTH_RATINGS_COLLECTION = "ground_truth_ratings"
GROUND_TRUTH_RATINGS_HISTORY_COLLECTION = "ground_truth_ratings_history"
GROUND_TRUTH_LEADERBOARDS_COLLECTION = "ground_truth_leaderboards"
GROUND_TRUTH_COVERAGE_COLLECTION = "ground_truth_coverage"


class MongoDB:
    """
    MongoDB connection and operations for LangScope.
    
    Handles connection pooling, index creation, and CRUD operations
    for models, matches, domains, and correlations.
    """
    
    def __init__(
        self,
        connection_string: str,
        db_name: str = "langscope",
        auto_connect: bool = True
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI
            db_name: Database name
            auto_connect: Whether to connect immediately
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB operations. Install with: pip install pymongo")
        
        self.connection_string = connection_string
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connect_kwargs = {
                "serverSelectionTimeoutMS": 5000,
                "maxPoolSize": 10,
            }
            
            # Only use TLS for non-local connections (e.g., MongoDB Atlas)
            is_local = any(host in self.connection_string for host in 
                          ['localhost', '127.0.0.1', '0.0.0.0'])
            
            if CERTIFI_AVAILABLE and not is_local:
                connect_kwargs["tlsCAFile"] = certifi.where()
            
            self.client = MongoClient(self.connection_string, **connect_kwargs)
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            # Create indexes
            self._create_indexes()
            
            self.connected = True
            logger.info(f"Connected to MongoDB: {self.db_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        if self.db is None:
            return
        
        # Ensure time series collections exist
        self._ensure_time_series_collections()
        
        # Models collection indexes
        self.db[MODELS_COLLECTION].create_index(
            [("name", pymongo.ASCENDING)], unique=True
        )
        self.db[MODELS_COLLECTION].create_index(
            [("model_id", pymongo.ASCENDING)], unique=True
        )
        self.db[MODELS_COLLECTION].create_index(
            [("provider", pymongo.ASCENDING)]
        )
        self.db[MODELS_COLLECTION].create_index(
            [("trueskill.raw.mu", pymongo.DESCENDING)]
        )
        
        # Matches collection indexes
        self.db[MATCHES_COLLECTION].create_index(
            [("timestamp", pymongo.DESCENDING)]
        )
        self.db[MATCHES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING)]
        )
        self.db[MATCHES_COLLECTION].create_index(
            [("participants", pymongo.ASCENDING)]
        )
        self.db[MATCHES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)]
        )
        
        # Domains collection indexes
        self.db[DOMAINS_COLLECTION].create_index(
            [("name", pymongo.ASCENDING)], unique=True
        )
        self.db[DOMAINS_COLLECTION].create_index(
            [("parent_domain", pymongo.ASCENDING)]
        )
        
        # Correlations collection indexes
        self.db[CORRELATIONS_COLLECTION].create_index(
            [("domain_a", pymongo.ASCENDING), ("domain_b", pymongo.ASCENDING)],
            unique=True
        )
        
        # Phase 11: Base models indexes
        self.db[BASE_MODELS_COLLECTION].create_index(
            [("name", pymongo.ASCENDING)]
        )
        self.db[BASE_MODELS_COLLECTION].create_index(
            [("family", pymongo.ASCENDING)]
        )
        self.db[BASE_MODELS_COLLECTION].create_index(
            [("organization", pymongo.ASCENDING)]
        )
        
        # Phase 11: Model deployments indexes
        self.db[MODEL_DEPLOYMENTS_COLLECTION].create_index(
            [("base_model_id", pymongo.ASCENDING)]
        )
        self.db[MODEL_DEPLOYMENTS_COLLECTION].create_index(
            [("provider.id", pymongo.ASCENDING)]
        )
        self.db[MODEL_DEPLOYMENTS_COLLECTION].create_index(
            [("base_model_id", pymongo.ASCENDING),
             ("trueskill.cost_adjusted.mu", pymongo.DESCENDING)]
        )
        self.db[MODEL_DEPLOYMENTS_COLLECTION].create_index(
            [("availability.status", pymongo.ASCENDING)]
        )
        
        # Phase 11: Self-hosted deployments indexes
        self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].create_index(
            [("base_model_id", pymongo.ASCENDING)]
        )
        self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].create_index(
            [("owner.user_id", pymongo.ASCENDING)]
        )
        self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].create_index(
            [("owner.is_public", pymongo.ASCENDING)]
        )
        
        # Phase 16: Content hashes indexes
        self.db[CONTENT_HASHES_COLLECTION].create_index(
            [("content_type", pymongo.ASCENDING), ("domain", pymongo.ASCENDING)]
        )
        self.db[CONTENT_HASHES_COLLECTION].create_index(
            [("first_seen", pymongo.DESCENDING)]
        )
        
        # Phase 17-24: Ground truth indexes
        self.db[GROUND_TRUTH_SAMPLES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING)]
        )
        self.db[GROUND_TRUTH_SAMPLES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("difficulty", pymongo.ASCENDING)]
        )
        self.db[GROUND_TRUTH_SAMPLES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("last_used", pymongo.ASCENDING)]
        )
        self.db[GROUND_TRUTH_SAMPLES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("metadata.context_length", pymongo.ASCENDING)]
        )
        
        self.db[GROUND_TRUTH_MATCHES_COLLECTION].create_index(
            [("timestamp", pymongo.DESCENDING)]
        )
        self.db[GROUND_TRUTH_MATCHES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING)]
        )
        self.db[GROUND_TRUTH_MATCHES_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)]
        )
        self.db[GROUND_TRUTH_MATCHES_COLLECTION].create_index(
            [("sample_id", pymongo.ASCENDING)]
        )
        self.db[GROUND_TRUTH_MATCHES_COLLECTION].create_index(
            [("participants", pymongo.ASCENDING)]
        )
        
        self.db[GROUND_TRUTH_RATINGS_COLLECTION].create_index(
            [("domain", pymongo.ASCENDING), ("trueskill.mu", pymongo.DESCENDING)]
        )
        self.db[GROUND_TRUTH_RATINGS_COLLECTION].create_index(
            [("deployment_id", pymongo.ASCENDING)]
        )
        
        logger.debug("Database indexes created")
    
    # =========================================================================
    # Model Operations
    # =========================================================================
    
    def save_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Save or update model in database.
        
        Args:
            model_data: Model document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            model_data["metadata"] = model_data.get("metadata", {})
            model_data["metadata"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[MODELS_COLLECTION].update_one(
                {"name": model_data["name"]},
                {"$set": model_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model by name.
        
        Args:
            name: Model name
        
        Returns:
            Model document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[MODELS_COLLECTION].find_one({"name": name}, {"_id": 0})
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model by model_id.
        
        Args:
            model_id: Model ID
        
        Returns:
            Model document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[MODELS_COLLECTION].find_one({"model_id": model_id}, {"_id": 0})
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all models."""
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[MODELS_COLLECTION].find({}, {"_id": 0}))
    
    def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get models by provider."""
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[MODELS_COLLECTION].find(
            {"provider": provider}, {"_id": 0}
        ))
    
    def get_models_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get models evaluated in a specific domain.
        
        Args:
            domain: Domain name
        
        Returns:
            List of model documents
        """
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[MODELS_COLLECTION].find(
            {f"trueskill_by_domain.{domain}": {"$exists": True}},
            {"_id": 0}
        ))
    
    def delete_model(self, name: str) -> bool:
        """Delete model by name."""
        if not self.connected or self.db is None:
            return False
        
        try:
            result = self.db[MODELS_COLLECTION].delete_one({"name": name})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    # =========================================================================
    # Match Operations (Multi-player)
    # =========================================================================
    
    def save_match(self, match_data: Dict[str, Any]) -> bool:
        """
        Save multi-player match result.
        
        Args:
            match_data: Match document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            self.db[MATCHES_COLLECTION].insert_one(match_data)
            return True
        except Exception as e:
            logger.error(f"Error saving match: {e}")
            return False
    
    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get match by ID.
        
        Args:
            match_id: Match ID
        
        Returns:
            Match document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[MATCHES_COLLECTION].find_one({"_id": match_id})
    
    def get_matches_by_domain(
        self,
        domain: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get matches for a domain.
        
        Args:
            domain: Domain name
            limit: Maximum matches to return
            skip: Number to skip (for pagination)
        
        Returns:
            List of match documents
        """
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[MATCHES_COLLECTION].find(
            {"domain": domain},
            {"_id": 1, "timestamp": 1, "participants": 1, "judgment.raw_ranking": 1}
        ).sort("timestamp", -1).skip(skip).limit(limit))
    
    def get_matches_by_model(
        self,
        model_id: str,
        domain: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get matches involving a specific model.
        
        Args:
            model_id: Model ID
            domain: Optional domain filter
            limit: Maximum matches to return
        
        Returns:
            List of match documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"participants": model_id}
        if domain:
            query["domain"] = domain
        
        return list(self.db[MATCHES_COLLECTION].find(
            query, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
    
    def get_match_count(self, domain: str = None) -> int:
        """Get total match count."""
        if not self.connected or self.db is None:
            return 0
        
        query = {"domain": domain} if domain else {}
        return self.db[MATCHES_COLLECTION].count_documents(query)
    
    # =========================================================================
    # Domain Operations
    # =========================================================================
    
    def save_domain(self, domain_data: Dict[str, Any]) -> bool:
        """Save or update domain."""
        if not self.connected or self.db is None:
            return False
        
        try:
            domain_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[DOMAINS_COLLECTION].update_one(
                {"_id": domain_data["_id"]},
                {"$set": domain_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving domain: {e}")
            return False
    
    def get_domain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get domain by name."""
        if not self.connected or self.db is None:
            return None
        
        return self.db[DOMAINS_COLLECTION].find_one({"_id": name})
    
    def get_all_domains(self) -> List[Dict[str, Any]]:
        """Get all domains."""
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[DOMAINS_COLLECTION].find({}))
    
    def delete_domain(self, name: str) -> bool:
        """Delete domain by name."""
        if not self.connected or self.db is None:
            return False
        
        try:
            result = self.db[DOMAINS_COLLECTION].delete_one({"_id": name})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting domain: {e}")
            return False
    
    # =========================================================================
    # Correlation Operations
    # =========================================================================
    
    def save_correlation(self, correlation_data: Dict[str, Any]) -> bool:
        """Save or update domain correlation."""
        if not self.connected or self.db is None:
            return False
        
        try:
            correlation_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[CORRELATIONS_COLLECTION].update_one(
                {"_id": correlation_data["_id"]},
                {"$set": correlation_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving correlation: {e}")
            return False
    
    def get_correlation(
        self,
        domain_a: str,
        domain_b: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get correlation between two domains.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
        
        Returns:
            Correlation document or None
        """
        if not self.connected or self.db is None:
            return None
        
        # Try both orderings
        correlation_id = f"{domain_a}|{domain_b}"
        result = self.db[CORRELATIONS_COLLECTION].find_one({"_id": correlation_id})
        
        if not result:
            correlation_id = f"{domain_b}|{domain_a}"
            result = self.db[CORRELATIONS_COLLECTION].find_one({"_id": correlation_id})
        
        return result
    
    def get_correlations_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all correlations involving a domain."""
        if not self.connected or self.db is None:
            return []
        
        return list(self.db[CORRELATIONS_COLLECTION].find({
            "$or": [
                {"domain_a": domain},
                {"domain_b": domain}
            ]
        }))
    
    # =========================================================================
    # Facet Similarity Operations (Transfer Learning)
    # =========================================================================
    
    def save_facet_similarity(self, similarity_data: Dict[str, Any]) -> bool:
        """
        Save or update facet similarity data.
        
        Args:
            similarity_data: Dict with _id, facet, value_a, value_b, similarities
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            similarity_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db["facet_similarities"].update_one(
                {"_id": similarity_data["_id"]},
                {"$set": similarity_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving facet similarity: {e}")
            return False
    
    def get_facet_similarity(
        self,
        facet: str,
        value_a: str,
        value_b: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get similarity between two facet values.
        
        Args:
            facet: Facet name (e.g., "language", "field")
            value_a: First value
            value_b: Second value
        
        Returns:
            Similarity document or None
        """
        if not self.connected or self.db is None:
            return None
        
        # Canonical key (sorted)
        a, b = sorted([value_a.lower(), value_b.lower()])
        similarity_id = f"{facet}|{a}|{b}"
        
        return self.db["facet_similarities"].find_one({"_id": similarity_id})
    
    def get_facet_similarities(self, facet: str) -> List[Dict[str, Any]]:
        """
        Get all learned similarities for a facet.
        
        Args:
            facet: Facet name
        
        Returns:
            List of similarity documents
        """
        if not self.connected or self.db is None:
            return []
        
        return list(self.db["facet_similarities"].find({"facet": facet}))
    
    def save_domain_descriptor(self, descriptor_data: Dict[str, Any]) -> bool:
        """
        Save or update domain descriptor (facets mapping).
        
        Args:
            descriptor_data: Dict with name and facets
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            descriptor_data["_id"] = descriptor_data.get("name", "")
            descriptor_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db["domain_descriptors"].update_one(
                {"_id": descriptor_data["_id"]},
                {"$set": descriptor_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving domain descriptor: {e}")
            return False
    
    def get_domain_descriptor(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Get domain descriptor by name.
        
        Args:
            domain_name: Domain name
        
        Returns:
            Descriptor document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db["domain_descriptors"].find_one({"_id": domain_name})
    
    def get_all_domain_descriptors(self) -> List[Dict[str, Any]]:
        """Get all domain descriptors."""
        if not self.connected or self.db is None:
            return []
        
        return list(self.db["domain_descriptors"].find({}))
    
    # =========================================================================
    # Leaderboard Operations
    # =========================================================================
    
    def get_leaderboard(
        self,
        domain: str = None,
        ranking_type: str = "raw",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard for a domain.
        
        Args:
            domain: Domain name (None for global)
            ranking_type: "raw" or "cost_adjusted"
            limit: Maximum entries to return
        
        Returns:
            List of model summaries sorted by rating
        """
        if not self.connected or self.db is None:
            return []
        
        # Build sort key based on domain and ranking type
        if domain:
            sort_key = f"trueskill_by_domain.{domain}.{ranking_type}.mu"
            query = {f"trueskill_by_domain.{domain}": {"$exists": True}}
        else:
            sort_key = f"trueskill.{ranking_type}.mu"
            query = {}
        
        projection = {
            "_id": 0,
            "name": 1,
            "model_id": 1,
            "provider": 1,
            "trueskill": 1,
            f"trueskill_by_domain.{domain}": 1 if domain else None,
            "performance": 1,
            f"performance_by_domain.{domain}": 1 if domain else None,
        }
        
        # Remove None values from projection
        projection = {k: v for k, v in projection.items() if v is not None}
        
        return list(self.db[MODELS_COLLECTION].find(
            query, projection
        ).sort(sort_key, pymongo.DESCENDING).limit(limit))
    
    # =========================================================================
    # User Session Operations (User Feedback Integration)
    # =========================================================================
    
    def save_user_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save user feedback session.
        
        Args:
            session_data: Session document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            self.db[USER_SESSIONS_COLLECTION].insert_one(session_data)
            return True
        except Exception as e:
            logger.error(f"Error saving user session: {e}")
            return False
    
    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[USER_SESSIONS_COLLECTION].find_one(
            {"_id": session_id}, {"_id": 0}
        )
    
    def get_sessions_by_use_case(
        self,
        use_case: str,
        domain: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a use case.
        
        Args:
            use_case: Use case category
            domain: Optional domain filter
            limit: Maximum sessions to return
        
        Returns:
            List of session documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"use_case": use_case}
        if domain:
            query["domain"] = domain
        
        return list(self.db[USER_SESSIONS_COLLECTION].find(
            query, {"_id": 0}
        ).sort("timestamp_end", -1).limit(limit))
    
    def get_sessions_by_user(
        self,
        user_id: str,
        domain: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            domain: Optional domain filter
            limit: Maximum sessions to return
        
        Returns:
            List of session documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"user_id": user_id}
        if domain:
            query["domain"] = domain
        
        return list(self.db[USER_SESSIONS_COLLECTION].find(
            query, {"_id": 0}
        ).sort("timestamp_end", -1).limit(limit))
    
    def get_model_deltas(
        self,
        model_id: str,
        use_case: str = None,
        domain: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get all deltas for a model across sessions.
        
        Args:
            model_id: Model identifier
            use_case: Optional use case filter
            domain: Optional domain filter
        
        Returns:
            List of delta documents for the model
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {f"deltas.{model_id}": {"$exists": True}}
        if use_case:
            query["use_case"] = use_case
        if domain:
            query["domain"] = domain
        
        sessions = self.db[USER_SESSIONS_COLLECTION].find(query)
        deltas = []
        for session in sessions:
            if model_id in session.get("deltas", {}):
                delta = session["deltas"][model_id]
                delta["session_id"] = session.get("session_id", session.get("_id"))
                delta["use_case"] = session.get("use_case")
                delta["domain"] = session.get("domain")
                deltas.append(delta)
        
        return deltas
    
    def get_aggregate_deltas_by_use_case(
        self,
        use_case: str,
        domain: str = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get aggregate delta statistics by use case.
        
        Args:
            use_case: Use case category
            domain: Optional domain filter
        
        Returns:
            {model_id: {"avg_delta": float, "count": int, "sum_delta": float}}
        """
        if not self.connected or self.db is None:
            return {}
        
        query: Dict[str, Any] = {"use_case": use_case}
        if domain:
            query["domain"] = domain
        
        sessions = list(self.db[USER_SESSIONS_COLLECTION].find(query))
        
        # Aggregate deltas by model
        model_stats: Dict[str, Dict[str, float]] = {}
        
        for session in sessions:
            for model_id, delta_data in session.get("deltas", {}).items():
                if model_id not in model_stats:
                    model_stats[model_id] = {"sum_delta": 0.0, "count": 0}
                
                model_stats[model_id]["sum_delta"] += delta_data.get("delta", 0.0)
                model_stats[model_id]["count"] += 1
        
        # Compute averages
        for model_id in model_stats:
            count = model_stats[model_id]["count"]
            if count > 0:
                model_stats[model_id]["avg_delta"] = (
                    model_stats[model_id]["sum_delta"] / count
                )
            else:
                model_stats[model_id]["avg_delta"] = 0.0
        
        return model_stats
    
    def get_session_count(
        self,
        use_case: str = None,
        domain: str = None
    ) -> int:
        """Get total session count with optional filters."""
        if not self.connected or self.db is None:
            return 0
        
        query: Dict[str, Any] = {}
        if use_case:
            query["use_case"] = use_case
        if domain:
            query["domain"] = domain
        
        return self.db[USER_SESSIONS_COLLECTION].count_documents(query)
    
    # =========================================================================
    # Judge Calibration Operations
    # =========================================================================
    
    def save_judge_calibration(self, calibration_data: Dict[str, Any]) -> bool:
        """
        Save or update judge calibration data.
        
        Args:
            calibration_data: Calibration document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            self.db[JUDGE_CALIBRATIONS_COLLECTION].update_one(
                {"_id": calibration_data.get("_id", calibration_data.get("judge_id"))},
                {"$set": calibration_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving judge calibration: {e}")
            return False
    
    def get_judge_calibration(
        self,
        judge_id: str,
        domain: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get calibration data for a judge.
        
        Args:
            judge_id: Judge identifier
            domain: Optional domain for domain-specific calibration
        
        Returns:
            Calibration document or None
        """
        if not self.connected or self.db is None:
            return None
        
        query_id = f"{judge_id}:{domain}" if domain else judge_id
        return self.db[JUDGE_CALIBRATIONS_COLLECTION].find_one(
            {"_id": query_id}
        )
    
    def get_all_judge_calibrations(
        self,
        domain: str = None
    ) -> List[Dict[str, Any]]:
        """Get all judge calibrations, optionally filtered by domain."""
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {}
        if domain:
            query["domain"] = domain
        
        return list(self.db[JUDGE_CALIBRATIONS_COLLECTION].find(query, {"_id": 0}))
    
    # =========================================================================
    # Parameter Management Operations
    # =========================================================================
    
    def _stringify_keys(self, obj: Any) -> Any:
        """Recursively convert all dict keys to strings (MongoDB requirement)."""
        if isinstance(obj, dict):
            return {str(k): self._stringify_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._stringify_keys(item) for item in obj]
        return obj
    
    def save_params(
        self,
        param_type: str,
        params: Dict[str, Any],
        domain: str = None
    ) -> bool:
        """
        Save or update parameters.
        
        Args:
            param_type: Parameter type identifier
            params: Parameter values to save
            domain: Optional domain for domain-specific override
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            # Create document ID
            doc_id = f"{param_type}:{domain}" if domain else param_type
            
            # Convert all keys to strings (MongoDB requirement)
            safe_params = self._stringify_keys(params)
            
            doc = {
                "_id": doc_id,
                "param_type": param_type,
                "domain": domain,
                "params": safe_params,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            
            self.db[PARAMETERS_COLLECTION].update_one(
                {"_id": doc_id},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving params: {e}")
            return False
    
    def get_params(
        self,
        param_type: str,
        domain: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get parameters for a type and optional domain.
        
        Args:
            param_type: Parameter type identifier
            domain: Optional domain for domain-specific params
        
        Returns:
            Parameter values dict or None
        """
        if not self.connected or self.db is None:
            return None
        
        doc_id = f"{param_type}:{domain}" if domain else param_type
        result = self.db[PARAMETERS_COLLECTION].find_one({"_id": doc_id})
        
        if result:
            return result.get("params")
        return None
    
    def list_param_overrides(self, param_type: str = None) -> List[str]:
        """
        List domains with parameter overrides.
        
        Args:
            param_type: Optional filter by parameter type
        
        Returns:
            List of domain names with overrides
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"domain": {"$ne": None}}
        if param_type:
            query["param_type"] = param_type
        
        results = self.db[PARAMETERS_COLLECTION].find(query, {"domain": 1})
        return [r["domain"] for r in results if r.get("domain")]
    
    def delete_param_override(self, param_type: str, domain: str) -> bool:
        """
        Delete a domain-specific parameter override.
        
        Args:
            param_type: Parameter type identifier
            domain: Domain name
        
        Returns:
            True if successfully deleted
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            doc_id = f"{param_type}:{domain}"
            result = self.db[PARAMETERS_COLLECTION].delete_one({"_id": doc_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting param override: {e}")
            return False
    
    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all parameters (global and domain-specific).
        
        Returns:
            Dictionary mapping param types to values
        """
        if not self.connected or self.db is None:
            return {}
        
        results = {}
        for doc in self.db[PARAMETERS_COLLECTION].find({}):
            key = doc["_id"]
            results[key] = {
                "param_type": doc.get("param_type"),
                "domain": doc.get("domain"),
                "params": doc.get("params"),
                "updated_at": doc.get("updated_at")
            }
        return results
    
    # =========================================================================
    # Phase 11: Base Model Operations
    # =========================================================================
    
    def save_base_model(self, base_model_data: Dict[str, Any]) -> bool:
        """
        Save or update a base model.
        
        Args:
            base_model_data: Base model document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            base_model_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[BASE_MODELS_COLLECTION].update_one(
                {"_id": base_model_data["_id"]},
                {"$set": base_model_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving base model: {e}")
            return False
    
    def get_base_model(self, base_model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get base model by ID.
        
        Args:
            base_model_id: Base model ID
        
        Returns:
            Base model document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[BASE_MODELS_COLLECTION].find_one({"_id": base_model_id})
    
    def get_all_base_models(
        self,
        family: str = None,
        organization: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all base models with optional filters.
        
        Args:
            family: Optional family filter
            organization: Optional organization filter
            limit: Maximum results
        
        Returns:
            List of base model documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {}
        if family:
            query["family"] = family
        if organization:
            query["organization"] = organization
        
        return list(self.db[BASE_MODELS_COLLECTION].find(query).limit(limit))
    
    def delete_base_model(self, base_model_id: str) -> bool:
        """Delete base model by ID."""
        if not self.connected or self.db is None:
            return False
        
        try:
            result = self.db[BASE_MODELS_COLLECTION].delete_one({"_id": base_model_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting base model: {e}")
            return False
    
    # =========================================================================
    # Phase 11: Model Deployment Operations
    # =========================================================================
    
    def save_deployment(self, deployment_data: Dict[str, Any]) -> bool:
        """
        Save or update a model deployment.
        
        Args:
            deployment_data: Deployment document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            deployment_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[MODEL_DEPLOYMENTS_COLLECTION].update_one(
                {"_id": deployment_data["_id"]},
                {"$set": deployment_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving deployment: {e}")
            return False
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get deployment by ID.
        
        Args:
            deployment_id: Deployment ID
        
        Returns:
            Deployment document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[MODEL_DEPLOYMENTS_COLLECTION].find_one({"_id": deployment_id})
    
    def get_deployments_by_base_model(
        self,
        base_model_id: str,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all deployments for a base model.
        
        Args:
            base_model_id: Base model ID
            include_inactive: Include deprecated/offline deployments
        
        Returns:
            List of deployment documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"base_model_id": base_model_id}
        if not include_inactive:
            query["availability.status"] = {"$in": ["active", "beta"]}
        
        return list(self.db[MODEL_DEPLOYMENTS_COLLECTION].find(query))
    
    def get_deployments_by_provider(
        self,
        provider_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all deployments for a provider.
        
        Args:
            provider_id: Provider ID
            limit: Maximum results
        
        Returns:
            List of deployment documents
        """
        if not self.connected or self.db is None:
            return []
        
        return list(
            self.db[MODEL_DEPLOYMENTS_COLLECTION].find(
                {"provider.id": provider_id}
            ).limit(limit)
        )
    
    def get_all_deployments(
        self,
        max_price: float = None,
        min_rating: float = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all deployments with optional filters.
        
        Args:
            max_price: Maximum input price per million
            min_rating: Minimum TrueSkill mu
            limit: Maximum results
        
        Returns:
            List of deployment documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"availability.status": {"$in": ["active", "beta"]}}
        if max_price is not None:
            query["pricing.input_cost_per_million"] = {"$lte": max_price}
        if min_rating is not None:
            query["trueskill.raw.mu"] = {"$gte": min_rating}
        
        return list(
            self.db[MODEL_DEPLOYMENTS_COLLECTION].find(query)
            .sort("trueskill.raw.mu", pymongo.DESCENDING)
            .limit(limit)
        )
    
    def get_best_deployment(
        self,
        base_model_id: str,
        domain: str = None,
        dimension: str = "cost_adjusted"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best deployment for a base model.
        
        Args:
            base_model_id: Base model ID
            domain: Optional domain for domain-specific ranking
            dimension: Rating dimension to sort by
        
        Returns:
            Best deployment document or None
        """
        if not self.connected or self.db is None:
            return None
        
        query = {
            "base_model_id": base_model_id,
            "availability.status": {"$in": ["active", "beta"]}
        }
        
        if domain:
            sort_key = f"trueskill_by_domain.{domain}.{dimension}.mu"
        else:
            sort_key = f"trueskill.{dimension}.mu"
        
        result = list(
            self.db[MODEL_DEPLOYMENTS_COLLECTION].find(query)
            .sort(sort_key, pymongo.DESCENDING)
            .limit(1)
        )
        
        return result[0] if result else None
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete deployment by ID."""
        if not self.connected or self.db is None:
            return False
        
        try:
            result = self.db[MODEL_DEPLOYMENTS_COLLECTION].delete_one(
                {"_id": deployment_id}
            )
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting deployment: {e}")
            return False
    
    # =========================================================================
    # Phase 11: Self-Hosted Deployment Operations
    # =========================================================================
    
    def save_self_hosted_deployment(
        self,
        deployment_data: Dict[str, Any]
    ) -> bool:
        """
        Save or update a self-hosted deployment.
        
        Args:
            deployment_data: Deployment document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            logger.error("Not connected to database")
            return False
        
        try:
            deployment_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].update_one(
                {"_id": deployment_data["_id"]},
                {"$set": deployment_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving self-hosted deployment: {e}")
            return False
    
    def get_self_hosted_deployment(
        self,
        deployment_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get self-hosted deployment by ID.
        
        Args:
            deployment_id: Deployment ID
        
        Returns:
            Deployment document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].find_one(
            {"_id": deployment_id}
        )
    
    def get_self_hosted_by_user(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all self-hosted deployments for a user.
        
        Args:
            user_id: Owner user ID
            limit: Maximum results
        
        Returns:
            List of deployment documents
        """
        if not self.connected or self.db is None:
            return []
        
        return list(
            self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].find(
                {"owner.user_id": user_id}
            ).limit(limit)
        )
    
    def get_public_self_hosted(
        self,
        base_model_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get public self-hosted deployments.
        
        Args:
            base_model_id: Optional filter by base model
            limit: Maximum results
        
        Returns:
            List of public deployment documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"owner.is_public": True}
        if base_model_id:
            query["base_model_id"] = base_model_id
        
        return list(
            self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].find(query).limit(limit)
        )
    
    def delete_self_hosted_deployment(
        self,
        deployment_id: str,
        user_id: str = None
    ) -> bool:
        """
        Delete self-hosted deployment by ID.
        
        Args:
            deployment_id: Deployment ID
            user_id: Optional user ID for ownership check
        
        Returns:
            True if deleted
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            query: Dict[str, Any] = {"_id": deployment_id}
            if user_id:
                query["owner.user_id"] = user_id
            
            result = self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting self-hosted deployment: {e}")
            return False
    
    # =========================================================================
    # Phase 11: Deployment Leaderboard
    # =========================================================================
    
    def get_deployment_leaderboard(
        self,
        domain: str = None,
        dimension: str = "raw",
        include_self_hosted: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of deployments (cloud + optionally self-hosted).
        
        Args:
            domain: Optional domain filter
            dimension: Rating dimension ("raw" or "cost_adjusted")
            include_self_hosted: Include public self-hosted deployments
            limit: Maximum entries
        
        Returns:
            List of deployment summaries sorted by rating
        """
        if not self.connected or self.db is None:
            return []
        
        results = []
        
        # Build sort key
        if domain:
            sort_key = f"trueskill_by_domain.{domain}.{dimension}.mu"
        else:
            sort_key = f"trueskill.{dimension}.mu"
        
        # Get cloud deployments
        cloud_query = {"availability.status": {"$in": ["active", "beta"]}}
        cloud = list(
            self.db[MODEL_DEPLOYMENTS_COLLECTION].find(cloud_query)
            .sort(sort_key, pymongo.DESCENDING)
            .limit(limit)
        )
        results.extend(cloud)
        
        # Get self-hosted if requested
        if include_self_hosted:
            self_hosted_query = {
                "owner.is_public": True,
                "availability.status": "active"
            }
            self_hosted = list(
                self.db[SELF_HOSTED_DEPLOYMENTS_COLLECTION].find(self_hosted_query)
                .sort(sort_key, pymongo.DESCENDING)
                .limit(limit)
            )
            results.extend(self_hosted)
        
        # Sort combined results
        def get_mu(doc):
            if domain and f"trueskill_by_domain" in doc:
                domain_ts = doc.get("trueskill_by_domain", {}).get(domain, {})
                return domain_ts.get(dimension, {}).get("mu", 0)
            return doc.get("trueskill", {}).get(dimension, {}).get("mu", 0)
        
        results.sort(key=get_mu, reverse=True)
        
        return results[:limit]
    
    # =========================================================================
    # Phase 14: Time Series Operations - Rating History
    # =========================================================================
    
    def _ensure_time_series_collections(self):
        """Create time series collections if they don't exist."""
        if not self.connected or self.db is None:
            return
        
        existing_collections = self.db.list_collection_names()
        
        # Model ratings history (1-year retention)
        if MODEL_RATINGS_HISTORY_COLLECTION not in existing_collections:
            try:
                self.db.create_collection(
                    MODEL_RATINGS_HISTORY_COLLECTION,
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "metadata",
                        "granularity": "minutes"
                    },
                    expireAfterSeconds=31536000  # 1 year
                )
                logger.info(f"Created time series collection: {MODEL_RATINGS_HISTORY_COLLECTION}")
            except Exception as e:
                # Collection may already exist or be a regular collection
                logger.debug(f"Could not create time series collection {MODEL_RATINGS_HISTORY_COLLECTION}: {e}")
        
        # Performance metrics (6-month retention)
        if PERFORMANCE_METRICS_COLLECTION not in existing_collections:
            try:
                self.db.create_collection(
                    PERFORMANCE_METRICS_COLLECTION,
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "metadata",
                        "granularity": "minutes"
                    },
                    expireAfterSeconds=15768000  # 6 months
                )
                logger.info(f"Created time series collection: {PERFORMANCE_METRICS_COLLECTION}")
            except Exception as e:
                logger.debug(f"Could not create time series collection {PERFORMANCE_METRICS_COLLECTION}: {e}")
        
        # Price history (no expiry)
        if PRICE_HISTORY_COLLECTION not in existing_collections:
            try:
                self.db.create_collection(
                    PRICE_HISTORY_COLLECTION,
                    timeseries={
                        "timeField": "effective_from",
                        "metaField": "metadata",
                        "granularity": "hours"
                    }
                )
                logger.info(f"Created time series collection: {PRICE_HISTORY_COLLECTION}")
            except Exception as e:
                logger.debug(f"Could not create time series collection {PRICE_HISTORY_COLLECTION}: {e}")
        
        # Source sync history (90-day retention)
        if SOURCE_SYNC_HISTORY_COLLECTION not in existing_collections:
            try:
                self.db.create_collection(
                    SOURCE_SYNC_HISTORY_COLLECTION,
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "metadata",
                        "granularity": "hours"
                    },
                    expireAfterSeconds=7776000  # 90 days
                )
                logger.info(f"Created time series collection: {SOURCE_SYNC_HISTORY_COLLECTION}")
            except Exception as e:
                logger.debug(f"Could not create time series collection {SOURCE_SYNC_HISTORY_COLLECTION}: {e}")
    
    def save_rating_history(
        self,
        deployment_id: str,
        domain: str,
        trueskill_raw: Dict[str, float],
        trueskill_cost: Dict[str, float],
        multi_trueskill: Dict[str, Any] = None,
        trigger: str = "match",
        trigger_id: str = "",
        stats: Dict[str, Any] = None
    ) -> bool:
        """
        Save a rating history point.
        
        Args:
            deployment_id: Deployment ID
            domain: Domain name
            trueskill_raw: Raw TrueSkill {mu, sigma}
            trueskill_cost: Cost-adjusted TrueSkill {mu, sigma}
            multi_trueskill: Optional 10D TrueSkill ratings
            trigger: What triggered this update ("match"/"feedback"/"transfer"/"calibration"/"manual")
            trigger_id: ID of the trigger (match_id, session_id, etc.)
            stats: Additional stats {matches_played, win_rate_last_10, avg_rank_last_10}
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            doc = {
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "deployment_id": deployment_id,
                    "domain": domain,
                },
                "trueskill_raw": trueskill_raw,
                "trueskill_cost": trueskill_cost,
                "trigger": trigger,
                "trigger_id": trigger_id,
            }
            
            if multi_trueskill:
                doc["multi_trueskill"] = multi_trueskill
            if stats:
                doc["stats"] = stats
            
            self.db[MODEL_RATINGS_HISTORY_COLLECTION].insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving rating history: {e}")
            return False
    
    def get_rating_history(
        self,
        deployment_id: str,
        domain: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get rating history for a deployment.
        
        Args:
            deployment_id: Deployment ID
            domain: Optional domain filter
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum records
        
        Returns:
            List of rating history documents
        """
        if not self.connected or self.db is None:
            return []
        
        query = {"metadata.deployment_id": deployment_id}
        if domain:
            query["metadata.domain"] = domain
        if start_date:
            query["timestamp"] = {"$gte": start_date}
        if end_date:
            if "timestamp" in query:
                query["timestamp"]["$lte"] = end_date
            else:
                query["timestamp"] = {"$lte": end_date}
        
        return list(
            self.db[MODEL_RATINGS_HISTORY_COLLECTION].find(query, {"_id": 0})
            .sort("timestamp", pymongo.DESCENDING)
            .limit(limit)
        )
    
    def get_rating_at_time(
        self,
        deployment_id: str,
        domain: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get the rating for a deployment at a specific point in time.
        
        Args:
            deployment_id: Deployment ID
            domain: Domain name
            timestamp: Point in time
        
        Returns:
            Rating document or None
        """
        if not self.connected or self.db is None:
            return None
        
        # Find the most recent rating before or at the given timestamp
        query = {
            "metadata.deployment_id": deployment_id,
            "metadata.domain": domain,
            "timestamp": {"$lte": timestamp}
        }
        
        result = list(
            self.db[MODEL_RATINGS_HISTORY_COLLECTION].find(query)
            .sort("timestamp", pymongo.DESCENDING)
            .limit(1)
        )
        
        return result[0] if result else None
    
    # =========================================================================
    # Phase 14: Time Series Operations - Performance Metrics
    # =========================================================================
    
    def save_performance_metrics(
        self,
        deployment_id: str,
        domain: str,
        match_id: str,
        latency_ms: float,
        ttft_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        raw_rank: int = 0,
        cost_rank: int = 0,
        dimension_ranks: Dict[str, int] = None,
        consistency_score: float = None,
        constraints_satisfied: int = None,
        hallucination_count: int = None
    ) -> bool:
        """
        Save performance metrics from a match.
        
        Args:
            deployment_id: Deployment ID
            domain: Domain name
            match_id: Match ID
            latency_ms: Total latency in milliseconds
            ttft_ms: Time to first token in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Total cost in USD
            raw_rank: Rank in raw quality
            cost_rank: Rank in cost-adjusted
            dimension_ranks: Ranks for all 10 dimensions
            consistency_score: Consistency score if computed
            constraints_satisfied: Constraints satisfied count
            hallucination_count: Detected hallucinations
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            doc = {
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "deployment_id": deployment_id,
                    "domain": domain,
                },
                "match_id": match_id,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "raw_rank": raw_rank,
                "cost_rank": cost_rank,
            }
            
            if dimension_ranks:
                doc["dimension_ranks"] = dimension_ranks
            if consistency_score is not None:
                doc["consistency_score"] = consistency_score
            if constraints_satisfied is not None:
                doc["constraints_satisfied"] = constraints_satisfied
            if hallucination_count is not None:
                doc["hallucination_count"] = hallucination_count
            
            self.db[PERFORMANCE_METRICS_COLLECTION].insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            return False
    
    def get_performance_metrics(
        self,
        deployment_id: str,
        domain: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics for a deployment.
        
        Args:
            deployment_id: Deployment ID
            domain: Optional domain filter
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum records
        
        Returns:
            List of performance metrics documents
        """
        if not self.connected or self.db is None:
            return []
        
        query = {"metadata.deployment_id": deployment_id}
        if domain:
            query["metadata.domain"] = domain
        if start_date:
            query["timestamp"] = {"$gte": start_date}
        if end_date:
            if "timestamp" in query:
                query["timestamp"]["$lte"] = end_date
            else:
                query["timestamp"] = {"$lte": end_date}
        
        return list(
            self.db[PERFORMANCE_METRICS_COLLECTION].find(query, {"_id": 0})
            .sort("timestamp", pymongo.DESCENDING)
            .limit(limit)
        )
    
    # =========================================================================
    # Phase 14: Time Series Operations - Price History
    # =========================================================================
    
    def save_price_history(
        self,
        deployment_id: str,
        provider: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
        previous_input_cost: float = None,
        previous_output_cost: float = None,
        source_id: str = "",
        source_url: str = "",
        price_hash: str = ""
    ) -> bool:
        """
        Save a price history entry.
        
        Args:
            deployment_id: Deployment ID
            provider: Provider name
            input_cost_per_million: Current input cost
            output_cost_per_million: Current output cost
            previous_input_cost: Previous input cost (if known)
            previous_output_cost: Previous output cost (if known)
            source_id: Data source ID
            source_url: Source URL
            price_hash: Hash for change detection
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            doc = {
                "effective_from": datetime.utcnow(),
                "metadata": {
                    "deployment_id": deployment_id,
                    "provider": provider,
                },
                "input_cost_per_million": input_cost_per_million,
                "output_cost_per_million": output_cost_per_million,
                "source_id": source_id,
                "source_url": source_url,
                "price_hash": price_hash,
            }
            
            if previous_input_cost is not None:
                doc["previous_input_cost"] = previous_input_cost
                if previous_input_cost != 0:
                    doc["change_pct_input"] = ((input_cost_per_million - previous_input_cost) / previous_input_cost) * 100
            
            if previous_output_cost is not None:
                doc["previous_output_cost"] = previous_output_cost
                if previous_output_cost != 0:
                    doc["change_pct_output"] = ((output_cost_per_million - previous_output_cost) / previous_output_cost) * 100
            
            self.db[PRICE_HISTORY_COLLECTION].insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving price history: {e}")
            return False
    
    def get_price_history(
        self,
        deployment_id: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get price history for a deployment.
        
        Args:
            deployment_id: Deployment ID
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum records
        
        Returns:
            List of price history documents
        """
        if not self.connected or self.db is None:
            return []
        
        query = {"metadata.deployment_id": deployment_id}
        if start_date:
            query["effective_from"] = {"$gte": start_date}
        if end_date:
            if "effective_from" in query:
                query["effective_from"]["$lte"] = end_date
            else:
                query["effective_from"] = {"$lte": end_date}
        
        return list(
            self.db[PRICE_HISTORY_COLLECTION].find(query, {"_id": 0})
            .sort("effective_from", pymongo.DESCENDING)
            .limit(limit)
        )
    
    # =========================================================================
    # Phase 13: Sync History Operations
    # =========================================================================
    
    def save_sync_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Save sync result to history.
        
        Args:
            result_data: Sync result document
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            doc = {
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "source_id": result_data.get("source_id", ""),
                    "provider": result_data.get("provider", ""),
                },
                "status": result_data.get("status", ""),
                "duration_ms": result_data.get("duration_ms", 0),
                "models_found": result_data.get("models_found", 0),
                "models_updated": result_data.get("models_updated", 0),
                "models_added": result_data.get("models_added", 0),
                "prices_changed": result_data.get("prices_changed", 0),
                "error": result_data.get("error"),
            }
            
            self.db[SOURCE_SYNC_HISTORY_COLLECTION].insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving sync result: {e}")
            return False
    
    def get_sync_history(
        self,
        source_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get sync history for a data source.
        
        Args:
            source_id: Data source ID
            days: Number of days to look back
        
        Returns:
            List of sync history documents
        """
        if not self.connected or self.db is None:
            return []
        
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return list(
            self.db[SOURCE_SYNC_HISTORY_COLLECTION].find(
                {
                    "metadata.source_id": source_id,
                    "timestamp": {"$gte": cutoff}
                },
                {"_id": 0}
            ).sort("timestamp", pymongo.DESCENDING)
        )
    
    # =========================================================================
    # Phase 16: Content Hashes Operations
    # =========================================================================
    
    def check_content_duplicate(
        self,
        content_hash: str,
        content_type: str = "case",
        domain: str = ""
    ) -> bool:
        """
        Check if content hash exists.
        
        Args:
            content_hash: The content hash to check
            content_type: Type of content ("case", "question", "response")
            domain: Domain name
        
        Returns:
            True if duplicate exists
        """
        if not self.connected or self.db is None:
            return False
        
        query = {
            "_id": content_hash,
            "content_type": content_type,
        }
        if domain:
            query["domain"] = domain
        
        return self.db[CONTENT_HASHES_COLLECTION].count_documents(query) > 0
    
    def register_content(
        self,
        content_hash: str,
        content_type: str = "case",
        domain: str = "",
        match_id: str = "",
        generator_model_id: str = "",
        content_preview: str = ""
    ) -> bool:
        """
        Register a content hash.
        
        Args:
            content_hash: The content hash
            content_type: Type of content ("case", "question", "response")
            domain: Domain name
            match_id: Match ID where this was first used
            generator_model_id: Model that generated this content
            content_preview: First N characters of content
        
        Returns:
            True if registered (or already exists)
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            now = datetime.utcnow().isoformat() + "Z"
            
            # Upsert: increment usage count if exists, create if not
            self.db[CONTENT_HASHES_COLLECTION].update_one(
                {"_id": content_hash},
                {
                    "$setOnInsert": {
                        "content_type": content_type,
                        "domain": domain,
                        "first_seen": now,
                        "first_match_id": match_id,
                        "generator_model_id": generator_model_id,
                        "content_preview": content_preview[:200] if content_preview else "",
                    },
                    "$inc": {"usage_count": 1},
                    "$set": {"last_used": now},
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error registering content: {e}")
            return False
    
    def get_content_hash_info(
        self,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a content hash.
        
        Args:
            content_hash: The content hash
        
        Returns:
            Content hash document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[CONTENT_HASHES_COLLECTION].find_one({"_id": content_hash})
    
    # =========================================================================
    # Phase 17-24: Ground Truth Sample Operations
    # =========================================================================
    
    def save_ground_truth_sample(self, sample_data: Dict[str, Any]) -> bool:
        """
        Save or update a ground truth sample.
        
        Args:
            sample_data: Sample document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            sample_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            if "created_at" not in sample_data:
                sample_data["created_at"] = sample_data["updated_at"]
            if "usage_count" not in sample_data:
                sample_data["usage_count"] = 0
            
            self.db[GROUND_TRUTH_SAMPLES_COLLECTION].update_one(
                {"_id": sample_data["_id"]},
                {"$set": sample_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving ground truth sample: {e}")
            return False
    
    def get_ground_truth_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get ground truth sample by ID."""
        if not self.connected or self.db is None:
            return None
        
        return self.db[GROUND_TRUTH_SAMPLES_COLLECTION].find_one({"_id": sample_id})
    
    def get_ground_truth_samples(
        self,
        domain: str,
        difficulty: str = None,
        limit: int = 100,
        exclude_recently_used_days: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get ground truth samples for a domain.
        
        Args:
            domain: Domain name
            difficulty: Optional difficulty filter
            limit: Maximum samples to return
            exclude_recently_used_days: Exclude samples used within N days
        
        Returns:
            List of sample documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"domain": domain}
        if difficulty:
            query["difficulty"] = difficulty
        
        if exclude_recently_used_days > 0:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=exclude_recently_used_days)
            query["$or"] = [
                {"last_used": {"$lt": cutoff.isoformat() + "Z"}},
                {"last_used": {"$exists": False}}
            ]
        
        return list(
            self.db[GROUND_TRUTH_SAMPLES_COLLECTION].find(query)
            .sort("usage_count", pymongo.ASCENDING)
            .limit(limit)
        )
    
    def get_random_ground_truth_sample(
        self,
        domain: str,
        filters: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a random ground truth sample for a domain.
        
        Args:
            domain: Domain name
            filters: Additional query filters
        
        Returns:
            Random sample document or None
        """
        if not self.connected or self.db is None:
            return None
        
        query = {"domain": domain}
        if filters:
            query.update(filters)
        
        result = list(
            self.db[GROUND_TRUTH_SAMPLES_COLLECTION].aggregate([
                {"$match": query},
                {"$sample": {"size": 1}}
            ])
        )
        
        return result[0] if result else None
    
    def update_sample_usage(self, sample_id: str) -> bool:
        """Mark a sample as used."""
        if not self.connected or self.db is None:
            return False
        
        try:
            self.db[GROUND_TRUTH_SAMPLES_COLLECTION].update_one(
                {"_id": sample_id},
                {
                    "$set": {"last_used": datetime.utcnow().isoformat() + "Z"},
                    "$inc": {"usage_count": 1}
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating sample usage: {e}")
            return False
    
    def get_ground_truth_sample_count(self, domain: str = None) -> int:
        """Get total sample count."""
        if not self.connected or self.db is None:
            return 0
        
        query = {"domain": domain} if domain else {}
        return self.db[GROUND_TRUTH_SAMPLES_COLLECTION].count_documents(query)
    
    # =========================================================================
    # Phase 17-24: Ground Truth Match Operations
    # =========================================================================
    
    def save_ground_truth_match(self, match_data: Dict[str, Any]) -> bool:
        """
        Save ground truth match result.
        
        Args:
            match_data: Match document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            self.db[GROUND_TRUTH_MATCHES_COLLECTION].insert_one(match_data)
            return True
        except Exception as e:
            logger.error(f"Error saving ground truth match: {e}")
            return False
    
    def get_ground_truth_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get ground truth match by ID."""
        if not self.connected or self.db is None:
            return None
        
        return self.db[GROUND_TRUTH_MATCHES_COLLECTION].find_one({"_id": match_id})
    
    def get_ground_truth_matches(
        self,
        domain: str,
        deployment_id: str = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get ground truth matches for a domain.
        
        Args:
            domain: Domain name
            deployment_id: Optional deployment filter
            limit: Maximum matches to return
            skip: Number to skip
        
        Returns:
            List of match documents
        """
        if not self.connected or self.db is None:
            return []
        
        query: Dict[str, Any] = {"domain": domain}
        if deployment_id:
            query["participants"] = deployment_id
        
        return list(
            self.db[GROUND_TRUTH_MATCHES_COLLECTION].find(query)
            .sort("timestamp", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
    
    def get_ground_truth_match_count(
        self,
        domain: str = None,
        deployment_id: str = None
    ) -> int:
        """Get ground truth match count."""
        if not self.connected or self.db is None:
            return 0
        
        query: Dict[str, Any] = {}
        if domain:
            query["domain"] = domain
        if deployment_id:
            query["participants"] = deployment_id
        
        return self.db[GROUND_TRUTH_MATCHES_COLLECTION].count_documents(query)
    
    # =========================================================================
    # Phase 17-24: Ground Truth Rating Operations
    # =========================================================================
    
    def save_ground_truth_rating(self, rating_data: Dict[str, Any]) -> bool:
        """
        Save or update ground truth rating.
        
        Args:
            rating_data: Rating document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            rating_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            if "created_at" not in rating_data:
                rating_data["created_at"] = rating_data["updated_at"]
            
            self.db[GROUND_TRUTH_RATINGS_COLLECTION].update_one(
                {"_id": rating_data["_id"]},
                {"$set": rating_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving ground truth rating: {e}")
            return False
    
    def get_ground_truth_rating(
        self,
        deployment_id: str,
        domain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get ground truth rating for a deployment in a domain.
        
        Args:
            deployment_id: Deployment ID
            domain: Domain name
        
        Returns:
            Rating document or None
        """
        if not self.connected or self.db is None:
            return None
        
        rating_id = f"{deployment_id}:{domain}"
        return self.db[GROUND_TRUTH_RATINGS_COLLECTION].find_one({"_id": rating_id})
    
    def get_ground_truth_leaderboard(
        self,
        domain: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get ground truth leaderboard for a domain.
        
        Args:
            domain: Domain name
            limit: Maximum entries
        
        Returns:
            List of rating documents sorted by TrueSkill mu
        """
        if not self.connected or self.db is None:
            return []
        
        return list(
            self.db[GROUND_TRUTH_RATINGS_COLLECTION].find({"domain": domain})
            .sort("trueskill.mu", pymongo.DESCENDING)
            .limit(limit)
        )
    
    def get_ground_truth_leaderboard_by_language(
        self,
        domain: str,
        language: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get language-specific ground truth leaderboard.
        
        Args:
            domain: Domain name
            language: Language code (e.g., "en", "hi", "bn")
            limit: Maximum entries
        
        Returns:
            List of rating documents filtered by language, sorted by TrueSkill mu
        """
        if not self.connected or self.db is None:
            return []
        
        # Language-specific ratings have IDs like "deployment_id:domain:language"
        # or have a "language" field
        query = {
            "domain": domain,
            "$or": [
                {"language": language},
                {"_id": {"$regex": f":{language}$"}}
            ]
        }
        
        return list(
            self.db[GROUND_TRUTH_RATINGS_COLLECTION].find(query)
            .sort("trueskill.mu", pymongo.DESCENDING)
            .limit(limit)
        )
    
    # =========================================================================
    # Phase 17-24: Ground Truth Coverage Operations
    # =========================================================================
    
    def save_ground_truth_coverage(self, coverage_data: Dict[str, Any]) -> bool:
        """
        Save ground truth coverage statistics.
        
        Args:
            coverage_data: Coverage document to save
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            coverage_data["last_computed"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[GROUND_TRUTH_COVERAGE_COLLECTION].update_one(
                {"_id": coverage_data["_id"]},
                {"$set": coverage_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving ground truth coverage: {e}")
            return False
    
    def get_ground_truth_coverage(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get ground truth coverage for a domain."""
        if not self.connected or self.db is None:
            return None
        
        return self.db[GROUND_TRUTH_COVERAGE_COLLECTION].find_one({"_id": domain})
    
    def compute_ground_truth_coverage(self, domain: str) -> Dict[str, Any]:
        """
        Compute and save coverage statistics for a domain.
        
        Args:
            domain: Domain name
        
        Returns:
            Coverage statistics
        """
        if not self.connected or self.db is None:
            return {}
        
        total = self.db[GROUND_TRUTH_SAMPLES_COLLECTION].count_documents(
            {"domain": domain}
        )
        used = self.db[GROUND_TRUTH_SAMPLES_COLLECTION].count_documents(
            {"domain": domain, "usage_count": {"$gt": 0}}
        )
        
        coverage = {
            "_id": domain,
            "domain": domain,
            "total_samples": total,
            "used_samples": used,
            "coverage_percentage": (used / total * 100) if total > 0 else 0,
        }
        
        self.save_ground_truth_coverage(coverage)
        return coverage
    
    # =========================================================================
    # Phase 12: Benchmark Operations
    # =========================================================================
    
    def get_all_benchmark_definitions(
        self,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all benchmark definitions.
        
        Args:
            filters: Optional query filters
        
        Returns:
            List of benchmark definition documents
        """
        if not self.connected or self.db is None:
            return []
        
        query = filters or {}
        return list(self.db[BENCHMARK_DEFINITIONS_COLLECTION].find(query))
    
    def get_benchmark_definition(
        self,
        benchmark_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a benchmark definition by ID.
        
        Args:
            benchmark_id: Benchmark ID
        
        Returns:
            Benchmark definition document or None
        """
        if not self.connected or self.db is None:
            return None
        
        return self.db[BENCHMARK_DEFINITIONS_COLLECTION].find_one({"_id": benchmark_id})
    
    def save_benchmark_definition(
        self,
        benchmark_data: Dict[str, Any]
    ) -> bool:
        """
        Save or update a benchmark definition.
        
        Args:
            benchmark_data: Benchmark definition document
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            benchmark_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.db[BENCHMARK_DEFINITIONS_COLLECTION].update_one(
                {"_id": benchmark_data["_id"]},
                {"$set": benchmark_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving benchmark definition: {e}")
            return False
    
    def delete_benchmark_definition(
        self,
        benchmark_id: str
    ) -> bool:
        """
        Delete a benchmark definition.
        
        Args:
            benchmark_id: Benchmark ID
        
        Returns:
            True if deleted
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            result = self.db[BENCHMARK_DEFINITIONS_COLLECTION].delete_one(
                {"_id": benchmark_id}
            )
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting benchmark definition: {e}")
            return False
    
    def get_benchmark_results(
        self,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark results.
        
        Args:
            filters: Query filters
        
        Returns:
            List of benchmark result documents
        """
        if not self.connected or self.db is None:
            return []
        
        query = filters or {}
        return list(self.db[BENCHMARK_RESULTS_COLLECTION].find(query))
    
    def save_benchmark_result(
        self,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Save a benchmark result.
        
        Args:
            result_data: Benchmark result document
        
        Returns:
            True if successful
        """
        if not self.connected or self.db is None:
            return False
        
        try:
            self.db[BENCHMARK_RESULTS_COLLECTION].insert_one(result_data)
            return True
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
            return False
    
    def get_benchmark_correlations(
        self,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark correlations with LangScope ratings.
        
        Args:
            filters: Query filters
        
        Returns:
            List of correlation documents
        """
        if not self.connected or self.db is None:
            return []
        
        # This would be stored in a separate collection if implemented
        # For now, return empty list
        return []


# =============================================================================
# Global Database Instance
# =============================================================================

_db_instance: Optional[MongoDB] = None


def get_database() -> Optional[MongoDB]:
    """Get the global database instance."""
    return _db_instance


def initialize_database(
    connection_string: str,
    db_name: str = "langscope"
) -> MongoDB:
    """
    Initialize the global database instance.
    
    Args:
        connection_string: MongoDB connection URI
        db_name: Database name
    
    Returns:
        MongoDB instance
    """
    global _db_instance
    _db_instance = MongoDB(connection_string, db_name)
    return _db_instance


