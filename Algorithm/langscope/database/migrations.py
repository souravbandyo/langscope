"""
Database migrations for LangScope.

Handles schema versioning and data migrations between versions.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Current schema version
CURRENT_VERSION = "0.1.0"

# Migration registry
_migrations: Dict[str, Callable] = {}


def migration(from_version: str, to_version: str):
    """
    Decorator to register a migration function.
    
    Args:
        from_version: Source version
        to_version: Target version
    """
    def decorator(func: Callable):
        _migrations[f"{from_version}->{to_version}"] = func
        return func
    return decorator


class MigrationManager:
    """
    Manages database schema migrations.
    
    Tracks schema versions and applies necessary migrations
    to bring the database up to the current version.
    """
    
    def __init__(self, db):
        """
        Initialize migration manager.
        
        Args:
            db: MongoDB instance
        """
        self.db = db
        self.meta_collection = "schema_metadata"
    
    def get_current_version(self) -> str:
        """Get current database schema version."""
        if not self.db.connected or not self.db.db:
            return "0.0.0"
        
        meta = self.db.db[self.meta_collection].find_one({"_id": "version"})
        if meta:
            return meta.get("version", "0.0.0")
        return "0.0.0"
    
    def set_version(self, version: str):
        """Set database schema version."""
        if not self.db.connected or not self.db.db:
            return
        
        self.db.db[self.meta_collection].update_one(
            {"_id": "version"},
            {
                "$set": {
                    "version": version,
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                }
            },
            upsert=True
        )
    
    def needs_migration(self) -> bool:
        """Check if database needs migration."""
        current = self.get_current_version()
        return current != CURRENT_VERSION
    
    def get_migration_path(self) -> List[str]:
        """
        Get ordered list of migrations to apply.
        
        Returns:
            List of migration keys ("from->to")
        """
        current = self.get_current_version()
        if current == CURRENT_VERSION:
            return []
        
        # Build migration path
        path = []
        version = current
        
        # Simple linear version path for now
        versions = ["0.0.0", "0.1.0"]
        
        try:
            current_idx = versions.index(version)
            target_idx = versions.index(CURRENT_VERSION)
            
            for i in range(current_idx, target_idx):
                migration_key = f"{versions[i]}->{versions[i+1]}"
                if migration_key in _migrations:
                    path.append(migration_key)
        except ValueError:
            logger.warning(f"Unknown version: {version}")
        
        return path
    
    def apply_migrations(self, dry_run: bool = False) -> bool:
        """
        Apply all pending migrations.
        
        Args:
            dry_run: If True, only log what would be done
        
        Returns:
            True if successful
        """
        path = self.get_migration_path()
        
        if not path:
            logger.info("No migrations needed")
            return True
        
        logger.info(f"Applying {len(path)} migrations...")
        
        for migration_key in path:
            if dry_run:
                logger.info(f"[DRY RUN] Would apply: {migration_key}")
                continue
            
            try:
                logger.info(f"Applying migration: {migration_key}")
                migration_func = _migrations[migration_key]
                migration_func(self.db)
                
                # Update version after successful migration
                to_version = migration_key.split("->")[1]
                self.set_version(to_version)
                
                logger.info(f"Migration {migration_key} completed")
            except Exception as e:
                logger.error(f"Migration {migration_key} failed: {e}")
                return False
        
        return True


# =============================================================================
# Migration Functions
# =============================================================================

@migration("0.0.0", "0.1.0")
def migrate_0_0_0_to_0_1_0(db):
    """
    Initial migration - set up base schema.
    
    This migration:
    1. Ensures all collections exist
    2. Creates required indexes
    3. Sets default values for missing fields
    """
    if not db.connected or not db.db:
        raise RuntimeError("Database not connected")
    
    # Ensure collections exist
    collections = db.db.list_collection_names()
    
    required_collections = ["models", "matches", "domains", "domain_correlations"]
    for collection in required_collections:
        if collection not in collections:
            db.db.create_collection(collection)
            logger.info(f"Created collection: {collection}")
    
    # Update existing models to have required fields
    db.db["models"].update_many(
        {"trueskill": {"$exists": False}},
        {
            "$set": {
                "trueskill": {
                    "raw": {"mu": 1500.0, "sigma": 166.0},
                    "cost_adjusted": {"mu": 1500.0, "sigma": 166.0}
                },
                "trueskill_by_domain": {},
                "performance": {
                    "total_matches_played": 0,
                    "total_races_participated": 0,
                    "avg_rank_raw": 0.0,
                    "avg_rank_cost": 0.0,
                    "rank_history_raw": [],
                    "rank_history_cost": [],
                    "total_tokens_used": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0.0
                },
                "performance_by_domain": {},
                "match_ids": {
                    "played": [],
                    "judged": [],
                    "cases_generated": [],
                    "questions_generated": []
                },
                "match_ids_by_domain": {},
                "metadata": {
                    "notes": "",
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                    "domains_evaluated": []
                }
            }
        }
    )
    
    logger.info("Migration 0.0.0 -> 0.1.0 completed")


# =============================================================================
# Utility Functions
# =============================================================================

def run_migrations(db, dry_run: bool = False) -> bool:
    """
    Convenience function to run all migrations.
    
    Args:
        db: MongoDB instance
        dry_run: If True, only log what would be done
    
    Returns:
        True if successful
    """
    manager = MigrationManager(db)
    return manager.apply_migrations(dry_run)


def check_migration_status(db) -> Dict[str, Any]:
    """
    Check migration status.
    
    Args:
        db: MongoDB instance
    
    Returns:
        Status dictionary
    """
    manager = MigrationManager(db)
    
    return {
        "current_version": manager.get_current_version(),
        "target_version": CURRENT_VERSION,
        "needs_migration": manager.needs_migration(),
        "pending_migrations": manager.get_migration_path()
    }


