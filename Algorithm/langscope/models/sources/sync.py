"""
Sync engine for automated data fetching.

Handles syncing from external data sources with change detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import hashlib
import json

from langscope.models.sources.data_sources import DataSource

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Sync result status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    NO_CHANGES = "no_changes"


@dataclass
class SyncResult:
    """Result of a sync operation."""
    source_id: str
    status: SyncStatus
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    duration_ms: int = 0
    
    # Counts
    models_found: int = 0
    models_updated: int = 0
    models_added: int = 0
    models_removed: int = 0
    prices_changed: int = 0
    
    # Changes requiring approval
    changes_pending_approval: int = 0
    
    # Error info
    error: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "models_found": self.models_found,
            "models_updated": self.models_updated,
            "models_added": self.models_added,
            "models_removed": self.models_removed,
            "prices_changed": self.prices_changed,
            "changes_pending_approval": self.changes_pending_approval,
            "error": self.error,
            "error_details": self.error_details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncResult':
        """Create from dictionary."""
        status = data.get("status", "failed")
        if isinstance(status, str):
            try:
                status = SyncStatus(status)
            except ValueError:
                status = SyncStatus.FAILED
        
        return cls(
            source_id=data.get("source_id", ""),
            status=status,
            timestamp=data.get("timestamp", ""),
            duration_ms=data.get("duration_ms", 0),
            models_found=data.get("models_found", 0),
            models_updated=data.get("models_updated", 0),
            models_added=data.get("models_added", 0),
            models_removed=data.get("models_removed", 0),
            prices_changed=data.get("prices_changed", 0),
            changes_pending_approval=data.get("changes_pending_approval", 0),
            error=data.get("error"),
            error_details=data.get("error_details"),
        )


class SyncEngine:
    """
    Engine for syncing data from external sources.
    
    Handles fetching, parsing, change detection, and applying updates.
    """
    
    def __init__(self, db=None):
        """
        Initialize sync engine.
        
        Args:
            db: MongoDB instance for persistence
        """
        self.db = db
        self._http_client = None
    
    def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=30.0)
            except ImportError:
                logger.warning("httpx not installed, sync will not work")
                return None
        return self._http_client
    
    def compute_hash(self, data: Any) -> str:
        """
        Compute hash of data for change detection.
        
        Args:
            data: Data to hash
        
        Returns:
            SHA256 hash string
        """
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def run_sync(self, source: DataSource) -> SyncResult:
        """
        Run sync for a single data source.
        
        Args:
            source: Data source to sync
        
        Returns:
            SyncResult with operation details
        """
        start_time = datetime.utcnow()
        
        try:
            # Fetch data
            data = await self._fetch_data(source)
            if data is None:
                return SyncResult(
                    source_id=source.id,
                    status=SyncStatus.FAILED,
                    error="Failed to fetch data",
                )
            
            # Parse and detect changes
            changes = self._detect_changes(source, data)
            
            if not changes:
                source.reliability.update_success()
                return SyncResult(
                    source_id=source.id,
                    status=SyncStatus.NO_CHANGES,
                    models_found=len(data) if isinstance(data, list) else 1,
                    duration_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                )
            
            # Apply changes (or queue for approval)
            result = await self._apply_changes(source, changes)
            
            result.duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            source.reliability.update_success()
            
            return result
            
        except Exception as e:
            logger.error(f"Sync failed for {source.id}: {e}")
            source.reliability.update_failure(str(e))
            
            return SyncResult(
                source_id=source.id,
                status=SyncStatus.FAILED,
                error=str(e),
                duration_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            )
    
    async def _fetch_data(self, source: DataSource) -> Optional[Any]:
        """Fetch data from source."""
        client = self._get_http_client()
        if client is None:
            return None
        
        try:
            # Build request
            url = source.source.url
            headers = dict(source.source.headers)
            
            # Add auth if configured
            if source.source.auth_type == "bearer" and source.source.auth_env_var:
                import os
                token = os.getenv(source.source.auth_env_var)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            
            # Make request
            if source.source.method.value == "GET":
                response = client.get(url, headers=headers, params=source.source.params)
            else:
                response = client.post(url, headers=headers, json=source.source.params)
            
            response.raise_for_status()
            
            # Parse response
            if source.source.response_format == "json":
                data = response.json()
                
                # Extract data from path if specified
                if source.source.data_path:
                    for key in source.source.data_path.split("."):
                        data = data.get(key, {})
                
                return data
            
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source.id}: {e}")
            return None
    
    def _detect_changes(
        self,
        source: DataSource,
        new_data: Any
    ) -> List[Dict[str, Any]]:
        """
        Detect changes between new data and existing data.
        
        Returns list of change dictionaries.
        """
        changes = []
        
        if not isinstance(new_data, list):
            new_data = [new_data]
        
        for item in new_data:
            # Map fields
            mapped = {}
            for our_field, their_field in source.field_mappings.items():
                value = item
                for key in their_field.split("."):
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        value = None
                        break
                mapped[our_field] = value
            
            if not mapped.get("model_id"):
                continue
            
            # TODO: Compare with existing data from DB
            # For now, mark all as potential updates
            changes.append({
                "type": "update",
                "model_id": mapped.get("model_id"),
                "data": mapped,
                "hash": self.compute_hash(mapped),
            })
        
        return changes
    
    async def _apply_changes(
        self,
        source: DataSource,
        changes: List[Dict[str, Any]]
    ) -> SyncResult:
        """Apply detected changes."""
        result = SyncResult(
            source_id=source.id,
            status=SyncStatus.SUCCESS,
            models_found=len(changes),
        )
        
        for change in changes:
            change_type = change.get("type")
            data = change.get("data", {})
            
            if change_type == "add":
                result.models_added += 1
                # Persist new model/deployment
                if self.db and hasattr(self.db, 'save_deployment'):
                    deployment_data = self._map_to_deployment(data, source)
                    if deployment_data:
                        self.db.save_deployment(deployment_data)
                        
            elif change_type == "update":
                result.models_updated += 1
                
                # Check if price changed and save to price history
                if "input_cost" in data or "output_cost" in data:
                    result.prices_changed += 1
                    
                    if self.db and hasattr(self.db, 'save_price_history'):
                        from langscope.models.hashing import price_hash
                        
                        model_id = data.get("model_id", "")
                        input_cost = data.get("input_cost", data.get("input_cost_per_million", 0))
                        output_cost = data.get("output_cost", data.get("output_cost_per_million", 0))
                        
                        self.db.save_price_history(
                            deployment_id=model_id,
                            provider=source.provider,
                            input_cost_per_million=input_cost,
                            output_cost_per_million=output_cost,
                            source_id=source.id,
                            source_url=source.source.url,
                            price_hash=price_hash(model_id, input_cost, output_cost),
                        )
                
                # Update deployment data
                if self.db and hasattr(self.db, 'save_deployment'):
                    deployment_data = self._map_to_deployment(data, source)
                    if deployment_data:
                        self.db.save_deployment(deployment_data)
                        
            elif change_type == "remove":
                result.models_removed += 1
        
        # Save sync result to history
        if self.db and hasattr(self.db, 'save_sync_result'):
            result_data = result.to_dict()
            result_data["provider"] = source.provider
            self.db.save_sync_result(result_data)
        
        return result
    
    def _map_to_deployment(
        self,
        data: Dict[str, Any],
        source: DataSource
    ) -> Optional[Dict[str, Any]]:
        """
        Map synced data to deployment format.
        
        Args:
            data: Raw synced data
            source: Data source
        
        Returns:
            Deployment document or None
        """
        model_id = data.get("model_id")
        if not model_id:
            return None
        
        # Create a basic deployment document
        deployment = {
            "_id": f"{source.provider}/{model_id}",
            "base_model_id": data.get("base_model_id", model_id),
            "provider": {
                "id": source.provider,
                "name": source.name,
            },
            "pricing": {
                "input_cost_per_million": data.get("input_cost", data.get("input_cost_per_million", 0)),
                "output_cost_per_million": data.get("output_cost", data.get("output_cost_per_million", 0)),
                "source_id": source.id,
                "last_verified": datetime.utcnow().isoformat() + "Z",
            },
            "availability": {
                "status": "active",
            },
        }
        
        # Add optional fields if present
        if "context_length" in data:
            deployment["deployment"] = {"context_length": data["context_length"]}
        
        return deployment
    
    async def run_all_syncs(
        self,
        sources: List[DataSource] = None
    ) -> List[SyncResult]:
        """
        Run sync for all enabled sources.
        
        Args:
            sources: Optional list of sources (uses predefined if None)
        
        Returns:
            List of SyncResults
        """
        from langscope.models.sources.data_sources import PREDEFINED_SOURCES
        
        if sources is None:
            sources = list(PREDEFINED_SOURCES.values())
        
        results = []
        for source in sources:
            if source.is_enabled():
                result = await self.run_sync(source)
                results.append(result)
        
        return results

