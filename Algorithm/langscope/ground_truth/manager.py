"""
Ground Truth Sample Manager.

Manages loading, indexing, and retrieval of ground truth samples.
Samples are stored on disk with metadata indexed in MongoDB.
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

from langscope.ground_truth.sampling import (
    StratifiedSampler,
    SamplingStrategy,
    get_default_strategy,
)

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthSample:
    """
    A ground truth sample for evaluation.
    
    Contains metadata and paths to actual content
    (audio, images, documents stored on disk).
    """
    # Identity
    sample_id: str
    domain: str
    category: str  # "multimodal" or "long_context"
    subdomain: Optional[str] = None
    
    # Location
    base_path: str = ""
    
    # Input paths (relative to base_path)
    inputs: Dict[str, str] = field(default_factory=dict)
    
    # Ground truth
    ground_truth: Any = None
    ground_truth_path: Optional[str] = None
    
    # Stratification
    difficulty: str = "medium"
    language: str = "en"
    
    # Domain-specific fields
    context_length: Optional[int] = None
    needle_position: Optional[float] = None
    audio_quality: Optional[str] = None
    duration_seconds: Optional[float] = None
    question_type: Optional[str] = None
    
    # Hashes for integrity
    input_hash: str = ""
    ground_truth_hash: str = ""
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[str] = None
    
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def get_input_path(self, input_type: str) -> Optional[str]:
        """Get full path for an input type."""
        if input_type not in self.inputs:
            return None
        return os.path.join(self.base_path, self.inputs[input_type])
    
    def load_ground_truth(self) -> Any:
        """Load ground truth from file if not already loaded."""
        if self.ground_truth is not None:
            return self.ground_truth
        
        if self.ground_truth_path:
            full_path = os.path.join(self.base_path, self.ground_truth_path)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    if full_path.endswith(".json"):
                        self.ground_truth = json.load(f)
                    else:
                        self.ground_truth = f.read()
        
        return self.ground_truth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "_id": self.sample_id,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "category": self.category,
            "subdomain": self.subdomain,
            "base_path": self.base_path,
            "inputs": self.inputs,
            "ground_truth_path": self.ground_truth_path,
            "difficulty": self.difficulty,
            "language": self.language,
            "context_length": self.context_length,
            "needle_position": self.needle_position,
            "audio_quality": self.audio_quality,
            "duration_seconds": self.duration_seconds,
            "question_type": self.question_type,
            "input_hash": self.input_hash,
            "ground_truth_hash": self.ground_truth_hash,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthSample":
        """Create from dictionary."""
        return cls(
            sample_id=data.get("sample_id", data.get("_id", "")),
            domain=data.get("domain", ""),
            category=data.get("category", ""),
            subdomain=data.get("subdomain"),
            base_path=data.get("base_path", ""),
            inputs=data.get("inputs", {}),
            ground_truth_path=data.get("ground_truth_path"),
            difficulty=data.get("difficulty", "medium"),
            language=data.get("language", "en"),
            context_length=data.get("context_length"),
            needle_position=data.get("needle_position"),
            audio_quality=data.get("audio_quality"),
            duration_seconds=data.get("duration_seconds"),
            question_type=data.get("question_type"),
            input_hash=data.get("input_hash", ""),
            ground_truth_hash=data.get("ground_truth_hash", ""),
            usage_count=data.get("usage_count", 0),
            last_used=data.get("last_used"),
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class GroundTruthManager:
    """
    Manages ground truth samples.
    
    Handles loading from disk, indexing in database,
    stratified sampling, and usage tracking.
    """
    
    def __init__(
        self,
        db: 'MongoDB' = None,
        base_dir: str = None,
        cache_samples: bool = True
    ):
        """
        Initialize manager.
        
        Args:
            db: MongoDB instance for metadata storage
            base_dir: Base directory for ground truth datasets
            cache_samples: Whether to cache loaded samples in memory
        """
        self.db = db
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(__file__), "datasets"
        )
        self.cache_samples = cache_samples
        
        # In-memory cache
        self._sample_cache: Dict[str, GroundTruthSample] = {}
        
        # Samplers per domain
        self._samplers: Dict[str, StratifiedSampler] = {}
    
    def load_sample(self, sample_id: str) -> Optional[GroundTruthSample]:
        """
        Load a sample by ID.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            GroundTruthSample or None
        """
        # Check cache
        if sample_id in self._sample_cache:
            return self._sample_cache[sample_id]
        
        # Load from database
        if self.db and self.db.connected:
            sample_data = self.db.db["ground_truth_samples"].find_one(
                {"_id": sample_id}
            )
            if sample_data:
                sample = GroundTruthSample.from_dict(sample_data)
                if self.cache_samples:
                    self._sample_cache[sample_id] = sample
                return sample
        
        return None
    
    def get_random_sample(
        self,
        domain: str,
        filters: Dict[str, Any] = None,
        exclude_ids: Set[str] = None
    ) -> Optional[GroundTruthSample]:
        """
        Get a stratified random sample for a domain.
        
        Args:
            domain: Domain name
            filters: Optional filters (e.g., {"language": "en"})
            exclude_ids: Sample IDs to exclude
        
        Returns:
            GroundTruthSample or None
        """
        # Get all samples for domain
        samples = self.get_samples_for_domain(domain, as_dicts=True)
        
        if not samples:
            return None
        
        # Get or create sampler
        sampler = self._get_sampler(domain)
        
        # Select sample
        selected = sampler.select_sample(samples, filters, exclude_ids)
        
        if selected:
            sample = GroundTruthSample.from_dict(selected)
            self._record_usage(sample.sample_id)
            return sample
        
        return None
    
    def get_batch_samples(
        self,
        domain: str,
        count: int,
        filters: Dict[str, Any] = None,
        stratification: Dict[str, float] = None
    ) -> List[GroundTruthSample]:
        """
        Get multiple stratified samples.
        
        Args:
            domain: Domain name
            count: Number of samples to get
            filters: Optional filters
            stratification: Custom stratification weights
        
        Returns:
            List of GroundTruthSample
        """
        # Get all samples for domain
        samples = self.get_samples_for_domain(domain, as_dicts=True)
        
        if not samples:
            return []
        
        # Get or create sampler
        sampler = self._get_sampler(domain)
        
        # Update stratification if provided
        if stratification:
            sampler.strategy.distributions.update(stratification)
        
        # Select samples
        selected = sampler.select_batch(samples, count, filters)
        
        result = []
        for s in selected:
            sample = GroundTruthSample.from_dict(s)
            self._record_usage(sample.sample_id)
            result.append(sample)
        
        return result
    
    def get_samples_for_domain(
        self,
        domain: str,
        filters: Dict[str, Any] = None,
        limit: int = 1000,
        as_dicts: bool = False
    ) -> List[Any]:
        """
        Get all samples for a domain.
        
        Args:
            domain: Domain name
            filters: Optional filters
            limit: Maximum samples to return
            as_dicts: Return as dicts instead of GroundTruthSample
        
        Returns:
            List of samples
        """
        if not self.db or not self.db.connected:
            return []
        
        query = {"domain": domain}
        if filters:
            query.update(filters)
        
        cursor = self.db.db["ground_truth_samples"].find(query).limit(limit)
        samples = list(cursor)
        
        if as_dicts:
            return samples
        
        return [GroundTruthSample.from_dict(s) for s in samples]
    
    def register_sample_usage(
        self,
        sample_id: str,
        match_id: str
    ) -> bool:
        """
        Record that a sample was used in a match.
        
        Args:
            sample_id: Sample identifier
            match_id: Match identifier
        
        Returns:
            True if successful
        """
        return self._record_usage(sample_id, match_id)
    
    def get_sample_coverage(
        self,
        domain: str,
        dimension: str = "difficulty"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get coverage analytics for a domain.
        
        Args:
            domain: Domain name
            dimension: Dimension to analyze
        
        Returns:
            Coverage statistics per stratum
        """
        samples = self.get_samples_for_domain(domain, as_dicts=True)
        sampler = self._get_sampler(domain)
        
        return sampler.get_coverage_stats(samples, dimension)
    
    def index_samples_from_disk(
        self,
        domain: str,
        category: str,
        directory: str
    ) -> int:
        """
        Index samples from a directory into the database.
        
        Args:
            domain: Domain name
            category: Category (multimodal/long_context)
            directory: Directory containing samples
        
        Returns:
            Number of samples indexed
        """
        if not self.db or not self.db.connected:
            logger.error("Database not connected")
            return 0
        
        indexed = 0
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0
        
        # Each subdirectory is a sample
        for sample_dir in directory.iterdir():
            if not sample_dir.is_dir():
                continue
            
            # Look for metadata.json
            metadata_path = sample_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Create sample
                sample = GroundTruthSample(
                    sample_id=metadata.get("sample_id", sample_dir.name),
                    domain=domain,
                    category=category,
                    subdomain=metadata.get("subdomain"),
                    base_path=str(sample_dir),
                    inputs=metadata.get("inputs", {}),
                    ground_truth_path=metadata.get("ground_truth_path"),
                    difficulty=metadata.get("difficulty", "medium"),
                    language=metadata.get("language", "en"),
                    context_length=metadata.get("context_length"),
                    needle_position=metadata.get("needle_position"),
                    audio_quality=metadata.get("audio_quality"),
                    duration_seconds=metadata.get("duration_seconds"),
                    question_type=metadata.get("question_type"),
                    tags=metadata.get("tags", []),
                    version=metadata.get("version", "1.0.0"),
                )
                
                # Compute hashes
                sample.input_hash = self._compute_input_hash(sample)
                sample.ground_truth_hash = self._compute_ground_truth_hash(sample)
                
                # Save to database
                self.db.db["ground_truth_samples"].update_one(
                    {"_id": sample.sample_id},
                    {"$set": sample.to_dict()},
                    upsert=True
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"Error indexing {sample_dir}: {e}")
        
        logger.info(f"Indexed {indexed} samples for domain {domain}")
        return indexed
    
    def _get_sampler(self, domain: str) -> StratifiedSampler:
        """Get or create sampler for domain."""
        if domain not in self._samplers:
            strategy = get_default_strategy(domain)
            self._samplers[domain] = StratifiedSampler(strategy)
        return self._samplers[domain]
    
    def _record_usage(self, sample_id: str, match_id: str = None) -> bool:
        """Record sample usage in database."""
        if not self.db or not self.db.connected:
            return False
        
        try:
            now = datetime.utcnow().isoformat() + "Z"
            update = {
                "$inc": {"usage_count": 1},
                "$set": {"last_used": now, "updated_at": now},
            }
            
            if match_id:
                update["$push"] = {"match_ids": match_id}
            
            self.db.db["ground_truth_samples"].update_one(
                {"_id": sample_id},
                update
            )
            return True
        except Exception as e:
            logger.error(f"Error recording usage: {e}")
            return False
    
    def _compute_input_hash(self, sample: GroundTruthSample) -> str:
        """Compute hash of sample inputs."""
        hasher = hashlib.sha256()
        
        for input_type, rel_path in sorted(sample.inputs.items()):
            full_path = os.path.join(sample.base_path, rel_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    # Hash in chunks for large files
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()[:16]
    
    def _compute_ground_truth_hash(self, sample: GroundTruthSample) -> str:
        """Compute hash of ground truth."""
        hasher = hashlib.sha256()
        
        if sample.ground_truth_path:
            full_path = os.path.join(sample.base_path, sample.ground_truth_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    hasher.update(f.read())
        elif sample.ground_truth:
            hasher.update(str(sample.ground_truth).encode("utf-8"))
        
        return hasher.hexdigest()[:16]


