"""
Stratified Sampling for Ground Truth Evaluation.

Provides balanced sampling across dimensions like:
- Difficulty (easy/medium/hard)
- Language (en/hi/bn/etc.)
- Context length (for long context tasks)
- Needle position (for needle in haystack)

Ensures fair evaluation by avoiding over-representation
of easy samples and tracks sample coverage.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum


class StratificationDimension(Enum):
    """Dimensions for stratified sampling."""
    DIFFICULTY = "difficulty"
    LANGUAGE = "language"
    CONTEXT_LENGTH = "context_length"
    NEEDLE_POSITION = "needle_position"
    AUDIO_QUALITY = "audio_quality"
    DURATION = "duration"
    QUESTION_TYPE = "question_type"
    DOCUMENT_TYPE = "document_type"


@dataclass
class SamplingStrategy:
    """Configuration for stratified sampling."""
    
    # Dimensions to stratify by
    dimensions: List[str] = field(default_factory=lambda: ["difficulty"])
    
    # Target distribution per dimension
    # e.g., {"difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3}}
    distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Cooldown period (hours) before reusing a sample
    cooldown_hours: int = 24
    
    # Minimum samples per stratum
    min_per_stratum: int = 5
    
    # Maximum consecutive samples from same stratum
    max_consecutive_same: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimensions": self.dimensions,
            "distributions": self.distributions,
            "cooldown_hours": self.cooldown_hours,
            "min_per_stratum": self.min_per_stratum,
            "max_consecutive_same": self.max_consecutive_same,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingStrategy":
        """Create from dictionary."""
        return cls(
            dimensions=data.get("dimensions", ["difficulty"]),
            distributions=data.get("distributions", {}),
            cooldown_hours=data.get("cooldown_hours", 24),
            min_per_stratum=data.get("min_per_stratum", 5),
            max_consecutive_same=data.get("max_consecutive_same", 3),
        )


@dataclass
class SampleUsage:
    """Tracks usage of a sample."""
    sample_id: str
    usage_count: int = 0
    last_used: Optional[datetime] = None


class StratifiedSampler:
    """
    Stratified sampler for ground truth samples.
    
    Ensures balanced representation across stratification
    dimensions and avoids recently used samples.
    """
    
    def __init__(self, strategy: SamplingStrategy = None, db: Any = None):
        """
        Initialize sampler.
        
        Args:
            strategy: Sampling strategy configuration
            db: Database instance for loading samples
        """
        self.strategy = strategy or SamplingStrategy()
        self.db = db
        
        # Track usage
        self.usage: Dict[str, SampleUsage] = {}
        
        # Track recent selections per stratum
        self.recent_selections: Dict[str, List[str]] = {}
    
    def select_sample(
        self,
        samples: List[Dict[str, Any]],
        filters: Dict[str, Any] = None,
        exclude_ids: Set[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select a stratified random sample.
        
        Args:
            samples: List of available samples
            filters: Optional filters to apply
            exclude_ids: Sample IDs to exclude
        
        Returns:
            Selected sample or None
        """
        if not samples:
            return None
        
        exclude_ids = exclude_ids or set()
        
        # Filter samples
        candidates = self._filter_samples(samples, filters, exclude_ids)
        
        if not candidates:
            return None
        
        # Remove recently used samples (cooldown)
        candidates = self._apply_cooldown(candidates)
        
        if not candidates:
            # If all are on cooldown, use any excluding explicit exclusions
            candidates = [s for s in samples if s.get("sample_id") not in exclude_ids]
        
        if not candidates:
            return None
        
        # Stratified selection
        selected = self._stratified_select(candidates)
        
        if selected:
            self._record_usage(selected.get("sample_id", ""))
        
        return selected
    
    def select_batch(
        self,
        samples: List[Dict[str, Any]],
        count: int,
        filters: Dict[str, Any] = None,
        ensure_diversity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Select multiple samples with stratification.
        
        Args:
            samples: List of available samples
            count: Number of samples to select
            filters: Optional filters to apply
            ensure_diversity: Ensure variety across strata
        
        Returns:
            List of selected samples
        """
        selected = []
        used_ids: Set[str] = set()
        
        for _ in range(count):
            sample = self.select_sample(samples, filters, used_ids)
            if sample:
                selected.append(sample)
                used_ids.add(sample.get("sample_id", ""))
            else:
                break
        
        return selected
    
    def get_stratified_batch(
        self,
        domain: str,
        count: int,
        stratification: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a stratified batch of samples from the database.
        
        Args:
            domain: Domain to sample from
            count: Number of samples to select
            stratification: Optional stratification config
                e.g., {"difficulty": {"easy": 2, "medium": 2, "hard": 1}}
                     or {"difficulty": "medium", "language": "en"}
        
        Returns:
            List of selected samples
        """
        if not self.db or not self.db.connected:
            return []
        
        selected = []
        used_ids: Set[str] = set()
        
        # Check if stratification specifies exact counts per stratum
        if stratification and any(isinstance(v, dict) for v in stratification.values()):
            # Count-based stratification
            for dim, distribution in stratification.items():
                if isinstance(distribution, dict):
                    for stratum_value, target_count in distribution.items():
                        if not isinstance(target_count, int):
                            continue
                        
                        # Get samples for this stratum
                        filters = {dim: stratum_value}
                        samples = self.db.get_ground_truth_samples(
                            domain=domain,
                            limit=target_count * 3  # Get extras for selection
                        )
                        
                        # Filter by stratum
                        stratum_samples = [
                            s for s in samples 
                            if s.get(dim) == stratum_value or 
                               s.get("metadata", {}).get(dim) == stratum_value
                        ]
                        
                        # Select from this stratum
                        for sample in self.select_batch(stratum_samples, target_count):
                            if sample.get("_id") not in used_ids:
                                selected.append(sample)
                                used_ids.add(sample.get("_id", sample.get("sample_id", "")))
        else:
            # Filter-based stratification
            filters = {}
            if stratification:
                for dim, value in stratification.items():
                    if isinstance(value, str):
                        filters[dim] = value
            
            # Get samples from database
            samples = self.db.get_ground_truth_samples(
                domain=domain,
                difficulty=filters.get("difficulty"),
                limit=count * 5  # Get extras for selection
            )
            
            # Apply additional filters manually
            filtered = []
            for sample in samples:
                matches = True
                for k, v in filters.items():
                    if k == "difficulty":
                        continue  # Already filtered
                    if sample.get(k) != v and sample.get("metadata", {}).get(k) != v:
                        matches = False
                        break
                if matches:
                    filtered.append(sample)
            
            # Select stratified batch
            selected = self.select_batch(filtered, count)
        
        return selected[:count]
    
    def get_coverage_stats(
        self,
        samples: List[Dict[str, Any]],
        dimension: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get coverage statistics for a dimension.
        
        Args:
            samples: All available samples
            dimension: Dimension to analyze
        
        Returns:
            Coverage statistics per stratum
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        # Group by stratum value
        for sample in samples:
            stratum = sample.get(dimension, "unknown")
            if stratum not in stats:
                stats[stratum] = {
                    "total": 0,
                    "used": 0,
                    "usage_rate": 0.0,
                    "avg_usage_count": 0.0,
                }
            
            stats[stratum]["total"] += 1
            
            sample_id = sample.get("sample_id", "")
            if sample_id in self.usage:
                usage = self.usage[sample_id]
                if usage.usage_count > 0:
                    stats[stratum]["used"] += 1
        
        # Compute rates
        for stratum, data in stats.items():
            if data["total"] > 0:
                data["usage_rate"] = data["used"] / data["total"]
        
        return stats
    
    def _filter_samples(
        self,
        samples: List[Dict[str, Any]],
        filters: Dict[str, Any],
        exclude_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """Filter samples by criteria."""
        candidates = []
        
        for sample in samples:
            sample_id = sample.get("sample_id", "")
            
            # Skip excluded
            if sample_id in exclude_ids:
                continue
            
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if sample.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            candidates.append(sample)
        
        return candidates
    
    def _apply_cooldown(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove samples that are on cooldown."""
        cutoff = datetime.utcnow() - timedelta(hours=self.strategy.cooldown_hours)
        
        available = []
        for sample in samples:
            sample_id = sample.get("sample_id", "")
            usage = self.usage.get(sample_id)
            
            if usage is None or usage.last_used is None:
                available.append(sample)
            elif usage.last_used < cutoff:
                available.append(sample)
        
        return available
    
    def _stratified_select(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select from candidates using stratification.
        
        Weights selection to match target distribution
        and reduce overrepresentation.
        """
        if not candidates:
            return None
        
        if not self.strategy.dimensions:
            # No stratification, random select
            return random.choice(candidates)
        
        # Build strata
        strata: Dict[str, List[Dict[str, Any]]] = {}
        for sample in candidates:
            stratum_key = self._get_stratum_key(sample)
            if stratum_key not in strata:
                strata[stratum_key] = []
            strata[stratum_key].append(sample)
        
        # Compute weights for each stratum
        weights = self._compute_stratum_weights(strata)
        
        # Select stratum
        stratum_keys = list(strata.keys())
        stratum_weights = [weights.get(k, 1.0) for k in stratum_keys]
        
        if sum(stratum_weights) == 0:
            stratum_weights = [1.0] * len(stratum_keys)
        
        selected_stratum = random.choices(stratum_keys, weights=stratum_weights, k=1)[0]
        
        # Select sample from stratum (weighted by inverse usage)
        stratum_samples = strata[selected_stratum]
        sample_weights = []
        for sample in stratum_samples:
            sample_id = sample.get("sample_id", "")
            usage = self.usage.get(sample_id)
            count = usage.usage_count if usage else 0
            # Inverse usage weight
            weight = 1.0 / (1.0 + count)
            sample_weights.append(weight)
        
        return random.choices(stratum_samples, weights=sample_weights, k=1)[0]
    
    def _get_stratum_key(self, sample: Dict[str, Any]) -> str:
        """Get stratum key for a sample."""
        parts = []
        for dim in self.strategy.dimensions:
            value = sample.get(dim, "unknown")
            parts.append(f"{dim}={value}")
        return "|".join(parts)
    
    def _compute_stratum_weights(
        self,
        strata: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Compute sampling weights for each stratum.
        
        Weights based on:
        1. Target distribution (if specified)
        2. Inverse of current representation
        """
        weights = {}
        total_samples = sum(len(s) for s in strata.values())
        
        for stratum_key, samples in strata.items():
            # Base weight from target distribution
            target_weight = 1.0
            
            # Parse stratum key to get dimension values
            for part in stratum_key.split("|"):
                if "=" in part:
                    dim, value = part.split("=", 1)
                    dist = self.strategy.distributions.get(dim, {})
                    if value in dist:
                        target_weight *= dist[value]
            
            # Adjust for current representation
            current_rate = len(samples) / total_samples if total_samples > 0 else 0
            
            # Boost underrepresented strata
            if current_rate > 0:
                weights[stratum_key] = target_weight / current_rate
            else:
                weights[stratum_key] = target_weight * 2
        
        return weights
    
    def _record_usage(self, sample_id: str):
        """Record that a sample was used."""
        if sample_id not in self.usage:
            self.usage[sample_id] = SampleUsage(sample_id=sample_id)
        
        self.usage[sample_id].usage_count += 1
        self.usage[sample_id].last_used = datetime.utcnow()
    
    def reset_usage(self):
        """Reset all usage tracking."""
        self.usage.clear()
        self.recent_selections.clear()


# =============================================================================
# Pre-defined Sampling Strategies
# =============================================================================

def get_default_strategy(domain: str) -> SamplingStrategy:
    """Get default sampling strategy for a domain."""
    
    if domain == "asr":
        return SamplingStrategy(
            dimensions=["difficulty", "language", "audio_quality"],
            distributions={
                "difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
                "audio_quality": {"clean": 0.4, "noisy": 0.3, "accented": 0.3},
            },
            cooldown_hours=48,
        )
    
    elif domain == "needle_in_haystack":
        return SamplingStrategy(
            dimensions=["context_length", "needle_position"],
            distributions={
                "context_length": {
                    "4096": 0.15, "8192": 0.15, "16384": 0.2,
                    "32768": 0.2, "65536": 0.15, "131072": 0.15
                },
                "needle_position": {
                    "0.0": 0.2, "0.25": 0.2, "0.5": 0.2,
                    "0.75": 0.2, "1.0": 0.2
                },
            },
            cooldown_hours=24,
        )
    
    elif domain == "visual_qa":
        return SamplingStrategy(
            dimensions=["difficulty", "question_type"],
            distributions={
                "difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
                "question_type": {
                    "color": 0.15, "count": 0.2, "spatial": 0.25,
                    "action": 0.2, "reasoning": 0.2
                },
            },
            cooldown_hours=24,
        )
    
    elif domain in ("long_document_qa", "code_completion"):
        return SamplingStrategy(
            dimensions=["difficulty", "context_length"],
            distributions={
                "difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
            },
            cooldown_hours=24,
        )
    
    # Default strategy
    return SamplingStrategy(
        dimensions=["difficulty"],
        distributions={
            "difficulty": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
        },
        cooldown_hours=24,
    )

