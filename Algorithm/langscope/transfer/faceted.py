"""
Multi-Faceted Domain Transfer Learning for LangScope.

Enables knowledge transfer between domains using compositional domain similarity.
Domains are decomposed into facets (language, field, modality, task, specialty)
and correlations are computed as weighted combinations of per-facet similarities.

Key Features:
- DomainDescriptor: Represents domains as facet compositions
- FacetSimilarityLearner: Per-facet Bayesian similarity learning
- CompositeDomainSimilarity: Weighted combination of facet similarities
- DomainIndex: In-memory index with pre-computed Top-K similar domains
- FacetedTransferLearning: Transfer learning using faceted correlations

Key Formulas:
- Composite correlation: ρ(D_S, D_T) = Σ_k β_k × sim_k(source[k], target[k])
- Bayesian blending: sim = α × prior + (1 - α) × data, α = 1/(1 + n/τ)
- Multi-source transfer: μ_target = μ₀ + Σⱼ wⱼ × ρⱼ × (μⱼ - μ₀)

See docs/transfer_learning.md for comprehensive documentation.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from langscope.core.constants import (
    TRUESKILL_MU_0,
    TRUESKILL_SIGMA_0,
    SIGMA_BASE,
)
from langscope.core.rating import TrueSkillRating

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB
    from langscope.core.model import LLMModel
    from langscope.domain.domain_manager import DomainManager
    from langscope.domain.domain_config import Domain


# =============================================================================
# Constants
# =============================================================================

# Facet keys
FACET_LANGUAGE = "language"
FACET_FIELD = "field"
FACET_MODALITY = "modality"
FACET_TASK = "task"
FACET_SPECIALTY = "specialty"

ALL_FACETS = [FACET_LANGUAGE, FACET_FIELD, FACET_MODALITY, FACET_TASK, FACET_SPECIALTY]

# Default facet weights (β) - sum to 1.0
DEFAULT_FACET_WEIGHTS = {
    FACET_FIELD: 0.35,      # Domain expertise transfers strongly
    FACET_LANGUAGE: 0.20,   # Linguistic patterns matter
    FACET_MODALITY: 0.20,   # Input/output format affects capability
    FACET_TASK: 0.15,       # Task structure has moderate impact
    FACET_SPECIALTY: 0.10,  # Fine-grained specialization
}

# Default τ (smoothing) values per facet
DEFAULT_FACET_TAU = {
    FACET_LANGUAGE: 15.0,   # Languages need fewer samples
    FACET_FIELD: 20.0,      # Fields are broader
    FACET_MODALITY: 10.0,   # Modalities are distinct
    FACET_TASK: 25.0,       # Tasks have more nuance
    FACET_SPECIALTY: 30.0,  # Specialties need more data
}

# Default facet values
DEFAULT_FACET_VALUES = {
    FACET_LANGUAGE: "english",
    FACET_FIELD: "general",
    FACET_MODALITY: "text",
    FACET_TASK: "general",
    FACET_SPECIALTY: "general",
}


# =============================================================================
# DomainDescriptor
# =============================================================================

@dataclass
class DomainDescriptor:
    """
    Represents a domain as a composition of facet values.
    
    Example:
        DomainDescriptor(
            name="odia_medical_imaging_thyroid",
            facets={
                "language": "odia",
                "field": "medical",
                "modality": "imaging",
                "task": "detection",
                "specialty": "thyroid"
            }
        )
    
    Facets:
        - language: Linguistic context (english, hindi, bengali, odia, etc.)
        - field: Domain expertise (medical, legal, coding, general, etc.)
        - modality: Input/output type (text, imaging, audio, video, multimodal)
        - task: Task type (qa, classification, generation, detection, etc.)
        - specialty: Sub-specialization (thyroid, cardiology, algorithms, etc.)
    """
    name: str
    facets: Dict[str, str] = field(default_factory=dict)
    
    # Facet key constants
    LANGUAGE = FACET_LANGUAGE
    FIELD = FACET_FIELD
    MODALITY = FACET_MODALITY
    TASK = FACET_TASK
    SPECIALTY = FACET_SPECIALTY
    
    # Default values
    DEFAULTS = DEFAULT_FACET_VALUES
    
    def get(self, facet: str) -> str:
        """Get facet value with default fallback."""
        return self.facets.get(facet, self.DEFAULTS.get(facet, "general"))
    
    def set(self, facet: str, value: str) -> None:
        """Set a facet value."""
        self.facets[facet] = value.lower().strip()
    
    def has_facet(self, facet: str) -> bool:
        """Check if facet is explicitly set."""
        return facet in self.facets
    
    def get_all_facets(self) -> Dict[str, str]:
        """Get all facets with defaults applied."""
        result = dict(self.DEFAULTS)
        result.update(self.facets)
        return result
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, DomainDescriptor):
            return self.name == other.name
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "facets": self.facets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainDescriptor':
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            facets=data.get("facets", {}),
        )


# =============================================================================
# FacetSimilarityData
# =============================================================================

@dataclass
class FacetSimilarityData:
    """Similarity data between two values of a facet."""
    facet: str
    value_a: str
    value_b: str
    prior_similarity: float = 0.5
    data_similarity: float = 0.0
    blended_similarity: float = 0.5
    sample_count: int = 0
    alpha: float = 1.0
    confidence: float = 0.0
    updated_at: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "_id": f"{self.facet}|{self.value_a}|{self.value_b}",
            "facet": self.facet,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "prior_similarity": self.prior_similarity,
            "data_similarity": self.data_similarity,
            "blended_similarity": self.blended_similarity,
            "sample_count": self.sample_count,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FacetSimilarityData':
        """Create from dictionary."""
        return cls(
            facet=data.get("facet", ""),
            value_a=data.get("value_a", ""),
            value_b=data.get("value_b", ""),
            prior_similarity=data.get("prior_similarity", 0.5),
            data_similarity=data.get("data_similarity", 0.0),
            blended_similarity=data.get("blended_similarity", 0.5),
            sample_count=data.get("sample_count", 0),
            alpha=data.get("alpha", 1.0),
            confidence=data.get("confidence", 0.0),
            updated_at=data.get("updated_at", ""),
        )


# =============================================================================
# FacetSimilarityLearner
# =============================================================================

class FacetSimilarityLearner:
    """
    Learns similarities between values within a single facet.
    
    Uses Bayesian smoothing: sim = α·prior + (1-α)·data
    where α = 1/(1 + n/τ)
    
    As more data is observed, α decreases and the estimate relies
    more on observed data than the prior.
    
    Example:
        learner = FacetSimilarityLearner("language", tau=15.0)
        learner.set_prior("bengali", "odia", 0.75)
        sim = learner.get_similarity("bengali", "odia")  # Returns 0.75 initially
        
        # After observing transfer accuracy
        learner.update_from_observation("bengali", "odia", 0.80, sample_size=10)
        sim = learner.get_similarity("bengali", "odia")  # Blended estimate
    """
    
    def __init__(
        self,
        facet: str,
        tau: float = None,
        db: 'MongoDB' = None
    ):
        """
        Initialize facet similarity learner.
        
        Args:
            facet: Facet name (e.g., "language", "field")
            tau: Bayesian smoothing parameter (higher = slower learning)
            db: Database instance for persistence
        """
        self.facet = facet
        self.tau = tau if tau is not None else DEFAULT_FACET_TAU.get(facet, 20.0)
        self.db = db
        self._similarities: Dict[Tuple[str, str], FacetSimilarityData] = {}
    
    def _get_key(self, value_a: str, value_b: str) -> Tuple[str, str]:
        """Get canonical key for value pair (sorted for consistency)."""
        a = value_a.lower().strip()
        b = value_b.lower().strip()
        return tuple(sorted([a, b]))
    
    def set_prior(self, value_a: str, value_b: str, prior: float) -> None:
        """
        Set expert prior for similarity between two facet values.
        
        Args:
            value_a: First value
            value_b: Second value
            prior: Prior similarity estimate (0 to 1)
        """
        if prior < 0 or prior > 1:
            raise ValueError("Prior must be between 0 and 1")
        
        key = self._get_key(value_a, value_b)
        
        if key in self._similarities:
            self._similarities[key].prior_similarity = prior
            self._update_blend(key)
        else:
            self._similarities[key] = FacetSimilarityData(
                facet=self.facet,
                value_a=key[0],
                value_b=key[1],
                prior_similarity=prior,
                blended_similarity=prior,
                alpha=1.0,
            )
        
        if self.db:
            self._persist(key)
    
    def update_from_observation(
        self,
        value_a: str,
        value_b: str,
        observed_similarity: float,
        sample_size: int = 1
    ) -> None:
        """
        Update similarity estimate with observed data.
        
        Args:
            value_a: First value
            value_b: Second value
            observed_similarity: Observed similarity (0 to 1)
            sample_size: Number of observations this represents
        """
        key = self._get_key(value_a, value_b)
        
        if key not in self._similarities:
            self._similarities[key] = FacetSimilarityData(
                facet=self.facet,
                value_a=key[0],
                value_b=key[1],
            )
        
        data = self._similarities[key]
        old_n = data.sample_count
        new_n = old_n + sample_size
        
        # Running average of observed similarities
        if old_n == 0:
            data.data_similarity = observed_similarity
        else:
            data.data_similarity = (
                old_n * data.data_similarity + sample_size * observed_similarity
            ) / new_n
        
        data.sample_count = new_n
        self._update_blend(key)
        
        if self.db:
            self._persist(key)
    
    def _update_blend(self, key: Tuple[str, str]) -> None:
        """Recompute blended similarity for a key."""
        data = self._similarities[key]
        
        # α = 1/(1 + n/τ)
        data.alpha = 1.0 / (1.0 + data.sample_count / self.tau)
        
        # Blend: sim = α·prior + (1-α)·data
        data.blended_similarity = (
            data.alpha * data.prior_similarity +
            (1 - data.alpha) * data.data_similarity
        )
        
        # Confidence = 1 - α (how much we trust the data)
        data.confidence = 1.0 - data.alpha
        data.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def _persist(self, key: Tuple[str, str]) -> None:
        """Persist similarity data to database."""
        if self.db and key in self._similarities:
            try:
                self.db.save_facet_similarity(self._similarities[key].to_dict())
            except Exception:
                pass  # Fail silently for non-critical persistence
    
    def get_similarity(self, value_a: str, value_b: str) -> float:
        """
        Get similarity between two facet values.
        
        Args:
            value_a: First value
            value_b: Second value
        
        Returns:
            Blended similarity estimate (0 to 1)
        """
        # Same value = perfect similarity
        if value_a.lower().strip() == value_b.lower().strip():
            return 1.0
        
        key = self._get_key(value_a, value_b)
        
        # Check cache
        if key in self._similarities:
            return self._similarities[key].blended_similarity
        
        # Check database
        if self.db:
            data = self.db.get_facet_similarity(self.facet, value_a, value_b)
            if data:
                self._similarities[key] = FacetSimilarityData.from_dict(data)
                return self._similarities[key].blended_similarity
        
        # Default: moderate similarity for unknown pairs
        return 0.5
    
    def get_similarity_data(
        self,
        value_a: str,
        value_b: str
    ) -> Optional[FacetSimilarityData]:
        """Get full similarity data for a value pair."""
        key = self._get_key(value_a, value_b)
        return self._similarities.get(key)
    
    def load_priors(self, priors: Dict[Tuple[str, str], float]) -> None:
        """
        Load multiple expert priors at once.
        
        Args:
            priors: Dict of {(value_a, value_b): prior_similarity}
        """
        for (value_a, value_b), prior in priors.items():
            self.set_prior(value_a, value_b, prior)
    
    def export_similarities(self) -> List[Dict]:
        """Export all learned similarities."""
        return [data.to_dict() for data in self._similarities.values()]


# =============================================================================
# CompositeDomainSimilarity
# =============================================================================

class CompositeDomainSimilarity:
    """
    Computes overall domain similarity by combining faceted similarities.
    
    Formula (weighted sum):
        ρ(D_source, D_target) = Σ_k β_k × sim_k(source[k], target[k])
    
    Alternative (geometric mean):
        ρ(D_source, D_target) = Π_k sim_k(source[k], target[k])^β_k
    
    Example:
        composite = CompositeDomainSimilarity()
        source = DomainDescriptor(name="bengali_medical", facets={...})
        target = DomainDescriptor(name="odia_medical_imaging", facets={...})
        correlation = composite.get_correlation(source, target)
    """
    
    def __init__(
        self,
        facet_learners: Dict[str, FacetSimilarityLearner] = None,
        weights: Dict[str, float] = None,
        combination: str = "weighted_sum",  # or "geometric"
        db: 'MongoDB' = None
    ):
        """
        Initialize composite domain similarity.
        
        Args:
            facet_learners: Dict of facet name → FacetSimilarityLearner
            weights: Facet importance weights (β values)
            combination: How to combine facets ("weighted_sum" or "geometric")
            db: Database for persistence
        """
        self.weights = weights or DEFAULT_FACET_WEIGHTS.copy()
        self.combination = combination
        self.db = db
        
        # Initialize facet learners
        if facet_learners:
            self.facet_learners = facet_learners
        else:
            self.facet_learners = {
                facet: FacetSimilarityLearner(facet, db=db)
                for facet in self.weights.keys()
            }
    
    def get_correlation(
        self,
        source: DomainDescriptor,
        target: DomainDescriptor
    ) -> float:
        """
        Compute composite correlation between two domains.
        
        Args:
            source: Source domain descriptor
            target: Target domain descriptor
        
        Returns:
            Correlation estimate (0 to 1)
        """
        if source.name == target.name:
            return 1.0
        
        if self.combination == "weighted_sum":
            return self._weighted_sum(source, target)
        elif self.combination == "geometric":
            return self._geometric_mean(source, target)
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")
    
    def _weighted_sum(
        self,
        source: DomainDescriptor,
        target: DomainDescriptor
    ) -> float:
        """Compute ρ = Σ_k β_k × sim_k(source[k], target[k])"""
        total = 0.0
        weight_sum = 0.0
        
        for facet, weight in self.weights.items():
            if facet in self.facet_learners:
                source_val = source.get(facet)
                target_val = target.get(facet)
                
                sim = self.facet_learners[facet].get_similarity(source_val, target_val)
                total += weight * sim
                weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.5
    
    def _geometric_mean(
        self,
        source: DomainDescriptor,
        target: DomainDescriptor
    ) -> float:
        """Compute ρ = Π_k sim_k(source[k], target[k])^β_k"""
        log_sum = 0.0
        weight_sum = 0.0
        
        for facet, weight in self.weights.items():
            if facet in self.facet_learners:
                source_val = source.get(facet)
                target_val = target.get(facet)
                
                sim = self.facet_learners[facet].get_similarity(source_val, target_val)
                sim = max(sim, 0.01)  # Avoid log(0)
                
                log_sum += weight * math.log(sim)
                weight_sum += weight
        
        return math.exp(log_sum / weight_sum) if weight_sum > 0 else 0.5
    
    def get_facet_breakdown(
        self,
        source: DomainDescriptor,
        target: DomainDescriptor
    ) -> Dict[str, Dict]:
        """
        Get detailed breakdown of similarity by facet.
        
        Returns:
            Dict with per-facet contribution details
        """
        breakdown = {}
        
        for facet, weight in self.weights.items():
            if facet in self.facet_learners:
                source_val = source.get(facet)
                target_val = target.get(facet)
                learner = self.facet_learners[facet]
                
                sim = learner.get_similarity(source_val, target_val)
                
                breakdown[facet] = {
                    "source_value": source_val,
                    "target_value": target_val,
                    "similarity": sim,
                    "weight": weight,
                    "contribution": weight * sim,
                }
        
        return breakdown
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update facet weights."""
        self.weights.update(new_weights)
    
    def get_learner(self, facet: str) -> Optional[FacetSimilarityLearner]:
        """Get the learner for a specific facet."""
        return self.facet_learners.get(facet)


# =============================================================================
# DomainNameParser
# =============================================================================

class DomainNameParser:
    """
    Heuristically extracts facets from domain names.
    
    Example:
        parser = DomainNameParser()
        facets = parser.parse("hindi_medical_imaging_thyroid")
        # Returns: {"language": "hindi", "field": "medical", 
        #           "modality": "imaging", "specialty": "thyroid"}
    """
    
    # Known values per facet
    LANGUAGES = {
        "english", "hindi", "bengali", "odia", "tamil", "telugu",
        "gujarati", "kannada", "malayalam", "punjabi", "marathi",
        "urdu", "assamese", "nepali", "sanskrit", "chinese", "japanese",
        "korean", "spanish", "french", "german", "arabic", "russian",
        "portuguese", "italian", "dutch", "polish", "vietnamese", "thai",
        "indonesian", "malay", "turkish", "hebrew", "persian", "swahili",
    }
    
    FIELDS = {
        "medical", "clinical", "legal", "coding", "code", "math",
        "mathematics", "general", "financial", "education", "scientific",
        "science", "engineering", "business", "marketing", "health",
        "healthcare", "radiology", "pathology", "cardiology", "oncology",
        "neurology", "pediatrics", "psychiatry", "surgery", "pharmacy",
        "nursing", "dentistry", "veterinary", "optometry", "dermatology",
        "algorithms", "programming", "software", "data", "analytics",
        "economics", "accounting", "insurance", "investment", "banking",
        "history", "geography", "philosophy", "psychology", "sociology",
        "anthropology", "political", "literature", "linguistics", "art",
        "music", "sports", "entertainment", "travel", "food", "fashion",
        "technology", "research", "academic", "creative", "writing",
    }
    
    MODALITIES = {
        "text", "imaging", "image", "audio", "video", "multimodal",
        "speech", "voice", "visual", "document", "ocr",
    }
    
    TASKS = {
        "qa", "qna", "question", "answering", "detection", "classification",
        "generation", "summarization", "reasoning", "extraction", "translation",
        "sentiment", "ner", "entity", "parsing", "completion", "chat",
        "dialogue", "conversation", "search", "retrieval", "recommendation",
        "captioning", "description", "transcription", "synthesis",
        "analysis", "prediction", "forecasting", "evaluation", "assessment",
        "grading", "scoring", "ranking", "comparison", "matching",
        "segmentation", "tagging", "labeling", "annotation", "editing",
        "rewriting", "paraphrasing", "compression", "expansion",
    }
    
    # Modality normalization
    MODALITY_ALIASES = {
        "image": "imaging",
        "visual": "imaging",
        "speech": "audio",
        "voice": "audio",
        "document": "text",
    }
    
    # Task normalization
    TASK_ALIASES = {
        "qna": "qa",
        "question": "qa",
        "answering": "qa",
    }
    
    def parse(self, domain_name: str) -> Dict[str, str]:
        """
        Parse domain name into facets.
        
        Args:
            domain_name: Domain name (e.g., "hindi_medical_imaging_qa")
        
        Returns:
            Dict of extracted facets
        """
        # Normalize: lowercase, replace hyphens with underscores
        normalized = domain_name.lower().replace("-", "_")
        parts = normalized.split("_")
        
        facets = {}
        used_parts = set()
        
        for i, part in enumerate(parts):
            if part in used_parts:
                continue
            
            # Check language
            if part in self.LANGUAGES and FACET_LANGUAGE not in facets:
                facets[FACET_LANGUAGE] = part
                used_parts.add(part)
                continue
            
            # Check field
            if part in self.FIELDS and FACET_FIELD not in facets:
                facets[FACET_FIELD] = part
                used_parts.add(part)
                continue
            
            # Check modality
            if part in self.MODALITIES and FACET_MODALITY not in facets:
                modality = self.MODALITY_ALIASES.get(part, part)
                facets[FACET_MODALITY] = modality
                used_parts.add(part)
                continue
            
            # Check task
            if part in self.TASKS and FACET_TASK not in facets:
                task = self.TASK_ALIASES.get(part, part)
                facets[FACET_TASK] = task
                used_parts.add(part)
                continue
        
        # Remaining parts become specialty
        remaining = [p for p in parts if p not in used_parts and len(p) > 1]
        if remaining and FACET_SPECIALTY not in facets:
            facets[FACET_SPECIALTY] = "_".join(remaining)
        
        return facets
    
    def create_descriptor(self, domain_name: str) -> DomainDescriptor:
        """Create a DomainDescriptor from a domain name."""
        facets = self.parse(domain_name)
        return DomainDescriptor(name=domain_name, facets=facets)


# =============================================================================
# DomainIndex
# =============================================================================

class DomainIndex:
    """
    Fast in-memory index for domain similarity lookups.
    
    Features:
    - O(1) lookup for pre-computed Top-K similar domains
    - Lazy computation for novel domains
    - Background refresh support
    
    Example:
        index = DomainIndex(domain_manager, composite)
        index.load()  # Load all domains
        similar = index.get_similar_domains("medical", k=5)
    """
    
    def __init__(
        self,
        domain_manager: 'DomainManager' = None,
        composite: CompositeDomainSimilarity = None,
        parser: DomainNameParser = None
    ):
        """
        Initialize domain index.
        
        Args:
            domain_manager: DomainManager instance for loading domains
            composite: CompositeDomainSimilarity for computing correlations
            parser: DomainNameParser for extracting facets from names
        """
        self.domain_manager = domain_manager
        self.composite = composite or CompositeDomainSimilarity()
        self.parser = parser or DomainNameParser()
        
        # Domain name → DomainDescriptor
        self.descriptors: Dict[str, DomainDescriptor] = {}
        
        # Pre-computed similarity cache: (source, target) → correlation
        self._sim_cache: Dict[Tuple[str, str], float] = {}
        
        # Top-K similar domains per domain: domain → [(similar_domain, correlation), ...]
        self._top_k_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Index metadata
        self._loaded = False
        self._last_refresh: Optional[datetime] = None
    
    def load(self) -> None:
        """Load all domains from DomainManager into the index."""
        if not self.domain_manager:
            self._loaded = True
            return
        
        try:
            for name in self.domain_manager.list_domains():
                domain = self.domain_manager.get_domain(name)
                if domain:
                    self.descriptors[name] = self._domain_to_descriptor(domain)
        except Exception:
            pass  # Fail gracefully
        
        self._loaded = True
    
    def _domain_to_descriptor(self, domain: 'Domain') -> DomainDescriptor:
        """Convert Domain object to DomainDescriptor."""
        # Check if facets are stored in metadata
        facets = domain.metadata.get("facets", {})
        
        # Fallback: parse from name
        if not facets:
            facets = self.parser.parse(domain.name)
        
        return DomainDescriptor(name=domain.name, facets=facets)
    
    def register_domain(
        self,
        descriptor: DomainDescriptor,
        precompute: bool = False
    ) -> None:
        """
        Register a new domain descriptor.
        
        Args:
            descriptor: Domain descriptor to register
            precompute: Whether to pre-compute similarities immediately
        """
        self.descriptors[descriptor.name] = descriptor
        
        if precompute:
            self._precompute_for_domain(descriptor.name)
    
    def get_or_create_descriptor(self, domain_name: str) -> DomainDescriptor:
        """Get descriptor for a domain, creating if not exists."""
        if domain_name in self.descriptors:
            return self.descriptors[domain_name]
        
        # Create from name
        descriptor = self.parser.create_descriptor(domain_name)
        self.descriptors[domain_name] = descriptor
        return descriptor
    
    def get_correlation(
        self,
        source_name: str,
        target_name: str
    ) -> float:
        """
        Get correlation between two domains.
        
        Args:
            source_name: Source domain name
            target_name: Target domain name
        
        Returns:
            Correlation (0 to 1)
        """
        if source_name == target_name:
            return 1.0
        
        # Check cache
        cache_key = (source_name, target_name)
        if cache_key in self._sim_cache:
            return self._sim_cache[cache_key]
        
        # Compute
        source = self.get_or_create_descriptor(source_name)
        target = self.get_or_create_descriptor(target_name)
        rho = self.composite.get_correlation(source, target)
        
        # Cache both directions
        self._sim_cache[(source_name, target_name)] = rho
        self._sim_cache[(target_name, source_name)] = rho
        
        return rho
    
    def get_similar_domains(
        self,
        target_name: str,
        k: int = 5,
        min_correlation: float = 0.25
    ) -> List[Tuple[str, float]]:
        """
        Get top-K most similar domains.
        
        Args:
            target_name: Target domain name
            k: Maximum number of domains to return
            min_correlation: Minimum correlation threshold
        
        Returns:
            List of (domain_name, correlation) tuples, sorted by correlation desc
        """
        # Check pre-computed cache
        if target_name in self._top_k_cache:
            cached = self._top_k_cache[target_name]
            return [
                (name, rho) for name, rho in cached[:k]
                if rho >= min_correlation
            ]
        
        # Compute on-the-fly
        target = self.get_or_create_descriptor(target_name)
        similarities = []
        
        for name, source in self.descriptors.items():
            if name != target_name:
                rho = self.composite.get_correlation(source, target)
                if rho >= min_correlation:
                    similarities.append((name, rho))
        
        # Sort by correlation descending
        similarities.sort(key=lambda x: -x[1])
        
        return similarities[:k]
    
    def precompute_top_k(self, k: int = 10) -> None:
        """
        Pre-compute top-K similar domains for all registered domains.
        
        Args:
            k: Number of similar domains to cache per domain
        """
        domains = list(self.descriptors.values())
        
        for target in domains:
            similarities = []
            for source in domains:
                if source.name != target.name:
                    rho = self.composite.get_correlation(source, target)
                    similarities.append((source.name, rho))
            
            similarities.sort(key=lambda x: -x[1])
            self._top_k_cache[target.name] = similarities[:k]
        
        self._last_refresh = datetime.utcnow()
    
    def _precompute_for_domain(self, domain_name: str, k: int = 10) -> None:
        """Pre-compute top-K for a single domain."""
        target = self.descriptors.get(domain_name)
        if not target:
            return
        
        similarities = []
        for name, source in self.descriptors.items():
            if name != domain_name:
                rho = self.composite.get_correlation(source, target)
                similarities.append((name, rho))
        
        similarities.sort(key=lambda x: -x[1])
        self._top_k_cache[domain_name] = similarities[:k]
    
    def get_facet_breakdown(
        self,
        source_name: str,
        target_name: str
    ) -> Dict[str, Dict]:
        """Get detailed facet breakdown for domain pair."""
        source = self.get_or_create_descriptor(source_name)
        target = self.get_or_create_descriptor(target_name)
        return self.composite.get_facet_breakdown(source, target)
    
    def list_domains(self) -> List[str]:
        """List all registered domain names."""
        return list(self.descriptors.keys())
    
    def get_descriptor(self, domain_name: str) -> Optional[DomainDescriptor]:
        """Get descriptor for a domain."""
        return self.descriptors.get(domain_name)
    
    @property
    def is_loaded(self) -> bool:
        """Check if index has been loaded."""
        return self._loaded
    
    @property
    def last_refresh(self) -> Optional[datetime]:
        """Get timestamp of last Top-K refresh."""
        return self._last_refresh
    
    def clear_cache(self) -> None:
        """Clear all cached similarities."""
        self._sim_cache.clear()
        self._top_k_cache.clear()


# =============================================================================
# TransferResult (Extended)
# =============================================================================

@dataclass
class FacetedTransferResult:
    """
    Result of a faceted transfer operation.
    
    Extends the basic TransferResult with facet breakdown information.
    """
    target_mu: float
    target_sigma: float
    source_domains: List[str]
    source_weights: Dict[str, float]
    correlation_used: float
    confidence: float
    
    # Facet breakdown for interpretability
    facet_contributions: Dict[str, Dict] = field(default_factory=dict)
    
    # Transfer source indicator
    source: str = "transfer"  # "direct" or "transfer"
    
    def to_rating(self) -> TrueSkillRating:
        """Convert to TrueSkillRating."""
        return TrueSkillRating(mu=self.target_mu, sigma=self.target_sigma)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "target_mu": self.target_mu,
            "target_sigma": self.target_sigma,
            "conservative_estimate": self.target_mu - 3 * self.target_sigma,
            "source_domains": self.source_domains,
            "source_weights": self.source_weights,
            "correlation_used": self.correlation_used,
            "confidence": self.confidence,
            "facet_contributions": self.facet_contributions,
            "source": self.source,
        }


# =============================================================================
# FacetedTransferLearning
# =============================================================================

class FacetedTransferLearning:
    """
    Transfer learning using faceted domain similarity.
    
    Extends TransferLearning to use CompositeDomainSimilarity
    instead of flat domain correlations.
    
    Supports:
    - Multi-source transfer with weighted combination
    - All 10 rating dimensions
    - Interpretable facet breakdowns
    - Learning from transfer accuracy
    
    Example:
        index = DomainIndex(domain_manager)
        index.load()
        
        ftl = FacetedTransferLearning(index)
        result = ftl.predict_rating(model, "odia_medical_imaging", "raw_quality")
    """
    
    def __init__(
        self,
        domain_index: DomainIndex,
        mu_0: float = TRUESKILL_MU_0,
        sigma_0: float = TRUESKILL_SIGMA_0,
        sigma_base: float = SIGMA_BASE,
        max_sources: int = 7,
        min_correlation: float = 0.25,
        cumulative_weight_threshold: float = 0.95,
    ):
        """
        Initialize faceted transfer learning.
        
        Args:
            domain_index: Pre-loaded domain index
            mu_0: Default mean rating
            sigma_0: Default uncertainty
            sigma_base: Baseline domain uncertainty
            max_sources: Maximum source domains to use
            min_correlation: Minimum correlation for source selection
            cumulative_weight_threshold: Stop when cumulative weight reaches this
        """
        self.domain_index = domain_index
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma_base = sigma_base
        self.max_sources = max_sources
        self.min_correlation = min_correlation
        self.cumulative_weight_threshold = cumulative_weight_threshold
    
    def predict_rating(
        self,
        model: 'LLMModel',
        target_domain: str,
        dimension: str = "raw"
    ) -> FacetedTransferResult:
        """
        Predict rating in target domain using multi-source transfer.
        
        Args:
            model: Model to predict rating for
            target_domain: Domain to predict in
            dimension: Rating dimension ("raw", "cost_adjusted", etc.)
        
        Returns:
            FacetedTransferResult with predicted rating
        """
        # Step 1: Get similar domains from index
        similar = self.domain_index.get_similar_domains(
            target_domain,
            k=self.max_sources * 2,  # Get more candidates for filtering
            min_correlation=self.min_correlation
        )
        
        # Step 2: Filter to domains where model has confident ratings
        source_ratings = {}
        correlations = {}
        
        for domain_name, rho in similar:
            rating = self._get_model_rating(model, domain_name, dimension)
            if rating and rating.sigma < self.sigma_0 * 0.8:
                source_ratings[domain_name] = rating
                correlations[domain_name] = rho
                
                if len(source_ratings) >= self.max_sources:
                    break
        
        # Step 3: Handle no sources - return default rating
        if not source_ratings:
            return FacetedTransferResult(
                target_mu=self.mu_0,
                target_sigma=self.sigma_0,
                source_domains=[],
                source_weights={},
                correlation_used=0.0,
                confidence=0.0,
                facet_contributions={},
                source="transfer",
            )
        
        # Step 4: Compute weights (reliability-weighted)
        weights = {}
        total_weight = 0.0
        
        for domain, rating in source_ratings.items():
            rho = correlations[domain]
            w = abs(rho) / rating.sigma
            weights[domain] = w
            total_weight += w
        
        # Normalize weights
        weights = {d: w / total_weight for d, w in weights.items()}
        
        # Apply cumulative weight threshold
        sorted_domains = sorted(weights.keys(), key=lambda d: -weights[d])
        cumulative = 0.0
        selected_domains = []
        
        for domain in sorted_domains:
            selected_domains.append(domain)
            cumulative += weights[domain]
            if cumulative >= self.cumulative_weight_threshold:
                break
        
        # Re-normalize weights for selected domains
        selected_weight_sum = sum(weights[d] for d in selected_domains)
        weights = {d: weights[d] / selected_weight_sum for d in selected_domains}
        
        # Step 5: Transfer mean
        mu_target = self.mu_0
        for domain in selected_domains:
            rating = source_ratings[domain]
            rho = correlations[domain]
            w = weights[domain]
            mu_target += w * rho * (rating.mu - self.mu_0)
        
        # Step 6: Transfer uncertainty
        variance_sum = 0.0
        for domain in selected_domains:
            rating = source_ratings[domain]
            rho = correlations[domain]
            w = weights[domain]
            individual_var = (
                rating.sigma ** 2 +
                (1 - rho ** 2) * self.sigma_0 ** 2 +
                self.sigma_base ** 2
            )
            variance_sum += w * individual_var
        
        sigma_target = math.sqrt(variance_sum)
        
        # Step 7: Compute confidence
        avg_rho = sum(weights[d] * correlations[d] for d in selected_domains)
        confidence = abs(avg_rho) * (1 - sigma_target / self.sigma_0)
        confidence = max(0, min(1, confidence))
        
        # Step 8: Get facet breakdown for top source
        facet_contributions = {}
        if selected_domains:
            top_source = selected_domains[0]
            facet_contributions = self.domain_index.get_facet_breakdown(
                top_source, target_domain
            )
        
        return FacetedTransferResult(
            target_mu=mu_target,
            target_sigma=sigma_target,
            source_domains=selected_domains,
            source_weights={d: weights[d] for d in selected_domains},
            correlation_used=avg_rho,
            confidence=confidence,
            facet_contributions=facet_contributions,
            source="transfer",
        )
    
    def _get_model_rating(
        self,
        model: 'LLMModel',
        domain: str,
        dimension: str
    ) -> Optional[TrueSkillRating]:
        """Get model's rating in a domain for a specific dimension."""
        if domain not in model.trueskill_by_domain:
            return None
        
        dual_ts = model.trueskill_by_domain[domain]
        
        # Get the appropriate dimension
        if dimension == "raw" or dimension == "raw_quality":
            return dual_ts.raw
        elif dimension == "cost_adjusted" or dimension == "cost":
            return dual_ts.cost_adjusted
        elif hasattr(dual_ts, dimension):
            return getattr(dual_ts, dimension)
        else:
            # Default to raw
            return dual_ts.raw
    
    def predict_all_dimensions(
        self,
        model: 'LLMModel',
        target_domain: str
    ) -> Dict[str, FacetedTransferResult]:
        """
        Predict ratings for all 10 dimensions.
        
        Args:
            model: Model to predict ratings for
            target_domain: Domain to predict in
        
        Returns:
            Dict of dimension → FacetedTransferResult
        """
        dimensions = [
            "raw_quality", "cost_adjusted", "latency", "ttft",
            "consistency", "token_efficiency", "instruction_following",
            "hallucination_resistance", "long_context", "combined"
        ]
        
        results = {}
        for dim in dimensions:
            results[dim] = self.predict_rating(model, target_domain, dim)
        
        return results
    
    def get_rating_or_transfer(
        self,
        model: 'LLMModel',
        domain: str,
        dimension: str = "raw"
    ) -> FacetedTransferResult:
        """
        Get model rating if exists, otherwise transfer.
        
        This is the main entry point for the Model Rank API.
        
        Args:
            model: Model to get rating for
            domain: Domain to get rating in
            dimension: Rating dimension
        
        Returns:
            FacetedTransferResult with source="direct" or source="transfer"
        """
        rating = self._get_model_rating(model, domain, dimension)
        
        if rating and rating.sigma < self.sigma_0 * 0.8:
            # Direct rating exists and is confident
            return FacetedTransferResult(
                target_mu=rating.mu,
                target_sigma=rating.sigma,
                source_domains=[domain],
                source_weights={domain: 1.0},
                correlation_used=1.0,
                confidence=1.0 - rating.sigma / self.sigma_0,
                facet_contributions={},
                source="direct",
            )
        else:
            # Use transfer learning
            return self.predict_rating(model, domain, dimension)
    
    def update_from_transfer_accuracy(
        self,
        source_domain: str,
        target_domain: str,
        predicted_mu: float,
        actual_mu: float,
        sigma: float
    ) -> None:
        """
        Update facet similarities based on transfer prediction accuracy.
        
        When we predict a rating via transfer and later observe actual performance,
        we can update our facet similarity estimates.
        
        Args:
            source_domain: Domain used as source
            target_domain: Domain transferred to
            predicted_mu: Predicted rating
            actual_mu: Observed actual rating
            sigma: Uncertainty of prediction
        """
        error = abs(predicted_mu - actual_mu)
        accuracy = max(0, 1 - error / (2 * sigma))
        
        source_desc = self.domain_index.get_or_create_descriptor(source_domain)
        target_desc = self.domain_index.get_or_create_descriptor(target_domain)
        
        # Update each facet that differs
        for facet in ALL_FACETS:
            source_val = source_desc.get(facet)
            target_val = target_desc.get(facet)
            
            if source_val != target_val:
                learner = self.domain_index.composite.get_learner(facet)
                if learner:
                    current_sim = learner.get_similarity(source_val, target_val)
                    # Adjust based on accuracy
                    observed_sim = current_sim * accuracy + 0.5 * (1 - accuracy)
                    learner.update_from_observation(
                        source_val, target_val, observed_sim
                    )
    
    def update_from_paired_performance(
        self,
        domain_a: str,
        domain_b: str,
        model_ratings_a: Dict[str, float],
        model_ratings_b: Dict[str, float]
    ) -> None:
        """
        Update facet similarities from paired domain performance.
        
        If models that do well in domain A also do well in domain B,
        then shared facet values are indeed similar.
        
        Args:
            domain_a: First domain
            domain_b: Second domain
            model_ratings_a: {model_id: mu} for domain A
            model_ratings_b: {model_id: mu} for domain B
        """
        common_models = set(model_ratings_a.keys()) & set(model_ratings_b.keys())
        
        if len(common_models) < 3:
            return
        
        # Compute observed correlation
        vals_a = [model_ratings_a[m] for m in common_models]
        vals_b = [model_ratings_b[m] for m in common_models]
        observed_corr = self._pearson_correlation(vals_a, vals_b)
        
        if observed_corr < 0:
            return  # Negative correlation doesn't make sense for similarity
        
        desc_a = self.domain_index.get_or_create_descriptor(domain_a)
        desc_b = self.domain_index.get_or_create_descriptor(domain_b)
        
        # Update facets that differ
        for facet in ALL_FACETS:
            val_a = desc_a.get(facet)
            val_b = desc_b.get(facet)
            
            if val_a != val_b:
                learner = self.domain_index.composite.get_learner(facet)
                if learner:
                    learner.update_from_observation(
                        val_a, val_b, observed_corr, sample_size=len(common_models)
                    )
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if var_x == 0 or var_y == 0:
            return 0.0
        
        return cov / math.sqrt(var_x * var_y)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_index: Optional[DomainIndex] = None
_default_transfer: Optional[FacetedTransferLearning] = None


def get_domain_index(
    domain_manager: 'DomainManager' = None,
    reload: bool = False
) -> DomainIndex:
    """Get or create the default domain index."""
    global _default_index
    
    if _default_index is None or reload:
        _default_index = DomainIndex(domain_manager)
        if domain_manager:
            _default_index.load()
    
    return _default_index


def get_faceted_transfer() -> FacetedTransferLearning:
    """Get or create the default faceted transfer learning instance."""
    global _default_transfer
    
    if _default_transfer is None:
        _default_transfer = FacetedTransferLearning(get_domain_index())
    
    return _default_transfer


def parse_domain_name(domain_name: str) -> Dict[str, str]:
    """Parse domain name into facets using default parser."""
    parser = DomainNameParser()
    return parser.parse(domain_name)


def get_domain_similarity(
    source_domain: str,
    target_domain: str
) -> float:
    """Get similarity between two domains using default index."""
    return get_domain_index().get_correlation(source_domain, target_domain)

