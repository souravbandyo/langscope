"""
Transfer learning module for LangScope.

Implements cross-domain knowledge transfer and specialist detection:
- Domain correlation learning with Bayesian smoothing
- Single and multi-source transfer
- Specialist and generalist detection
- Multi-faceted domain similarity (NEW)
- Faceted transfer learning for Model Rank API (NEW)

New in v2.0 (Transfer Learning Enhancement):
- DomainDescriptor: Faceted domain representation
- FacetSimilarityLearner: Per-facet Bayesian similarity learning
- CompositeDomainSimilarity: Weighted combination of facet similarities
- DomainIndex: In-memory index with pre-computed Top-K
- FacetedTransferLearning: Transfer using faceted correlations
- Expert priors for all facets
"""

from langscope.transfer.correlation import (
    CorrelationLearner,
    get_correlation,
    set_prior_correlation,
)
from langscope.transfer.transfer_learning import (
    TransferLearning,
    TransferResult,
    transfer_single_source,
    transfer_multi_source,
)
from langscope.transfer.specialist import (
    SpecialistDetector,
    SpecialistResult,
    detect_specialist,
)

# New faceted transfer learning exports
from langscope.transfer.faceted import (
    # Core data structures
    DomainDescriptor,
    FacetSimilarityData,
    FacetedTransferResult,
    # Learners and similarity
    FacetSimilarityLearner,
    CompositeDomainSimilarity,
    DomainNameParser,
    # Index and transfer
    DomainIndex,
    FacetedTransferLearning,
    # Constants
    FACET_LANGUAGE,
    FACET_FIELD,
    FACET_MODALITY,
    FACET_TASK,
    FACET_SPECIALTY,
    ALL_FACETS,
    DEFAULT_FACET_WEIGHTS,
    DEFAULT_FACET_TAU,
    # Convenience functions
    get_domain_index,
    get_faceted_transfer,
    parse_domain_name,
    get_domain_similarity,
)

from langscope.transfer.priors import (
    LANGUAGE_PRIORS,
    FIELD_PRIORS,
    MODALITY_PRIORS,
    TASK_PRIORS,
    SPECIALTY_PRIORS,
    ALL_PRIORS,
    load_priors_into_learner,
    load_all_priors,
    create_initialized_composite,
    get_prior_statistics,
)

__all__ = [
    # Legacy transfer learning
    "CorrelationLearner",
    "get_correlation",
    "set_prior_correlation",
    "TransferLearning",
    "TransferResult",
    "transfer_single_source",
    "transfer_multi_source",
    "SpecialistDetector",
    "SpecialistResult",
    "detect_specialist",
    # Core data structures
    "DomainDescriptor",
    "FacetSimilarityData",
    "FacetedTransferResult",
    # Learners and similarity
    "FacetSimilarityLearner",
    "CompositeDomainSimilarity",
    "DomainNameParser",
    # Index and transfer
    "DomainIndex",
    "FacetedTransferLearning",
    # Constants
    "FACET_LANGUAGE",
    "FACET_FIELD",
    "FACET_MODALITY",
    "FACET_TASK",
    "FACET_SPECIALTY",
    "ALL_FACETS",
    "DEFAULT_FACET_WEIGHTS",
    "DEFAULT_FACET_TAU",
    # Convenience functions
    "get_domain_index",
    "get_faceted_transfer",
    "parse_domain_name",
    "get_domain_similarity",
    # Priors
    "LANGUAGE_PRIORS",
    "FIELD_PRIORS",
    "MODALITY_PRIORS",
    "TASK_PRIORS",
    "SPECIALTY_PRIORS",
    "ALL_PRIORS",
    "load_priors_into_learner",
    "load_all_priors",
    "create_initialized_composite",
    "get_prior_statistics",
]


