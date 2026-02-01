"""
GraphQL type definitions for LangScope.

All GraphQL types are defined using Strawberry's decorator-based approach.
"""

import strawberry
from typing import Optional, List
from datetime import datetime


# === Provider & Pricing Types ===

@strawberry.type
class ProviderType:
    """Provider information for a model deployment."""
    name: str
    region: str
    endpoint: str


@strawberry.type
class PricingType:
    """Pricing information for a deployment."""
    input_cost_per_million: float
    output_cost_per_million: float
    currency: str
    effective_date: str


@strawberry.type
class PerformanceType:
    """Performance metrics for a deployment."""
    median_latency_ms: float
    p95_latency_ms: float
    median_ttft_ms: float
    p95_ttft_ms: float
    tokens_per_second: float
    uptime_pct: float
    sample_size: int


# === Rating Types ===

@strawberry.type
class TrueSkillRatingType:
    """A TrueSkill rating (mu, sigma)."""
    mu: float
    sigma: float


@strawberry.type
class DimensionRatingsType:
    """Ratings across all 10 dimensions."""
    raw_quality: TrueSkillRatingType
    cost_adjusted: TrueSkillRatingType
    latency: TrueSkillRatingType
    ttft: TrueSkillRatingType
    consistency: TrueSkillRatingType
    token_efficiency: TrueSkillRatingType
    instruction_following: TrueSkillRatingType
    hallucination_resistance: TrueSkillRatingType
    long_context: TrueSkillRatingType
    combined: TrueSkillRatingType


# === Architecture Types ===

@strawberry.type
class ArchitectureType:
    """Model architecture details."""
    type: str
    parameters: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int


@strawberry.type
class CapabilitiesType:
    """Model capabilities."""
    modalities: List[str]
    languages: List[str]
    supports_function_calling: bool
    supports_vision: bool
    supports_audio: bool


@strawberry.type
class ContextType:
    """Context window information."""
    max_context: int
    max_output: int
    optimal_context: int


@strawberry.type
class LicenseType:
    """License information."""
    type: str
    commercial_use: bool
    attribution_required: bool
    source_url: str


# === Main Entity Types ===

@strawberry.type
class BaseModelType:
    """A base LLM model (architecture/weights)."""
    id: str
    name: str
    family: str
    version: str
    organization: str
    architecture: ArchitectureType
    capabilities: CapabilitiesType
    context: ContextType
    license: LicenseType
    released_at: datetime
    created_at: datetime
    updated_at: datetime
    
    @strawberry.field
    async def deployments(self, info) -> List["ModelDeploymentType"]:
        """Get all deployments of this base model."""
        db = info.context.get("db")
        if db:
            deployments = await db.get_deployments_by_base_model(self.id)
            return [_deployment_to_type(d) for d in deployments]
        return []


@strawberry.type
class ModelDeploymentType:
    """A specific deployment of a base model by a provider."""
    id: str
    base_model_id: str
    name: str
    provider: ProviderType
    pricing: PricingType
    performance: PerformanceType
    available: bool
    created_at: datetime
    updated_at: datetime
    
    @strawberry.field
    async def base_model(self, info) -> Optional[BaseModelType]:
        """Get the base model for this deployment."""
        db = info.context.get("db")
        if db:
            base = await db.get_base_model(self.base_model_id)
            if base:
                return _base_model_to_type(base)
        return None
    
    @strawberry.field
    async def ratings(self, info, domain: Optional[str] = None) -> Optional[DimensionRatingsType]:
        """Get ratings for this deployment, optionally in a specific domain."""
        # TODO: Resolve from database
        return None


@strawberry.type
class SelfHostedDeploymentType:
    """A self-hosted deployment of a base model."""
    id: str
    base_model_id: str
    owner_id: str
    name: str
    available: bool
    created_at: datetime
    updated_at: datetime
    
    @strawberry.field
    async def base_model(self, info) -> Optional[BaseModelType]:
        """Get the base model for this deployment."""
        db = info.context.get("db")
        if db:
            base = await db.get_base_model(self.base_model_id)
            if base:
                return _base_model_to_type(base)
        return None


@strawberry.type
class DomainType:
    """A domain for evaluation."""
    id: str
    name: str
    description: str
    case_prompt: str
    evaluation_criteria: List[str]
    created_at: datetime


@strawberry.type
class MatchResultType:
    """Result of a match evaluation."""
    match_id: str
    domain: str
    rankings: List[str]
    scores: List[float]
    judge_model: str
    created_at: datetime


@strawberry.type
class LeaderboardEntryType:
    """Entry in the leaderboard."""
    rank: int
    deployment_id: str
    name: str
    provider: str
    rating_mu: float
    rating_sigma: float
    conservative_rating: float
    matches_played: int


@strawberry.type
class BenchmarkScoreType:
    """A benchmark score for a base model."""
    benchmark_id: str
    score: float
    variant: str
    source_url: str
    measured_at: datetime


# === Ground Truth Types (Phase 17-24) ===

@strawberry.type
class GroundTruthDomainType:
    """A ground truth evaluation domain."""
    name: str
    category: str
    primary_metric: str
    metrics: List[str]
    evaluation_mode: str
    sample_count: int


@strawberry.type
class GroundTruthSampleType:
    """A ground truth sample."""
    id: str
    domain: str
    category: str
    difficulty: str
    language: str
    usage_count: int


@strawberry.type
class GroundTruthMetricType:
    """A single metric value."""
    name: str
    value: float
    is_primary: bool


@strawberry.type
class GroundTruthScoreType:
    """Score from ground truth evaluation."""
    model_id: str
    sample_id: str
    overall: float
    metrics: List[GroundTruthMetricType]
    semantic_match: Optional[float]
    evaluation_mode: str


@strawberry.type
class GroundTruthMatchType:
    """Result of a ground truth match."""
    match_id: str
    domain: str
    timestamp: datetime
    sample_id: str
    participants: List[str]
    rankings: List[str]  # Ordered list of model IDs
    scores: List[GroundTruthScoreType]
    status: str
    duration_ms: float


@strawberry.type
class GroundTruthLeaderboardEntryType:
    """Entry in ground truth leaderboard."""
    rank: int
    deployment_id: str
    trueskill_mu: float
    trueskill_sigma: float
    primary_metric_avg: float
    total_evaluations: int


@strawberry.type
class NeedleHeatmapCellType:
    """Single cell in needle heatmap."""
    context_length: int
    needle_position: float
    accuracy: float
    sample_count: int


@strawberry.type
class NeedleHeatmapType:
    """Needle in haystack accuracy heatmap for a model."""
    model_id: str
    cells: List[NeedleHeatmapCellType]
    overall_accuracy: float


@strawberry.type
class GroundTruthCoverageType:
    """Sample coverage statistics for a domain."""
    domain: str
    total_samples: int
    used_samples: int
    coverage_percentage: float


# === Connection/Pagination Types ===

@strawberry.type
class PageInfo:
    """Pagination information."""
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]


@strawberry.type
class ModelDeploymentEdge:
    """Edge in a connection."""
    cursor: str
    node: ModelDeploymentType


@strawberry.type
class ModelDeploymentConnection:
    """Paginated list of model deployments."""
    edges: List[ModelDeploymentEdge]
    page_info: PageInfo
    total_count: int


# === Helper Functions ===

def _base_model_to_type(data: dict) -> BaseModelType:
    """Convert database document to GraphQL type."""
    arch = data.get("architecture", {})
    caps = data.get("capabilities", {})
    ctx = data.get("context", {})
    lic = data.get("license", {})
    
    return BaseModelType(
        id=str(data.get("_id", "")),
        name=data.get("name", ""),
        family=data.get("family", ""),
        version=data.get("version", ""),
        organization=data.get("organization", ""),
        architecture=ArchitectureType(
            type=arch.get("type", ""),
            parameters=arch.get("parameters", ""),
            hidden_size=arch.get("hidden_size", 0),
            num_layers=arch.get("num_layers", 0),
            num_attention_heads=arch.get("num_attention_heads", 0),
            vocab_size=arch.get("vocab_size", 0),
        ),
        capabilities=CapabilitiesType(
            modalities=caps.get("modalities", []),
            languages=caps.get("languages", []),
            supports_function_calling=caps.get("supports_function_calling", False),
            supports_vision=caps.get("supports_vision", False),
            supports_audio=caps.get("supports_audio", False),
        ),
        context=ContextType(
            max_context=ctx.get("max_context", 0),
            max_output=ctx.get("max_output", 0),
            optimal_context=ctx.get("optimal_context", 0),
        ),
        license=LicenseType(
            type=lic.get("type", ""),
            commercial_use=lic.get("commercial_use", False),
            attribution_required=lic.get("attribution_required", False),
            source_url=lic.get("source_url", ""),
        ),
        released_at=data.get("released_at", datetime.utcnow()),
        created_at=data.get("created_at", datetime.utcnow()),
        updated_at=data.get("updated_at", datetime.utcnow()),
    )


def _deployment_to_type(data: dict) -> ModelDeploymentType:
    """Convert database document to GraphQL type."""
    prov = data.get("provider", {})
    price = data.get("pricing", {})
    perf = data.get("performance", {})
    
    return ModelDeploymentType(
        id=str(data.get("_id", "")),
        base_model_id=data.get("base_model_id", ""),
        name=data.get("name", ""),
        provider=ProviderType(
            name=prov.get("name", ""),
            region=prov.get("region", ""),
            endpoint=prov.get("endpoint", ""),
        ),
        pricing=PricingType(
            input_cost_per_million=price.get("input_cost_per_million", 0),
            output_cost_per_million=price.get("output_cost_per_million", 0),
            currency=price.get("currency", "USD"),
            effective_date=price.get("effective_date", ""),
        ),
        performance=PerformanceType(
            median_latency_ms=perf.get("median_latency_ms", 0),
            p95_latency_ms=perf.get("p95_latency_ms", 0),
            median_ttft_ms=perf.get("median_ttft_ms", 0),
            p95_ttft_ms=perf.get("p95_ttft_ms", 0),
            tokens_per_second=perf.get("tokens_per_second", 0),
            uptime_pct=perf.get("uptime_pct", 0),
            sample_size=perf.get("sample_size", 0),
        ),
        available=data.get("available", True),
        created_at=data.get("created_at", datetime.utcnow()),
        updated_at=data.get("updated_at", datetime.utcnow()),
    )


# === Prompt Management Types (Phase PM) ===

@strawberry.type
class ClassificationResultType:
    """Result of domain classification."""
    category: str
    base_domain: str
    variant: Optional[str]
    confidence: float
    is_ground_truth: bool
    full_domain_name: str
    template_name: str


@strawberry.type
class PromptProcessingResultType:
    """Result of processing a prompt."""
    prompt: str
    domain: ClassificationResultType
    cache_hit: bool
    cache_layer: Optional[str]
    cache_similarity: Optional[float]
    evaluation_type: str
    processing_time_ms: float


@strawberry.type
class PromptMetricsType:
    """Prompt processing metrics."""
    exact_hits: int
    semantic_hits: int
    misses: int
    classifications: int
    total_requests: int
    exact_hit_rate: float
    semantic_hit_rate: float
    overall_hit_rate: float
    avg_time_ms: float


# === Cache Management Types (Phase C) ===

@strawberry.type
class CacheLayerHitsType:
    """Cache hits by layer."""
    local: int
    redis: int
    qdrant: int
    mongo: int


@strawberry.type
class CacheCategoryStatsType:
    """Statistics for a cache category."""
    category: str
    total_hits: int
    misses: int
    errors: int
    writes: int
    hit_rate: float
    local_entries: int


@strawberry.type
class CacheTotalsType:
    """Total cache statistics."""
    hits: int
    misses: int
    errors: int
    writes: int
    hit_rate: float


@strawberry.type
class CacheConnectionsType:
    """Cache connection status."""
    redis: bool
    mongodb: bool
    qdrant: bool


@strawberry.type
class CacheStatsType:
    """Comprehensive cache statistics."""
    by_category: List[CacheCategoryStatsType]
    totals: CacheTotalsType
    connections: CacheConnectionsType


@strawberry.type
class SessionType:
    """Session information."""
    session_id: str
    session_type: str
    user_id: Optional[str]
    status: str
    created_at: str
    last_activity: str
    expires_at: str
    domain: Optional[str]


@strawberry.type
class RateLimitStatusType:
    """Rate limit status."""
    current_count: int
    limit: int
    window: int
    remaining: int


# === Faceted Transfer Learning Types (Phase TL) ===

@strawberry.type
class FacetContributionType:
    """Contribution of a single facet to domain similarity."""
    facet: str
    source_value: str
    target_value: str
    similarity: float
    weight: float
    contribution: float


@strawberry.type
class TransferDetailsType:
    """Details of transfer learning computation."""
    source_domains: List[str]
    source_weights: List[float]  # Parallel to source_domains
    correlation_used: float
    facet_contributions: List[FacetContributionType]


@strawberry.type
class ModelRatingType:
    """Model rating in a domain with source indicator."""
    model_id: str
    domain: str
    dimension: str
    mu: float
    sigma: float
    conservative_estimate: float
    source: str  # "direct" or "transfer"
    confidence: float
    match_count: Optional[int]
    transfer_details: Optional[TransferDetailsType]
    last_updated: Optional[str]


@strawberry.type
class DimensionRatingType:
    """Rating for a specific dimension."""
    dimension: str
    mu: float
    sigma: float
    conservative_estimate: float
    source: str
    confidence: float


@strawberry.type
class ModelAllRatingsType:
    """All dimension ratings for a model in a domain."""
    model_id: str
    domain: str
    ratings: List[DimensionRatingType]


@strawberry.type
class DomainFacetsType:
    """Domain facets representation."""
    name: str
    language: str
    field: str
    modality: str
    task: str
    specialty: str


@strawberry.type
class SimilarDomainType:
    """Similar domain entry."""
    name: str
    correlation: float
    facet_breakdown: List[FacetContributionType]


@strawberry.type
class FacetedLeaderboardEntryType:
    """Leaderboard entry with transfer indicator."""
    rank: int
    model_id: str
    deployment_id: Optional[str]
    mu: float
    sigma: float
    conservative_estimate: float
    source: str  # "direct" or "transfer"
    confidence: float
    transfer_note: Optional[str]


@strawberry.type
class FacetedLeaderboardType:
    """Leaderboard with transfer-included entries."""
    domain: str
    dimension: str
    evaluation_type: str
    entries: List[FacetedLeaderboardEntryType]
    total_models: int
    direct_count: int
    transferred_count: int
    generated_at: str


@strawberry.type
class DomainSimilarityExplanationType:
    """Explanation of similarity between two domains."""
    source: str
    target: str
    correlation: float
    facet_contributions: List[FacetContributionType]


@strawberry.type
class FacetSimilarityType:
    """Learned similarity between two facet values."""
    facet: str
    value_a: str
    value_b: str
    prior_similarity: float
    data_similarity: float
    blended_similarity: float
    sample_count: int
    alpha: float
    confidence: float


@strawberry.type
class FacetSimilaritiesType:
    """All similarities for a facet."""
    facet: str
    tau: float
    count: int
    similarities: List[FacetSimilarityType]


@strawberry.type
class DomainIndexStatsType:
    """Statistics about the domain index."""
    total_domains: int
    domains_with_facets: int
    precomputed_similarities: int
    last_refresh: Optional[str]

