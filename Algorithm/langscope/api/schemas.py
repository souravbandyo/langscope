"""
Pydantic schemas for API request/response validation.

Provides type-safe request and response models for all endpoints.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


# =============================================================================
# Model Schemas
# =============================================================================

class ModelCreate(BaseModel):
    """Request to create a new model."""
    name: str = Field(..., description="Unique model name")
    model_id: str = Field(..., description="Provider-specific model identifier")
    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    input_cost_per_million: float = Field(0.0, description="Cost per million input tokens")
    output_cost_per_million: float = Field(0.0, description="Cost per million output tokens")
    pricing_source: str = Field("", description="Source of pricing information")
    max_matches: int = Field(1000, description="Maximum matches this model can participate in")


class ModelUpdate(BaseModel):
    """Request to update a model."""
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    pricing_source: Optional[str] = None
    max_matches: Optional[int] = None
    notes: Optional[str] = None


class TrueSkillRatingSchema(BaseModel):
    """TrueSkill rating."""
    mu: float
    sigma: float


class DualTrueSkillSchema(BaseModel):
    """Dual TrueSkill rating (raw + cost-adjusted)."""
    raw: TrueSkillRatingSchema
    cost_adjusted: TrueSkillRatingSchema


class MultiDimensionalTrueSkillSchema(BaseModel):
    """10-Dimensional TrueSkill ratings."""
    raw_quality: TrueSkillRatingSchema
    cost_adjusted: TrueSkillRatingSchema
    latency: TrueSkillRatingSchema
    ttft: TrueSkillRatingSchema
    consistency: TrueSkillRatingSchema
    token_efficiency: TrueSkillRatingSchema
    instruction_following: TrueSkillRatingSchema
    hallucination_resistance: TrueSkillRatingSchema
    long_context: TrueSkillRatingSchema
    combined: TrueSkillRatingSchema


class BattleMetricsSchema(BaseModel):
    """Battle metrics for 10D evaluation."""
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    consistency_runs: int = 0
    response_variance: float = 0.0
    constraints_satisfied: int = 0
    total_constraints: int = 0
    hallucination_count: int = 0
    verifiable_claims: int = 0
    context_length: int = 0
    quality_at_length: float = 0.0


class DimensionWeightsSchema(BaseModel):
    """Weights for Combined dimension calculation."""
    raw_quality: float = 0.20
    cost_adjusted: float = 0.10
    latency: float = 0.10
    ttft: float = 0.05
    consistency: float = 0.10
    token_efficiency: float = 0.10
    instruction_following: float = 0.15
    hallucination_resistance: float = 0.15
    long_context: float = 0.05


class ModelResponse(BaseModel):
    """Model response."""
    name: str
    model_id: str
    provider: str
    input_cost_per_million: float
    output_cost_per_million: float
    pricing_source: str
    trueskill: DualTrueSkillSchema
    trueskill_by_domain: Dict[str, DualTrueSkillSchema] = {}
    multi_trueskill: Optional[MultiDimensionalTrueSkillSchema] = None
    multi_trueskill_by_domain: Dict[str, MultiDimensionalTrueSkillSchema] = {}
    total_matches_played: int = 0
    domains_evaluated: List[str] = []
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0


class ModelListResponse(BaseModel):
    """List of models response."""
    models: List[ModelResponse]
    total: int


# =============================================================================
# Domain Schemas
# =============================================================================

class DomainSettingsSchema(BaseModel):
    """Domain settings."""
    strata_elite_threshold: int = 1520
    strata_high_threshold: int = 1450
    strata_mid_threshold: int = 1400
    players_per_match: int = 6
    min_players: int = 5
    judge_count: int = 5


class DomainPromptsSchema(BaseModel):
    """Domain prompts."""
    case_prompt: str = ""
    question_prompt: str = ""
    answer_prompt: str = ""
    judge_prompt: str = ""


class DomainCreate(BaseModel):
    """Request to create a new domain."""
    name: str = Field(..., description="Unique domain identifier")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    description: str = Field("", description="Domain description")
    parent_domain: Optional[str] = Field(None, description="Parent domain for correlation")
    template: Optional[str] = Field(None, description="Template to base domain on")
    settings: Optional[DomainSettingsSchema] = None
    prompts: Optional[DomainPromptsSchema] = None


class DomainUpdate(BaseModel):
    """Request to update a domain."""
    display_name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[DomainSettingsSchema] = None
    prompts: Optional[DomainPromptsSchema] = None


class DomainStatisticsSchema(BaseModel):
    """Domain statistics."""
    total_matches: int = 0
    total_models_evaluated: int = 0
    top_model_raw: Optional[str] = None
    top_model_cost: Optional[str] = None
    last_match_timestamp: Optional[str] = None


class DomainResponse(BaseModel):
    """Domain response."""
    name: str
    display_name: str
    description: str
    parent_domain: Optional[str]
    statistics: DomainStatisticsSchema
    created_at: str
    updated_at: str


class DomainListResponse(BaseModel):
    """List of domains response."""
    domains: List[str]
    total: int


# =============================================================================
# Match Schemas
# =============================================================================

class MatchTrigger(BaseModel):
    """Request to trigger a match."""
    domain: str = Field(..., description="Domain for the match")
    model_ids: Optional[List[str]] = Field(None, description="Specific models to include (optional)")


class MatchStatus(BaseModel):
    """Match status response."""
    match_id: str
    status: str  # "pending", "running", "completed", "failed"
    domain: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class MatchParticipantResult(BaseModel):
    """Result for a single participant in a match."""
    model_id: str
    raw_rank: int
    cost_rank: int
    response_text: str
    tokens_used: int
    cost_usd: float


class MatchResult(BaseModel):
    """Complete match result."""
    match_id: str
    domain: str
    timestamp: str
    participants: List[str]
    case_text: str
    question_text: str
    case_creator: str
    question_creator: str
    judges: List[str]
    raw_ranking: Dict[str, int]
    cost_adjusted_ranking: Dict[str, int]
    pl_strengths: Dict[str, float]
    info_bits: float


class MatchListResponse(BaseModel):
    """List of matches response."""
    matches: List[MatchResult]
    total: int


# =============================================================================
# Leaderboard Schemas
# =============================================================================

class LeaderboardEntry(BaseModel):
    """Single leaderboard entry."""
    rank: int
    name: str
    model_id: str
    provider: str
    mu: float
    sigma: float
    conservative_estimate: float
    matches_played: int
    avg_rank: float
    dimension: Optional[str] = None  # The dimension this ranking is for


class LeaderboardResponse(BaseModel):
    """Leaderboard response."""
    domain: Optional[str]
    ranking_type: str  # Dimension name (raw_quality, cost_adjusted, latency, etc.)
    entries: List[LeaderboardEntry]
    total: int
    generated_at: str


class DimensionRatingSchema(BaseModel):
    """Rating for a single dimension."""
    mu: float
    sigma: float
    conservative_estimate: float


class MultiDimensionalLeaderboardEntry(BaseModel):
    """Leaderboard entry with all dimension ratings."""
    rank: int
    name: str
    model_id: str
    provider: str
    dimension_ratings: Dict[str, DimensionRatingSchema]  # dim_name -> rating
    matches_played: int


class MultiDimensionalLeaderboardResponse(BaseModel):
    """Multi-dimensional leaderboard response."""
    domain: Optional[str]
    sort_by: str  # Dimension used for sorting
    entries: List[MultiDimensionalLeaderboardEntry]
    total: int
    generated_at: str


# =============================================================================
# Transfer Learning Schemas
# =============================================================================

class TransferPrediction(BaseModel):
    """Transfer learning prediction request."""
    model_id: str = Field(..., description="Model to predict for")
    target_domain: str = Field(..., description="Target domain")
    source_domains: Optional[List[str]] = Field(None, description="Source domains to use")


class TransferPredictionResult(BaseModel):
    """Transfer learning prediction result."""
    model_id: str
    target_domain: str
    predicted_mu: float
    predicted_sigma: float
    source_weights: Dict[str, float]
    confidence: float


class CorrelationUpdate(BaseModel):
    """Request to update domain correlation."""
    domain_a: str
    domain_b: str
    correlation: float = Field(..., ge=-1.0, le=1.0)
    prior: bool = Field(True, description="Whether this is a prior estimate")


class CorrelationResponse(BaseModel):
    """Domain correlation response."""
    domain_a: str
    domain_b: str
    correlation: float
    n_observations: int
    alpha: float


# =============================================================================
# Specialist Detection Schemas
# =============================================================================

class SpecialistQuery(BaseModel):
    """Query for specialist detection."""
    model_id: str
    target_domain: Optional[str] = None  # None = check all domains


class SpecialistResultSchema(BaseModel):
    """Specialist detection result."""
    model_id: str
    domain: str
    is_specialist: bool
    z_score: float
    actual_mu: float
    predicted_mu: float
    p_value: float
    category: str  # "specialist", "weak_spot", "normal"


class SpecialistProfileResponse(BaseModel):
    """Complete specialist profile for a model."""
    model_id: str
    model_name: str
    domains_evaluated: List[str]
    specialist_domains: List[str]
    weak_spot_domains: List[str]
    specialization_score: float
    is_generalist: bool
    detailed_results: List[SpecialistResultSchema]


# =============================================================================
# Arena Mode (User Feedback) Schemas
# =============================================================================

class ArenaSessionStart(BaseModel):
    """Request to start an arena session."""
    domain: str = Field(..., description="Domain for evaluation")
    use_case: str = Field(..., description="User's use case category")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    model_ids: Optional[List[str]] = Field(None, description="Specific models to test")


class ArenaSessionStartResponse(BaseModel):
    """Response when arena session is started."""
    session_id: str
    domain: str
    use_case: str
    models_available: List[str]
    predictions: Dict[str, TrueSkillRatingSchema]
    started_at: str


class ArenaBattle(BaseModel):
    """Submit a battle judgment in arena mode."""
    participant_ids: List[str] = Field(..., min_length=2, description="Models in this battle")
    user_ranking: Dict[str, int] = Field(..., description="User's ranking (1=best)")


class ArenaBattleResponse(BaseModel):
    """Response after processing a battle."""
    battle_number: int
    participants_updated: List[str]
    message: str


class ArenaSessionComplete(BaseModel):
    """Request to complete an arena session."""
    judge_rankings: Optional[Dict[str, Dict[str, int]]] = Field(
        None, description="Optional judge rankings for calibration"
    )


class ArenaSessionResult(BaseModel):
    """Results of a completed arena session."""
    session_id: str
    domain: str
    use_case: str
    n_battles: int
    n_models: int
    prediction_accuracy: float
    kendall_tau: float
    conservation_satisfied: bool
    delta_sum: float
    biggest_winner: Dict[str, Any]
    biggest_loser: Dict[str, Any]
    n_specialists: int
    n_underperformers: int
    deltas: Dict[str, Dict[str, float]]


# =============================================================================
# Recommendation Schemas
# =============================================================================

class RecommendationQuery(BaseModel):
    """Query for use-case recommendations."""
    use_case: str
    domain: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)


class RecommendationEntry(BaseModel):
    """Single recommendation entry."""
    model_id: str
    model_name: str
    adjusted_mu: float
    global_mu: float
    adjustment: float
    provider: str


class RecommendationResponse(BaseModel):
    """Use-case recommendation response."""
    use_case: str
    domain: Optional[str]
    beta: float
    n_users: int
    recommendations: List[RecommendationEntry]


# =============================================================================
# Common Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database_connected: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: str


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool
    message: str


# =============================================================================
# Phase 11: Base Model Schemas
# =============================================================================

class ArchitectureSchema(BaseModel):
    """Model architecture details."""
    type: str = "decoder-only"
    parameters: int = 0
    parameters_display: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    native_precision: str = "bfloat16"
    native_size_gb: float = 0.0


class CapabilitiesSchema(BaseModel):
    """Model capabilities."""
    modalities: List[str] = ["text"]
    languages: List[str] = ["en"]
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_system_prompt: bool = True
    supports_streaming: bool = True
    trained_for: List[str] = ["chat"]


class ContextWindowSchema(BaseModel):
    """Context window configuration."""
    max_context_length: int = 4096
    recommended_context: int = 4096
    max_output_tokens: int = 4096
    quality_at_context: Dict[str, float] = {}


class LicenseSchema(BaseModel):
    """License information."""
    type: str = "unknown"
    commercial_use: bool = False
    requires_agreement: bool = False
    restrictions: List[str] = []
    url: str = ""


class QuantizationOptionSchema(BaseModel):
    """Quantization option."""
    bits: float
    vram_gb: float
    ram_gb: Optional[float] = None
    quality_retention: float = 1.0
    huggingface_id: Optional[str] = None
    supported_frameworks: List[str] = []
    notes: str = ""


class BenchmarkScoreSchema(BaseModel):
    """Benchmark score."""
    score: float
    variant: str = ""
    percentile: Optional[int] = None
    updated_at: str = ""


class BenchmarkAggregatesSchema(BaseModel):
    """Benchmark aggregates."""
    open_llm_average: float = 0.0
    knowledge_average: float = 0.0
    reasoning_average: float = 0.0
    coding_average: float = 0.0
    math_average: float = 0.0
    chat_average: float = 0.0
    overall_rank: int = 0
    total_models_ranked: int = 0


class BaseModelCreate(BaseModel):
    """Request to create a base model."""
    id: str = Field(..., description="Unique ID (e.g., 'meta-llama/llama-3.1-70b')")
    name: str = Field(..., description="Human-readable name")
    family: str = Field("", description="Model family")
    version: str = Field("", description="Version within family")
    organization: str = Field("", description="Creating organization")
    architecture: Optional[ArchitectureSchema] = None
    capabilities: Optional[CapabilitiesSchema] = None
    context: Optional[ContextWindowSchema] = None
    license: Optional[LicenseSchema] = None


class BaseModelResponse(BaseModel):
    """Base model response."""
    id: str
    name: str
    family: str
    version: str
    organization: str
    architecture: ArchitectureSchema
    capabilities: CapabilitiesSchema
    context: ContextWindowSchema
    license: LicenseSchema
    quantizations: Dict[str, QuantizationOptionSchema] = {}
    benchmarks: Dict[str, BenchmarkScoreSchema] = {}
    benchmark_aggregates: BenchmarkAggregatesSchema
    deployment_count: int = 0
    created_at: str
    updated_at: str


class BaseModelListResponse(BaseModel):
    """List of base models."""
    models: List[BaseModelResponse]
    total: int


# =============================================================================
# Phase 11: Deployment Schemas
# =============================================================================

class ProviderSchema(BaseModel):
    """Provider information."""
    id: str
    name: str
    type: str = "cloud"
    api_base: str = ""
    api_compatible: str = "openai"
    website: str = ""
    docs: str = ""


class DeploymentConfigSchema(BaseModel):
    """Deployment configuration."""
    model_id: str
    display_name: str = ""
    quantization: str = ""
    serving_framework: str = ""
    max_context_length: int = 4096
    max_output_tokens: int = 4096
    notes: str = ""


class PricingSchema(BaseModel):
    """Pricing information."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    currency: str = "USD"
    source_id: str = ""
    source_url: str = ""
    last_verified: str = ""


class PerformanceSchema(BaseModel):
    """Performance metrics."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    uptime_30d: float = 1.0
    error_rate_30d: float = 0.0


class AvailabilitySchema(BaseModel):
    """Availability status."""
    status: str = "active"
    regions: List[str] = []
    requires_waitlist: bool = False
    requires_enterprise: bool = False


class DeploymentCreate(BaseModel):
    """Request to create a deployment."""
    id: str = Field(..., description="Deployment ID (e.g., 'groq/llama-3.1-70b-versatile')")
    base_model_id: str = Field(..., description="ID of the base model")
    provider: ProviderSchema
    deployment: DeploymentConfigSchema
    pricing: Optional[PricingSchema] = None


class DeploymentResponse(BaseModel):
    """Deployment response."""
    id: str
    base_model_id: str
    provider: ProviderSchema
    deployment: DeploymentConfigSchema
    pricing: PricingSchema
    performance: PerformanceSchema
    availability: AvailabilitySchema
    trueskill: DualTrueSkillSchema
    performance_stats: Dict[str, Any] = {}
    created_at: str
    updated_at: str


class DeploymentListResponse(BaseModel):
    """List of deployments."""
    deployments: List[DeploymentResponse]
    total: int


class ProviderComparisonEntry(BaseModel):
    """Provider comparison entry."""
    deployment_id: str
    provider: ProviderSchema
    pricing: PricingSchema
    performance: PerformanceSchema
    trueskill: DualTrueSkillSchema


class ProviderComparisonResponse(BaseModel):
    """Provider comparison for a base model."""
    base_model_id: str
    base_model_name: str
    providers: List[ProviderComparisonEntry]


# =============================================================================
# Phase 11: Self-Hosted Deployment Schemas
# =============================================================================

class HardwareConfigSchema(BaseModel):
    """Hardware configuration."""
    gpu_type: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    ram_gb: float = 0.0
    cloud_provider: str = "other"
    instance_type: str = ""
    region: str = ""


class SoftwareConfigSchema(BaseModel):
    """Software configuration."""
    serving_framework: str = "other"
    framework_version: str = ""
    quantization: str = ""
    quantization_source: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9


class SelfHostedCostsSchema(BaseModel):
    """Self-hosted costs."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    hourly_compute_cost: float = 0.0
    notes: str = ""


class SelfHostedCreate(BaseModel):
    """Request to create a self-hosted deployment."""
    deployment_name: str = Field(..., description="Deployment name (user_id/name will be the ID)")
    base_model_id: str = Field(..., description="ID of the base model")
    hardware: HardwareConfigSchema
    software: SoftwareConfigSchema
    costs: SelfHostedCostsSchema
    is_public: bool = Field(False, description="Make visible to other users")


class SelfHostedResponse(BaseModel):
    """Self-hosted deployment response."""
    id: str
    base_model_id: str
    owner_user_id: str
    is_public: bool
    hardware: HardwareConfigSchema
    software: SoftwareConfigSchema
    costs: SelfHostedCostsSchema
    trueskill: DualTrueSkillSchema
    performance_stats: Dict[str, Any] = {}
    created_at: str
    updated_at: str


class SelfHostedListResponse(BaseModel):
    """List of self-hosted deployments."""
    deployments: List[SelfHostedResponse]
    total: int


class CostEstimateRequest(BaseModel):
    """Request to estimate per-token costs."""
    hourly_compute_cost: float
    expected_throughput_tps: float
    utilization: float = 0.7


class CostEstimateResponse(BaseModel):
    """Cost estimate response."""
    input_cost_per_million: float
    output_cost_per_million: float
    assumptions: Dict[str, Any]


# =============================================================================
# Faceted Transfer Learning Schemas (Model Rank API)
# =============================================================================

class FacetContributionSchema(BaseModel):
    """Contribution of a single facet to domain similarity."""
    facet: str
    source_value: str
    target_value: str
    similarity: float
    weight: float
    contribution: float


class TransferDetailsSchema(BaseModel):
    """Details of transfer learning computation."""
    source_domains: List[str]
    source_weights: Dict[str, float]
    correlation_used: float
    facet_contributions: Dict[str, FacetContributionSchema] = {}


class ModelRatingSchema(BaseModel):
    """Model rating in a domain with source indicator."""
    model_id: str
    domain: str
    dimension: str = "raw_quality"
    rating: TrueSkillRatingSchema
    conservative_estimate: float
    source: str = "direct"  # "direct" or "transfer"
    confidence: float
    match_count: Optional[int] = None
    transfer_details: Optional[TransferDetailsSchema] = None
    last_updated: Optional[str] = None


class ModelRatingsSchema(BaseModel):
    """All dimension ratings for a model in a domain."""
    model_id: str
    domain: str
    ratings: Dict[str, ModelRatingSchema]


class ModelRatingRequest(BaseModel):
    """Request for model rating."""
    model_id: str = Field(..., description="Model identifier")
    domain: str = Field(..., description="Domain to get rating for")
    dimension: str = Field("raw_quality", description="Rating dimension")
    explain: bool = Field(False, description="Include transfer explanation")


class DomainFacetsSchema(BaseModel):
    """Domain facets representation."""
    name: str
    facets: Dict[str, str]


class SimilarDomainSchema(BaseModel):
    """Similar domain entry."""
    name: str
    correlation: float
    facet_breakdown: Dict[str, FacetContributionSchema] = {}


class SimilarDomainsResponse(BaseModel):
    """Response for similar domains query."""
    domain: str
    facets: Dict[str, str]
    similar_domains: List[SimilarDomainSchema]


class FacetedLeaderboardEntry(BaseModel):
    """Leaderboard entry with transfer indicator."""
    rank: int
    model_id: str
    deployment_id: Optional[str] = None
    rating: TrueSkillRatingSchema
    conservative_estimate: float
    source: str = "direct"  # "direct" or "transfer"
    confidence: float
    transfer_note: Optional[str] = None


class FacetedLeaderboardResponse(BaseModel):
    """Leaderboard with transfer-included entries."""
    domain: str
    dimension: str
    evaluation_type: str = "subjective"
    entries: List[FacetedLeaderboardEntry]
    total_models: int
    direct_count: int
    transferred_count: int
    generated_at: str


class FacetSimilaritySchema(BaseModel):
    """Similarity between two facet values."""
    facet: str
    value_a: str
    value_b: str
    prior_similarity: float
    data_similarity: float
    blended_similarity: float
    sample_count: int
    alpha: float
    confidence: float


class FacetPriorUpdate(BaseModel):
    """Request to update a facet prior."""
    facet: str = Field(..., description="Facet name (language, field, etc.)")
    value_a: str = Field(..., description="First value")
    value_b: str = Field(..., description="Second value")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Prior similarity")


class DomainIndexStats(BaseModel):
    """Statistics about the domain index."""
    total_domains: int
    domains_with_facets: int
    precomputed_similarities: int
    last_refresh: Optional[str] = None


class RefreshIndexResponse(BaseModel):
    """Response after refreshing the similarity index."""
    success: bool
    domains_indexed: int
    similarities_computed: int
    duration_ms: float


# =============================================================================
# User Model Schemas (My Models / Private Testing)
# =============================================================================

class ModelAPIConfigSchema(BaseModel):
    """API configuration for a user model."""
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key (only for creation, not returned)")
    model_id: str = Field(..., description="Model identifier for API calls")
    api_format: str = Field("openai", description="API format (openai, anthropic, google, custom)")
    has_api_key: bool = Field(False, description="Whether an API key is configured")
    headers: Optional[Dict[str, str]] = None
    extra_params: Optional[Dict[str, Any]] = None


class ModelAPIConfigResponse(BaseModel):
    """API configuration response (without sensitive data)."""
    endpoint: str
    model_id: str
    api_format: str
    has_api_key: bool


class ModelTypeConfigSchema(BaseModel):
    """Type-specific configuration."""
    language: Optional[str] = None
    sample_rate: Optional[int] = None
    image_detail: Optional[str] = None
    image_size: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    embedding_dimension: Optional[int] = None
    normalize: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ModelCostsSchema(BaseModel):
    """Cost configuration."""
    input_cost_per_million: float = Field(0.0, ge=0)
    output_cost_per_million: float = Field(0.0, ge=0)
    currency: str = "USD"
    is_estimate: bool = True
    notes: Optional[str] = None


class UserModelCreate(BaseModel):
    """Request to create a new user model."""
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    model_type: str = Field(..., description="Model type (LLM, ASR, TTS, VLM, etc.)")
    version: str = Field("1.0", description="Model version")
    base_model_id: Optional[str] = Field(None, description="Link to base model")
    api_config: ModelAPIConfigSchema
    type_config: Optional[ModelTypeConfigSchema] = None
    costs: ModelCostsSchema
    is_public: bool = Field(False, description="Make model visible on public leaderboard")


class UserModelUpdate(BaseModel):
    """Request to update a user model."""
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    api_config: Optional[ModelAPIConfigSchema] = None
    type_config: Optional[ModelTypeConfigSchema] = None
    costs: Optional[ModelCostsSchema] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class UserModelResponse(BaseModel):
    """User model response."""
    id: str
    user_id: str
    name: str
    description: Optional[str]
    model_type: str
    version: str
    base_model_id: Optional[str]
    api_config: ModelAPIConfigResponse
    type_config: Dict[str, Any] = {}
    costs: ModelCostsSchema
    is_public: bool
    is_active: bool
    trueskill: Optional[MultiDimensionalTrueSkillSchema] = None
    ground_truth_metrics: Optional[Dict[str, float]] = None
    total_evaluations: int
    domains_evaluated: List[str]
    last_evaluated_at: Optional[str]
    created_at: str
    updated_at: str


class UserModelListResponse(BaseModel):
    """List of user models response."""
    models: List[UserModelResponse]
    total: int
    by_type: Dict[str, int] = {}


class UserModelPerformanceResponse(BaseModel):
    """Performance data for a user model."""
    model_id: str
    model_type: str
    trueskill: Optional[MultiDimensionalTrueSkillSchema] = None
    trueskill_by_domain: Optional[Dict[str, MultiDimensionalTrueSkillSchema]] = None
    ground_truth_metrics: Optional[Dict[str, float]] = None
    ground_truth_by_domain: Optional[Dict[str, Dict[str, float]]] = None
    evaluation_history: List[Dict[str, Any]] = []
    public_rank: Optional[int] = None
    public_total: Optional[int] = None
    percentile: Optional[float] = None


class ModelComparisonEntry(BaseModel):
    """Entry in model comparison."""
    model_id: str
    name: str
    is_user_model: bool
    model_type: str
    provider: Optional[str] = None
    metrics: Dict[str, float]
    rank: int
    costs: Optional[ModelCostsSchema] = None


class ModelComparisonResponse(BaseModel):
    """Comparison between user models and public leaderboard."""
    domain: str
    model_type: str
    metric: str
    entries: List[ModelComparisonEntry]
    user_model_ids: List[str]


class RunEvaluationRequest(BaseModel):
    """Request to run evaluation on a user model."""
    model_id: str = Field(..., description="User model ID")
    domain: str = Field(..., description="Domain to evaluate in")
    evaluation_type: str = Field(..., description="'subjective' or 'ground_truth'")
    sample_count: Optional[int] = Field(10, ge=1, le=100)
    competitors: Optional[List[str]] = Field(None, description="Model IDs to compete against")


class RunEvaluationResponse(BaseModel):
    """Response after starting an evaluation."""
    evaluation_id: str
    status: str  # queued, running, completed, failed
    model_id: str
    domain: str
    estimated_duration_ms: Optional[int] = None
    queue_position: Optional[int] = None


class EvaluationStatusResponse(BaseModel):
    """Status of a running evaluation."""
    evaluation_id: str
    status: str
    progress: Optional[float] = None
    current_step: Optional[str] = None
    results: Optional[UserModelPerformanceResponse] = None
    error: Optional[str] = None


class TestConnectionRequest(BaseModel):
    """Request to test model API connection."""
    endpoint: str
    api_key: str
    model_id: str
    api_format: str = "openai"


class TestConnectionResponse(BaseModel):
    """Response from connection test."""
    success: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class UpdateApiKeyRequest(BaseModel):
    """Request to update API key."""
    api_key: str = Field(..., description="New API key")
