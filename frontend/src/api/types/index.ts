/**
 * TypeScript types for LangScope API
 * Generated from backend Pydantic schemas
 */

// =============================================================================
// Rating Types
// =============================================================================

export interface TrueSkillRating {
  mu: number
  sigma: number
}

export interface DualTrueSkill {
  raw: TrueSkillRating
  cost_adjusted: TrueSkillRating
}

export interface MultiDimensionalTrueSkill {
  raw_quality: TrueSkillRating
  cost_adjusted: TrueSkillRating
  latency: TrueSkillRating
  ttft: TrueSkillRating
  consistency: TrueSkillRating
  token_efficiency: TrueSkillRating
  instruction_following: TrueSkillRating
  hallucination_resistance: TrueSkillRating
  long_context: TrueSkillRating
  combined: TrueSkillRating
}

export type RatingDimension = keyof MultiDimensionalTrueSkill

// =============================================================================
// Model Types
// =============================================================================

export interface ModelResponse {
  name: string
  model_id: string
  provider: string
  input_cost_per_million: number
  output_cost_per_million: number
  pricing_source: string
  trueskill: DualTrueSkill
  trueskill_by_domain: Record<string, DualTrueSkill>
  multi_trueskill?: MultiDimensionalTrueSkill
  multi_trueskill_by_domain: Record<string, MultiDimensionalTrueSkill>
  total_matches_played: number
  domains_evaluated: string[]
  avg_latency_ms: number
  avg_ttft_ms: number
}

export interface ModelListResponse {
  models: ModelResponse[]
  total: number
}

// =============================================================================
// Domain Types
// =============================================================================

export interface DomainStatistics {
  total_matches: number
  total_models_evaluated: number
  top_model_raw: string | null
  top_model_cost: string | null
  last_match_timestamp: string | null
}

export interface DomainResponse {
  name: string
  display_name: string
  description: string
  parent_domain: string | null
  statistics: DomainStatistics
  created_at: string
  updated_at: string
}

export interface DomainListResponse {
  domains: string[]
  total: number
}

// =============================================================================
// Leaderboard Types
// =============================================================================

export interface LeaderboardEntry {
  rank: number
  name: string
  model_id: string
  provider: string
  mu: number
  sigma: number
  conservative_estimate: number
  matches_played: number
  avg_rank: number
  dimension?: string
}

export interface LeaderboardResponse {
  domain: string | null
  ranking_type: string
  entries: LeaderboardEntry[]
  total: number
  generated_at: string
}

export interface DimensionRating {
  mu: number
  sigma: number
  conservative_estimate: number
}

export interface MultiDimensionalLeaderboardEntry {
  rank: number
  name: string
  model_id: string
  provider: string
  dimension_ratings: Record<string, DimensionRating>
  matches_played: number
}

// =============================================================================
// Match Types
// =============================================================================

export interface MatchResult {
  match_id: string
  domain: string
  timestamp: string
  participants: string[]
  case_text: string
  question_text: string
  case_creator: string
  question_creator: string
  judges: string[]
  raw_ranking: Record<string, number>
  cost_adjusted_ranking: Record<string, number>
  pl_strengths: Record<string, number>
  info_bits: number
}

export interface MatchListResponse {
  matches: MatchResult[]
  total: number
}

// =============================================================================
// Arena Types
// =============================================================================

export interface ArenaSessionStart {
  domain: string
  use_case: string
  user_id?: string
  model_ids?: string[]
}

export interface ArenaSessionStartResponse {
  session_id: string
  domain: string
  use_case: string
  models_available: string[]
  predictions: Record<string, TrueSkillRating>
  started_at: string
}

export interface ArenaBattle {
  participant_ids: string[]
  user_ranking: Record<string, number>
}

export interface ArenaBattleResponse {
  battle_number: number
  participants_updated: string[]
  message: string
}

export interface ArenaSessionResult {
  session_id: string
  domain: string
  use_case: string
  n_battles: number
  n_models: number
  prediction_accuracy: number
  kendall_tau: number
  conservation_satisfied: boolean
  delta_sum: number
  biggest_winner: Record<string, unknown>
  biggest_loser: Record<string, unknown>
  n_specialists: number
  n_underperformers: number
  deltas: Record<string, Record<string, number>>
}

// =============================================================================
// Recommendation Types
// =============================================================================

export interface RecommendationEntry {
  model_id: string
  model_name: string
  adjusted_mu: number
  global_mu: number
  adjustment: number
  provider: string
}

export interface RecommendationResponse {
  use_case: string
  domain: string | null
  beta: number
  n_users: number
  recommendations: RecommendationEntry[]
}

// =============================================================================
// Common Types
// =============================================================================

export interface HealthResponse {
  status: string
  version: string
  database_connected: boolean
  timestamp: string
}

export interface ErrorResponse {
  error: string
  detail?: string
  code: string
}

export interface SuccessResponse {
  success: boolean
  message: string
}

// =============================================================================
// Auth Types
// =============================================================================

export interface UserResponse {
  id: string
  email: string
  name?: string
  avatar_url?: string
  created_at: string
  last_sign_in?: string
}

export interface AuthStatusResponse {
  authenticated: boolean
  user_id?: string
  expires_at?: string
}

export interface AuthInfoResponse {
  provider: string
  requires_auth: boolean
  login_url?: string
}

export interface TokenVerifyResponse {
  valid: boolean
  user_id?: string
  expires_at?: string
}

// =============================================================================
// Transfer Learning Types
// =============================================================================

export interface TransferPrediction {
  model_id: string
  target_domain: string
  source_domains?: string[]
}

export interface TransferPredictionResult {
  model_id: string
  target_domain: string
  predicted_mu: number
  predicted_sigma: number
  source_weights: Record<string, number>
  confidence: number
}

export interface CorrelationUpdate {
  domain_a: string
  domain_b: string
  correlation: number
  prior?: boolean
}

export interface CorrelationResponse {
  domain_a: string
  domain_b: string
  correlation: number
  n_observations: number
  alpha: number
}

export interface FacetContribution {
  facet: string
  source_value: string
  target_value: string
  similarity: number
  weight: number
  contribution: number
}

export interface TransferDetails {
  source_domains: string[]
  source_weights: Record<string, number>
  correlation_used: number
  facet_contributions?: Record<string, FacetContribution>
}

export interface ModelRating {
  model_id: string
  domain: string
  dimension: string
  rating: TrueSkillRating
  conservative_estimate: number
  source: 'direct' | 'transfer'
  confidence: number
  match_count?: number
  transfer_details?: TransferDetails
  last_updated?: string
}

export interface ModelRatings {
  model_id: string
  domain: string
  ratings: Record<string, ModelRating>
}

export interface DomainFacets {
  name: string
  facets: Record<string, string>
}

export interface SimilarDomain {
  name: string
  correlation: number
  facet_breakdown?: Record<string, FacetContribution>
}

export interface SimilarDomainsResponse {
  domain: string
  facets: Record<string, string>
  similar_domains: SimilarDomain[]
}

export interface FacetedLeaderboardEntry {
  rank: number
  model_id: string
  deployment_id?: string
  rating: TrueSkillRating
  conservative_estimate: number
  source: 'direct' | 'transfer'
  confidence: number
  transfer_note?: string
}

export interface FacetedLeaderboardResponse {
  domain: string
  dimension: string
  evaluation_type: string
  entries: FacetedLeaderboardEntry[]
  total_models: number
  direct_count: number
  transferred_count: number
  generated_at: string
}

export interface FacetSimilarity {
  facet: string
  value_a: string
  value_b: string
  prior_similarity: number
  data_similarity: number
  blended_similarity: number
  sample_count: number
  alpha: number
  confidence: number
}

export interface FacetPriorUpdate {
  facet: string
  value_a: string
  value_b: string
  similarity: number
}

export interface DomainIndexStats {
  total_domains: number
  domains_with_facets: number
  precomputed_similarities: number
  last_refresh?: string
}

export interface RefreshIndexResponse {
  success: boolean
  domains_indexed: number
  similarities_computed: number
  duration_ms: number
}

// =============================================================================
// Specialist Detection Types
// =============================================================================

export interface SpecialistQuery {
  model_id: string
  target_domain?: string
}

export interface SpecialistResult {
  model_id: string
  domain: string
  is_specialist: boolean
  z_score: number
  actual_mu: number
  predicted_mu: number
  p_value: number
  category: 'specialist' | 'weak_spot' | 'normal'
}

export interface SpecialistProfile {
  model_id: string
  model_name: string
  domains_evaluated: string[]
  specialist_domains: string[]
  weak_spot_domains: string[]
  specialization_score: number
  is_generalist: boolean
  detailed_results: SpecialistResult[]
}

// =============================================================================
// Base Model Types
// =============================================================================

export interface Architecture {
  type: string
  parameters: number
  parameters_display: string
  hidden_size: number
  num_layers: number
  num_attention_heads: number
  num_kv_heads: number
  vocab_size: number
  max_position_embeddings: number
  native_precision: string
  native_size_gb: number
}

export interface Capabilities {
  modalities: string[]
  languages: string[]
  supports_function_calling: boolean
  supports_json_mode: boolean
  supports_vision: boolean
  supports_audio: boolean
  supports_system_prompt: boolean
  supports_streaming: boolean
  trained_for: string[]
}

export interface ContextWindow {
  max_context_length: number
  recommended_context: number
  max_output_tokens: number
  quality_at_context: Record<string, number>
}

export interface License {
  type: string
  commercial_use: boolean
  requires_agreement: boolean
  restrictions: string[]
  url: string
}

export interface QuantizationOption {
  bits: number
  vram_gb: number
  ram_gb?: number
  quality_retention: number
  huggingface_id?: string
  supported_frameworks: string[]
  notes: string
}

export interface BenchmarkScore {
  score: number
  variant: string
  percentile?: number
  updated_at: string
}

export interface BenchmarkAggregates {
  open_llm_average: number
  knowledge_average: number
  reasoning_average: number
  coding_average: number
  math_average: number
  chat_average: number
  overall_rank: number
  total_models_ranked: number
}

export interface BaseModelCreate {
  id: string
  name: string
  family?: string
  version?: string
  organization?: string
  architecture?: Partial<Architecture>
  capabilities?: Partial<Capabilities>
  context?: Partial<ContextWindow>
  license?: Partial<License>
}

export interface BaseModelResponse {
  id: string
  name: string
  family: string
  version: string
  organization: string
  architecture: Architecture
  capabilities: Capabilities
  context: ContextWindow
  license: License
  quantizations: Record<string, QuantizationOption>
  benchmarks: Record<string, BenchmarkScore>
  benchmark_aggregates: BenchmarkAggregates
  deployment_count: number
  created_at: string
  updated_at: string
}

export interface BaseModelListResponse {
  models: BaseModelResponse[]
  total: number
}

// =============================================================================
// Deployment Types
// =============================================================================

export interface Provider {
  id: string
  name: string
  type: string
  api_base: string
  api_compatible: string
  website: string
  docs: string
}

export interface DeploymentConfig {
  model_id: string
  display_name: string
  quantization: string
  serving_framework: string
  max_context_length: number
  max_output_tokens: number
  notes: string
}

export interface Pricing {
  input_cost_per_million: number
  output_cost_per_million: number
  currency: string
  source_id: string
  source_url: string
  last_verified: string
}

export interface Performance {
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  avg_ttft_ms: number
  tokens_per_second: number
  uptime_30d: number
  error_rate_30d: number
}

export interface Availability {
  status: string
  regions: string[]
  requires_waitlist: boolean
  requires_enterprise: boolean
}

export interface DeploymentCreate {
  id: string
  base_model_id: string
  provider: Provider
  deployment: DeploymentConfig
  pricing?: Partial<Pricing>
}

export interface DeploymentResponse {
  id: string
  base_model_id: string
  provider: Provider
  deployment: DeploymentConfig
  pricing: Pricing
  performance: Performance
  availability: Availability
  trueskill: DualTrueSkill
  performance_stats: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface DeploymentListResponse {
  deployments: DeploymentResponse[]
  total: number
}

export interface ProviderComparisonEntry {
  deployment_id: string
  provider: Provider
  pricing: Pricing
  performance: Performance
  trueskill: DualTrueSkill
}

export interface ProviderComparisonResponse {
  base_model_id: string
  base_model_name: string
  providers: ProviderComparisonEntry[]
}

// =============================================================================
// Self-Hosted Deployment Types
// =============================================================================

export interface HardwareConfig {
  gpu_type: string
  gpu_count: number
  gpu_memory_gb: number
  cpu_cores: number
  ram_gb: number
  cloud_provider: string
  instance_type: string
  region: string
}

export interface SoftwareConfig {
  serving_framework: string
  framework_version: string
  quantization: string
  quantization_source: string
  tensor_parallel_size: number
  max_model_len: number
  gpu_memory_utilization: number
}

export interface SelfHostedCosts {
  input_cost_per_million: number
  output_cost_per_million: number
  hourly_compute_cost: number
  notes: string
}

export interface SelfHostedCreate {
  deployment_name: string
  base_model_id: string
  hardware: HardwareConfig
  software: SoftwareConfig
  costs: SelfHostedCosts
  is_public?: boolean
}

export interface SelfHostedResponse {
  id: string
  base_model_id: string
  owner_user_id: string
  is_public: boolean
  hardware: HardwareConfig
  software: SoftwareConfig
  costs: SelfHostedCosts
  trueskill: DualTrueSkill
  performance_stats: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface SelfHostedListResponse {
  deployments: SelfHostedResponse[]
  total: number
}

export interface CostEstimateRequest {
  hourly_compute_cost: number
  expected_throughput_tps: number
  utilization?: number
}

export interface CostEstimateResponse {
  input_cost_per_million: number
  output_cost_per_million: number
  assumptions: Record<string, unknown>
}

// =============================================================================
// Benchmark Types
// =============================================================================

export interface BenchmarkDefinition {
  _id: string  // Backend uses _id
  id?: string  // Alias for compatibility
  name: string
  category: string
  description: string
  what_it_tests?: string[]
  scoring?: {
    method: string
    min_value: number
    max_value: number
    higher_is_better: boolean
  }
  metric_name?: string  // Legacy field
  metric_direction?: 'asc' | 'desc'  // Legacy field
  source_url?: string
  automation_available?: boolean
  variants?: string[]
  created_at: string
}

export interface CreateBenchmarkDefinition {
  id: string
  name: string
  category: string
  description?: string
  metric_name: string
  metric_direction: 'asc' | 'desc'
  source_url?: string
  variants?: string[]
}

export interface BenchmarkResult {
  id: string
  base_model_id: string
  benchmark_id: string
  score: number
  variant?: string
  percentile?: number
  source_url?: string
  evaluated_at: string
}

export interface CreateBenchmarkResult {
  base_model_id: string
  benchmark_id: string
  score: number
  variant?: string
  source_url?: string
}

export interface BenchmarkCorrelation {
  benchmark_id: string
  dimension: string
  correlation: number
  sample_count: number
  p_value: number
}

export interface BenchmarkLeaderboardEntry {
  base_model_id: string
  base_model_name: string
  score: number
  percentile: number
  rank: number
}

// =============================================================================
// Prompt Types
// =============================================================================

export interface ClassifyRequest {
  prompt: string
  context?: string
}

export interface ClassifyResponse {
  domain: string
  confidence: number
  language: string
  language_confidence: number
  secondary_domains?: Array<{ domain: string; confidence: number }>
}

export interface ProcessRequest {
  prompt: string
  model_id?: string
  domain?: string
}

export interface ProcessResponse {
  processed_prompt: string
  detected_domain: string
  recommended_models: string[]
  cache_key?: string
}

export interface CacheResponseRequest {
  cache_key: string
  response: string
  model_id: string
  latency_ms: number
}

export interface PromptMetrics {
  total_classified: number
  total_processed: number
  cache_hits: number
  cache_misses: number
  avg_classification_time_ms: number
  domain_distribution: Record<string, number>
}

// =============================================================================
// Cache Types
// =============================================================================

export interface CacheStats {
  total_entries: number
  total_size_bytes: number
  hit_rate: number
  miss_rate: number
  eviction_count: number
  categories: Record<string, { entries: number; size_bytes: number }>
}

export interface InvalidateResponse {
  success: boolean
  entries_invalidated: number
  message: string
}

export interface SessionInfo {
  session_id: string
  user_id?: string
  created_at: string
  last_activity: string
  request_count: number
  data: Record<string, unknown>
}

export interface CreateSessionRequest {
  user_id?: string
  data?: Record<string, unknown>
}

export interface UpdateSessionRequest {
  data: Record<string, unknown>
}

export interface RateLimitStatus {
  identifier: string
  endpoint?: string
  requests_made: number
  requests_limit: number
  window_seconds: number
  reset_at: string
  is_limited: boolean
}

// =============================================================================
// Ground Truth Types
// =============================================================================

export interface GroundTruthDomainInfo {
  name: string
  category: string
  description: string
  sample_count: number
  primary_metric: string
  supported_languages: string[]
  difficulty_distribution: Record<string, number>
}

export interface GroundTruthSample {
  id: string
  domain: string
  category: string
  difficulty: string
  language: string
  inputs: Record<string, unknown>
  ground_truth: Record<string, unknown>
  metadata: Record<string, unknown>
  usage_count: number
}

export interface GroundTruthMatchResponse {
  id: string
  domain: string
  timestamp: string
  sample_id: string
  participants: string[]
  scores: Record<string, { metrics: Record<string, number>; overall: number }>
  rankings: Record<string, number>
  status: string
  duration_ms: number
}

export interface EvaluationRequest {
  domain: string
  deployment_ids: string[]
  sample_id?: string
  sample_count?: number
  stratification?: Record<string, string>
}

export interface GroundTruthLeaderboardEntry {
  deployment_id: string
  rank: number
  trueskill_mu: number
  trueskill_sigma: number
  primary_metric_avg: number
  total_evaluations: number
  last_evaluation: string
}

export interface NeedleHeatmapResponse {
  model_id: string
  domain: string
  heatmap: Array<{
    context_length: number
    needle_position: number
    accuracy: number
    sample_count: number
  }>
  overall_accuracy: number
}

export interface ModelPerformanceResponse {
  model_id: string
  domain: string
  metrics: Record<string, number>
  by_difficulty: Record<string, Record<string, number>>
  by_language: Record<string, Record<string, number>>
  total_evaluations: number
}

export interface GroundTruthCoverage {
  domain: string
  total_samples: number
  used_samples: number
  coverage_percentage: number
  stratification_coverage: Record<string, Record<string, { total: number; used: number; coverage: number }>>
}

// =============================================================================
// Monitoring Types
// =============================================================================

export interface DashboardResponse {
  system_status: string
  database_status: string
  cache_status: string
  total_models: number
  total_domains: number
  total_matches_24h: number
  active_sessions: number
  error_rate_24h: number
  avg_latency_ms: number
  alerts_count: number
}

export interface MonitoringHealthResponse {
  status: string
  components: Record<string, { status: string; latency_ms?: number; error?: string }>
  timestamp: string
}

export interface Alert {
  id: string
  level: 'info' | 'warning' | 'error' | 'critical'
  domain?: string
  message: string
  details: Record<string, unknown>
  created_at: string
  resolved_at?: string
  is_resolved: boolean
}

export interface CoverageResponse {
  domains: Record<string, {
    total_models: number
    evaluated_models: number
    coverage_percentage: number
    last_evaluation: string
  }>
  overall_coverage: number
}

export interface FreshnessResponse {
  domains: Record<string, {
    last_match: string
    hours_since_last: number
    is_stale: boolean
  }>
  stale_domains: string[]
  threshold_hours: number
}

export interface ErrorSummary {
  total_errors: number
  by_type: Record<string, number>
  by_domain: Record<string, number>
  recent_errors: Array<{
    timestamp: string
    type: string
    message: string
    domain?: string
  }>
}

// =============================================================================
// Parameters Types
// =============================================================================

export interface ParamType {
  id: string
  name: string
  description: string
  schema: Record<string, unknown>
}

export interface ParamTypesResponse {
  types: ParamType[]
}

export interface ParamResponse {
  param_type: string
  domain?: string
  params: Record<string, unknown>
  updated_at: string
  is_default: boolean
}

export interface ParamUpdateRequest {
  params: Record<string, unknown>
  domain?: string
}

export interface ParamCacheStats {
  total_cached: number
  cache_hits: number
  cache_misses: number
  hit_rate: number
}

export interface ParamExportResponse {
  params: Record<string, Record<string, unknown>>
  exported_at: string
  version: string
}

export interface ParamImportResponse {
  success: boolean
  imported_count: number
  skipped_count: number
  errors: string[]
}

// =============================================================================
// User Model Types (Private Testing)
// =============================================================================

/**
 * Model types supported by the platform
 */
export type ModelType = 
  | 'LLM'
  | 'ASR'
  | 'TTS'
  | 'VLM'
  | 'V2V'
  | 'STT'
  | 'ImageGen'
  | 'VideoGen'
  | 'Embedding'
  | 'Reranker'

/**
 * API configuration for a user's private model
 */
export interface ModelAPIConfig {
  endpoint: string
  apiKey?: string  // Stored encrypted, not returned in responses
  hasApiKey: boolean  // Whether an API key is configured
  modelId: string
  apiFormat: 'openai' | 'anthropic' | 'google' | 'custom'
  headers?: Record<string, string>
  extraParams?: Record<string, unknown>
}

/**
 * Cost configuration for user models
 */
export interface ModelCosts {
  inputCostPerMillion: number
  outputCostPerMillion: number
  currency: string
  isEstimate: boolean
  notes?: string
}

/**
 * Type-specific configuration for different model types
 */
export interface ModelTypeSpecificConfig {
  // ASR/TTS specific
  language?: string
  sampleRate?: number
  // VLM specific
  imageDetail?: 'auto' | 'low' | 'high'
  // ImageGen specific
  imageSize?: string
  steps?: number
  guidanceScale?: number
  // Embedding specific
  embeddingDimension?: number
  normalize?: boolean
  // General
  maxTokens?: number
  temperature?: number
}

/**
 * User's private model registration
 */
export interface UserModel {
  id: string
  userId: string
  name: string
  description?: string
  modelType: ModelType
  version: string
  baseModelId?: string  // Link to base model if applicable
  apiConfig: Omit<ModelAPIConfig, 'apiKey'>  // API key not returned
  typeConfig: ModelTypeSpecificConfig
  costs: ModelCosts
  isPublic: boolean
  isActive: boolean
  trueskill?: MultiDimensionalTrueSkill
  groundTruthMetrics?: Record<string, number>
  totalEvaluations: number
  domainsEvaluated: string[]
  lastEvaluatedAt?: string
  createdAt: string
  updatedAt: string
}

/**
 * Request to create a new user model
 */
export interface UserModelCreate {
  name: string
  description?: string
  modelType: ModelType
  version: string
  baseModelId?: string
  apiConfig: ModelAPIConfig
  typeConfig?: ModelTypeSpecificConfig
  costs: ModelCosts
  isPublic?: boolean
}

/**
 * Request to update a user model
 */
export interface UserModelUpdate {
  name?: string
  description?: string
  version?: string
  apiConfig?: Partial<ModelAPIConfig>
  typeConfig?: Partial<ModelTypeSpecificConfig>
  costs?: Partial<ModelCosts>
  isPublic?: boolean
  isActive?: boolean
}

/**
 * List response for user models
 */
export interface UserModelListResponse {
  models: UserModel[]
  total: number
  byType: Record<ModelType, number>
}

/**
 * Performance data for a user model
 */
export interface UserModelPerformance {
  modelId: string
  modelType: ModelType
  // TrueSkill scores (for LLM/VLM)
  trueskill?: MultiDimensionalTrueSkill
  trueskillByDomain?: Record<string, MultiDimensionalTrueSkill>
  // Ground truth metrics (type-specific)
  groundTruthMetrics?: Record<string, number>
  groundTruthByDomain?: Record<string, Record<string, number>>
  // Evaluation history
  evaluationHistory: Array<{
    timestamp: string
    domain: string
    metricType: 'trueskill' | 'ground_truth'
    scores: Record<string, number>
  }>
  // Comparison with public models
  publicRank?: number
  publicTotal?: number
  percentile?: number
}

/**
 * Comparison between user models and public leaderboard
 */
export interface ModelComparisonEntry {
  modelId: string
  name: string
  isUserModel: boolean
  modelType: ModelType
  provider?: string
  metrics: Record<string, number>
  rank: number
  costs?: ModelCosts
}

export interface ModelComparisonResponse {
  domain: string
  modelType: ModelType
  metric: string
  entries: ModelComparisonEntry[]
  userModelIds: string[]
}

/**
 * Request to run evaluation on user model
 */
export interface RunEvaluationRequest {
  modelId: string
  domain: string
  evaluationType: 'subjective' | 'ground_truth'
  sampleCount?: number
  competitors?: string[]  // Model IDs to compete against
}

/**
 * Response from running evaluation
 */
export interface RunEvaluationResponse {
  evaluationId: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  modelId: string
  domain: string
  estimatedDurationMs?: number
  queuePosition?: number
}

/**
 * Evaluation status check
 */
export interface EvaluationStatus {
  evaluationId: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress?: number  // 0-100
  currentStep?: string
  results?: UserModelPerformance
  error?: string
}

// =============================================================================
// User Profile Types
// =============================================================================

export interface UserProfile {
  user_id: string
  email: string
  display_name?: string
  avatar_url?: string
  phone?: string
  timezone: string
  language: string
  organization_id?: string
  role_in_org?: string
  plan: 'free' | 'pro' | 'enterprise'
  created_at?: string
  updated_at?: string
}

export interface UpdateProfileRequest {
  display_name?: string
  phone?: string
  timezone?: string
  language?: string
}

export interface ChangePasswordRequest {
  current_password: string
  new_password: string
}

export interface Session {
  session_id: string
  device: string
  ip_address: string
  location?: string
  last_active: string
  is_current: boolean
}

export interface SessionsResponse {
  sessions: Session[]
}

export interface AvatarResponse {
  avatar_url: string
  message: string
}

// =============================================================================
// Organization Types
// =============================================================================

export interface OrganizationSettings {
  default_domain?: string
  allowed_email_domains: string[]
  sso_enabled: boolean
}

export interface Organization {
  id: string
  name: string
  slug: string
  logo_url?: string
  description?: string
  website?: string
  owner_id: string
  plan: 'free' | 'pro' | 'enterprise'
  settings: OrganizationSettings
  created_at: string
  member_count: number
}

export interface CreateOrganizationRequest {
  name: string
  description?: string
  website?: string
}

export interface UpdateOrganizationRequest {
  name?: string
  description?: string
  website?: string
  settings?: Partial<OrganizationSettings>
}

export interface TeamMember {
  id: string
  user_id: string
  organization_id: string
  email: string
  display_name?: string
  avatar_url?: string
  role: 'owner' | 'admin' | 'member' | 'viewer'
  status: 'active' | 'pending' | 'suspended'
  joined_at: string
}

export interface TeamMembersResponse {
  members: TeamMember[]
  total: number
}

export interface InviteMemberRequest {
  email: string
  role: 'admin' | 'member' | 'viewer'
}

export interface UpdateMemberRoleRequest {
  role: 'admin' | 'member' | 'viewer'
}

export interface Invitation {
  id: string
  organization_id: string
  organization_name: string
  email: string
  role: string
  invited_by: string
  invite_code: string
  expires_at: string
  status: 'pending' | 'accepted' | 'expired' | 'revoked'
}

export interface InvitationsResponse {
  invitations: Invitation[]
}

export interface JoinOrganizationRequest {
  invite_code: string
}

// =============================================================================
// Billing & Subscription Types
// =============================================================================

export interface PlanFeatures {
  evaluations_per_month: number
  team_members: number
  api_access: boolean
  all_domains: boolean
  custom_domains: boolean
  priority_support: boolean
  export_reports: boolean
  sso_saml: boolean
  sla_guarantee: boolean
}

export interface Plan {
  id: string
  name: string
  price_monthly: number
  price_yearly: number
  features: PlanFeatures
  popular: boolean
}

export interface PlansResponse {
  plans: Plan[]
}

export interface UsageStats {
  evaluations: number
  evaluations_limit: number
  api_calls: number
  api_calls_limit: number
  team_members: number
  team_members_limit: number
  period_start: string
  period_end: string
}

export interface Subscription {
  id: string
  organization_id: string
  plan: 'free' | 'pro' | 'enterprise'
  plan_name: string
  status: 'active' | 'past_due' | 'canceled' | 'trialing'
  current_period_start: string
  current_period_end: string
  cancel_at_period_end: boolean
  usage: UsageStats
}

export interface SubscribeRequest {
  plan_id: string
  payment_method_id?: string
}

export interface ChangePlanRequest {
  plan_id: string
}

export interface PaymentMethod {
  id: string
  type: 'card' | 'bank_account'
  last_four: string
  brand?: string
  exp_month?: number
  exp_year?: number
  is_default: boolean
}

export interface PaymentMethodsResponse {
  payment_methods: PaymentMethod[]
}

export interface AddPaymentMethodRequest {
  token: string
}

export interface Invoice {
  id: string
  organization_id: string
  amount: number
  currency: string
  status: 'paid' | 'pending' | 'failed'
  description: string
  created_at: string
  pdf_url?: string
}

export interface InvoicesResponse {
  invoices: Invoice[]
  total: number
}
