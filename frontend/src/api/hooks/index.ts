/**
 * React Query hooks for LangScope API
 */

// Existing hooks
export { useModels, useModel } from './useModels'
export { useDomains, useDomain } from './useDomains'
export { useLeaderboard, useMultiDimensionalLeaderboard } from './useLeaderboard'
export { useMatches } from './useMatches'
export { useArenaSession, useArenaBattle, useCompleteArenaSession, useArenaSessionStatus } from './useArena'
export { useRecommendations } from './useRecommendations'

// Auth hooks
export {
  useCurrentUser,
  useAuthStatus,
  useAuthInfo,
  useVerifyToken,
} from './useAuth'

// Transfer Learning hooks
export {
  useTransferPrediction,
  useCorrelation,
  useDomainCorrelations,
  useModelRating,
  useModelRatings,
  useSimilarDomains,
  useDomainFacets,
  useFacetSimilarities,
  useDomainIndexStats,
  useTransferLeaderboard,
  useSetCorrelation,
  useSetDomainFacets,
  useSetFacetPrior,
  useRefreshSimilarityIndex,
  useTransferRatings,
} from './useTransfer'

// Specialist Detection hooks
export {
  useDetectSpecialist,
  useSpecialistProfile,
  useDomainSpecialists,
  useGeneralists,
  useSpecialistSummary,
} from './useSpecialists'

// Base Models hooks
export {
  useBaseModels,
  useBaseModel,
  useProviderComparison,
  useCreateBaseModel,
  useDeleteBaseModel,
} from './useBaseModels'

// Deployments hooks
export {
  useDeployments,
  useDeployment,
  useDeploymentsByBaseModel,
  useBestDeployment,
  useCreateDeployment,
  useDeleteDeployment,
} from './useDeployments'

// Self-Hosted Deployments hooks
export {
  useSelfHostedDeployments,
  usePublicSelfHosted,
  useSelfHostedDeployment,
  useCreateSelfHosted,
  useEstimateCosts,
  useDeleteSelfHosted,
} from './useSelfHosted'

// Benchmarks hooks
export {
  useBenchmarkDefinitions,
  useBenchmarkDefinition,
  useBenchmarkResults,
  useBenchmarkComparison,
  useBenchmarkCorrelations,
  useBenchmarkLeaderboard,
  useCreateBenchmarkDefinition,
  useDeleteBenchmarkDefinition,
  useCreateBenchmarkResult,
} from './useBenchmarks'

// Prompts hooks
export {
  useClassifyPrompt,
  useProcessPrompt,
  useCacheResponse,
  usePromptMetrics,
  useResetPromptMetrics,
  usePromptDomains,
  usePromptLanguages,
} from './usePrompts'

// Cache hooks
export {
  useCacheStats,
  useInvalidateCategory,
  useInvalidateLeaderboard,
  useInvalidateAllCache,
  useResetCacheStats,
  useCreateSession,
  useSession,
  useUpdateSession,
  useEndSession,
  useRateLimitStatus,
  useResetRateLimit,
} from './useCache'

// Ground Truth hooks
export {
  useGroundTruthDomains,
  useGroundTruthDomainInfo,
  useGroundTruthSamples,
  useGroundTruthSample,
  useRandomSample,
  useGroundTruthMatches,
  useGroundTruthMatch,
  useGroundTruthLeaderboard,
  useGroundTruthLanguageLeaderboard,
  useNeedleHeatmap,
  useModelPerformance,
  useGroundTruthCoverage,
  useTriggerEvaluation,
  useGetBatchSamples,
} from './useGroundTruth'

// Monitoring hooks
export {
  useDashboard,
  useMonitoringHealth,
  useAlerts,
  useCheckAlerts,
  useResolveAlert,
  useCoverageSummary,
  useDomainCoverage,
  useErrorSummary,
  useFreshness,
} from './useMonitoring'

// Parameters hooks
export {
  useParamTypes,
  useParams,
  useUpdateParams,
  useRemoveDomainOverride,
  useResetParams,
  useParamCacheStats,
  useInvalidateParamCache,
  useExportParams,
  useImportParams,
} from './useParams'
