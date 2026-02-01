/**
 * React Query hooks for Ground Truth Evaluation API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  GroundTruthDomainInfo,
  GroundTruthSample,
  GroundTruthMatchResponse,
  EvaluationRequest,
  GroundTruthLeaderboardEntry,
  NeedleHeatmapResponse,
  ModelPerformanceResponse,
  GroundTruthCoverage,
} from '../types'

/**
 * List available ground truth domains
 */
export function useGroundTruthDomains() {
  return useQuery({
    queryKey: ['ground-truth', 'domains'],
    queryFn: () => api.get<GroundTruthDomainInfo[]>('/ground-truth/domains'),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Get domain info
 */
export function useGroundTruthDomainInfo(domain: string) {
  return useQuery({
    queryKey: ['ground-truth', 'domains', domain, 'info'],
    queryFn: () => api.get<GroundTruthDomainInfo>(`/ground-truth/domains/${domain}/info`),
    enabled: !!domain,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * List ground truth samples
 */
export function useGroundTruthSamples(params?: {
  domain?: string
  difficulty?: string
  limit?: number
}) {
  return useQuery({
    queryKey: ['ground-truth', 'samples', params],
    queryFn: () => api.get<GroundTruthSample[]>('/ground-truth/samples', params),
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get a specific sample
 */
export function useGroundTruthSample(sampleId: string) {
  return useQuery({
    queryKey: ['ground-truth', 'samples', sampleId],
    queryFn: () => api.get<GroundTruthSample>(`/ground-truth/samples/${sampleId}`),
    enabled: !!sampleId,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get a random sample
 */
export function useRandomSample(params?: {
  domain?: string
  difficulty?: string
  language?: string
}) {
  return useQuery({
    queryKey: ['ground-truth', 'samples', 'random', params],
    queryFn: () => api.get<GroundTruthSample>('/ground-truth/samples/random', params),
    staleTime: 0, // Always refetch for random
    enabled: false, // Manual trigger only
  })
}

/**
 * List ground truth matches
 */
export function useGroundTruthMatches(params?: {
  domain?: string
  limit?: number
  skip?: number
}) {
  return useQuery({
    queryKey: ['ground-truth', 'matches', params],
    queryFn: () => api.get<GroundTruthMatchResponse[]>('/ground-truth/matches', params),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Get a specific ground truth match
 */
export function useGroundTruthMatch(matchId: string) {
  return useQuery({
    queryKey: ['ground-truth', 'matches', matchId],
    queryFn: () => api.get<GroundTruthMatchResponse>(`/ground-truth/matches/${matchId}`),
    enabled: !!matchId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get ground truth leaderboard for a domain
 */
export function useGroundTruthLeaderboard(
  domain: string,
  params?: { limit?: number }
) {
  return useQuery({
    queryKey: ['ground-truth', 'leaderboards', domain, params],
    queryFn: () => api.get<GroundTruthLeaderboardEntry[]>(`/ground-truth/leaderboards/${domain}`, params),
    enabled: !!domain,
    staleTime: 60 * 1000,
  })
}

/**
 * Get ground truth leaderboard for a domain and language
 */
export function useGroundTruthLanguageLeaderboard(
  domain: string,
  language: string,
  params?: { limit?: number }
) {
  return useQuery({
    queryKey: ['ground-truth', 'leaderboards', domain, language, params],
    queryFn: () => api.get<GroundTruthLeaderboardEntry[]>(`/ground-truth/leaderboards/${domain}/${language}`, params),
    enabled: !!domain && !!language,
    staleTime: 60 * 1000,
  })
}

/**
 * Get needle-in-haystack heatmap for a model
 */
export function useNeedleHeatmap(modelId: string) {
  return useQuery({
    queryKey: ['ground-truth', 'analytics', 'needle-heatmap', modelId],
    queryFn: () => api.get<NeedleHeatmapResponse>(`/ground-truth/analytics/needle-heatmap/${modelId}`),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get model performance analytics
 */
export function useModelPerformance(
  modelId: string,
  params?: { domain?: string }
) {
  return useQuery({
    queryKey: ['ground-truth', 'analytics', 'model-performance', modelId, params],
    queryFn: () => api.get<ModelPerformanceResponse>(`/ground-truth/analytics/model-performance/${modelId}`, params),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get coverage statistics for a domain
 */
export function useGroundTruthCoverage(domain: string) {
  return useQuery({
    queryKey: ['ground-truth', 'coverage', domain],
    queryFn: () => api.get<GroundTruthCoverage>(`/ground-truth/coverage/${domain}`),
    enabled: !!domain,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Trigger ground truth evaluation
 */
export function useTriggerEvaluation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: EvaluationRequest) =>
      api.post<GroundTruthMatchResponse>('/ground-truth/evaluate', data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['ground-truth', 'matches'] })
      queryClient.invalidateQueries({ queryKey: ['ground-truth', 'leaderboards', variables.domain] })
    },
  })
}

/**
 * Get batch samples for evaluation
 */
export function useGetBatchSamples() {
  return useMutation({
    mutationFn: (params: { domain: string; count: number; stratification?: Record<string, string> }) =>
      api.post<GroundTruthSample[]>(`/ground-truth/samples/batch?domain=${params.domain}&count=${params.count}`, params.stratification),
  })
}
