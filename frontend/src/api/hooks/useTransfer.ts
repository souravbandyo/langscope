/**
 * React Query hooks for Transfer Learning API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  TransferPrediction,
  TransferPredictionResult,
  CorrelationResponse,
  CorrelationUpdate,
  ModelRating,
  ModelRatings,
  SimilarDomainsResponse,
  DomainFacets,
  FacetedLeaderboardResponse,
  FacetSimilarity,
  FacetPriorUpdate,
  DomainIndexStats,
  RefreshIndexResponse,
} from '../types'

// =============================================================================
// Query Hooks
// =============================================================================

/**
 * Predict model performance in a target domain using transfer learning
 */
export function useTransferPrediction(params: TransferPrediction | null) {
  return useQuery({
    queryKey: ['transfer', 'predict', params],
    queryFn: () => api.post<TransferPredictionResult>('/transfer/predict', params),
    enabled: !!params?.model_id && !!params?.target_domain,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Get correlation between two domains
 */
export function useCorrelation(domainA: string, domainB: string) {
  return useQuery({
    queryKey: ['transfer', 'correlation', domainA, domainB],
    queryFn: () => api.get<CorrelationResponse>(`/transfer/correlation/${domainA}/${domainB}`),
    enabled: !!domainA && !!domainB,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Get all correlations for a domain
 */
export function useDomainCorrelations(domain: string) {
  return useQuery({
    queryKey: ['transfer', 'correlations', domain],
    queryFn: () => api.get<CorrelationResponse[]>(`/transfer/correlations/${domain}`),
    enabled: !!domain,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get model rating for a specific domain and dimension
 */
export function useModelRating(
  modelId: string,
  params?: { domain?: string; dimension?: string; explain?: boolean }
) {
  return useQuery({
    queryKey: ['transfer', 'models', modelId, 'rating', params],
    queryFn: () => api.get<ModelRating>(`/transfer/models/${modelId}/rating`, params),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get all dimension ratings for a model in a domain
 */
export function useModelRatings(
  modelId: string,
  params?: { domain?: string; dimensions?: string }
) {
  return useQuery({
    queryKey: ['transfer', 'models', modelId, 'ratings', params],
    queryFn: () => api.get<ModelRatings>(`/transfer/models/${modelId}/ratings`, params),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get similar domains
 */
export function useSimilarDomains(
  domain: string,
  params?: { limit?: number; min_correlation?: number }
) {
  return useQuery({
    queryKey: ['transfer', 'domains', domain, 'similar', params],
    queryFn: () => api.get<SimilarDomainsResponse>(`/transfer/domains/${domain}/similar`, params),
    enabled: !!domain,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get domain facets
 */
export function useDomainFacets(domain: string) {
  return useQuery({
    queryKey: ['transfer', 'domains', domain, 'facets'],
    queryFn: () => api.get<DomainFacets>(`/transfer/domains/${domain}/facets`),
    enabled: !!domain,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get facet similarities
 */
export function useFacetSimilarities(facet: string) {
  return useQuery({
    queryKey: ['transfer', 'similarity', 'facets', facet],
    queryFn: () => api.get<FacetSimilarity[]>(`/transfer/similarity/facets/${facet}`),
    enabled: !!facet,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get domain similarity index stats
 */
export function useDomainIndexStats() {
  return useQuery({
    queryKey: ['transfer', 'similarity', 'index', 'stats'],
    queryFn: () => api.get<DomainIndexStats>('/transfer/similarity/index/stats'),
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get transfer-aware leaderboard for a domain
 */
export function useTransferLeaderboard(
  domain: string,
  params?: { dimension?: string; include_transferred?: boolean; min_confidence?: number; limit?: number }
) {
  return useQuery({
    queryKey: ['transfer', 'leaderboard', domain, params],
    queryFn: () => api.get<FacetedLeaderboardResponse>(`/transfer/leaderboard/${domain}`, params),
    enabled: !!domain,
    staleTime: 60 * 1000, // 1 minute
  })
}

// =============================================================================
// Mutation Hooks
// =============================================================================

/**
 * Set correlation between two domains
 */
export function useSetCorrelation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: CorrelationUpdate) =>
      api.put<CorrelationResponse>('/transfer/correlation', data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['transfer', 'correlation', variables.domain_a] })
      queryClient.invalidateQueries({ queryKey: ['transfer', 'correlation', variables.domain_b] })
      queryClient.invalidateQueries({ queryKey: ['transfer', 'correlations'] })
    },
  })
}

/**
 * Set domain facets
 */
export function useSetDomainFacets() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ domain, facets }: { domain: string; facets: Record<string, string> }) =>
      api.post<DomainFacets>(`/transfer/domains/${domain}/facets`, facets),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['transfer', 'domains', variables.domain, 'facets'] })
      queryClient.invalidateQueries({ queryKey: ['transfer', 'similarity'] })
    },
  })
}

/**
 * Update facet prior similarity
 */
export function useSetFacetPrior() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ facet, ...data }: FacetPriorUpdate) =>
      api.put(`/transfer/similarity/facets/${facet}/prior`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transfer', 'similarity', 'facets'] })
    },
  })
}

/**
 * Refresh similarity index
 */
export function useRefreshSimilarityIndex() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<RefreshIndexResponse>('/transfer/similarity/index/refresh'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transfer', 'similarity'] })
      queryClient.invalidateQueries({ queryKey: ['transfer', 'domains'] })
    },
  })
}

/**
 * Transfer ratings to a target domain
 */
export function useTransferRatings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (params: { target_domain: string; source_domains?: string; model_ids?: string }) =>
      api.post(`/transfer/transfer-ratings?target_domain=${params.target_domain}${params.source_domains ? `&source_domains=${params.source_domains}` : ''}${params.model_ids ? `&model_ids=${params.model_ids}` : ''}`),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['transfer', 'leaderboard', variables.target_domain] })
      queryClient.invalidateQueries({ queryKey: ['transfer', 'models'] })
    },
  })
}
