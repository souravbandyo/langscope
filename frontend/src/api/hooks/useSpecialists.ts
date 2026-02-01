/**
 * React Query hooks for Specialist Detection API
 */

import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../client'
import type {
  SpecialistQuery,
  SpecialistResult,
  SpecialistProfile,
} from '../types'

/**
 * Detect if a model is a specialist in domains
 */
export function useDetectSpecialist() {
  return useMutation({
    mutationFn: (query: SpecialistQuery) =>
      api.post<SpecialistResult[]>('/specialists/detect', query),
  })
}

/**
 * Get specialist profile for a model
 */
export function useSpecialistProfile(modelId: string) {
  return useQuery({
    queryKey: ['specialists', 'profile', modelId],
    queryFn: () => api.get<SpecialistProfile>(`/specialists/profile/${modelId}`),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Find specialists for a specific domain
 */
export function useDomainSpecialists(
  domain: string,
  params?: { include_weak_spots?: boolean }
) {
  return useQuery({
    queryKey: ['specialists', 'domain', domain, params],
    queryFn: () => api.get<SpecialistResult[]>(`/specialists/domain/${domain}`, params),
    enabled: !!domain,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Find generalist models
 */
export function useGeneralists(params?: { min_domains?: number }) {
  return useQuery({
    queryKey: ['specialists', 'generalists', params],
    queryFn: () => api.get<SpecialistProfile[]>('/specialists/generalists', params),
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get specialist detection summary
 */
export function useSpecialistSummary() {
  return useQuery({
    queryKey: ['specialists', 'summary'],
    queryFn: () => api.get<{
      total_specialists: number
      total_generalists: number
      specialist_by_domain: Record<string, number>
    }>('/specialists/summary'),
    staleTime: 5 * 60 * 1000,
  })
}
