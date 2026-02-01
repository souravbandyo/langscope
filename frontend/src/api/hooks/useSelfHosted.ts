/**
 * React Query hooks for Self-Hosted Deployments API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  SelfHostedCreate,
  SelfHostedResponse,
  SelfHostedListResponse,
  CostEstimateRequest,
  CostEstimateResponse,
  SuccessResponse,
} from '../types'

/**
 * List user's self-hosted deployments
 */
export function useSelfHostedDeployments(params?: { skip?: number; limit?: number }) {
  return useQuery({
    queryKey: ['self-hosted', params],
    queryFn: () => api.get<SelfHostedListResponse>('/self-hosted', params),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * List public self-hosted deployments
 */
export function usePublicSelfHosted(params?: { base_model_id?: string; limit?: number }) {
  return useQuery({
    queryKey: ['self-hosted', 'public', params],
    queryFn: () => api.get<SelfHostedListResponse>('/self-hosted/public', params),
    staleTime: 60 * 1000,
  })
}

/**
 * Get a specific self-hosted deployment
 */
export function useSelfHostedDeployment(deploymentId: string) {
  return useQuery({
    queryKey: ['self-hosted', deploymentId],
    queryFn: () => api.get<SelfHostedResponse>(`/self-hosted/${deploymentId}`),
    enabled: !!deploymentId,
    staleTime: 60 * 1000,
  })
}

/**
 * Create a new self-hosted deployment
 */
export function useCreateSelfHosted() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: SelfHostedCreate) =>
      api.post<SelfHostedResponse>('/self-hosted', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['self-hosted'] })
    },
  })
}

/**
 * Estimate per-token costs for self-hosted deployment
 */
export function useEstimateCosts() {
  return useMutation({
    mutationFn: (data: CostEstimateRequest) =>
      api.post<CostEstimateResponse>('/self-hosted/estimate-costs', data),
  })
}

/**
 * Delete a self-hosted deployment
 */
export function useDeleteSelfHosted() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (deploymentId: string) =>
      api.delete<SuccessResponse>(`/self-hosted/${deploymentId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['self-hosted'] })
    },
  })
}
