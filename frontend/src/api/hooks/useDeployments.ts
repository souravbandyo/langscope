/**
 * React Query hooks for Deployments API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  DeploymentCreate,
  DeploymentResponse,
  DeploymentListResponse,
  SuccessResponse,
} from '../types'

/**
 * List all deployments
 */
export function useDeployments(params?: {
  base_model_id?: string
  provider?: string
  max_price?: number
  min_rating?: number
  skip?: number
  limit?: number
}) {
  return useQuery({
    queryKey: ['deployments', params],
    queryFn: () => api.get<DeploymentListResponse>('/deployments', params),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Get a specific deployment by ID
 */
export function useDeployment(deploymentId: string) {
  return useQuery({
    queryKey: ['deployments', deploymentId],
    queryFn: () => api.get<DeploymentResponse>(`/deployments/${deploymentId}`),
    enabled: !!deploymentId,
    staleTime: 60 * 1000,
  })
}

/**
 * Get deployments by base model
 */
export function useDeploymentsByBaseModel(
  baseModelId: string,
  params?: { include_inactive?: boolean }
) {
  return useQuery({
    queryKey: ['deployments', 'by-base-model', baseModelId, params],
    queryFn: () => api.get<DeploymentListResponse>(`/deployments/by-base-model/${baseModelId}`, params),
    enabled: !!baseModelId,
    staleTime: 60 * 1000,
  })
}

/**
 * Get best deployment for a base model
 */
export function useBestDeployment(
  baseModelId: string,
  params?: { domain?: string; dimension?: string }
) {
  return useQuery({
    queryKey: ['deployments', 'best', baseModelId, params],
    queryFn: () => api.get<DeploymentResponse>(`/deployments/best/${baseModelId}`, params),
    enabled: !!baseModelId,
    staleTime: 60 * 1000,
  })
}

/**
 * Create a new deployment
 */
export function useCreateDeployment() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: DeploymentCreate) =>
      api.post<DeploymentResponse>('/deployments', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}

/**
 * Delete a deployment
 */
export function useDeleteDeployment() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (deploymentId: string) =>
      api.delete<SuccessResponse>(`/deployments/${deploymentId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}
