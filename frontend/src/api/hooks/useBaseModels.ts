/**
 * React Query hooks for Base Models API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  BaseModelCreate,
  BaseModelResponse,
  BaseModelListResponse,
  ProviderComparisonResponse,
  SuccessResponse,
} from '../types'

/**
 * List all base models
 */
export function useBaseModels(params?: {
  family?: string
  organization?: string
  skip?: number
  limit?: number
}) {
  return useQuery({
    queryKey: ['base-models', params],
    queryFn: () => api.get<BaseModelListResponse>('/base-models', params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Get a specific base model by ID
 */
export function useBaseModel(baseModelId: string) {
  return useQuery({
    queryKey: ['base-models', baseModelId],
    queryFn: () => api.get<BaseModelResponse>(`/base-models/${baseModelId}`),
    enabled: !!baseModelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Compare providers for a base model
 */
export function useProviderComparison(baseModelId: string) {
  return useQuery({
    queryKey: ['base-models', baseModelId, 'compare-providers'],
    queryFn: () => api.get<ProviderComparisonResponse>(`/base-models/${baseModelId}/compare-providers`),
    enabled: !!baseModelId,
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Create a new base model
 */
export function useCreateBaseModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: BaseModelCreate) =>
      api.post<BaseModelResponse>('/base-models', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['base-models'] })
    },
  })
}

/**
 * Delete a base model
 */
export function useDeleteBaseModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (baseModelId: string) =>
      api.delete<SuccessResponse>(`/base-models/${baseModelId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['base-models'] })
    },
  })
}
