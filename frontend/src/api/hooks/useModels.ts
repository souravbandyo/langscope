import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { ModelListResponse, ModelResponse } from '../types'

/**
 * Fetch all models
 */
export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelListResponse>('/models'),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Fetch a single model by ID
 */
export function useModel(modelId: string) {
  return useQuery({
    queryKey: ['models', modelId],
    queryFn: () => api.get<ModelResponse>(`/models/${modelId}`),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  })
}
