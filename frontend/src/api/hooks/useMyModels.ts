/**
 * React Query hooks for User's Private Models API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  UserModel,
  UserModelCreate,
  UserModelUpdate,
  UserModelListResponse,
  UserModelPerformance,
  ModelComparisonResponse,
  RunEvaluationRequest,
  RunEvaluationResponse,
  EvaluationStatus,
  SuccessResponse,
  ModelType,
} from '../types'

// =============================================================================
// Query Keys
// =============================================================================

export const myModelsKeys = {
  all: ['my-models'] as const,
  lists: () => [...myModelsKeys.all, 'list'] as const,
  list: (params?: { type?: ModelType; active?: boolean }) => 
    [...myModelsKeys.lists(), params] as const,
  details: () => [...myModelsKeys.all, 'detail'] as const,
  detail: (id: string) => [...myModelsKeys.details(), id] as const,
  performance: (id: string) => [...myModelsKeys.all, 'performance', id] as const,
  comparison: (params: { domain: string; modelType: ModelType; metric?: string }) =>
    [...myModelsKeys.all, 'comparison', params] as const,
  evaluations: () => [...myModelsKeys.all, 'evaluations'] as const,
  evaluation: (id: string) => [...myModelsKeys.evaluations(), id] as const,
}

// =============================================================================
// List User Models
// =============================================================================

/**
 * List all user's private models
 */
export function useMyModels(params?: { 
  type?: ModelType
  active?: boolean 
  skip?: number
  limit?: number
}) {
  return useQuery({
    queryKey: myModelsKeys.list(params),
    queryFn: () => api.get<UserModelListResponse>('/my-models', params),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Get models grouped by type
 */
export function useMyModelsByType() {
  const { data, ...rest } = useMyModels()
  
  const groupedByType = data?.models?.reduce((acc, model) => {
    if (!acc[model.modelType]) {
      acc[model.modelType] = []
    }
    acc[model.modelType].push(model)
    return acc
  }, {} as Record<ModelType, UserModel[]>)
  
  return {
    ...rest,
    data,
    groupedByType,
  }
}

// =============================================================================
// Get Single Model
// =============================================================================

/**
 * Get a specific user model by ID
 */
export function useMyModel(modelId: string) {
  return useQuery({
    queryKey: myModelsKeys.detail(modelId),
    queryFn: () => api.get<UserModel>(`/my-models/${modelId}`),
    enabled: !!modelId,
    staleTime: 60 * 1000,
  })
}

// =============================================================================
// Create Model
// =============================================================================

/**
 * Create a new user model
 */
export function useCreateMyModel() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: UserModelCreate) =>
      api.post<UserModel>('/my-models', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: myModelsKeys.lists() })
    },
  })
}

// =============================================================================
// Update Model
// =============================================================================

/**
 * Update a user model
 */
export function useUpdateMyModel() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ modelId, data }: { modelId: string; data: UserModelUpdate }) =>
      api.patch<UserModel>(`/my-models/${modelId}`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: myModelsKeys.lists() })
      queryClient.invalidateQueries({ queryKey: myModelsKeys.detail(variables.modelId) })
    },
  })
}

// =============================================================================
// Delete Model
// =============================================================================

/**
 * Delete a user model
 */
export function useDeleteMyModel() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (modelId: string) =>
      api.delete<SuccessResponse>(`/my-models/${modelId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: myModelsKeys.lists() })
    },
  })
}

// =============================================================================
// Model Performance
// =============================================================================

/**
 * Get performance data for a user model
 */
export function useMyModelPerformance(modelId: string) {
  return useQuery({
    queryKey: myModelsKeys.performance(modelId),
    queryFn: () => api.get<UserModelPerformance>(`/my-models/${modelId}/performance`),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// =============================================================================
// Model Comparison
// =============================================================================

/**
 * Compare user models against public leaderboard
 */
export function useModelComparison(params: {
  domain: string
  modelType: ModelType
  metric?: string
  includePublic?: boolean
  limit?: number
}) {
  return useQuery({
    queryKey: myModelsKeys.comparison(params),
    queryFn: () => api.get<ModelComparisonResponse>('/my-models/compare', params),
    enabled: !!params.domain && !!params.modelType,
    staleTime: 5 * 60 * 1000,
  })
}

// =============================================================================
// Run Evaluation
// =============================================================================

/**
 * Run evaluation on a user model
 */
export function useRunEvaluation() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: RunEvaluationRequest) =>
      api.post<RunEvaluationResponse>('/my-models/evaluate', data),
    onSuccess: (_, variables) => {
      // Invalidate performance data when evaluation is started
      queryClient.invalidateQueries({ 
        queryKey: myModelsKeys.performance(variables.modelId) 
      })
    },
  })
}

/**
 * Check evaluation status
 */
export function useEvaluationStatus(evaluationId: string, options?: { 
  refetchInterval?: number | false 
}) {
  return useQuery({
    queryKey: myModelsKeys.evaluation(evaluationId),
    queryFn: () => api.get<EvaluationStatus>(`/my-models/evaluations/${evaluationId}`),
    enabled: !!evaluationId,
    refetchInterval: options?.refetchInterval ?? 5000, // Poll every 5 seconds by default
  })
}

// =============================================================================
// Test Connection
// =============================================================================

/**
 * Test API connection for a model configuration
 */
export function useTestModelConnection() {
  return useMutation({
    mutationFn: (data: { 
      endpoint: string
      apiKey: string
      modelId: string
      apiFormat: string
    }) => api.post<{ success: boolean; latencyMs: number; error?: string }>(
      '/my-models/test-connection', 
      data
    ),
  })
}

// =============================================================================
// API Key Management
// =============================================================================

/**
 * Update API key for a model (separate endpoint for security)
 */
export function useUpdateModelApiKey() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ modelId, apiKey }: { modelId: string; apiKey: string }) =>
      api.post<SuccessResponse>(`/my-models/${modelId}/api-key`, { apiKey }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: myModelsKeys.detail(variables.modelId) })
    },
  })
}

/**
 * Verify API key is still valid
 */
export function useVerifyModelApiKey() {
  return useMutation({
    mutationFn: (modelId: string) =>
      api.post<{ valid: boolean; error?: string }>(`/my-models/${modelId}/verify-key`),
  })
}
