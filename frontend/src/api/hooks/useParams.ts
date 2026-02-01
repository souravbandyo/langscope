/**
 * React Query hooks for Parameters API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  ParamTypesResponse,
  ParamResponse,
  ParamUpdateRequest,
  ParamCacheStats,
  ParamExportResponse,
  ParamImportResponse,
  SuccessResponse,
} from '../types'

/**
 * List available parameter types
 */
export function useParamTypes() {
  return useQuery({
    queryKey: ['params', 'types'],
    queryFn: () => api.get<ParamTypesResponse>('/params'),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Get parameters for a specific type
 */
export function useParams(paramType: string, params?: { domain?: string }) {
  return useQuery({
    queryKey: ['params', paramType, params],
    queryFn: () => api.get<ParamResponse>(`/params/${paramType}`, params),
    enabled: !!paramType,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Update parameters
 */
export function useUpdateParams() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ paramType, data }: { paramType: string; data: ParamUpdateRequest }) =>
      api.put<ParamResponse>(`/params/${paramType}`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['params', variables.paramType] })
    },
  })
}

/**
 * Remove domain override for parameters
 */
export function useRemoveDomainOverride() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ paramType, domain }: { paramType: string; domain: string }) =>
      api.delete<SuccessResponse>(`/params/${paramType}/domain/${domain}`),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['params', variables.paramType] })
    },
  })
}

/**
 * Reset parameters to defaults
 */
export function useResetParams() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ paramType, domain }: { paramType: string; domain?: string }) =>
      api.post<ParamResponse>(`/params/reset/${paramType}${domain ? `?domain=${domain}` : ''}`),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['params', variables.paramType] })
    },
  })
}

/**
 * Get parameter cache statistics
 */
export function useParamCacheStats() {
  return useQuery({
    queryKey: ['params', 'cache', 'stats'],
    queryFn: () => api.get<ParamCacheStats>('/params/cache/stats'),
    staleTime: 30 * 1000, // 30 seconds
  })
}

/**
 * Invalidate parameter cache
 */
export function useInvalidateParamCache() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (params?: { param_type?: string; domain?: string }) => {
      const queryParams = new URLSearchParams()
      if (params?.param_type) queryParams.set('param_type', params.param_type)
      if (params?.domain) queryParams.set('domain', params.domain)
      const queryString = queryParams.toString()
      return api.post<SuccessResponse>(`/params/cache/invalidate${queryString ? `?${queryString}` : ''}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['params'] })
    },
  })
}

/**
 * Export all parameters
 */
export function useExportParams() {
  return useQuery({
    queryKey: ['params', 'export'],
    queryFn: () => api.get<ParamExportResponse>('/params/export'),
    enabled: false, // Manual trigger only
    staleTime: 0,
  })
}

/**
 * Import parameters
 */
export function useImportParams() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: Record<string, unknown>) =>
      api.post<ParamImportResponse>('/params/import', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['params'] })
    },
  })
}
