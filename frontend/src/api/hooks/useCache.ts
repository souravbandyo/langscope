/**
 * React Query hooks for Cache API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  CacheStats,
  InvalidateResponse,
  SessionInfo,
  CreateSessionRequest,
  UpdateSessionRequest,
  RateLimitStatus,
  SuccessResponse,
} from '../types'

/**
 * Get cache statistics
 */
export function useCacheStats() {
  return useQuery({
    queryKey: ['cache', 'stats'],
    queryFn: () => api.get<CacheStats>('/cache/stats'),
    staleTime: 30 * 1000, // 30 seconds
  })
}

/**
 * Invalidate cache by category
 */
export function useInvalidateCategory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (category: string) =>
      api.post<InvalidateResponse>(`/cache/invalidate/${category}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cache', 'stats'] })
    },
  })
}

/**
 * Invalidate leaderboard cache for a domain
 */
export function useInvalidateLeaderboard() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (domain: string) =>
      api.post<InvalidateResponse>(`/cache/invalidate/leaderboard/${domain}`),
    onSuccess: (_, domain) => {
      queryClient.invalidateQueries({ queryKey: ['leaderboard', domain] })
      queryClient.invalidateQueries({ queryKey: ['cache', 'stats'] })
    },
  })
}

/**
 * Invalidate all cache
 */
export function useInvalidateAllCache() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.delete<SuccessResponse>('/cache/all'),
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })
}

/**
 * Reset cache statistics
 */
export function useResetCacheStats() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<SuccessResponse>('/cache/stats/reset'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cache', 'stats'] })
    },
  })
}

/**
 * Create a new session
 */
export function useCreateSession() {
  return useMutation({
    mutationFn: (data: CreateSessionRequest) =>
      api.post<SessionInfo>('/cache/sessions', data),
  })
}

/**
 * Get session info
 */
export function useSession(sessionId: string) {
  return useQuery({
    queryKey: ['cache', 'sessions', sessionId],
    queryFn: () => api.get<SessionInfo>(`/cache/sessions/${sessionId}`),
    enabled: !!sessionId,
    staleTime: 30 * 1000,
  })
}

/**
 * Update session data
 */
export function useUpdateSession() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ sessionId, data }: { sessionId: string; data: UpdateSessionRequest }) =>
      api.put<SuccessResponse>(`/cache/sessions/${sessionId}`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['cache', 'sessions', variables.sessionId] })
    },
  })
}

/**
 * End a session
 */
export function useEndSession() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (sessionId: string) =>
      api.delete<SuccessResponse>(`/cache/sessions/${sessionId}`),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ['cache', 'sessions', sessionId] })
    },
  })
}

/**
 * Get rate limit status
 */
export function useRateLimitStatus(params: { identifier: string; endpoint?: string }) {
  return useQuery({
    queryKey: ['cache', 'rate-limit', 'status', params],
    queryFn: () => api.get<RateLimitStatus>('/cache/rate-limit/status', params),
    enabled: !!params.identifier,
    staleTime: 10 * 1000, // 10 seconds
  })
}

/**
 * Reset rate limit for an identifier
 */
export function useResetRateLimit() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ identifier, endpoint }: { identifier: string; endpoint?: string }) =>
      api.post<SuccessResponse>(`/cache/rate-limit/reset/${identifier}${endpoint ? `?endpoint=${endpoint}` : ''}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cache', 'rate-limit'] })
    },
  })
}
