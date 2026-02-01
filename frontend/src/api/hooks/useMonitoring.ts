/**
 * React Query hooks for Monitoring API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  DashboardResponse,
  MonitoringHealthResponse,
  Alert,
  CoverageResponse,
  FreshnessResponse,
  ErrorSummary,
  SuccessResponse,
} from '../types'

/**
 * Get monitoring dashboard data
 */
export function useDashboard() {
  return useQuery({
    queryKey: ['monitoring', 'dashboard'],
    queryFn: () => api.get<DashboardResponse>('/monitoring/dashboard'),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
  })
}

/**
 * Get system health status
 */
export function useMonitoringHealth() {
  return useQuery({
    queryKey: ['monitoring', 'health'],
    queryFn: () => api.get<MonitoringHealthResponse>('/monitoring/health'),
    staleTime: 10 * 1000, // 10 seconds
    refetchInterval: 30 * 1000, // Auto-refresh every 30 seconds
  })
}

/**
 * Get active alerts
 */
export function useAlerts(params?: {
  level?: string
  domain?: string
  include_resolved?: boolean
}) {
  return useQuery({
    queryKey: ['monitoring', 'alerts', params],
    queryFn: () => api.get<Alert[]>('/monitoring/alerts', params),
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
  })
}

/**
 * Check for new alerts
 */
export function useCheckAlerts() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<{ alerts: Alert[] }>('/monitoring/alerts/check'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'alerts'] })
    },
  })
}

/**
 * Resolve an alert
 */
export function useResolveAlert() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (alertId: string) =>
      api.post<SuccessResponse>(`/monitoring/alerts/${alertId}/resolve`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'alerts'] })
      queryClient.invalidateQueries({ queryKey: ['monitoring', 'dashboard'] })
    },
  })
}

/**
 * Get coverage summary
 */
export function useCoverageSummary(params?: { domain?: string }) {
  return useQuery({
    queryKey: ['monitoring', 'coverage', params],
    queryFn: () => api.get<CoverageResponse>('/monitoring/coverage', params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Get domain-specific coverage
 */
export function useDomainCoverage(domain: string) {
  return useQuery({
    queryKey: ['monitoring', 'coverage', domain],
    queryFn: () => api.get<CoverageResponse>(`/monitoring/coverage/${domain}`),
    enabled: !!domain,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get error summary
 */
export function useErrorSummary(params?: { hours?: number; domain?: string }) {
  return useQuery({
    queryKey: ['monitoring', 'errors', params],
    queryFn: () => api.get<ErrorSummary>('/monitoring/errors', params),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Get data freshness status
 */
export function useFreshness(params?: { threshold_hours?: number }) {
  return useQuery({
    queryKey: ['monitoring', 'freshness', params],
    queryFn: () => api.get<FreshnessResponse>('/monitoring/freshness', params),
    staleTime: 5 * 60 * 1000,
  })
}
