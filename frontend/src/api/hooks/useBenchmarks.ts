/**
 * React Query hooks for Benchmarks API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  BenchmarkDefinition,
  CreateBenchmarkDefinition,
  BenchmarkResult,
  CreateBenchmarkResult,
  BenchmarkCorrelation,
  BenchmarkLeaderboardEntry,
  SuccessResponse,
} from '../types'

/**
 * List benchmark definitions
 */
export function useBenchmarkDefinitions(params?: { category?: string }) {
  return useQuery({
    queryKey: ['benchmarks', 'definitions', params],
    queryFn: () => api.get<BenchmarkDefinition[]>('/benchmarks/definitions', params),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Get a specific benchmark definition
 */
export function useBenchmarkDefinition(benchmarkId: string) {
  return useQuery({
    queryKey: ['benchmarks', 'definitions', benchmarkId],
    queryFn: () => api.get<BenchmarkDefinition>(`/benchmarks/definitions/${benchmarkId}`),
    enabled: !!benchmarkId,
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get benchmark results for a base model
 */
export function useBenchmarkResults(
  baseModelId: string,
  params?: { benchmark_id?: string }
) {
  return useQuery({
    queryKey: ['benchmarks', 'results', baseModelId, params],
    queryFn: () => api.get<BenchmarkResult[]>(`/benchmarks/results/${baseModelId}`, params),
    enabled: !!baseModelId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Compare models on a benchmark
 */
export function useBenchmarkComparison(params: {
  base_model_ids: string
  benchmark_id?: string
}) {
  return useQuery({
    queryKey: ['benchmarks', 'compare', params],
    queryFn: () => api.get<Record<string, Record<string, number>>>('/benchmarks/compare', params),
    enabled: !!params.base_model_ids,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Get correlations between benchmarks and TrueSkill dimensions
 */
export function useBenchmarkCorrelations(params?: {
  benchmark_id?: string
  dimension?: string
  min_correlation?: number
}) {
  return useQuery({
    queryKey: ['benchmarks', 'correlations', params],
    queryFn: () => api.get<BenchmarkCorrelation[]>('/benchmarks/correlations', params),
    staleTime: 10 * 60 * 1000,
  })
}

/**
 * Get benchmark leaderboard
 */
export function useBenchmarkLeaderboard(
  benchmarkId: string,
  params?: { variant?: string; limit?: number }
) {
  return useQuery({
    queryKey: ['benchmarks', 'leaderboard', benchmarkId, params],
    queryFn: () => api.get<BenchmarkLeaderboardEntry[]>(`/benchmarks/leaderboard/${benchmarkId}`, params),
    enabled: !!benchmarkId,
    staleTime: 5 * 60 * 1000,
  })
}

/**
 * Create a benchmark definition
 */
export function useCreateBenchmarkDefinition() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: CreateBenchmarkDefinition) =>
      api.post<BenchmarkDefinition>('/benchmarks/definitions', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['benchmarks', 'definitions'] })
    },
  })
}

/**
 * Delete a benchmark definition
 */
export function useDeleteBenchmarkDefinition() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (benchmarkId: string) =>
      api.delete<SuccessResponse>(`/benchmarks/definitions/${benchmarkId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['benchmarks', 'definitions'] })
    },
  })
}

/**
 * Create a benchmark result
 */
export function useCreateBenchmarkResult() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: CreateBenchmarkResult) =>
      api.post<BenchmarkResult>('/benchmarks/results', data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['benchmarks', 'results', variables.base_model_id] })
      queryClient.invalidateQueries({ queryKey: ['benchmarks', 'leaderboard'] })
    },
  })
}
