/**
 * React Query hooks for Prompts API
 */

import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../client'
import type {
  ClassifyRequest,
  ClassifyResponse,
  ProcessRequest,
  ProcessResponse,
  CacheResponseRequest,
  PromptMetrics,
  SuccessResponse,
} from '../types'

/**
 * Classify a prompt to detect domain and language
 */
export function useClassifyPrompt() {
  return useMutation({
    mutationFn: (data: ClassifyRequest) =>
      api.post<ClassifyResponse>('/prompts/classify', data),
  })
}

/**
 * Process a prompt and get recommendations
 */
export function useProcessPrompt() {
  return useMutation({
    mutationFn: (data: ProcessRequest) =>
      api.post<ProcessResponse>('/prompts/process', data),
  })
}

/**
 * Cache a response for a prompt
 */
export function useCacheResponse() {
  return useMutation({
    mutationFn: (data: CacheResponseRequest) =>
      api.post<SuccessResponse>('/prompts/cache', data),
  })
}

/**
 * Get prompt processing metrics
 */
export function usePromptMetrics() {
  return useQuery({
    queryKey: ['prompts', 'metrics'],
    queryFn: () => api.get<PromptMetrics>('/prompts/metrics'),
    staleTime: 30 * 1000, // 30 seconds
  })
}

/**
 * Reset prompt metrics
 */
export function useResetPromptMetrics() {
  return useMutation({
    mutationFn: () => api.post<SuccessResponse>('/prompts/metrics/reset'),
  })
}

/**
 * List available domains for prompt classification
 */
export function usePromptDomains() {
  return useQuery({
    queryKey: ['prompts', 'domains'],
    queryFn: () => api.get<{ domains: string[] }>('/prompts/domains'),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * List supported languages for prompt classification
 */
export function usePromptLanguages() {
  return useQuery({
    queryKey: ['prompts', 'languages'],
    queryFn: () => api.get<{ languages: string[] }>('/prompts/languages'),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}
