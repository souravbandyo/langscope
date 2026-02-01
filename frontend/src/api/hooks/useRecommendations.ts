import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { RecommendationResponse } from '../types'

interface RecommendationParams {
  useCase: string
  domain?: string
  topK?: number
}

/**
 * Get model recommendations for a use case
 */
export function useRecommendations(params: RecommendationParams) {
  const { useCase, domain, topK = 10 } = params

  return useQuery({
    queryKey: ['recommendations', useCase, domain, topK],
    queryFn: () =>
      api.get<RecommendationResponse>(`/recommendations/${encodeURIComponent(useCase)}`, {
        domain,
        top_k: topK,
      }),
    enabled: !!useCase,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}
