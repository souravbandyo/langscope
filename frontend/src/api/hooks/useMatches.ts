import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { MatchListResponse } from '../types'

interface MatchesParams {
  domain?: string
  modelId?: string
  limit?: number
  offset?: number
}

/**
 * Fetch match history
 */
export function useMatches(params: MatchesParams = {}) {
  const { domain, modelId, limit = 50, offset = 0 } = params

  return useQuery({
    queryKey: ['matches', domain, modelId, limit, offset],
    queryFn: () =>
      api.get<MatchListResponse>('/matches', {
        domain,
        model_id: modelId,
        limit,
        offset,
      }),
    staleTime: 30 * 1000, // 30 seconds
  })
}
