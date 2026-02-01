import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { LeaderboardResponse } from '../types'

interface LeaderboardParams {
  domain?: string
  rankingType?: string
  limit?: number
}

/**
 * Fetch leaderboard for a domain
 * - For global leaderboard, use domain='all' or undefined
 * - For domain-specific, use the domain name
 */
export function useLeaderboard(params: LeaderboardParams = {}) {
  const { domain, rankingType = 'raw_quality', limit } = params

  // Determine the correct endpoint path
  // Global leaderboard: /leaderboard
  // Domain-specific: /leaderboard/domain/{domain}
  const endpoint = !domain || domain === 'all' 
    ? '/leaderboard'
    : `/leaderboard/domain/${domain}`

  return useQuery({
    queryKey: ['leaderboard', domain || 'all', rankingType, limit],
    queryFn: () =>
      api.get<LeaderboardResponse>(endpoint, {
        dimension: rankingType,
        limit,
      }),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Fetch multi-dimensional leaderboard
 */
export function useMultiDimensionalLeaderboard(domain?: string) {
  return useQuery({
    queryKey: ['leaderboard', 'multi', domain || 'all'],
    queryFn: () => api.get<LeaderboardResponse>('/leaderboard/multi-dimensional', {
      domain: domain && domain !== 'all' ? domain : undefined,
    }),
    staleTime: 60 * 1000,
  })
}
