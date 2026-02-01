import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../client'
import type {
  ArenaSessionStart,
  ArenaSessionStartResponse,
  ArenaBattle,
  ArenaBattleResponse,
  ArenaSessionResult,
} from '../types'

/**
 * Start an arena session
 */
export function useArenaSession() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: ArenaSessionStart) =>
      api.post<ArenaSessionStartResponse>('/arena/session/start', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['arena'] })
    },
  })
}

/**
 * Submit a battle result
 */
export function useArenaBattle(sessionId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: ArenaBattle) =>
      api.post<ArenaBattleResponse>(`/arena/session/${sessionId}/battle`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['arena', sessionId] })
    },
  })
}

/**
 * Complete an arena session
 */
export function useCompleteArenaSession(sessionId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () =>
      api.post<ArenaSessionResult>(`/arena/session/${sessionId}/complete`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['arena'] })
      queryClient.invalidateQueries({ queryKey: ['leaderboard'] })
    },
  })
}

/**
 * Get arena session status
 */
export function useArenaSessionStatus(sessionId: string) {
  return useQuery({
    queryKey: ['arena', sessionId],
    queryFn: () => api.get<ArenaSessionResult>(`/arena/session/${sessionId}`),
    enabled: !!sessionId,
  })
}
