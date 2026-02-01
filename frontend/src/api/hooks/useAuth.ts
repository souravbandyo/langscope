/**
 * React Query hooks for Authentication API
 */

import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import type {
  UserResponse,
  AuthStatusResponse,
  AuthInfoResponse,
  TokenVerifyResponse,
} from '../types'

/**
 * Get current authenticated user
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: ['auth', 'me'],
    queryFn: () => api.get<UserResponse>('/auth/me'),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: false, // Don't retry auth requests
  })
}

/**
 * Check authentication status
 */
export function useAuthStatus() {
  return useQuery({
    queryKey: ['auth', 'status'],
    queryFn: () => api.get<AuthStatusResponse>('/auth/status'),
    staleTime: 60 * 1000, // 1 minute
    retry: false,
  })
}

/**
 * Get authentication configuration info
 */
export function useAuthInfo() {
  return useQuery({
    queryKey: ['auth', 'info'],
    queryFn: () => api.get<AuthInfoResponse>('/auth/info'),
    staleTime: 10 * 60 * 1000, // 10 minutes (config rarely changes)
  })
}

/**
 * Verify current token validity
 */
export function useVerifyToken() {
  return useQuery({
    queryKey: ['auth', 'verify'],
    queryFn: () => api.get<TokenVerifyResponse>('/auth/verify'),
    staleTime: 30 * 1000, // 30 seconds
    retry: false,
  })
}
