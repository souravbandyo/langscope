/**
 * React Query hooks for User Profile API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  UserProfile,
  UpdateProfileRequest,
  ChangePasswordRequest,
  SessionsResponse,
  AvatarResponse,
} from '../types'

/**
 * Get current user's profile
 */
export function useProfile() {
  return useQuery({
    queryKey: ['users', 'profile'],
    queryFn: () => api.get<UserProfile>('/users/me/profile'),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Update user profile
 */
export function useUpdateProfile() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: UpdateProfileRequest) =>
      api.put<UserProfile>('/users/me/profile', data),
    onSuccess: (data) => {
      queryClient.setQueryData(['users', 'profile'], data)
      queryClient.invalidateQueries({ queryKey: ['auth', 'me'] })
    },
  })
}

/**
 * Upload avatar image
 */
export function useUploadAvatar() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${api.baseUrl}/users/me/avatar`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${api.getToken()}`,
        },
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to upload avatar')
      }

      return response.json() as Promise<AvatarResponse>
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
    },
  })
}

/**
 * Delete avatar
 */
export function useDeleteAvatar() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.delete<{ message: string }>('/users/me/avatar'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
    },
  })
}

/**
 * Change password
 */
export function useChangePassword() {
  return useMutation({
    mutationFn: (data: ChangePasswordRequest) =>
      api.put<{ message: string }>('/users/me/password', data),
  })
}

/**
 * Get active sessions
 */
export function useActiveSessions() {
  return useQuery({
    queryKey: ['users', 'sessions'],
    queryFn: () => api.get<SessionsResponse>('/users/me/sessions'),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Revoke a specific session
 */
export function useRevokeSession() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (sessionId: string) =>
      api.delete<{ message: string }>(`/users/me/sessions/${sessionId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'sessions'] })
    },
  })
}

/**
 * Revoke all sessions (sign out everywhere)
 */
export function useRevokeAllSessions() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.delete<{ message: string }>('/users/me/sessions'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'sessions'] })
    },
  })
}

/**
 * Delete account
 */
export function useDeleteAccount() {
  return useMutation({
    mutationFn: () => api.delete<{ message: string }>('/users/me'),
  })
}
