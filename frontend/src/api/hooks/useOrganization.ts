/**
 * React Query hooks for Organization API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  Organization,
  CreateOrganizationRequest,
  UpdateOrganizationRequest,
  TeamMember,
  TeamMembersResponse,
  InviteMemberRequest,
  UpdateMemberRoleRequest,
  Invitation,
  InvitationsResponse,
  JoinOrganizationRequest,
} from '../types'

/**
 * Get current user's organization
 */
export function useMyOrganization() {
  return useQuery({
    queryKey: ['organizations', 'me'],
    queryFn: () => api.get<Organization | null>('/organizations/me'),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Create a new organization
 */
export function useCreateOrganization() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateOrganizationRequest) =>
      api.post<Organization>('/organizations', data),
    onSuccess: (data) => {
      queryClient.setQueryData(['organizations', 'me'], data)
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
    },
  })
}

/**
 * Update organization
 */
export function useUpdateOrganization() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ orgId, data }: { orgId: string; data: UpdateOrganizationRequest }) =>
      api.put<Organization>(`/organizations/${orgId}`, data),
    onSuccess: (data) => {
      queryClient.setQueryData(['organizations', 'me'], data)
    },
  })
}

/**
 * Upload organization logo
 */
export function useUploadOrgLogo() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ orgId, file }: { orgId: string; file: File }) => {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${api.baseUrl}/organizations/${orgId}/logo`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${api.getToken()}`,
        },
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to upload logo')
      }

      return response.json() as Promise<{ logo_url: string; message: string }>
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['organizations', 'me'] })
    },
  })
}

/**
 * Delete organization
 */
export function useDeleteOrganization() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (orgId: string) =>
      api.delete<{ message: string }>(`/organizations/${orgId}`),
    onSuccess: () => {
      queryClient.setQueryData(['organizations', 'me'], null)
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
      queryClient.invalidateQueries({ queryKey: ['organizations'] })
    },
  })
}

/**
 * Get organization members
 */
export function useOrganizationMembers(orgId: string | undefined) {
  return useQuery({
    queryKey: ['organizations', orgId, 'members'],
    queryFn: () => api.get<TeamMembersResponse>(`/organizations/${orgId}/members`),
    enabled: !!orgId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

/**
 * Invite a new member
 */
export function useInviteMember() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ orgId, data }: { orgId: string; data: InviteMemberRequest }) =>
      api.post<Invitation>(`/organizations/${orgId}/members`, data),
    onSuccess: (_, { orgId }) => {
      queryClient.invalidateQueries({ queryKey: ['organizations', orgId, 'invitations'] })
    },
  })
}

/**
 * Update member role
 */
export function useUpdateMemberRole() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      orgId,
      memberId,
      data,
    }: {
      orgId: string
      memberId: string
      data: UpdateMemberRoleRequest
    }) => api.put<TeamMember>(`/organizations/${orgId}/members/${memberId}`, data),
    onSuccess: (_, { orgId }) => {
      queryClient.invalidateQueries({ queryKey: ['organizations', orgId, 'members'] })
    },
  })
}

/**
 * Remove member from organization
 */
export function useRemoveMember() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ orgId, memberId }: { orgId: string; memberId: string }) =>
      api.delete<{ message: string }>(`/organizations/${orgId}/members/${memberId}`),
    onSuccess: (_, { orgId }) => {
      queryClient.invalidateQueries({ queryKey: ['organizations', orgId, 'members'] })
      queryClient.invalidateQueries({ queryKey: ['organizations', 'me'] })
    },
  })
}

/**
 * Get pending invitations
 */
export function usePendingInvitations(orgId: string | undefined) {
  return useQuery({
    queryKey: ['organizations', orgId, 'invitations'],
    queryFn: () => api.get<InvitationsResponse>(`/organizations/${orgId}/invitations`),
    enabled: !!orgId,
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Revoke an invitation
 */
export function useRevokeInvitation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ orgId, invitationId }: { orgId: string; invitationId: string }) =>
      api.delete<{ message: string }>(`/organizations/${orgId}/invitations/${invitationId}`),
    onSuccess: (_, { orgId }) => {
      queryClient.invalidateQueries({ queryKey: ['organizations', orgId, 'invitations'] })
    },
  })
}

/**
 * Join organization with invite code
 */
export function useJoinOrganization() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: JoinOrganizationRequest) =>
      api.post<Organization>('/organizations/join', data),
    onSuccess: (data) => {
      queryClient.setQueryData(['organizations', 'me'], data)
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
    },
  })
}
