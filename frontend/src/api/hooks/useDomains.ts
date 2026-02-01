import { useQuery } from '@tanstack/react-query'
import api from '../client'
import type { DomainListResponse, DomainResponse } from '../types'

/**
 * Fetch all domains
 */
export function useDomains() {
  return useQuery({
    queryKey: ['domains'],
    queryFn: () => api.get<DomainListResponse>('/domains'),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Fetch a single domain by name
 */
export function useDomain(domainName: string) {
  return useQuery({
    queryKey: ['domains', domainName],
    queryFn: () => api.get<DomainResponse>(`/domains/${domainName}`),
    enabled: !!domainName,
    staleTime: 5 * 60 * 1000,
  })
}
