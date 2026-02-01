/**
 * React Query hooks for Billing API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import type {
  Subscription,
  SubscribeRequest,
  ChangePlanRequest,
  PlansResponse,
  UsageStats,
  InvoicesResponse,
  PaymentMethodsResponse,
  PaymentMethod,
  AddPaymentMethodRequest,
} from '../types'

/**
 * Get available plans
 */
export function useAvailablePlans() {
  return useQuery({
    queryKey: ['billing', 'plans'],
    queryFn: () => api.get<PlansResponse>('/billing/plans'),
    staleTime: 60 * 60 * 1000, // 1 hour - plans don't change often
  })
}

/**
 * Get current subscription
 */
export function useSubscription() {
  return useQuery({
    queryKey: ['billing', 'subscription'],
    queryFn: () => api.get<Subscription | null>('/billing/subscription'),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Subscribe to a plan
 */
export function useSubscribeToPlan() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: SubscribeRequest) =>
      api.post<Subscription>('/billing/subscribe', data),
    onSuccess: (data) => {
      queryClient.setQueryData(['billing', 'subscription'], data)
      queryClient.invalidateQueries({ queryKey: ['organizations', 'me'] })
      queryClient.invalidateQueries({ queryKey: ['users', 'profile'] })
    },
  })
}

/**
 * Change subscription plan
 */
export function useChangePlan() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: ChangePlanRequest) =>
      api.put<Subscription>('/billing/subscription', data),
    onSuccess: (data) => {
      queryClient.setQueryData(['billing', 'subscription'], data)
      queryClient.invalidateQueries({ queryKey: ['organizations', 'me'] })
    },
  })
}

/**
 * Cancel subscription
 */
export function useCancelSubscription() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.delete<{ message: string }>('/billing/subscription'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['billing', 'subscription'] })
    },
  })
}

/**
 * Get usage statistics
 */
export function useUsageStats() {
  return useQuery({
    queryKey: ['billing', 'usage'],
    queryFn: () => api.get<UsageStats>('/billing/usage'),
    staleTime: 60 * 1000, // 1 minute
  })
}

/**
 * Get invoices
 */
export function useInvoices(limit: number = 12) {
  return useQuery({
    queryKey: ['billing', 'invoices', limit],
    queryFn: () => api.get<InvoicesResponse>(`/billing/invoices?limit=${limit}`),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Get invoice PDF URL
 */
export function useInvoicePdf(invoiceId: string) {
  return useQuery({
    queryKey: ['billing', 'invoices', invoiceId, 'pdf'],
    queryFn: () => api.get<{ pdf_url: string; invoice_id: string }>(`/billing/invoices/${invoiceId}/pdf`),
    enabled: !!invoiceId,
  })
}

/**
 * Get payment methods
 */
export function usePaymentMethods() {
  return useQuery({
    queryKey: ['billing', 'payment-methods'],
    queryFn: () => api.get<PaymentMethodsResponse>('/billing/payment-methods'),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Add payment method
 */
export function useAddPaymentMethod() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: AddPaymentMethodRequest) =>
      api.post<PaymentMethod>('/billing/payment-method', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['billing', 'payment-methods'] })
    },
  })
}

/**
 * Remove payment method
 */
export function useRemovePaymentMethod() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (paymentMethodId: string) =>
      api.delete<{ message: string }>(`/billing/payment-method/${paymentMethodId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['billing', 'payment-methods'] })
    },
  })
}
