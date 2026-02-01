/**
 * API Client for LangScope Backend
 * Handles authentication and requests to FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>
}

class ApiClient {
  private baseUrl: string
  private authToken: string | null = null

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  /**
   * Set the authentication token
   */
  setAuthToken(token: string | null) {
    this.authToken = token
  }

  /**
   * Build URL with query parameters
   */
  private buildUrl(endpoint: string, params?: RequestOptions['params']): string {
    const url = new URL(`${this.baseUrl}${endpoint}`)
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value))
        }
      })
    }
    
    return url.toString()
  }

  /**
   * Make an HTTP request
   */
  private async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const { params, ...fetchOptions } = options
    const url = this.buildUrl(endpoint, params)

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    }

    if (this.authToken) {
      (headers as Record<string, string>)['Authorization'] = `Bearer ${this.authToken}`
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }))
      throw new ApiError(response.status, error.detail || error.error || 'Request failed')
    }

    return response.json()
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, params?: RequestOptions['params']): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET', params })
  }

  /**
   * POST request
   */
  async post<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  /**
   * PATCH request (partial update)
   */
  async patch<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  /**
   * DELETE request
   */
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
}

/**
 * Custom API error class
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// Export singleton instance
export const api = new ApiClient(API_BASE_URL)

// Re-export for direct usage
export default api
