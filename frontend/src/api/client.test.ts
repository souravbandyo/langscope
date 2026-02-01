import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { api, ApiError } from './client'

describe('ApiClient', () => {
  const mockFetch = vi.fn()

  beforeEach(() => {
    global.fetch = mockFetch
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('get', () => {
    it('makes a GET request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: 'test' }),
      })

      const result = await api.get('/test')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/test'),
        expect.objectContaining({ method: 'GET' })
      )
      expect(result).toEqual({ data: 'test' })
    })

    it('includes query parameters', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await api.get('/test', { param1: 'value1', param2: 123 })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('param1=value1'),
        expect.anything()
      )
    })
  })

  describe('post', () => {
    it('makes a POST request with body', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true }),
      })

      const result = await api.post('/test', { name: 'test' })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/test'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: 'test' }),
        })
      )
      expect(result).toEqual({ success: true })
    })
  })

  describe('error handling', () => {
    it('throws ApiError on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'Not found' }),
      })

      await expect(api.get('/notfound')).rejects.toThrow(ApiError)
    })

    it('includes status code in ApiError', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      })

      try {
        await api.get('/error')
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError)
        expect((error as ApiError).status).toBe(500)
      }
    })
  })

  describe('authentication', () => {
    it('includes auth token in headers when set', async () => {
      api.setAuthToken('test-token')
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await api.get('/protected')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      )

      // Clean up
      api.setAuthToken(null)
    })
  })
})
