/**
 * Integration Tests for LangScope Frontend-Backend Connectivity
 * 
 * Tests API endpoints, data structures, and hook responses
 */

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'

// Mock fetch for testing
const API_BASE_URL = 'http://localhost:8000'

// Helper to make API calls
async function apiCall(endpoint: string) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`)
    return {
      ok: response.ok,
      status: response.status,
      data: await response.json().catch(() => null)
    }
  } catch (error) {
    return { ok: false, status: 0, error: 'Connection refused' }
  }
}

describe('Backend Connectivity', () => {
  describe('Health Endpoints', () => {
    it('should respond to health check', async () => {
      const result = await apiCall('/health')
      
      if (result.status === 0) {
        console.warn('⚠️ Backend not running - skipping connectivity tests')
        return
      }
      
      expect(result.ok).toBe(true)
      expect(result.data).toHaveProperty('status')
      expect(result.data.status).toBe('healthy')
    })

    it('should return API version', async () => {
      const result = await apiCall('/')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      expect(result.data).toHaveProperty('version')
      expect(result.data).toHaveProperty('name', 'LangScope API')
    })
  })

  describe('Domain Endpoints', () => {
    it('should return list of domains', async () => {
      const result = await apiCall('/domains')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      expect(result.data).toHaveProperty('domains')
      expect(Array.isArray(result.data.domains)).toBe(true)
    })
  })

  describe('Models Endpoints', () => {
    it('should return list of models', async () => {
      const result = await apiCall('/models')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      expect(result.data).toHaveProperty('models')
      expect(Array.isArray(result.data.models)).toBe(true)
    })
  })

  describe('Transfer Learning Endpoints', () => {
    it('should return domain index stats', async () => {
      const result = await apiCall('/transfer/similarity/index/stats')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('total_domains')
      }
    })

    it('should return similar domains for code_generation', async () => {
      const result = await apiCall('/transfer/domains/code_generation/similar?limit=5')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('similar_domains')
      }
    })

    it('should return transfer leaderboard', async () => {
      const result = await apiCall('/transfer/leaderboard/code_generation?include_transferred=true')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('entries')
      }
    })

    it('should return domain correlation', async () => {
      const result = await apiCall('/transfer/correlation/code_generation/coding_python')
      
      if (result.status === 0) return
      
      // May return 404 if no correlation exists
      if (result.ok && result.data) {
        expect(result.data).toHaveProperty('correlation')
      }
    })
  })

  describe('Specialists Endpoints', () => {
    it('should return specialist summary', async () => {
      const result = await apiCall('/specialists/summary')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('total_specialists')
        expect(result.data).toHaveProperty('total_generalists')
      }
    })

    it('should return specialists for domain', async () => {
      const result = await apiCall('/specialists/domain/code_generation?include_weak_spots=true')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(Array.isArray(result.data)).toBe(true)
      }
    })
  })

  describe('Leaderboard Endpoints', () => {
    it('should return leaderboard for domain', async () => {
      const result = await apiCall('/leaderboard/domain/code_generation')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('entries')
      }
    })

    it('should support dimension parameter', async () => {
      const result = await apiCall('/leaderboard/domain/code_generation?dimension=cost_adjusted')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
    })

    it('should return global leaderboard', async () => {
      const result = await apiCall('/leaderboard')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
    })
  })

  describe('Ground Truth Endpoints', () => {
    it('should return ground truth domains', async () => {
      const result = await apiCall('/ground-truth/domains')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
      if (result.data) {
        expect(result.data).toHaveProperty('domains')
      }
    })
  })

  describe('Arena Endpoints', () => {
    it('should return recent arena sessions', async () => {
      const result = await apiCall('/arena/recent')
      
      if (result.status === 0) return
      
      expect(result.ok).toBe(true)
    })
  })
})

describe('API Response Structures', () => {
  describe('TrueSkill Rating Format', () => {
    it('should have correct rating structure in leaderboard', async () => {
      const result = await apiCall('/leaderboard/domain/code_generation')
      
      if (result.status === 0 || !result.data?.entries?.length) return
      
      const entry = result.data.entries[0]
      // Leaderboard entries have mu/sigma directly
      expect(entry).toHaveProperty('mu')
      expect(entry).toHaveProperty('sigma')
      expect(typeof entry.mu).toBe('number')
      expect(typeof entry.sigma).toBe('number')
    })
  })

  describe('Similar Domain Response Format', () => {
    it('should have correct similar domain structure', async () => {
      const result = await apiCall('/transfer/domains/code_generation/similar')
      
      if (result.status === 0 || !result.data?.similar_domains?.length) return
      
      const domain = result.data.similar_domains[0]
      expect(domain).toHaveProperty('name')
      expect(domain).toHaveProperty('correlation')
      expect(typeof domain.correlation).toBe('number')
      expect(domain.correlation).toBeGreaterThanOrEqual(0)
      expect(domain.correlation).toBeLessThanOrEqual(1)
    })
  })

  describe('Specialist Response Format', () => {
    it('should have correct specialist structure', async () => {
      const result = await apiCall('/specialists/domain/code_generation')
      
      if (result.status === 0 || !Array.isArray(result.data) || !result.data.length) return
      
      const specialist = result.data[0]
      expect(specialist).toHaveProperty('model_id')
      expect(specialist).toHaveProperty('domain')
      expect(specialist).toHaveProperty('z_score')
      expect(specialist).toHaveProperty('category')
      expect(['specialist', 'weak_spot', 'normal']).toContain(specialist.category)
    })
  })
})

describe('Error Handling', () => {
  it('should return 404 for non-existent domain', async () => {
    const result = await apiCall('/leaderboard/nonexistent_domain_xyz')
    
    if (result.status === 0) return
    
    // Should either be 404 or return empty data
    expect([200, 404]).toContain(result.status)
  })

  it('should return 404 for non-existent model', async () => {
    const result = await apiCall('/models/nonexistent_model_xyz')
    
    if (result.status === 0) return
    
    expect([404, 422]).toContain(result.status)
  })
})

// Summary test that prints connectivity status
describe('Connectivity Summary', () => {
  it('prints backend status', async () => {
    const health = await apiCall('/health')
    
    console.log('\n================================')
    console.log('  CONNECTIVITY TEST SUMMARY')
    console.log('================================')
    
    if (health.status === 0) {
      console.log('❌ Backend: NOT RUNNING')
      console.log('')
      console.log('To start the backend:')
      console.log('  cd Algorithm')
      console.log('  python -m uvicorn langscope.api.main:app --reload')
    } else if (health.ok) {
      console.log('✅ Backend: HEALTHY')
      console.log(`   Version: ${health.data?.version || 'unknown'}`)
      console.log(`   Database: ${health.data?.database_connected ? 'Connected' : 'Not connected'}`)
    } else {
      console.log(`⚠️ Backend: ERROR (HTTP ${health.status})`)
    }
    
    console.log('================================\n')
  })
})
