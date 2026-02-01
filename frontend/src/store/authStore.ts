/**
 * Auth Store - Zustand store for authentication state
 * 
 * Manages:
 * - User session state
 * - Authentication token for API requests
 * - Login/logout actions
 * - Supports both Supabase and local auth modes
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User, Session } from '@supabase/supabase-js'
import { supabase, isLocalAuthMode } from '@/lib/supabase'
import { api } from '@/api/client'

// Local auth API URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Local storage key for local auth session
const LOCAL_AUTH_KEY = 'langscope-local-auth'

interface LocalAuthSession {
  access_token: string
  user: {
    id: string
    email: string
    role: string
  }
  expires_at: number
}

interface AuthState {
  user: User | null
  session: Session | null
  localSession: LocalAuthSession | null
  isLoading: boolean
  isInitialized: boolean
  error: string | null
  isLocalAuth: boolean
  
  // Actions
  initialize: () => Promise<void>
  signIn: (email: string, password: string) => Promise<{ error: string | null }>
  signUp: (email: string, password: string) => Promise<{ error: string | null }>
  signOut: () => Promise<void>
  clearError: () => void
}

// Local auth helper functions
async function localSignIn(email: string, password: string): Promise<{ 
  success: boolean
  session?: LocalAuthSession
  error?: string 
}> {
  try {
    // For local auth, we call our backend's auth endpoint
    const response = await fetch(`${API_BASE_URL}/auth/local/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    
    if (response.ok) {
      const data = await response.json()
      // Transform response to LocalAuthSession format
      const session: LocalAuthSession = {
        access_token: data.access_token,
        user: data.user,
        expires_at: data.expires_at,
      }
      return { success: true, session }
    }
    
    const errorData = await response.json().catch(() => ({}))
    return { 
      success: false, 
      error: errorData.detail || errorData.error || 'Login failed' 
    }
  } catch (err) {
    return { 
      success: false, 
      error: err instanceof Error ? err.message : 'Connection failed' 
    }
  }
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      session: null,
      localSession: null,
      isLoading: false,
      isInitialized: false,
      error: null,
      isLocalAuth: isLocalAuthMode,

      initialize: async () => {
        if (get().isInitialized) return
        
        set({ isLoading: true })
        
        try {
          if (isLocalAuthMode) {
            // Local auth mode - check localStorage for saved session
            const savedSession = localStorage.getItem(LOCAL_AUTH_KEY)
            if (savedSession) {
              try {
                const session: LocalAuthSession = JSON.parse(savedSession)
                // Check if token is expired
                if (session.expires_at > Date.now() / 1000) {
                  api.setAuthToken(session.access_token)
                  set({
                    localSession: session,
                    user: session.user as unknown as User,
                    isLoading: false,
                    isInitialized: true,
                    error: null
                  })
                  return
                }
              } catch {
                localStorage.removeItem(LOCAL_AUTH_KEY)
              }
            }
            set({ isLoading: false, isInitialized: true })
          } else if (supabase) {
            // Supabase auth mode
            const { data: { session }, error } = await supabase.auth.getSession()
            
            if (error) {
              console.error('Failed to get session:', error)
              set({ isLoading: false, isInitialized: true, error: error.message })
              return
            }
            
            if (session) {
              api.setAuthToken(session.access_token)
              set({ 
                user: session.user, 
                session, 
                isLoading: false, 
                isInitialized: true,
                error: null 
              })
            } else {
              set({ isLoading: false, isInitialized: true })
            }
            
            // Listen for auth state changes
            supabase.auth.onAuthStateChange((_event, session) => {
              if (session) {
                api.setAuthToken(session.access_token)
                set({ user: session.user, session, error: null })
              } else {
                api.setAuthToken(null)
                set({ user: null, session: null })
              }
            })
          } else {
            // No auth configured
            set({ isLoading: false, isInitialized: true })
          }
        } catch (err) {
          console.error('Auth initialization error:', err)
          set({ 
            isLoading: false, 
            isInitialized: true, 
            error: 'Failed to initialize authentication' 
          })
        }
      },

      signIn: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        
        try {
          if (isLocalAuthMode) {
            // Local auth mode - use backend's local login endpoint
            const result = await localSignIn(email, password)
            
            if (result.success && result.session) {
              localStorage.setItem(LOCAL_AUTH_KEY, JSON.stringify(result.session))
              api.setAuthToken(result.session.access_token)
              set({
                localSession: result.session,
                user: result.session.user as unknown as User,
                isLoading: false,
                error: null
              })
              return { error: null }
            }
            
            set({ isLoading: false, error: result.error || 'Invalid credentials' })
            return { error: result.error || 'Invalid credentials' }
          } else if (supabase) {
            // Supabase auth mode
            const { data, error } = await supabase.auth.signInWithPassword({
              email,
              password,
            })
            
            if (error) {
              set({ isLoading: false, error: error.message })
              return { error: error.message }
            }
            
            if (data.session) {
              api.setAuthToken(data.session.access_token)
              set({ 
                user: data.user, 
                session: data.session, 
                isLoading: false,
                error: null 
              })
            }
            
            return { error: null }
          } else {
            set({ isLoading: false, error: 'Authentication not configured' })
            return { error: 'Authentication not configured' }
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Sign in failed'
          set({ isLoading: false, error: message })
          return { error: message }
        }
      },

      signUp: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        
        try {
          if (isLocalAuthMode) {
            // Local auth mode - signup is not needed, credentials are preset
            set({ isLoading: false, error: 'Use test@langscope.dev / TestPassword123! for local dev' })
            return { error: null }
          } else if (supabase) {
            const { data, error } = await supabase.auth.signUp({
              email,
              password,
            })
            
            if (error) {
              set({ isLoading: false, error: error.message })
              return { error: error.message }
            }
            
            if (data.session) {
              api.setAuthToken(data.session.access_token)
              set({ 
                user: data.user, 
                session: data.session, 
                isLoading: false,
                error: null 
              })
            } else {
              set({ isLoading: false })
            }
            
            return { error: null }
          } else {
            set({ isLoading: false, error: 'Authentication not configured' })
            return { error: 'Authentication not configured' }
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Sign up failed'
          set({ isLoading: false, error: message })
          return { error: message }
        }
      },

      signOut: async () => {
        set({ isLoading: true })
        
        try {
          if (isLocalAuthMode) {
            localStorage.removeItem(LOCAL_AUTH_KEY)
            localStorage.removeItem('sb-localhost-auth-token')
          } else if (supabase) {
            await supabase.auth.signOut()
          }
          
          api.setAuthToken(null)
          set({ 
            user: null, 
            session: null, 
            localSession: null,
            isLoading: false, 
            error: null 
          })
        } catch (err) {
          console.error('Sign out error:', err)
          api.setAuthToken(null)
          set({ 
            user: null, 
            session: null, 
            localSession: null,
            isLoading: false 
          })
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'langscope-auth',
      partialize: () => ({}),
    }
  )
)

export default useAuthStore
