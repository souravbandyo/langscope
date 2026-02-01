/**
 * Auth Initializer Component
 * 
 * Initializes authentication state on app mount.
 * Must wrap the entire app to ensure auth is initialized before routing.
 */

import { useEffect } from 'react'
import { useAuthStore } from '@/store/authStore'
import { LoadingState } from '@/components/common'

interface AuthInitializerProps {
  children: React.ReactNode
}

export function AuthInitializer({ children }: AuthInitializerProps) {
  const { initialize, isInitialized, isLoading } = useAuthStore()

  useEffect(() => {
    initialize()
  }, [initialize])

  // Show loading screen while initializing
  if (!isInitialized && isLoading) {
    return (
      <div style={{ 
        height: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: 'var(--color-paper, #fdf6e3)'
      }}>
        <LoadingState message="Loading LangScope..." />
      </div>
    )
  }

  return <>{children}</>
}

export default AuthInitializer
