/**
 * Protected Route Component
 * 
 * Redirects unauthenticated users to the login page.
 * Preserves the intended destination for redirect after login.
 */

import { Navigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { LoadingState } from '@/components/common'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, isLoading, isInitialized } = useAuthStore()
  const location = useLocation()

  // Show loading while initializing auth
  if (!isInitialized || isLoading) {
    return (
      <div style={{ padding: '2rem', display: 'flex', justifyContent: 'center' }}>
        <LoadingState message="Checking authentication..." />
      </div>
    )
  }

  // Redirect to login if not authenticated
  if (!user) {
    return <Navigate to="/auth" state={{ from: location.pathname }} replace />
  }

  return <>{children}</>
}

export default ProtectedRoute
