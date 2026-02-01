/**
 * Auth Page - Login and Sign Up
 * 
 * Stateless JWT authentication via Supabase.
 */

import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { PageLayout } from '@/components/layout'
import { SketchCard, SketchButton, SketchInput } from '@/components/sketch'
import { useAuthStore } from '@/store/authStore'
import styles from './Auth.module.css'

type AuthMode = 'signin' | 'signup'

export default function AuthPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { signIn, signUp, isLoading, error, clearError } = useAuthStore()
  
  const [mode, setMode] = useState<AuthMode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  // Get the redirect path from location state, default to home
  const from = (location.state as { from?: string })?.from || '/'

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLocalError(null)
    setSuccessMessage(null)
    clearError()

    // Basic validation
    if (!email || !password) {
      setLocalError('Please fill in all fields')
      return
    }

    if (mode === 'signup') {
      if (password !== confirmPassword) {
        setLocalError('Passwords do not match')
        return
      }
      if (password.length < 6) {
        setLocalError('Password must be at least 6 characters')
        return
      }
    }

    if (mode === 'signin') {
      const result = await signIn(email, password)
      if (!result.error) {
        navigate(from, { replace: true })
      }
    } else {
      const result = await signUp(email, password)
      if (!result.error) {
        setSuccessMessage('Check your email to verify your account!')
        setMode('signin')
      }
    }
  }

  const toggleMode = () => {
    setMode(mode === 'signin' ? 'signup' : 'signin')
    setLocalError(null)
    setSuccessMessage(null)
    clearError()
  }

  const displayError = localError || error

  return (
    <PageLayout>
      <div className={styles.container}>
        <SketchCard className={styles.authCard}>
          <div className={styles.header}>
            <h1 className={styles.title}>
              {mode === 'signin' ? 'Welcome Back' : 'Create Account'}
            </h1>
            <p className={styles.subtitle}>
              {mode === 'signin' 
                ? 'Sign in to access LangScope' 
                : 'Sign up to start evaluating LLMs'}
            </p>
          </div>

          {successMessage && (
            <div className={styles.success}>
              {successMessage}
            </div>
          )}

          {displayError && (
            <div className={styles.error}>
              {displayError}
            </div>
          )}

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.field}>
              <label htmlFor="email" className={styles.label}>Email</label>
              <SketchInput
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                disabled={isLoading}
                autoComplete="email"
              />
            </div>

            <div className={styles.field}>
              <label htmlFor="password" className={styles.label}>Password</label>
              <SketchInput
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                disabled={isLoading}
                autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              />
            </div>

            {mode === 'signup' && (
              <div className={styles.field}>
                <label htmlFor="confirmPassword" className={styles.label}>
                  Confirm Password
                </label>
                <SketchInput
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  disabled={isLoading}
                  autoComplete="new-password"
                />
              </div>
            )}

            <SketchButton
              type="submit"
              disabled={isLoading}
              className={styles.submitButton}
            >
              {isLoading 
                ? 'Please wait...' 
                : mode === 'signin' ? 'Sign In' : 'Sign Up'}
            </SketchButton>
          </form>

          <div className={styles.footer}>
            <p className={styles.toggleText}>
              {mode === 'signin' 
                ? "Don't have an account? " 
                : 'Already have an account? '}
              <button 
                type="button"
                onClick={toggleMode}
                className={styles.toggleButton}
                disabled={isLoading}
              >
                {mode === 'signin' ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </div>
        </SketchCard>

        <div className={styles.info}>
          <h2 className={styles.infoTitle}>Why LangScope?</h2>
          <ul className={styles.infoList}>
            <li>Domain-specific LLM rankings</li>
            <li>Multi-dimensional evaluations</li>
            <li>Real-time arena battles</li>
            <li>Transfer learning insights</li>
          </ul>
        </div>
      </div>
    </PageLayout>
  )
}
