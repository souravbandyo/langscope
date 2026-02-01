import { SketchCard } from '@/components/sketch'
import styles from './ErrorState.module.css'

interface ErrorStateProps {
  title?: string
  message?: string
  error?: Error | null
  onRetry?: () => void
  compact?: boolean
}

/**
 * Reusable error state component with retry functionality
 */
export function ErrorState({
  title = 'Something went wrong',
  message,
  error,
  onRetry,
  compact = false,
}: ErrorStateProps) {
  // Determine the error message
  const getErrorMessage = () => {
    if (message) return message

    if (error?.message?.includes('Failed to fetch') || 
        error?.message?.includes('NetworkError') ||
        error?.message?.includes('ERR_CONNECTION_REFUSED')) {
      return 'Unable to connect to the server. Please check if the API is running.'
    }

    if (error?.message?.includes('404')) {
      return 'The requested resource was not found.'
    }

    if (error?.message?.includes('500')) {
      return 'Server error. Please try again later.'
    }

    return error?.message || 'An unexpected error occurred.'
  }

  if (compact) {
    return (
      <div className={styles.compactError}>
        <span className={styles.errorIcon}>⚠️</span>
        <span className={styles.errorText}>{getErrorMessage()}</span>
        {onRetry && (
          <button className={styles.retryButtonSmall} onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    )
  }

  return (
    <SketchCard padding="lg">
      <div className={styles.errorState}>
        <div className={styles.errorIcon}>⚠️</div>
        <h3 className={styles.errorTitle}>{title}</h3>
        <p className={styles.errorMessage}>{getErrorMessage()}</p>
        {onRetry && (
          <button className={styles.retryButton} onClick={onRetry}>
            Try Again
          </button>
        )}
        <p className={styles.helpText}>
          If the problem persists, make sure the backend API is running at{' '}
          <code>localhost:8000</code>
        </p>
      </div>
    </SketchCard>
  )
}

/**
 * Loading state component
 */
export function LoadingState({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className={styles.loadingState}>
      <div className={styles.spinner} />
      <span className={styles.loadingText}>{message}</span>
    </div>
  )
}

/**
 * Empty state component
 */
export function EmptyState({
  title = 'No data available',
  message,
  icon = '📭',
}: {
  title?: string
  message?: string
  icon?: string
}) {
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyIcon}>{icon}</div>
      <h3 className={styles.emptyTitle}>{title}</h3>
      {message && <p className={styles.emptyMessage}>{message}</p>}
    </div>
  )
}
