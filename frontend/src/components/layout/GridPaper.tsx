import clsx from 'clsx'
import styles from './GridPaper.module.css'

export interface GridPaperProps {
  children: React.ReactNode
  className?: string
  variant?: 'simple' | 'detailed'
}

/**
 * Grid paper background wrapper
 * Provides the graph paper aesthetic for the entire app
 */
export function GridPaper({
  children,
  className,
  variant = 'detailed',
}: GridPaperProps) {
  return (
    <div
      className={clsx(
        styles.gridPaper,
        variant === 'detailed' && styles.detailed,
        className
      )}
    >
      {children}
    </div>
  )
}
