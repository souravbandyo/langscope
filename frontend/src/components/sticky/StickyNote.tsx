import { useMemo } from 'react'
import clsx from 'clsx'
import { getRandomRotation } from '@/styles/sketch-theme'
import styles from './StickyNote.module.css'

export type StickyColor = 'yellow' | 'pink' | 'blue' | 'green' | 'orange'

export interface StickyNoteProps {
  children: React.ReactNode
  title?: string
  color?: StickyColor
  rotation?: number
  pinned?: boolean
  className?: string
  onClick?: () => void
  size?: 'sm' | 'md' | 'lg'
}

/**
 * Sticky note component with hand-drawn aesthetic
 * Features rotation, pin, and various colors
 */
export function StickyNote({
  children,
  title,
  color = 'yellow',
  rotation,
  pinned = false,
  className,
  onClick,
  size = 'md',
}: StickyNoteProps) {
  // Use provided rotation or generate random one
  const noteRotation = useMemo(
    () => rotation ?? getRandomRotation(),
    [rotation]
  )

  const colorClasses: Record<StickyColor, string> = {
    yellow: styles.colorYellow,
    pink: styles.colorPink,
    blue: styles.colorBlue,
    green: styles.colorGreen,
    orange: styles.colorOrange,
  }

  const sizeClasses = {
    sm: styles.sizeSm,
    md: styles.sizeMd,
    lg: styles.sizeLg,
  }

  return (
    <div
      className={clsx(
        styles.stickyNote,
        colorClasses[color],
        sizeClasses[size],
        onClick && styles.clickable,
        className
      )}
      style={{ '--rotation': `${noteRotation}deg` } as React.CSSProperties}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {pinned && (
        <div className={styles.pin}>
          <div className={styles.pinHead} />
        </div>
      )}
      {title && <h3 className={styles.title}>{title}</h3>}
      <div className={styles.content}>{children}</div>
    </div>
  )
}

/**
 * Alert-style sticky note with icon
 */
export interface AlertStickyProps extends Omit<StickyNoteProps, 'pinned'> {
  icon?: React.ReactNode
  type?: 'info' | 'warning' | 'success' | 'error'
}

export function AlertSticky({
  icon,
  type = 'info',
  color,
  children,
  ...props
}: AlertStickyProps) {
  // Default colors based on type
  const typeColors: Record<string, StickyColor> = {
    info: 'blue',
    warning: 'orange',
    success: 'green',
    error: 'pink',
  }

  const defaultIcons: Record<string, string> = {
    info: 'ph ph-info',
    warning: 'ph ph-warning',
    success: 'ph ph-check-circle',
    error: 'ph ph-x-circle',
  }

  return (
    <StickyNote color={color ?? typeColors[type]} pinned {...props}>
      <div className={styles.alertContent}>
        {icon ?? <i className={`${defaultIcons[type]} ${styles.alertIcon}`}></i>}
        <div>{children}</div>
      </div>
    </StickyNote>
  )
}

/**
 * Leaderboard-style sticky note for displaying rankings
 */
export interface LeaderboardStickyProps {
  title: string
  domain: string
  entries: Array<{ rank: number; name: string; score?: number }>
  color?: StickyColor
  rotation?: number
  onClick?: () => void
}

export function LeaderboardSticky({
  title,
  domain,
  entries,
  color = 'yellow',
  rotation,
  onClick,
}: LeaderboardStickyProps) {
  return (
    <StickyNote
      title={title}
      color={color}
      rotation={rotation}
      pinned
      onClick={onClick}
      size="md"
    >
      <div className={styles.leaderboardContent}>
        <span className={styles.domainTag}>#{domain}</span>
        <ol className={styles.leaderboardList}>
          {entries.slice(0, 5).map((entry) => (
            <li key={entry.rank} className={styles.leaderboardEntry}>
              <span className={styles.rank}>{entry.rank}.</span>
              <span className={styles.name}>{entry.name}</span>
              {entry.score !== undefined && (
                <span className={styles.score}>{entry.score}</span>
              )}
            </li>
          ))}
        </ol>
      </div>
    </StickyNote>
  )
}
