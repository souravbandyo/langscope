/**
 * UsageBar Component
 * Progress bar showing usage vs limits
 */

import styles from './UsageBar.module.css'

interface UsageBarProps {
  label: string
  current: number
  limit: number
  unit?: string
  showPercentage?: boolean
}

export function UsageBar({
  label,
  current,
  limit,
  unit = '',
  showPercentage = true,
}: UsageBarProps) {
  const percentage = limit > 0 ? Math.min((current / limit) * 100, 100) : 0
  const isNearLimit = percentage >= 80
  const isOverLimit = percentage >= 100

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toLocaleString()
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.label}>{label}</span>
        <span className={styles.values}>
          {formatNumber(current)} / {formatNumber(limit)}
          {unit && ` ${unit}`}
          {showPercentage && (
            <span
              className={`${styles.percentage} ${isOverLimit ? styles.over : isNearLimit ? styles.warning : ''}`}
            >
              ({percentage.toFixed(0)}%)
            </span>
          )}
        </span>
      </div>
      <div className={styles.track}>
        <div
          className={`${styles.fill} ${isOverLimit ? styles.fillOver : isNearLimit ? styles.fillWarning : ''}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {isNearLimit && !isOverLimit && (
        <span className={styles.hint}>Approaching limit</span>
      )}
      {isOverLimit && <span className={styles.hintOver}>Limit exceeded</span>}
    </div>
  )
}
