import { SketchCard } from '@/components/sketch'
import { useMonitoringHealth } from '@/api/hooks'
import styles from './SystemHealth.module.css'

/**
 * System health indicator
 */
export function SystemHealth() {
  const { data, isLoading, error } = useMonitoringHealth()

  const getStatusDisplay = () => {
    if (isLoading) return { text: 'Checking...', className: styles.statusLoading }
    if (error) return { text: 'Error', className: styles.statusError }
    if (!data) return { text: 'Unknown', className: styles.statusUnknown }
    
    const status = data.status?.toLowerCase()
    if (status === 'healthy' || status === 'ok') {
      return { text: 'Good', className: styles.statusGood }
    } else if (status === 'degraded') {
      return { text: 'Degraded', className: styles.statusDegraded }
    } else {
      return { text: 'Unhealthy', className: styles.statusError }
    }
  }

  const statusDisplay = getStatusDisplay()

  return (
    <SketchCard padding="md">
      <div className={styles.container}>
        <i className={`ph ph-gear ${styles.icon}`}></i>
        <div className={styles.content}>
          <span className={styles.label}>System Health:</span>
          <span className={`${styles.status} ${statusDisplay.className}`}>
            {statusDisplay.text}
          </span>
        </div>
        <div className={`${styles.indicator} ${statusDisplay.className}`} />
      </div>
    </SketchCard>
  )
}
