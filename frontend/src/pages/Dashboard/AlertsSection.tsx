import { StickyNote } from '@/components/sticky'
import { useAlerts } from '@/api/hooks'
import styles from './AlertsSection.module.css'

const alertColors = ['yellow', 'orange', 'pink', 'blue'] as const
const rotations = [-2, 1, -1, 2]

/**
 * Alert section with sticky notes
 */
export function AlertsSection() {
  const { data: alerts, isLoading, error } = useAlerts({ include_resolved: false })

  if (isLoading) {
    return (
      <section className={styles.section}>
        <div className={styles.stickyRow}>
          <StickyNote title="Loading..." color="yellow" rotation={0} pinned size="sm">
            <div>Fetching alerts...</div>
          </StickyNote>
        </div>
      </section>
    )
  }

  if (error || !alerts || alerts.length === 0) {
    return (
      <section className={styles.section}>
        <div className={styles.stickyRow}>
          <StickyNote title="All Clear!" color="green" rotation={-1} pinned size="sm">
            <div>No active alerts</div>
          </StickyNote>
        </div>
      </section>
    )
  }

  return (
    <section className={styles.section}>
      <div className={styles.stickyRow}>
        {alerts.slice(0, 4).map((alert, index) => (
          <StickyNote
            key={alert.id}
            title={`${alert.level.toUpperCase()}:`}
            color={alert.level === 'critical' ? 'orange' : alert.level === 'warning' ? 'yellow' : alertColors[index % alertColors.length]}
            rotation={rotations[index % rotations.length]}
            pinned
            size="sm"
          >
            <div>{alert.message}</div>
            {alert.domain && <div className={styles.domain}>Domain: {alert.domain}</div>}
          </StickyNote>
        ))}
      </div>
    </section>
  )
}
