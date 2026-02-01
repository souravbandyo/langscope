import { AlertsSection } from './AlertsSection'
import { PerformanceCharts } from './PerformanceCharts'
import { TopModelsTable } from './TopModelsTable'
import { RecentActivity } from './RecentActivity'
import { SystemHealth } from './SystemHealth'
import { LedgerVerification } from './LedgerVerification'
import styles from './Dashboard.module.css'

/**
 * Dashboard page with charts, alerts, and system overview
 * Layout matches wireframe: charts on top, sticky alerts, tables below
 */
export function Dashboard() {
  return (
    <div className={styles.dashboard}>
      <h1 className={styles.title}>Dashboard</h1>

      {/* Performance Charts Row */}
      <PerformanceCharts />

      {/* Alerts Section - Sticky Notes */}
      <AlertsSection />

      {/* Bottom Section - Tables and Stats */}
      <div className={styles.bottomSection}>
        <div className={styles.leftColumn}>
          <TopModelsTable />
          <RecentActivity />
        </div>
        <div className={styles.rightColumn}>
          <SystemHealth />
        </div>
      </div>

      {/* Ledger Verification Section */}
      <LedgerVerification />
    </div>
  )
}

export { Dashboard as default }
