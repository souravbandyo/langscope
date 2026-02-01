import { SketchCard } from '@/components/sketch'
import { useMatches } from '@/api/hooks'
import styles from './RecentActivity.module.css'

/**
 * Recent activity table showing recent matches
 */
export function RecentActivity() {
  const { data, isLoading, error } = useMatches({ limit: 5 })

  return (
    <SketchCard padding="md">
      <h3 className={styles.title}>Recent Activity</h3>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>#</th>
            <th>Domain</th>
            <th>Participants</th>
          </tr>
        </thead>
        <tbody>
          {isLoading ? (
            <tr>
              <td colSpan={3}>Loading...</td>
            </tr>
          ) : error ? (
            <tr>
              <td colSpan={3}>Failed to load</td>
            </tr>
          ) : data?.matches?.length === 0 ? (
            <tr>
              <td colSpan={3}>No recent activity</td>
            </tr>
          ) : (
            data?.matches?.slice(0, 5).map((match, index) => (
              <tr key={match.match_id}>
                <td>{index + 1}</td>
                <td>{match.domain.replace(/_/g, ' ')}</td>
                <td>{match.participants.length}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </SketchCard>
  )
}
