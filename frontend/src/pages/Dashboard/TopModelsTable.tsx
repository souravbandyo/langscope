import { SketchCard } from '@/components/sketch'
import { useLeaderboard } from '@/api/hooks'
import styles from './TopModelsTable.module.css'

/**
 * Top performing models table
 */
export function TopModelsTable() {
  const { data, isLoading, error } = useLeaderboard({ limit: 5 })

  return (
    <SketchCard padding="md">
      <h3 className={styles.title}>Top Performing Models</h3>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Model</th>
            <th>Rating (μ)</th>
            <th>Matches</th>
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
          ) : data?.entries?.length === 0 ? (
            <tr>
              <td colSpan={3}>No data</td>
            </tr>
          ) : (
            data?.entries?.slice(0, 5).map((model) => (
              <tr key={model.model_id}>
                <td>{model.name}</td>
                <td>{Math.round(model.mu)}</td>
                <td>{model.matches_played}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </SketchCard>
  )
}
