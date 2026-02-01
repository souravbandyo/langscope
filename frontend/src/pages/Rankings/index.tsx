import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { SketchCard } from '@/components/sketch'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useLeaderboard, useDomains } from '@/api/hooks'
import type { RatingDimension } from '@/api/types'
import styles from './Rankings.module.css'

const dimensions: { id: RatingDimension; name: string }[] = [
  { id: 'raw_quality', name: 'Raw Quality' },
  { id: 'cost_adjusted', name: 'Cost Adjusted' },
  { id: 'latency', name: 'Latency' },
  { id: 'ttft', name: 'Time to First Token' },
  { id: 'consistency', name: 'Consistency' },
  { id: 'token_efficiency', name: 'Token Efficiency' },
  { id: 'instruction_following', name: 'Instruction Following' },
  { id: 'hallucination_resistance', name: 'Hallucination Resistance' },
  { id: 'long_context', name: 'Long Context' },
  { id: 'combined', name: 'Combined' },
]

/**
 * Rankings page with leaderboard table and filters
 */
export function Rankings() {
  const { domain: urlDomain } = useParams()
  const navigate = useNavigate()
  const [selectedDomain, setSelectedDomain] = useState(urlDomain || '')
  const [selectedDimension, setSelectedDimension] = useState<RatingDimension>('raw_quality')

  // Update selected domain when URL changes
  useEffect(() => {
    if (urlDomain) {
      setSelectedDomain(urlDomain)
    }
  }, [urlDomain])

  // Fetch domains for filter
  const { data: domainsData } = useDomains()

  // Fetch leaderboard data
  const { data: leaderboardData, isLoading, error, refetch } = useLeaderboard({
    domain: selectedDomain || undefined,
    rankingType: selectedDimension,
    limit: 50,
  })

  const domainOptions = [
    { id: '', name: 'All Domains' },
    ...(domainsData?.domains || []).map((d) => ({
      id: d,
      name: d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    })),
  ]

  return (
    <div className={styles.rankings}>
      <h1 className={styles.title}>Rankings</h1>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Domain:</label>
          <select
            className={styles.select}
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
          >
            {domainOptions.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Dimension:</label>
          <select
            className={styles.select}
            value={selectedDimension}
            onChange={(e) => setSelectedDimension(e.target.value as RatingDimension)}
          >
            {dimensions.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Leaderboard Table */}
      {isLoading ? (
        <SketchCard padding="lg">
          <LoadingState message="Loading leaderboard..." />
        </SketchCard>
      ) : error ? (
        <ErrorState
          title="Failed to load leaderboard"
          error={error as Error}
          onRetry={() => refetch()}
        />
      ) : (
        <SketchCard padding="lg">
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Provider</th>
                <th>Rating (μ)</th>
                <th>Matches</th>
              </tr>
            </thead>
            <tbody>
              {leaderboardData?.entries?.length === 0 ? (
                <tr>
                  <td colSpan={5}>
                    <EmptyState
                      title="No rankings available"
                      message="No models have been evaluated for this domain yet."
                      icon="📊"
                    />
                  </td>
                </tr>
              ) : (
                leaderboardData?.entries?.map((entry) => (
                  <tr key={entry.model_id} className={styles.tableRow}>
                    <td className={styles.rank}>
                      {entry.rank <= 3 ? (
                        <span className={styles.medal}>
                          {entry.rank === 1 && '🥇'}
                          {entry.rank === 2 && '🥈'}
                          {entry.rank === 3 && '🥉'}
                        </span>
                      ) : (
                        entry.rank
                      )}
                    </td>
                    <td className={styles.modelName}>
                      <button
                        className={styles.modelLink}
                        onClick={() => navigate(`/models/${entry.model_id}`)}
                      >
                        {entry.name}
                      </button>
                    </td>
                    <td className={styles.provider}>{entry.provider}</td>
                    <td className={styles.rating}>{Math.round(entry.mu)}</td>
                    <td className={styles.matches}>{entry.matches_played}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </SketchCard>
      )}
    </div>
  )
}

export { Rankings as default }
