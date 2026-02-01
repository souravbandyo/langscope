import { useState } from 'react'
import { SketchCard } from '@/components/sketch'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useBenchmarkDefinitions, useBenchmarkLeaderboard, useBenchmarkCorrelations } from '@/api/hooks'
import styles from './Benchmarks.module.css'

/**
 * Benchmarks page displaying external benchmark results
 */
export function Benchmarks() {
  const [selectedBenchmark, setSelectedBenchmark] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('')

  const { data: definitions, isLoading: defsLoading, error: defsError, refetch: refetchDefs } = useBenchmarkDefinitions({
    category: selectedCategory || undefined,
  })
  const { data: leaderboard, isLoading: leaderboardLoading } = useBenchmarkLeaderboard(
    selectedBenchmark,
    { limit: 20 }
  )
  const { data: correlations } = useBenchmarkCorrelations({
    benchmark_id: selectedBenchmark || undefined,
    min_correlation: 0.3,
  })

  const categories = [...new Set(definitions?.map((d) => d.category) || [])]

  return (
    <div className={styles.benchmarks}>
      <h1 className={styles.title}>External Benchmarks</h1>
      <p className={styles.subtitle}>
        Standard benchmark results and their correlation with TrueSkill ratings
      </p>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Category:</label>
          <select
            className={styles.select}
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
          >
            <option value="">All Categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className={styles.content}>
        {/* Benchmark Definitions */}
        <SketchCard padding="md" className={styles.definitionsCard}>
          <h2 className={styles.sectionTitle}>Benchmarks</h2>

          {defsLoading ? (
            <LoadingState message="Loading benchmarks..." />
          ) : defsError ? (
            <ErrorState
              title="Failed to load benchmarks"
              error={defsError as Error}
              onRetry={() => refetchDefs()}
              compact
            />
          ) : !definitions?.length ? (
            <EmptyState
              title="No benchmarks found"
              message="Benchmark definitions will appear here when available."
              icon="ph ph-chart-bar"
            />
          ) : (
            <div className={styles.benchmarkList}>
              {definitions.map((def) => (
                <button
                  key={def._id}
                  className={`${styles.benchmarkItem} ${selectedBenchmark === def._id ? styles.selected : ''}`}
                  onClick={() => setSelectedBenchmark(def._id)}
                >
                  <div className={styles.benchmarkName}>{def.name}</div>
                  <div className={styles.benchmarkMeta}>
                    <span className={styles.benchmarkCategory}>{def.category}</span>
                    {def.scoring && (
                      <span className={styles.benchmarkMetric}>
                        {def.scoring.method} ({def.scoring.higher_is_better ? '↑' : '↓'})
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </SketchCard>

        {/* Leaderboard */}
        <div className={styles.mainContent}>
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>
              {selectedBenchmark
                ? `${definitions?.find((d) => d.id === selectedBenchmark)?.name || selectedBenchmark} Leaderboard`
                : 'Select a Benchmark'}
            </h2>

            {!selectedBenchmark ? (
              <div className={styles.empty}>Choose a benchmark from the list</div>
            ) : leaderboardLoading ? (
              <div className={styles.loading}>Loading leaderboard...</div>
            ) : !leaderboard?.length ? (
              <div className={styles.empty}>No results for this benchmark</div>
            ) : (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Score</th>
                    <th>Percentile</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((entry) => (
                    <tr key={entry.base_model_id} className={styles.tableRow}>
                      <td className={styles.rank}>
                        {entry.rank <= 3 ? (
                          <span className={`${styles.medal} ${styles[`rank${entry.rank}`]}`}>
                            <i className="ph-fill ph-medal"></i>
                          </span>
                        ) : (
                          entry.rank
                        )}
                      </td>
                      <td className={styles.modelName}>{entry.base_model_name}</td>
                      <td className={styles.score}>{entry.score.toFixed(2)}</td>
                      <td className={styles.percentile}>
                        <span className={styles.percentileBadge}>
                          Top {100 - entry.percentile}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </SketchCard>

          {/* Correlations */}
          {selectedBenchmark && correlations && correlations.length > 0 && (
            <SketchCard padding="md" className={styles.correlationsCard}>
              <h3 className={styles.sectionTitle}>TrueSkill Correlations</h3>
              <p className={styles.correlationInfo}>
                How this benchmark correlates with our rating dimensions
              </p>
              <div className={styles.correlationsList}>
                {correlations.map((corr) => (
                  <div key={corr.dimension} className={styles.correlationItem}>
                    <span className={styles.dimensionName}>
                      {corr.dimension.replace(/_/g, ' ')}
                    </span>
                    <div className={styles.correlationBar}>
                      <div
                        className={styles.correlationFill}
                        style={{ width: `${Math.abs(corr.correlation) * 100}%` }}
                      />
                    </div>
                    <span className={`${styles.correlationValue} ${corr.correlation >= 0 ? styles.positive : styles.negative}`}>
                      {corr.correlation.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </SketchCard>
          )}
        </div>
      </div>
    </div>
  )
}

export { Benchmarks as default }
