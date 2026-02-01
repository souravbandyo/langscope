import { useState } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useGroundTruthDomains, useGroundTruthLeaderboard, useNeedleHeatmap, useGroundTruthCoverage } from '@/api/hooks'
import styles from './GroundTruth.module.css'

/**
 * Ground Truth page for objective evaluation results
 */
export function GroundTruth() {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [selectedModel, setSelectedModel] = useState('')

  const { data: domains, isLoading: domainsLoading, error: domainsError, refetch: refetchDomains } = useGroundTruthDomains()
  const { data: leaderboard, isLoading: leaderboardLoading } = useGroundTruthLeaderboard(
    selectedDomain,
    { limit: 20 }
  )
  const { data: coverage } = useGroundTruthCoverage(selectedDomain)
  const { data: heatmap } = useNeedleHeatmap(selectedModel)

  const selectedDomainInfo = domains?.find((d) => d.name === selectedDomain)

  return (
    <div className={styles.groundTruth}>
      <h1 className={styles.title}>🎯 Ground Truth Evaluation</h1>
      <p className={styles.subtitle}>
        Objective model performance with verifiable answers
      </p>

      {/* Domain Cards */}
      <div className={styles.domainGrid}>
        {domainsLoading ? (
          <LoadingState message="Loading domains..." />
        ) : domainsError ? (
          <ErrorState
            title="Failed to load domains"
            error={domainsError as Error}
            onRetry={() => refetchDomains()}
          />
        ) : !domains?.length ? (
          <EmptyState
            title="No ground truth domains"
            message="Ground truth evaluation domains will appear here when configured."
            icon="🎯"
          />
        ) : (
          domains.map((domain) => (
            <button
              key={domain.name}
              className={`${styles.domainCard} ${selectedDomain === domain.name ? styles.selected : ''}`}
              onClick={() => setSelectedDomain(domain.name)}
            >
              <div className={styles.domainIcon}>
                {domain.category === 'multimodal' && '🖼️'}
                {domain.category === 'long_context' && '📚'}
                {!domain.category && '📋'}
              </div>
              <div className={styles.domainInfo}>
                <h3 className={styles.domainName}>{domain.name.replace(/_/g, ' ')}</h3>
                <span className={styles.domainCategory}>{domain.category}</span>
                <span className={styles.sampleCount}>{domain.sample_count} samples</span>
              </div>
            </button>
          ))
        )}
      </div>

      {/* Selected Domain Info */}
      {selectedDomainInfo && (
        <div className={styles.domainDetails}>
          <StickyNote title="Domain Info" color="blue" rotation={-1}>
            <div className={styles.infoContent}>
              <p><strong>Primary Metric:</strong> {selectedDomainInfo.primary_metric}</p>
              <p><strong>Languages:</strong> {selectedDomainInfo.supported_languages?.join(', ') || 'All'}</p>
              {selectedDomainInfo.description && <p>{selectedDomainInfo.description}</p>}
            </div>
          </StickyNote>

          {coverage && (
            <StickyNote title="Coverage" color="green" rotation={1}>
              <div className={styles.infoContent}>
                <p><strong>Total Samples:</strong> {coverage.total_samples}</p>
                <p><strong>Used Samples:</strong> {coverage.used_samples}</p>
                <p><strong>Coverage:</strong> {coverage.coverage_percentage.toFixed(1)}%</p>
              </div>
            </StickyNote>
          )}
        </div>
      )}

      {/* Leaderboard */}
      {selectedDomain && (
        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>
            {selectedDomain.replace(/_/g, ' ')} Leaderboard
          </h2>

          {leaderboardLoading ? (
            <div className={styles.loading}>Loading leaderboard...</div>
          ) : !leaderboard?.length ? (
            <div className={styles.empty}>No results for this domain</div>
          ) : (
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Rating (μ)</th>
                  <th>Avg Score</th>
                  <th>Evaluations</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry) => (
                  <tr
                    key={entry.deployment_id}
                    className={`${styles.tableRow} ${selectedModel === entry.deployment_id ? styles.selectedRow : ''}`}
                    onClick={() => setSelectedModel(entry.deployment_id)}
                  >
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
                    <td className={styles.modelName}>{entry.deployment_id}</td>
                    <td className={styles.rating}>{Math.round(entry.trueskill_mu)}</td>
                    <td className={styles.score}>{entry.primary_metric_avg.toFixed(2)}</td>
                    <td className={styles.evaluations}>{entry.total_evaluations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </SketchCard>
      )}

      {/* Needle Heatmap for selected model */}
      {selectedModel && selectedDomain === 'needle_in_haystack' && heatmap && (
        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>
            Needle-in-Haystack Performance: {selectedModel}
          </h2>
          <p className={styles.heatmapInfo}>
            Overall accuracy: {(heatmap.overall_accuracy * 100).toFixed(1)}%
          </p>
          <div className={styles.heatmapContainer}>
            <div className={styles.heatmapGrid}>
              {heatmap.heatmap.map((cell, index) => (
                <div
                  key={index}
                  className={styles.heatmapCell}
                  style={{
                    backgroundColor: `rgba(74, 144, 217, ${cell.accuracy})`,
                  }}
                  title={`Context: ${cell.context_length}, Position: ${(cell.needle_position * 100).toFixed(0)}%, Accuracy: ${(cell.accuracy * 100).toFixed(1)}%`}
                >
                  {(cell.accuracy * 100).toFixed(0)}
                </div>
              ))}
            </div>
            <div className={styles.heatmapLegend}>
              <span>Context Length →</span>
              <span>Needle Position ↓</span>
            </div>
          </div>
        </SketchCard>
      )}
    </div>
  )
}

export { GroundTruth as default }
