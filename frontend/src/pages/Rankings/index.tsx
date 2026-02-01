import { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { SketchCard } from '@/components/sketch'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useLeaderboard, useDomains, useModels } from '@/api/hooks'
import type { RatingDimension } from '@/api/types'
import styles from './Rankings.module.css'

const dimensions: { id: RatingDimension; name: string; icon: string; description: string }[] = [
  { id: 'raw_quality', name: 'Raw', icon: 'ph ph-star', description: 'Pure response quality' },
  { id: 'cost_adjusted', name: 'Cost', icon: 'ph ph-currency-dollar', description: 'Quality per dollar' },
  { id: 'latency', name: 'Latency', icon: 'ph ph-lightning', description: 'Response speed' },
  { id: 'ttft', name: 'TTFT', icon: 'ph ph-rocket', description: 'Time to first token' },
  { id: 'consistency', name: 'Consistent', icon: 'ph ph-target', description: 'Output reliability' },
  { id: 'token_efficiency', name: 'Efficient', icon: 'ph ph-chart-bar', description: 'Quality per token' },
  { id: 'instruction_following', name: 'Instruct', icon: 'ph ph-clipboard-text', description: 'Format compliance' },
  { id: 'hallucination_resistance', name: 'Factual', icon: 'ph ph-check-circle', description: 'Hallucination resistance' },
  { id: 'long_context', name: 'Context', icon: 'ph ph-books', description: 'Long context quality' },
  { id: 'combined', name: 'Combined', icon: 'ph ph-trophy', description: 'Weighted aggregate' },
]

/**
 * Rankings page with leaderboard table, dimension tabs, and provider filtering
 */
export function Rankings() {
  const { domain: urlDomain } = useParams()
  const navigate = useNavigate()
  const [selectedDomain, setSelectedDomain] = useState(urlDomain || '')
  const [selectedDimension, setSelectedDimension] = useState<RatingDimension>('raw_quality')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [showBaseModels, setShowBaseModels] = useState(false)

  // Update selected domain when URL changes
  useEffect(() => {
    if (urlDomain) {
      setSelectedDomain(urlDomain)
    }
  }, [urlDomain])

  // Fetch domains for filter
  const { data: domainsData } = useDomains()
  
  // Fetch models to get provider list
  const { data: modelsData } = useModels()

  // Fetch leaderboard data
  const { data: leaderboardData, isLoading, error, refetch } = useLeaderboard({
    domain: selectedDomain || undefined,
    rankingType: selectedDimension,
    limit: 100,
  })

  // Extract unique providers from models
  const providers = useMemo(() => {
    if (!modelsData?.models) return []
    const providerSet = new Set(modelsData.models.map((m) => m.provider))
    return Array.from(providerSet).sort()
  }, [modelsData])

  // Filter entries by provider
  const filteredEntries = useMemo(() => {
    if (!leaderboardData?.entries) return []
    let entries = leaderboardData.entries
    
    if (selectedProvider) {
      entries = entries.filter((e) => e.provider === selectedProvider)
    }
    
    // Re-rank after filtering
    return entries.map((entry, index) => ({
      ...entry,
      displayRank: index + 1,
    }))
  }, [leaderboardData, selectedProvider])

  const domainOptions = [
    { id: '', name: 'All Domains' },
    ...(domainsData?.domains || []).map((d) => ({
      id: d,
      name: d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    })),
  ]

  return (
    <div className={styles.rankings}>
      <div className={styles.header}>
        <h1 className={styles.title}>Rankings</h1>
        <p className={styles.subtitle}>
          Compare model performance across 10 dimensions
        </p>
      </div>

      {/* Dimension Tabs */}
      <div className={styles.dimensionTabs}>
        {dimensions.map((dim) => (
          <button
            key={dim.id}
            className={`${styles.dimensionTab} ${selectedDimension === dim.id ? styles.active : ''}`}
            onClick={() => setSelectedDimension(dim.id)}
            title={dim.description}
          >
            <i className={`${dim.icon} ${styles.tabIcon}`}></i>
            <span className={styles.tabName}>{dim.name}</span>
          </button>
        ))}
      </div>

      {/* Active dimension description */}
      <div className={styles.dimensionInfo}>
        <span className={styles.dimensionLabel}>
          <i className={dimensions.find((d) => d.id === selectedDimension)?.icon}></i>{' '}
          {dimensions.find((d) => d.id === selectedDimension)?.name}:
        </span>
        <span className={styles.dimensionDesc}>
          {dimensions.find((d) => d.id === selectedDimension)?.description}
        </span>
      </div>

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
          <label className={styles.filterLabel}>Provider:</label>
          <select
            className={styles.select}
            value={selectedProvider}
            onChange={(e) => setSelectedProvider(e.target.value)}
          >
            <option value="">All Providers</option>
            {providers.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.filterGroup}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showBaseModels}
              onChange={(e) => setShowBaseModels(e.target.checked)}
            />
            <span>Show base model info</span>
          </label>
        </div>
      </div>

      {/* Stats Bar */}
      {leaderboardData && (
        <div className={styles.statsBar}>
          <span className={styles.statItem}>
            <strong>{filteredEntries.length}</strong> models
            {selectedProvider && ` from ${selectedProvider}`}
          </span>
          <span className={styles.statItem}>
            Domain: <strong>{selectedDomain || 'All'}</strong>
          </span>
          <span className={styles.statItem}>
            Total evaluations: <strong>{leaderboardData.total || filteredEntries.length}</strong>
          </span>
        </div>
      )}

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
                <th className={styles.thRank}>Rank</th>
                <th className={styles.thModel}>Model</th>
                {showBaseModels && <th className={styles.thBase}>Base Model</th>}
                <th className={styles.thProvider}>Provider</th>
                <th className={styles.thRating}>Rating (μ)</th>
                <th className={styles.thUncertainty}>±σ</th>
                <th className={styles.thConservative}>Conservative</th>
                <th className={styles.thMatches}>Matches</th>
              </tr>
            </thead>
            <tbody>
              {filteredEntries.length === 0 ? (
                <tr>
                  <td colSpan={showBaseModels ? 8 : 7}>
                    <EmptyState
                      title="No rankings available"
                      message="No models have been evaluated for this domain yet."
                      icon="ph ph-chart-bar"
                    />
                  </td>
                </tr>
              ) : (
                filteredEntries.map((entry) => (
                  <tr key={entry.model_id} className={styles.tableRow}>
                    <td className={styles.rank}>
                      {entry.displayRank <= 3 ? (
                        <span className={`${styles.medal} ${styles[`rank${entry.displayRank}`]}`}>
                          <i className="ph-fill ph-medal"></i>
                        </span>
                      ) : (
                        <span className={styles.rankNumber}>{entry.displayRank}</span>
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
                    {showBaseModels && (
                      <td className={styles.baseModel}>
                        {entry.model_id.split('/')[0] || '-'}
                      </td>
                    )}
                    <td className={styles.provider}>
                      <span className={styles.providerBadge}>{entry.provider}</span>
                    </td>
                    <td className={styles.rating}>
                      <span className={styles.muValue}>{Math.round(entry.mu)}</span>
                    </td>
                    <td className={styles.uncertainty}>
                      <span className={styles.sigmaValue}>±{Math.round(entry.sigma)}</span>
                    </td>
                    <td className={styles.conservative}>
                      <span className={styles.conservativeValue}>
                        {Math.round(entry.conservative_estimate || entry.mu - 2 * entry.sigma)}
                      </span>
                    </td>
                    <td className={styles.matches}>{entry.matches_played}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </SketchCard>
      )}

      {/* Legend */}
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendLabel}>μ (mu):</span>
          <span className={styles.legendDesc}>Expected skill rating</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLabel}>σ (sigma):</span>
          <span className={styles.legendDesc}>Uncertainty (lower = more confident)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLabel}>Conservative:</span>
          <span className={styles.legendDesc}>μ - 2σ (95% confidence lower bound)</span>
        </div>
      </div>
    </div>
  )
}

export { Rankings as default }
