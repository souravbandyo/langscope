import { useState } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useDomains, useSimilarDomains, useTransferLeaderboard, useDomainIndexStats } from '@/api/hooks'
import styles from './Transfer.module.css'

// Default domains when API is unavailable
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']

/**
 * Transfer Learning page for domain correlations and predictions
 */
export function Transfer() {
  const [selectedDomain, setSelectedDomain] = useState('')

  const { data: domainsData, error: domainsError } = useDomains()
  const { data: indexStats } = useDomainIndexStats()
  const { data: similarDomains, isLoading: similarLoading } = useSimilarDomains(
    selectedDomain,
    { limit: 10, min_correlation: 0.3 }
  )
  const { data: transferLeaderboard, isLoading: leaderboardLoading } = useTransferLeaderboard(
    selectedDomain,
    { include_transferred: true, limit: 15 }
  )

  const domains = domainsError ? defaultDomains : (domainsData?.domains || [])

  return (
    <div className={styles.transfer}>
      <h1 className={styles.title}>🔄 Transfer Learning</h1>
      <p className={styles.subtitle}>
        Predict model performance in new domains using ratings from similar evaluated domains
      </p>

      {/* Index Stats */}
      {indexStats && (
        <div className={styles.statsRow}>
          <StickyNote title="Domain Index" color="yellow" rotation={-1}>
            <div className={styles.statValue}>{indexStats.total_domains}</div>
            <div className={styles.statLabel}>Total domains</div>
          </StickyNote>
          <StickyNote title="With Facets" color="blue" rotation={1}>
            <div className={styles.statValue}>{indexStats.domains_with_facets}</div>
            <div className={styles.statLabel}>Domains with facets</div>
          </StickyNote>
          <StickyNote title="Similarities" color="green" rotation={-0.5}>
            <div className={styles.statValue}>{indexStats.precomputed_similarities}</div>
            <div className={styles.statLabel}>Precomputed pairs</div>
          </StickyNote>
        </div>
      )}

      {/* Domain Selection */}
      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}>Select Target Domain</h2>
        <p className={styles.sectionDesc}>
          Choose a domain to see similar domains and transfer-based ratings
        </p>

        <div className={styles.domainSelect}>
          <select
            className={styles.select}
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
          >
            <option value="">Choose a domain...</option>
            {domains.map((d) => (
              <option key={d} value={d}>
                {d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </SketchCard>

      {selectedDomain && (
        <div className={styles.resultsGrid}>
          {/* Similar Domains */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Similar Domains</h2>
            <p className={styles.sectionDesc}>
              Domains with correlated model performance
            </p>

            {similarLoading ? (
              <div className={styles.loading}>Finding similar domains...</div>
            ) : !similarDomains?.similar_domains?.length ? (
              <div className={styles.empty}>No similar domains found</div>
            ) : (
              <div className={styles.similarList}>
                {similarDomains.similar_domains.map((domain) => (
                  <div key={domain.name} className={styles.similarCard}>
                    <div className={styles.similarHeader}>
                      <span className={styles.similarName}>
                        {domain.name.replace(/_/g, ' ')}
                      </span>
                      <span className={styles.correlation}>
                        ρ = {domain.correlation.toFixed(2)}
                      </span>
                    </div>
                    <div className={styles.correlationBar}>
                      <div
                        className={styles.correlationFill}
                        style={{ width: `${domain.correlation * 100}%` }}
                      />
                    </div>
                    {domain.facet_breakdown && Object.keys(domain.facet_breakdown).length > 0 && (
                      <div className={styles.facetBreakdown}>
                        {Object.entries(domain.facet_breakdown).slice(0, 3).map(([facet, contribution]) => (
                          <span key={facet} className={styles.facetTag}>
                            {facet}: {(contribution.contribution * 100).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {similarDomains?.facets && Object.keys(similarDomains.facets).length > 0 && (
              <div className={styles.domainFacets}>
                <h3 className={styles.facetsTitle}>Domain Facets</h3>
                <div className={styles.facetsList}>
                  {Object.entries(similarDomains.facets).map(([facet, value]) => (
                    <div key={facet} className={styles.facetItem}>
                      <span className={styles.facetName}>{facet}:</span>
                      <span className={styles.facetValue}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </SketchCard>

          {/* Transfer Leaderboard */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Transfer-Aware Leaderboard</h2>
            <p className={styles.sectionDesc}>
              Includes both direct ratings and transfer predictions
            </p>

            {leaderboardLoading ? (
              <div className={styles.loading}>Loading leaderboard...</div>
            ) : !transferLeaderboard?.entries?.length ? (
              <div className={styles.empty}>No data available</div>
            ) : (
              <>
                <div className={styles.leaderboardMeta}>
                  <span className={styles.metaItem}>
                    Direct: {transferLeaderboard.direct_count}
                  </span>
                  <span className={styles.metaItem}>
                    Transferred: {transferLeaderboard.transferred_count}
                  </span>
                </div>

                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Model</th>
                      <th>Rating (μ)</th>
                      <th>Source</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transferLeaderboard.entries.map((entry) => (
                      <tr 
                        key={entry.model_id}
                        className={`${styles.tableRow} ${entry.source === 'transfer' ? styles.transferred : ''}`}
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
                        <td className={styles.modelName}>{entry.model_id}</td>
                        <td className={styles.rating}>{Math.round(entry.rating.mu)}</td>
                        <td className={styles.source}>
                          <span className={`${styles.sourceBadge} ${styles[entry.source]}`}>
                            {entry.source === 'direct' ? '📊' : '🔄'} {entry.source}
                          </span>
                        </td>
                        <td className={styles.confidence}>
                          {(entry.confidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </SketchCard>
        </div>
      )}
    </div>
  )
}

export { Transfer as default }
