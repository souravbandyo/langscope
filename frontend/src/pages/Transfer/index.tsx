import { useState, useMemo } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useDomains, useSimilarDomains, useTransferLeaderboard, useDomainIndexStats } from '@/api/hooks'
import styles from './Transfer.module.css'

// Default domains when API is unavailable
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization', 'creative_writing', 'math', 'analysis']

// Domain categories for the matrix
const domainCategories = {
  'Language': ['writing', 'translation', 'summarization', 'creative_writing'],
  'Technical': ['code', 'math', 'analysis', 'reasoning'],
  'Regional': ['hindi', 'odia', 'tamil', 'bengali'],
  'Specialized': ['medical', 'legal', 'finance'],
}

// Simulated correlation data for matrix display (would come from API)
const getCorrelation = (domain1: string, domain2: string): number => {
  if (domain1 === domain2) return 1.0
  // Simulated correlations based on domain similarity
  const sameCategory = Object.values(domainCategories).some(
    cat => cat.includes(domain1) && cat.includes(domain2)
  )
  if (sameCategory) return 0.6 + Math.random() * 0.3
  return 0.2 + Math.random() * 0.3
}

/**
 * Transfer Learning page for domain correlations and predictions
 */
export function Transfer() {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [viewMode, setViewMode] = useState<'similar' | 'matrix'>('similar')
  const [hoveredCell, setHoveredCell] = useState<{ row: string; col: string } | null>(null)

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

  // Create correlation matrix data
  const matrixDomains = useMemo(() => {
    // Use first 8 domains for the matrix
    return domains.slice(0, 8)
  }, [domains])

  const correlationMatrix = useMemo(() => {
    return matrixDomains.map(row => 
      matrixDomains.map(col => ({
        row,
        col,
        correlation: getCorrelation(row, col)
      }))
    )
  }, [matrixDomains])

  return (
    <div className={styles.transfer}>
      <div className={styles.header}>
        <h1 className={styles.title}>Transfer Learning</h1>
        <p className={styles.subtitle}>
          Predict model performance in new domains using ratings from similar evaluated domains
        </p>
      </div>

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

      {/* View Mode Toggle */}
      <div className={styles.viewModeToggle}>
        <button
          className={`${styles.viewModeBtn} ${viewMode === 'similar' ? styles.active : ''}`}
          onClick={() => setViewMode('similar')}
        >
          <i className={`ph ph-clipboard-text ${styles.viewIcon}`}></i>
          Similar Domains
        </button>
        <button
          className={`${styles.viewModeBtn} ${viewMode === 'matrix' ? styles.active : ''}`}
          onClick={() => setViewMode('matrix')}
        >
          <i className={`ph ph-grid-four ${styles.viewIcon}`}></i>
          Correlation Matrix
        </button>
      </div>

      {/* Correlation Matrix View */}
      {viewMode === 'matrix' && (
        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>Domain Similarity Matrix</h2>
          <p className={styles.sectionDesc}>
            Visual representation of correlation between domains. Darker = higher correlation.
          </p>

          <div className={styles.matrixContainer}>
            <div className={styles.matrixWrapper}>
              {/* Column headers */}
              <div className={styles.matrixHeader}>
                <div className={styles.matrixCorner}></div>
                {matrixDomains.map(domain => (
                  <div key={domain} className={styles.matrixColHeader}>
                    {domain.slice(0, 4)}
                  </div>
                ))}
              </div>

              {/* Matrix rows */}
              {correlationMatrix.map((row, rowIdx) => (
                <div key={matrixDomains[rowIdx]} className={styles.matrixRow}>
                  <div className={styles.matrixRowHeader}>
                    {matrixDomains[rowIdx].slice(0, 6)}
                  </div>
                  {row.map((cell, colIdx) => {
                    const isHovered = hoveredCell?.row === cell.row && hoveredCell?.col === cell.col
                    const hue = cell.correlation * 120 // 0-120 (red to green)
                    const saturation = 60 + cell.correlation * 20
                    const lightness = 85 - cell.correlation * 40

                    return (
                      <div
                        key={`${rowIdx}-${colIdx}`}
                        className={`${styles.matrixCell} ${isHovered ? styles.hovered : ''}`}
                        style={{
                          backgroundColor: `hsl(${hue}, ${saturation}%, ${lightness}%)`
                        }}
                        onMouseEnter={() => setHoveredCell({ row: cell.row, col: cell.col })}
                        onMouseLeave={() => setHoveredCell(null)}
                        onClick={() => setSelectedDomain(cell.row)}
                        title={`${cell.row} ↔ ${cell.col}: ρ = ${cell.correlation.toFixed(2)}`}
                      >
                        <span className={styles.cellValue}>
                          {(cell.correlation * 100).toFixed(0)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className={styles.matrixLegend}>
              <span className={styles.legendLabel}>Correlation:</span>
              <div className={styles.legendGradient}>
                <span>0%</span>
                <div className={styles.gradientBar} />
                <span>100%</span>
              </div>
            </div>
          </div>

          {hoveredCell && (
            <div className={styles.matrixTooltip}>
              <strong>{hoveredCell.row}</strong> ↔ <strong>{hoveredCell.col}</strong>
              <br />
              Correlation: {(getCorrelation(hoveredCell.row, hoveredCell.col) * 100).toFixed(0)}%
            </div>
          )}
        </SketchCard>
      )}

      {/* Domain Selection */}
      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}>Select Target Domain</h2>
        <p className={styles.sectionDesc}>
          Choose a domain to see similar domains and transfer-based ratings
        </p>

        <div className={styles.domainSelectRow}>
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
          {selectedDomain && (
            <div className={styles.coldStartIndicator}>
              {similarDomains?.is_cold_start ? (
                <span className={styles.coldStartBadge}>
                  <i className="ph ph-snowflake"></i> Cold Start - Using transfer learning
                </span>
              ) : (
                <span className={styles.evaluatedBadge}>
                  <i className="ph ph-check-circle"></i> Directly evaluated
                </span>
              )}
            </div>
          )}
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
                {similarDomains.similar_domains.map((domain, index) => (
                  <div key={domain.name} className={styles.similarCard}>
                    <div className={styles.similarRank}>#{index + 1}</div>
                    <div className={styles.similarContent}>
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
                          style={{ 
                            width: `${domain.correlation * 100}%`,
                            background: domain.correlation > 0.7 ? '#4caf50' : 
                                       domain.correlation > 0.5 ? '#ff9800' : '#f44336'
                          }}
                        />
                      </div>
                      <div className={styles.transferStats}>
                        <span className={styles.transferStat}>
                          {domain.shared_models || 0} shared models
                        </span>
                        <span className={styles.transferStat}>
                          {domain.transfer_confidence ? `${(domain.transfer_confidence * 100).toFixed(0)}% reliable` : ''}
                        </span>
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
                  </div>
                ))}
              </div>
            )}

            {similarDomains?.facets && Object.keys(similarDomains.facets).length > 0 && (
              <div className={styles.domainFacets}>
                <h3 className={styles.facetsTitle}>Domain Facets</h3>
                <p className={styles.facetsDesc}>
                  Characteristics used for similarity calculation
                </p>
                <div className={styles.facetsGrid}>
                  {Object.entries(similarDomains.facets).map(([facet, value]) => (
                    <div key={facet} className={styles.facetCard}>
                      <span className={styles.facetName}>{facet}</span>
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
                  <div className={styles.metaItem}>
                    <i className={`ph ph-chart-bar ${styles.metaIcon}`}></i>
                    <span className={styles.metaCount}>{transferLeaderboard.direct_count}</span>
                    <span className={styles.metaLabel}>Direct</span>
                  </div>
                  <div className={styles.metaItem}>
                    <i className={`ph ph-arrows-clockwise ${styles.metaIcon}`}></i>
                    <span className={styles.metaCount}>{transferLeaderboard.transferred_count}</span>
                    <span className={styles.metaLabel}>Transferred</span>
                  </div>
                </div>

                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th className={styles.thRank}>Rank</th>
                      <th className={styles.thModel}>Model</th>
                      <th className={styles.thRating}>Rating (μ)</th>
                      <th className={styles.thSource}>Source</th>
                      <th className={styles.thConf}>Confidence</th>
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
                            <span className={`${styles.medal} ${styles[`rank${entry.rank}`]}`}>
                              <i className="ph-fill ph-medal"></i>
                            </span>
                          ) : (
                            <span className={styles.rankNumber}>{entry.rank}</span>
                          )}
                        </td>
                        <td className={styles.modelName}>{entry.model_id}</td>
                        <td className={styles.rating}>
                          <div className={styles.ratingContainer}>
                            <span className={styles.muValue}>{Math.round(entry.rating.mu)}</span>
                            <span className={styles.sigmaValue}>±{Math.round(entry.rating.sigma)}</span>
                          </div>
                        </td>
                        <td className={styles.source}>
                          <span className={`${styles.sourceBadge} ${styles[entry.source]}`}>
                            <i className={entry.source === 'direct' ? 'ph ph-chart-bar' : 'ph ph-arrows-clockwise'}></i> {entry.source}
                          </span>
                        </td>
                        <td className={styles.confidence}>
                          <div className={styles.confidenceBar}>
                            <div 
                              className={styles.confidenceFill}
                              style={{ 
                                width: `${entry.confidence * 100}%`,
                                backgroundColor: entry.confidence > 0.8 ? '#4caf50' : 
                                                entry.confidence > 0.5 ? '#ff9800' : '#f44336'
                              }}
                            />
                          </div>
                          <span className={styles.confidenceText}>
                            {(entry.confidence * 100).toFixed(0)}%
                          </span>
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

      {/* Help Section */}
      <div className={styles.helpSection}>
        <h3 className={styles.helpTitle}>How Transfer Learning Works</h3>
        <div className={styles.helpGrid}>
          <div className={styles.helpCard}>
            <i className={`ph ph-chart-bar ${styles.helpIcon}`}></i>
            <h4>Domain Facets</h4>
            <p>Each domain is characterized by facets like complexity, formality, and required skills. Similar facets indicate transferable performance.</p>
          </div>
          <div className={styles.helpCard}>
            <i className={`ph ph-link ${styles.helpIcon}`}></i>
            <h4>Correlation Calculation</h4>
            <p>We calculate Pearson correlation between model rankings across domains. High correlation means similar model preferences.</p>
          </div>
          <div className={styles.helpCard}>
            <i className={`ph ph-snowflake ${styles.helpIcon}`}></i>
            <h4>Cold Start Handling</h4>
            <p>For new domains with no direct evaluations, we use weighted transfer from similar domains to bootstrap initial rankings.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export { Transfer as default }
