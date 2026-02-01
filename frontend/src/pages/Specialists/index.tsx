import { useState, useMemo } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useSpecialistSummary, useDomainSpecialists, useGeneralists, useDomains } from '@/api/hooks'
import styles from './Specialists.module.css'

// Default domains when API is unavailable
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']

// Z-score thresholds
const SPECIALIST_THRESHOLD = 2.0
const WEAK_SPOT_THRESHOLD = -2.0

// Z-score visualization component
interface ZScoreBarProps {
  zScore: number
  maxRange?: number
}

function ZScoreBar({ zScore, maxRange = 4 }: ZScoreBarProps) {
  const clampedScore = Math.max(-maxRange, Math.min(maxRange, zScore))
  const percentage = ((clampedScore + maxRange) / (2 * maxRange)) * 100
  const isPositive = zScore > 0
  const isSpecialist = zScore >= SPECIALIST_THRESHOLD
  const isWeakSpot = zScore <= WEAK_SPOT_THRESHOLD

  return (
    <div className={styles.zScoreBarContainer}>
      <div className={styles.zScoreBar}>
        <div className={styles.zScoreMarker} style={{ left: '50%' }} />
        <div className={styles.zScoreZone + ' ' + styles.weakZone} />
        <div className={styles.zScoreZone + ' ' + styles.strongZone} />
        <div 
          className={`${styles.zScoreIndicator} ${isPositive ? styles.positive : styles.negative} ${isSpecialist ? styles.specialist : ''} ${isWeakSpot ? styles.weakspot : ''}`}
          style={{ left: `${percentage}%` }}
        />
      </div>
      <div className={styles.zScoreLabels}>
        <span>-{maxRange}</span>
        <span>0</span>
        <span>+{maxRange}</span>
      </div>
    </div>
  )
}

/**
 * Specialists page showing models that excel in specific domains
 */
export function Specialists() {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [viewMode, setViewMode] = useState<'specialists' | 'generalists' | 'weakspots'>('specialists')
  const [sortBy, setSortBy] = useState<'z_score' | 'actual_mu'>('z_score')

  const { data: domainsData, error: domainsError } = useDomains()
  const { data: summary, isLoading: summaryLoading } = useSpecialistSummary()
  const { data: specialists, isLoading: specialistsLoading } = useDomainSpecialists(
    selectedDomain,
    { include_weak_spots: true }
  )
  const { data: generalists, isLoading: generalistsLoading } = useGeneralists({ min_domains: 3 })

  const domains = domainsError ? defaultDomains : (domainsData?.domains || [])
  const isLoading = summaryLoading || specialistsLoading || generalistsLoading

  // Sort and filter specialists data
  const sortedSpecialists = useMemo(() => {
    if (!specialists) return []
    
    let filtered = [...specialists]
    
    // Filter based on view mode
    if (viewMode === 'specialists') {
      filtered = filtered.filter(s => s.category === 'specialist' || s.z_score > 0)
    } else if (viewMode === 'weakspots') {
      filtered = filtered.filter(s => s.category === 'weak_spot' || s.z_score < 0)
    }

    // Sort
    return filtered.sort((a, b) => {
      if (sortBy === 'z_score') {
        return viewMode === 'weakspots' ? a.z_score - b.z_score : b.z_score - a.z_score
      }
      return b.actual_mu - a.actual_mu
    })
  }, [specialists, viewMode, sortBy])

  // Calculate statistics
  const stats = useMemo(() => {
    if (!specialists) return null
    const specialistCount = specialists.filter(s => s.category === 'specialist').length
    const weakSpotCount = specialists.filter(s => s.category === 'weak_spot').length
    const avgZScore = specialists.length > 0 
      ? specialists.reduce((sum, s) => sum + s.z_score, 0) / specialists.length 
      : 0
    
    return { specialistCount, weakSpotCount, avgZScore }
  }, [specialists])

  return (
    <div className={styles.specialists}>
      <div className={styles.header}>
        <h1 className={styles.title}>Model Specialists</h1>
        <p className={styles.subtitle}>
          Discover models that excel in specific domains vs generalist all-rounders
        </p>
      </div>

      {/* Summary Stats */}
      {summary && (
        <div className={styles.summaryRow}>
          <StickyNote title="Specialists" color="yellow" rotation={-2}>
            <div className={styles.statValue}>{summary.total_specialists}</div>
            <div className={styles.statLabel}>Domain experts (z &gt; 2.0)</div>
          </StickyNote>
          <StickyNote title="Weak Spots" color="pink" rotation={0}>
            <div className={styles.statValue}>{summary.total_weak_spots || 0}</div>
            <div className={styles.statLabel}>Underperformers (z &lt; -2.0)</div>
          </StickyNote>
          <StickyNote title="Generalists" color="blue" rotation={1}>
            <div className={styles.statValue}>{summary.total_generalists}</div>
            <div className={styles.statLabel}>All-rounders</div>
          </StickyNote>
        </div>
      )}

      {/* View Toggle */}
      <div className={styles.viewToggle}>
        <button
          className={`${styles.toggleBtn} ${viewMode === 'specialists' ? styles.active : ''}`}
          onClick={() => setViewMode('specialists')}
        >
          <i className="ph ph-star"></i> Specialists
        </button>
        <button
          className={`${styles.toggleBtn} ${viewMode === 'weakspots' ? styles.active : ''}`}
          onClick={() => setViewMode('weakspots')}
        >
          <i className="ph ph-warning"></i> Weak Spots
        </button>
        <button
          className={`${styles.toggleBtn} ${viewMode === 'generalists' ? styles.active : ''}`}
          onClick={() => setViewMode('generalists')}
        >
          <i className="ph ph-target"></i> Generalists
        </button>
      </div>

      {/* Z-Score Legend */}
      <div className={styles.legendCard}>
        <h4 className={styles.legendTitle}>Z-Score Interpretation</h4>
        <div className={styles.legendContent}>
          <div className={styles.legendItem}>
            <i className={`ph ph-star ${styles.legendIcon}`}></i>
            <span>Specialist (z ≥ 2.0): Significantly outperforms expectations</span>
          </div>
          <div className={styles.legendItem}>
            <i className={`ph ph-minus ${styles.legendIcon}`}></i>
            <span>Normal (-2.0 &lt; z &lt; 2.0): Performs as expected</span>
          </div>
          <div className={styles.legendItem}>
            <i className={`ph ph-warning ${styles.legendIcon}`}></i>
            <span>Weak Spot (z ≤ -2.0): Significantly underperforms expectations</span>
          </div>
        </div>
      </div>

      {/* Specialists/Weak Spots View */}
      {(viewMode === 'specialists' || viewMode === 'weakspots') && (
        <>
          <div className={styles.filters}>
            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Select Domain:</label>
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

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Sort by:</label>
              <select
                className={styles.select}
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'z_score' | 'actual_mu')}
              >
                <option value="z_score">Z-Score</option>
                <option value="actual_mu">Rating (μ)</option>
              </select>
            </div>
          </div>

          <SketchCard padding="lg">
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <i className={viewMode === 'specialists' ? 'ph ph-star' : 'ph ph-warning'}></i>{' '}
                {selectedDomain 
                  ? `${viewMode === 'specialists' ? 'Specialists' : 'Weak Spots'} in ${selectedDomain.replace(/_/g, ' ')}`
                  : `Select a domain to view ${viewMode === 'specialists' ? 'specialists' : 'weak spots'}`}
              </h2>
              {stats && selectedDomain && (
                <div className={styles.sectionStats}>
                  <span className={styles.statBadge + ' ' + styles.specialist}>
                    {stats.specialistCount} specialists
                  </span>
                  <span className={styles.statBadge + ' ' + styles.weakspot}>
                    {stats.weakSpotCount} weak spots
                  </span>
                </div>
              )}
            </div>

            {!selectedDomain ? (
              <div className={styles.empty}>Choose a domain from the dropdown above</div>
            ) : isLoading ? (
              <div className={styles.loading}>Loading data...</div>
            ) : !sortedSpecialists.length ? (
              <div className={styles.empty}>
                No {viewMode === 'specialists' ? 'specialists' : 'weak spots'} found for this domain
              </div>
            ) : (
              <div className={styles.resultsList}>
                {sortedSpecialists.map((result, index) => (
                  <div 
                    key={`${result.model_id}-${result.domain}`}
                    className={`${styles.resultCard} ${styles[result.category]}`}
                  >
                    <div className={styles.rankBadge}>#{index + 1}</div>
                    <div className={styles.categoryBadge}>
                      {result.category === 'specialist' && <i className="ph ph-star"></i>}
                      {result.category === 'weak_spot' && <i className="ph ph-warning"></i>}
                      {result.category === 'normal' && <i className="ph ph-minus"></i>}
                    </div>
                    <div className={styles.resultInfo}>
                      <h3 className={styles.modelName}>{result.model_id}</h3>
                      <span className={styles.category}>
                        {result.category.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className={styles.zScoreSection}>
                      <div className={styles.zScoreValue}>
                        <span className={styles.scoreLabel}>Z-Score</span>
                        <span className={`${styles.bigScore} ${result.z_score > 0 ? styles.positive : styles.negative}`}>
                          {result.z_score > 0 ? '+' : ''}{result.z_score.toFixed(2)}
                        </span>
                      </div>
                      <ZScoreBar zScore={result.z_score} />
                    </div>
                    <div className={styles.scores}>
                      <div className={styles.scoreItem}>
                        <span className={styles.scoreLabel}>Actual μ</span>
                        <span className={styles.scoreValue}>{Math.round(result.actual_mu)}</span>
                      </div>
                      <div className={styles.scoreItem}>
                        <span className={styles.scoreLabel}>Predicted μ</span>
                        <span className={styles.scoreValue}>{Math.round(result.predicted_mu)}</span>
                      </div>
                      <div className={styles.scoreItem}>
                        <span className={styles.scoreLabel}>Δ (Difference)</span>
                        <span className={`${styles.scoreValue} ${result.actual_mu > result.predicted_mu ? styles.positive : styles.negative}`}>
                          {result.actual_mu > result.predicted_mu ? '+' : ''}
                          {Math.round(result.actual_mu - result.predicted_mu)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </SketchCard>
        </>
      )}

      {/* Generalists View */}
      {viewMode === 'generalists' && (
        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>Generalist Models</h2>
          <p className={styles.sectionSubtitle}>
            Models that perform consistently well across multiple domains
          </p>

          {generalistsLoading ? (
            <div className={styles.loading}>Loading generalists...</div>
          ) : !generalists?.length ? (
            <div className={styles.empty}>No generalist models found</div>
          ) : (
            <div className={styles.resultsList}>
              {generalists.map((profile) => (
                <div key={profile.model_id} className={styles.generalistCard}>
                  <div className={styles.generalistHeader}>
                    <h3 className={styles.modelName}>{profile.model_name}</h3>
                    <span className={styles.domainsCount}>
                      {profile.domains_evaluated.length} domains
                    </span>
                  </div>
                  <div className={styles.generalistStats}>
                    <div className={styles.statBlock}>
                      <span className={styles.statNumber}>{profile.specialist_domains.length}</span>
                      <span className={styles.statText}>Specialist domains</span>
                    </div>
                    <div className={styles.statBlock}>
                      <span className={styles.statNumber}>{profile.weak_spot_domains.length}</span>
                      <span className={styles.statText}>Weak spots</span>
                    </div>
                    <div className={styles.statBlock}>
                      <span className={styles.statNumber}>{profile.specialization_score.toFixed(2)}</span>
                      <span className={styles.statText}>Specialization score</span>
                    </div>
                  </div>
                  {profile.specialist_domains.length > 0 && (
                    <div className={styles.domainTags}>
                      <span className={styles.tagLabel}>Strong in:</span>
                      {profile.specialist_domains.slice(0, 3).map((d) => (
                        <span key={d} className={styles.tag}>{d.replace(/_/g, ' ')}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </SketchCard>
      )}
    </div>
  )
}

export { Specialists as default }
