import { useState } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useSpecialistSummary, useDomainSpecialists, useGeneralists, useDomains } from '@/api/hooks'
import styles from './Specialists.module.css'

// Default domains when API is unavailable
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']

/**
 * Specialists page showing models that excel in specific domains
 */
export function Specialists() {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [viewMode, setViewMode] = useState<'specialists' | 'generalists'>('specialists')

  const { data: domainsData, error: domainsError } = useDomains()
  const { data: summary, isLoading: summaryLoading } = useSpecialistSummary()
  const { data: specialists, isLoading: specialistsLoading } = useDomainSpecialists(
    selectedDomain,
    { include_weak_spots: true }
  )
  const { data: generalists, isLoading: generalistsLoading } = useGeneralists({ min_domains: 3 })

  const domains = domainsError ? defaultDomains : (domainsData?.domains || [])
  const isLoading = summaryLoading || (viewMode === 'specialists' ? specialistsLoading : generalistsLoading)

  return (
    <div className={styles.specialists}>
      <h1 className={styles.title}>🎖️ Model Specialists</h1>
      <p className={styles.subtitle}>
        Discover models that excel in specific domains vs generalist all-rounders
      </p>

      {/* Summary Stats */}
      {summary && (
        <div className={styles.summaryRow}>
          <StickyNote title="Specialists" color="yellow" rotation={-2}>
            <div className={styles.statValue}>{summary.total_specialists}</div>
            <div className={styles.statLabel}>Domain experts</div>
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
          Specialists by Domain
        </button>
        <button
          className={`${styles.toggleBtn} ${viewMode === 'generalists' ? styles.active : ''}`}
          onClick={() => setViewMode('generalists')}
        >
          Generalists
        </button>
      </div>

      {/* Specialists View */}
      {viewMode === 'specialists' && (
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
          </div>

          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>
              {selectedDomain 
                ? `Specialists in ${selectedDomain.replace(/_/g, ' ')}`
                : 'Select a domain to view specialists'}
            </h2>

            {!selectedDomain ? (
              <div className={styles.empty}>Choose a domain from the dropdown above</div>
            ) : isLoading ? (
              <div className={styles.loading}>Loading specialists...</div>
            ) : !specialists?.length ? (
              <div className={styles.empty}>No specialists found for this domain</div>
            ) : (
              <div className={styles.resultsList}>
                {specialists.map((result) => (
                  <div 
                    key={`${result.model_id}-${result.domain}`}
                    className={`${styles.resultCard} ${styles[result.category]}`}
                  >
                    <div className={styles.categoryBadge}>
                      {result.category === 'specialist' && '⭐'}
                      {result.category === 'weak_spot' && '⚠️'}
                      {result.category === 'normal' && '➖'}
                    </div>
                    <div className={styles.resultInfo}>
                      <h3 className={styles.modelName}>{result.model_id}</h3>
                      <span className={styles.category}>
                        {result.category.replace(/_/g, ' ')}
                      </span>
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
                        <span className={styles.scoreLabel}>Z-Score</span>
                        <span className={`${styles.scoreValue} ${result.z_score > 0 ? styles.positive : styles.negative}`}>
                          {result.z_score.toFixed(2)}
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
