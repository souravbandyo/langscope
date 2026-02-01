import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useRecommendations, useDomains } from '@/api/hooks'
import styles from './Recommendations.module.css'

const useCasePresets = [
  { id: 'general', name: 'General Purpose' },
  { id: 'accuracy', name: 'Accuracy-focused' },
  { id: 'speed', name: 'Speed-focused' },
  { id: 'cost', name: 'Cost-sensitive' },
  { id: 'creative', name: 'Creative Writing' },
  { id: 'coding', name: 'Code Generation' },
  { id: 'analysis', name: 'Data Analysis' },
]

// Default domains when API is unavailable
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']

/**
 * Recommendations page for use-case based model suggestions
 */
export function Recommendations() {
  const [searchParams, setSearchParams] = useSearchParams()
  const queryFromUrl = searchParams.get('q') || ''
  
  const [useCase, setUseCase] = useState(queryFromUrl || 'general')
  const [selectedDomain, setSelectedDomain] = useState('')
  const [customQuery, setCustomQuery] = useState(queryFromUrl)

  // Update use case when URL query changes
  useEffect(() => {
    if (queryFromUrl) {
      setUseCase(queryFromUrl)
      setCustomQuery(queryFromUrl)
    }
  }, [queryFromUrl])

  const { data: domainsData, error: domainsError } = useDomains()
  const { data, isLoading, error, refetch } = useRecommendations({
    useCase: useCase,
    domain: selectedDomain || undefined,
    topK: 10,
  })

  const domains = domainsError ? defaultDomains : (domainsData?.domains || [])

  // Handle custom search
  const handleSearch = () => {
    if (customQuery.trim()) {
      setUseCase(customQuery.trim())
      setSearchParams({ q: customQuery.trim() })
    }
  }

  // Handle preset selection
  const handlePresetChange = (presetId: string) => {
    setUseCase(presetId)
    setCustomQuery('')
    setSearchParams({})
  }

  return (
    <div className={styles.recommendations}>
      <h1 className={styles.title}>Model Recommendations</h1>
      <p className={styles.subtitle}>
        Get personalized model recommendations based on your use case
      </p>

      {/* Custom Search */}
      <div className={styles.searchSection}>
        <div className={styles.searchBox}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Describe your use case..."
            value={customQuery}
            onChange={(e) => setCustomQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button className={styles.searchButton} onClick={handleSearch}>
            Search
          </button>
        </div>
        {useCase && useCase !== 'general' && !useCasePresets.find(p => p.id === useCase) && (
          <div className={styles.currentQuery}>
            Searching for: <strong>{useCase}</strong>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Quick Presets:</label>
          <select
            className={styles.select}
            value={useCasePresets.find(p => p.id === useCase)?.id || ''}
            onChange={(e) => handlePresetChange(e.target.value)}
          >
            <option value="">Custom Search</option>
            {useCasePresets.map((uc) => (
              <option key={uc.id} value={uc.id}>
                {uc.name}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Domain (optional):</label>
          <select
            className={styles.select}
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
          >
            <option value="">All Domains</option>
            {domains.map((d) => (
              <option key={d} value={d}>
                {d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Info Card */}
      {data && (
        <StickyNote title="Recommendation Info" color="blue" rotation={-1}>
          <div className={styles.infoContent}>
            <p>Use case: {data.use_case}</p>
            {data.beta !== undefined && <p>Beta value: {data.beta.toFixed(2)}</p>}
            {data.n_users !== undefined && <p>Based on {data.n_users} user preferences</p>}
          </div>
        </StickyNote>
      )}

      {/* Recommendations List */}
      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}>Recommended Models</h2>

        {isLoading ? (
          <LoadingState message="Loading recommendations..." />
        ) : error ? (
          <ErrorState
            title="Failed to load recommendations"
            error={error as Error}
            onRetry={() => refetch()}
          />
        ) : !data?.recommendations?.length ? (
          <EmptyState
            title="No recommendations available"
            message="Try selecting a different use case or domain."
            icon="ph ph-target"
          />
        ) : (
          <div className={styles.recommendationsList}>
            {data.recommendations.map((rec, index) => (
              <div key={rec.model_id} className={styles.recommendationCard}>
                <div className={`${styles.rankBadge} ${index < 3 ? styles[`rank${index + 1}`] : ''}`}>
                  {index < 3 ? <i className="ph-fill ph-medal"></i> : `#${index + 1}`}
                </div>
                <div className={styles.modelInfo}>
                  <h3 className={styles.modelName}>{rec.model_name}</h3>
                  <span className={styles.provider}>{rec.provider}</span>
                </div>
                <div className={styles.scores}>
                  <div className={styles.scoreItem}>
                    <span className={styles.scoreLabel}>Adjusted μ</span>
                    <span className={styles.scoreValue}>{rec.adjusted_mu !== undefined ? Math.round(rec.adjusted_mu) : '-'}</span>
                  </div>
                  <div className={styles.scoreItem}>
                    <span className={styles.scoreLabel}>Global μ</span>
                    <span className={styles.scoreValue}>{rec.global_mu !== undefined ? Math.round(rec.global_mu) : '-'}</span>
                  </div>
                  {rec.adjustment !== undefined && (
                    <div className={styles.scoreItem}>
                      <span className={styles.scoreLabel}>Adjustment</span>
                      <span className={`${styles.scoreValue} ${rec.adjustment >= 0 ? styles.positive : styles.negative}`}>
                        {rec.adjustment >= 0 ? '+' : ''}{rec.adjustment.toFixed(1)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </SketchCard>
    </div>
  )
}

export { Recommendations as default }
