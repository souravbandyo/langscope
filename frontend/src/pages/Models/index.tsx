/**
 * Models Page
 * 
 * Lists all LLM models with their ratings and allows navigation to detail pages.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useModels } from '@/api/hooks'
import styles from './Models.module.css'

export function Models() {
  const navigate = useNavigate()
  const { data, isLoading, error, refetch } = useModels()
  const [providerFilter, setProviderFilter] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')

  if (isLoading) {
    return <LoadingState message="Loading models..." />
  }

  if (error) {
    const errorMessage = (error as Error)?.message || ''
    const isAuthError = errorMessage.includes('401') || errorMessage.includes('AUTH')
    
    return (
      <ErrorState 
        title={isAuthError ? "Authentication required" : "Failed to load models"}
        message={isAuthError ? "Please sign in to view models" : undefined}
        error={error as Error}
        onRetry={() => refetch()}
      />
    )
  }

  const models = data?.models || []
  
  // Get unique providers
  const providers = [...new Set(models.map(m => m.provider))].sort()
  
  // Filter models
  const filteredModels = models.filter(model => {
    const matchesProvider = !providerFilter || model.provider === providerFilter
    const matchesSearch = !searchQuery || 
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.model_id.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesProvider && matchesSearch
  })

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Models</h1>
        <p className={styles.subtitle}>
          Browse and compare {models.length} LLM models across providers
        </p>
      </header>

      {/* Filters */}
      <SketchCard padding="md" className={styles.filters}>
        <div className={styles.filterRow}>
          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Search</label>
            <input
              type="text"
              className={styles.searchInput}
              placeholder="Search by name or ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          
          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Provider</label>
            <select
              className={styles.select}
              value={providerFilter}
              onChange={(e) => setProviderFilter(e.target.value)}
            >
              <option value="">All Providers</option>
              {providers.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        </div>
      </SketchCard>

      {/* Models Grid */}
      {filteredModels.length === 0 ? (
        <EmptyState
          title="No models found"
          message="Try adjusting your filters"
          icon="🔍"
        />
      ) : (
        <div className={styles.modelsGrid}>
          {filteredModels.map(model => (
            <SketchCard
              key={model.model_id}
              padding="md"
              className={styles.modelCard}
              onClick={() => navigate(`/models/${model.model_id}`)}
            >
              <div className={styles.modelHeader}>
                <h3 className={styles.modelName}>{model.name}</h3>
                <span className={styles.providerBadge}>{model.provider}</span>
              </div>
              
              <div className={styles.modelStats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Rating</span>
                  <span className={styles.statValue}>
                    {model.trueskill?.raw?.mu?.toFixed(1) || 'N/A'}
                  </span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Matches</span>
                  <span className={styles.statValue}>{model.total_matches_played || 0}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Domains</span>
                  <span className={styles.statValue}>{model.domains_evaluated?.length || 0}</span>
                </div>
              </div>

              <div className={styles.modelPricing}>
                <span className={styles.pricingLabel}>Pricing:</span>
                <span className={styles.pricingValue}>
                  ${model.input_cost_per_million?.toFixed(2) || '?'} / ${model.output_cost_per_million?.toFixed(2) || '?'} per 1M tokens
                </span>
              </div>

              {model.domains_evaluated && model.domains_evaluated.length > 0 && (
                <div className={styles.domainTags}>
                  {model.domains_evaluated.slice(0, 3).map(domain => (
                    <span key={domain} className={styles.domainTag}>{domain}</span>
                  ))}
                  {model.domains_evaluated.length > 3 && (
                    <span className={styles.domainTag}>+{model.domains_evaluated.length - 3} more</span>
                  )}
                </div>
              )}
            </SketchCard>
          ))}
        </div>
      )}
    </div>
  )
}

export default Models
