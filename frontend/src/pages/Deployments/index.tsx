/**
 * Deployments Page
 * 
 * Lists all model deployments across providers with pricing and performance data.
 * Includes Provider Comparison view for comparing same base model across providers.
 */

import { useState, useMemo } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useDeployments } from '@/api/hooks'
import type { DeploymentResponse } from '@/api/types'
import styles from './Deployments.module.css'

type ViewMode = 'grid' | 'compare'

interface DeploymentDetailModalProps {
  deployment: DeploymentResponse
  onClose: () => void
}

function DeploymentDetailModal({ deployment, onClose }: DeploymentDetailModalProps) {
  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>{deployment.deployment.display_name}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalContent}>
          {/* Provider Info */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Provider</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Name</span>
                <span className={styles.detailValue}>{deployment.provider.name}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Type</span>
                <span className={styles.detailValue}>{deployment.provider.type}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>API Compatible</span>
                <span className={styles.detailValue}>{deployment.provider.api_compatible}</span>
              </div>
              {deployment.provider.website && (
                <div className={styles.detailItem}>
                  <span className={styles.detailLabel}>Website</span>
                  <a href={deployment.provider.website} target="_blank" rel="noopener noreferrer" className={styles.link}>
                    {deployment.provider.website}
                  </a>
                </div>
              )}
            </div>
          </section>

          {/* Deployment Config */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Configuration</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Model ID</span>
                <span className={styles.detailValue}>{deployment.deployment.model_id}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Base Model</span>
                <span className={styles.detailValue}>{deployment.base_model_id}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Quantization</span>
                <span className={styles.detailValue}>{deployment.deployment.quantization || 'Native'}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Framework</span>
                <span className={styles.detailValue}>{deployment.deployment.serving_framework || 'N/A'}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Max Context</span>
                <span className={styles.detailValue}>{deployment.deployment.max_context_length.toLocaleString()} tokens</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Max Output</span>
                <span className={styles.detailValue}>{deployment.deployment.max_output_tokens.toLocaleString()} tokens</span>
              </div>
            </div>
          </section>

          {/* Pricing */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Pricing</h3>
            <div className={styles.pricingCards}>
              <div className={styles.pricingCard}>
                <span className={styles.pricingLabel}>Input</span>
                <span className={styles.pricingValue}>${deployment.pricing.input_cost_per_million.toFixed(2)}</span>
                <span className={styles.pricingUnit}>per 1M tokens</span>
              </div>
              <div className={styles.pricingCard}>
                <span className={styles.pricingLabel}>Output</span>
                <span className={styles.pricingValue}>${deployment.pricing.output_cost_per_million.toFixed(2)}</span>
                <span className={styles.pricingUnit}>per 1M tokens</span>
              </div>
            </div>
            <div className={styles.pricingMeta}>
              <span>Source: {deployment.pricing.source_id}</span>
              <span>Last verified: {new Date(deployment.pricing.last_verified).toLocaleDateString()}</span>
            </div>
          </section>

          {/* Performance */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Performance</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Avg Latency</span>
                <span className={styles.detailValue}>{deployment.performance.avg_latency_ms.toFixed(0)} ms</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>P50 Latency</span>
                <span className={styles.detailValue}>{deployment.performance.p50_latency_ms.toFixed(0)} ms</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>P95 Latency</span>
                <span className={styles.detailValue}>{deployment.performance.p95_latency_ms.toFixed(0)} ms</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>P99 Latency</span>
                <span className={styles.detailValue}>{deployment.performance.p99_latency_ms.toFixed(0)} ms</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Avg TTFT</span>
                <span className={styles.detailValue}>{deployment.performance.avg_ttft_ms.toFixed(0)} ms</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Tokens/sec</span>
                <span className={styles.detailValue}>{deployment.performance.tokens_per_second.toFixed(1)}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>30d Uptime</span>
                <span className={styles.detailValue}>{(deployment.performance.uptime_30d * 100).toFixed(2)}%</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>30d Error Rate</span>
                <span className={styles.detailValue}>{(deployment.performance.error_rate_30d * 100).toFixed(3)}%</span>
              </div>
            </div>
          </section>

          {/* Availability */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Availability</h3>
            <div className={styles.availabilityInfo}>
              <span className={`${styles.statusBadge} ${styles[deployment.availability.status]}`}>
                {deployment.availability.status}
              </span>
              {deployment.availability.requires_waitlist && (
                <span className={styles.warningBadge}>Waitlist</span>
              )}
              {deployment.availability.requires_enterprise && (
                <span className={styles.warningBadge}>Enterprise Only</span>
              )}
            </div>
            {deployment.availability.regions.length > 0 && (
              <div className={styles.regions}>
                <span className={styles.detailLabel}>Regions:</span>
                <span>{deployment.availability.regions.join(', ')}</span>
              </div>
            )}
          </section>

          {/* TrueSkill Rating */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>TrueSkill Rating</h3>
            <div className={styles.ratingCards}>
              <div className={styles.ratingCard}>
                <span className={styles.ratingLabel}>Raw Quality</span>
                <span className={styles.ratingValue}>{deployment.trueskill.raw.mu.toFixed(0)}</span>
                <span className={styles.ratingUncertainty}>±{deployment.trueskill.raw.sigma.toFixed(0)}</span>
              </div>
              <div className={styles.ratingCard}>
                <span className={styles.ratingLabel}>Cost-Adjusted</span>
                <span className={styles.ratingValue}>{deployment.trueskill.cost_adjusted.mu.toFixed(0)}</span>
                <span className={styles.ratingUncertainty}>±{deployment.trueskill.cost_adjusted.sigma.toFixed(0)}</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

// Provider Comparison Component
function ProviderComparisonView({ 
  deployments, 
  selectedBaseModel,
  onSelectDeployment 
}: { 
  deployments: DeploymentResponse[]
  selectedBaseModel: string
  onSelectDeployment: (d: DeploymentResponse) => void
}) {
  // Get deployments for selected base model
  const modelDeployments = useMemo(() => {
    if (!selectedBaseModel) return []
    return deployments
      .filter(d => d.base_model_id === selectedBaseModel)
      .sort((a, b) => b.trueskill.raw.mu - a.trueskill.raw.mu)
  }, [deployments, selectedBaseModel])

  // Find best values for highlighting
  const bestValues = useMemo(() => {
    if (modelDeployments.length === 0) return null
    return {
      lowestInputPrice: Math.min(...modelDeployments.map(d => d.pricing.input_cost_per_million)),
      lowestOutputPrice: Math.min(...modelDeployments.map(d => d.pricing.output_cost_per_million)),
      lowestLatency: Math.min(...modelDeployments.map(d => d.performance.avg_latency_ms)),
      lowestTTFT: Math.min(...modelDeployments.map(d => d.performance.avg_ttft_ms)),
      highestRating: Math.max(...modelDeployments.map(d => d.trueskill.raw.mu)),
      highestCostAdjusted: Math.max(...modelDeployments.map(d => d.trueskill.cost_adjusted.mu)),
      highestThroughput: Math.max(...modelDeployments.map(d => d.performance.tokens_per_second)),
    }
  }, [modelDeployments])

  if (!selectedBaseModel) {
    return (
      <EmptyState
        title="Select a base model"
        message="Choose a base model from the dropdown to compare providers"
        icon="ph ph-chart-bar"
      />
    )
  }

  if (modelDeployments.length === 0) {
    return (
      <EmptyState
        title="No providers found"
        message="No deployments available for this base model"
        icon="ph ph-magnifying-glass"
      />
    )
  }

  return (
    <div className={styles.comparisonContainer}>
      <div className={styles.comparisonHeader}>
        <h2 className={styles.comparisonTitle}>
          Provider Comparison: {selectedBaseModel}
        </h2>
        <span className={styles.comparisonCount}>
          {modelDeployments.length} provider{modelDeployments.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Info Notes */}
      <div className={styles.infoNotes}>
        <StickyNote title="Best Values" color="green" rotation={-1}>
          <p><i className="ph ph-trophy"></i> = Best in category</p>
          <p>Highlighted cells show leading values</p>
        </StickyNote>
        <StickyNote title="Tip" color="yellow" rotation={1}>
          <p>Same base model, different providers</p>
          <p>Quality may vary due to quantization!</p>
        </StickyNote>
      </div>

      {/* Comparison Table */}
      <div className={styles.comparisonTableWrapper}>
        <table className={styles.comparisonTable}>
          <thead>
            <tr>
              <th className={styles.thProvider}>Provider</th>
              <th className={styles.thPrice}>Input $/1M</th>
              <th className={styles.thPrice}>Output $/1M</th>
              <th className={styles.thPerf}>Latency</th>
              <th className={styles.thPerf}>TTFT</th>
              <th className={styles.thPerf}>Tok/s</th>
              <th className={styles.thRating}>Raw μ</th>
              <th className={styles.thRating}>Cost-Adj μ</th>
              <th className={styles.thStatus}>Status</th>
            </tr>
          </thead>
          <tbody>
            {modelDeployments.map(deployment => (
              <tr 
                key={deployment.id} 
                className={styles.comparisonRow}
                onClick={() => onSelectDeployment(deployment)}
              >
                <td className={styles.providerCell}>
                  <span className={styles.providerName}>{deployment.provider.name}</span>
                  {deployment.deployment.quantization && deployment.deployment.quantization !== 'native' && (
                    <span className={styles.quantBadge}>{deployment.deployment.quantization}</span>
                  )}
                </td>
                <td className={`${styles.priceCell} ${deployment.pricing.input_cost_per_million === bestValues?.lowestInputPrice ? styles.bestValue : ''}`}>
                  ${deployment.pricing.input_cost_per_million.toFixed(2)}
                  {deployment.pricing.input_cost_per_million === bestValues?.lowestInputPrice && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.priceCell} ${deployment.pricing.output_cost_per_million === bestValues?.lowestOutputPrice ? styles.bestValue : ''}`}>
                  ${deployment.pricing.output_cost_per_million.toFixed(2)}
                  {deployment.pricing.output_cost_per_million === bestValues?.lowestOutputPrice && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.perfCell} ${deployment.performance.avg_latency_ms === bestValues?.lowestLatency ? styles.bestValue : ''}`}>
                  {deployment.performance.avg_latency_ms.toFixed(0)}ms
                  {deployment.performance.avg_latency_ms === bestValues?.lowestLatency && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.perfCell} ${deployment.performance.avg_ttft_ms === bestValues?.lowestTTFT ? styles.bestValue : ''}`}>
                  {deployment.performance.avg_ttft_ms.toFixed(0)}ms
                  {deployment.performance.avg_ttft_ms === bestValues?.lowestTTFT && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.perfCell} ${deployment.performance.tokens_per_second === bestValues?.highestThroughput ? styles.bestValue : ''}`}>
                  {deployment.performance.tokens_per_second.toFixed(0)}
                  {deployment.performance.tokens_per_second === bestValues?.highestThroughput && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.ratingCell} ${deployment.trueskill.raw.mu === bestValues?.highestRating ? styles.bestValue : ''}`}>
                  {deployment.trueskill.raw.mu.toFixed(0)}
                  {deployment.trueskill.raw.mu === bestValues?.highestRating && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={`${styles.ratingCell} ${deployment.trueskill.cost_adjusted.mu === bestValues?.highestCostAdjusted ? styles.bestValue : ''}`}>
                  {deployment.trueskill.cost_adjusted.mu.toFixed(0)}
                  {deployment.trueskill.cost_adjusted.mu === bestValues?.highestCostAdjusted && <span className={styles.bestBadge}><i className="ph ph-trophy"></i></span>}
                </td>
                <td className={styles.statusCell}>
                  <span className={`${styles.statusBadge} ${styles[deployment.availability.status]}`}>
                    {deployment.availability.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Cards */}
      <div className={styles.summaryCards}>
        <div className={styles.summaryCard}>
          <h4><i className="ph ph-currency-dollar"></i> Cheapest (Input)</h4>
          <p className={styles.summaryProvider}>
            {modelDeployments.find(d => d.pricing.input_cost_per_million === bestValues?.lowestInputPrice)?.provider.name}
          </p>
          <p className={styles.summaryValue}>${bestValues?.lowestInputPrice.toFixed(2)}/1M</p>
        </div>
        <div className={styles.summaryCard}>
          <h4><i className="ph ph-lightning"></i> Fastest (Latency)</h4>
          <p className={styles.summaryProvider}>
            {modelDeployments.find(d => d.performance.avg_latency_ms === bestValues?.lowestLatency)?.provider.name}
          </p>
          <p className={styles.summaryValue}>{bestValues?.lowestLatency.toFixed(0)}ms</p>
        </div>
        <div className={styles.summaryCard}>
          <h4><i className="ph ph-star"></i> Best Quality</h4>
          <p className={styles.summaryProvider}>
            {modelDeployments.find(d => d.trueskill.raw.mu === bestValues?.highestRating)?.provider.name}
          </p>
          <p className={styles.summaryValue}>μ = {bestValues?.highestRating.toFixed(0)}</p>
        </div>
        <div className={styles.summaryCard}>
          <h4><i className="ph ph-chart-bar"></i> Best Value</h4>
          <p className={styles.summaryProvider}>
            {modelDeployments.find(d => d.trueskill.cost_adjusted.mu === bestValues?.highestCostAdjusted)?.provider.name}
          </p>
          <p className={styles.summaryValue}>Cost-adj μ = {bestValues?.highestCostAdjusted.toFixed(0)}</p>
        </div>
      </div>
    </div>
  )
}

export function Deployments() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [providerFilter, setProviderFilter] = useState('')
  const [maxPriceFilter, setMaxPriceFilter] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedDeployment, setSelectedDeployment] = useState<DeploymentResponse | null>(null)
  const [selectedBaseModel, setSelectedBaseModel] = useState('')

  const { data, isLoading, error, refetch } = useDeployments({
    provider: providerFilter || undefined,
    max_price: maxPriceFilter ? parseFloat(maxPriceFilter) : undefined,
  })

  const deployments = data?.deployments || []

  // Get unique providers and base models
  const providers = useMemo(() => 
    [...new Set(deployments.map(d => d.provider.name))].sort(), 
    [deployments]
  )
  
  const baseModels = useMemo(() => {
    const models = [...new Set(deployments.map(d => d.base_model_id))].sort()
    // Add deployment count for each base model
    return models.map(model => ({
      id: model,
      count: deployments.filter(d => d.base_model_id === model).length
    }))
  }, [deployments])

  // Filter deployments
  const filteredDeployments = useMemo(() => {
    return deployments.filter(deployment => {
      const matchesSearch = !searchQuery ||
        deployment.deployment.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        deployment.base_model_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        deployment.provider.name.toLowerCase().includes(searchQuery.toLowerCase())
      return matchesSearch
    })
  }, [deployments, searchQuery])

  // Sort by rating
  const sortedDeployments = useMemo(() => 
    [...filteredDeployments].sort((a, b) => b.trueskill.raw.mu - a.trueskill.raw.mu),
    [filteredDeployments]
  )

  if (isLoading) {
    return <LoadingState message="Loading deployments..." />
  }

  if (error) {
    return (
      <ErrorState
        title="Failed to load deployments"
        error={error as Error}
        onRetry={() => refetch()}
      />
    )
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Deployments</h1>
        <p className={styles.subtitle}>
          Compare {deployments.length} model deployments across providers
        </p>
      </header>

      {/* View Mode Toggle */}
      <div className={styles.viewModeToggle}>
        <button
          className={`${styles.viewModeBtn} ${viewMode === 'grid' ? styles.active : ''}`}
          onClick={() => setViewMode('grid')}
        >
          <i className={`ph ph-folder-open ${styles.viewIcon}`}></i>
          All Deployments
        </button>
        <button
          className={`${styles.viewModeBtn} ${viewMode === 'compare' ? styles.active : ''}`}
          onClick={() => setViewMode('compare')}
        >
          <i className={`ph ph-scales ${styles.viewIcon}`}></i>
          Compare Providers
        </button>
      </div>

      {/* Filters */}
      <SketchCard padding="md" className={styles.filters}>
        <div className={styles.filterRow}>
          {viewMode === 'grid' ? (
            <>
              <div className={styles.filterGroup}>
                <label className={styles.filterLabel}>Search</label>
                <input
                  type="text"
                  className={styles.searchInput}
                  placeholder="Search by name, model, or provider..."
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

              <div className={styles.filterGroup}>
                <label className={styles.filterLabel}>Max Price ($/1M tokens)</label>
                <input
                  type="number"
                  className={styles.searchInput}
                  placeholder="e.g. 10.00"
                  value={maxPriceFilter}
                  onChange={(e) => setMaxPriceFilter(e.target.value)}
                  min="0"
                  step="0.01"
                />
              </div>
            </>
          ) : (
            <div className={styles.filterGroup} style={{ flex: 1 }}>
              <label className={styles.filterLabel}>Select Base Model to Compare</label>
              <select
                className={styles.select}
                value={selectedBaseModel}
                onChange={(e) => setSelectedBaseModel(e.target.value)}
                style={{ maxWidth: '500px' }}
              >
                <option value="">Choose a base model...</option>
                {baseModels.map(m => (
                  <option key={m.id} value={m.id}>
                    {m.id} ({m.count} provider{m.count !== 1 ? 's' : ''})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </SketchCard>

      {/* Content based on view mode */}
      {viewMode === 'grid' ? (
        /* Deployments Grid */
        sortedDeployments.length === 0 ? (
          <EmptyState
            title="No deployments found"
            message="Try adjusting your filters"
            icon="ph ph-magnifying-glass"
          />
        ) : (
          <div className={styles.deploymentsGrid}>
            {sortedDeployments.map(deployment => (
              <SketchCard
                key={deployment.id}
                padding="md"
                className={styles.deploymentCard}
                onClick={() => setSelectedDeployment(deployment)}
              >
                <div className={styles.deploymentHeader}>
                  <h3 className={styles.deploymentName}>{deployment.deployment.display_name}</h3>
                  <span className={styles.providerBadge}>{deployment.provider.name}</span>
                </div>

                <div className={styles.baseModel}>
                  Base: {deployment.base_model_id}
                </div>

                <div className={styles.deploymentStats}>
                  <div className={styles.stat}>
                    <span className={styles.statLabel}>Rating</span>
                    <span className={styles.statValue}>{deployment.trueskill.raw.mu.toFixed(0)}</span>
                  </div>
                  <div className={styles.stat}>
                    <span className={styles.statLabel}>Latency</span>
                    <span className={styles.statValue}>{deployment.performance.avg_latency_ms.toFixed(0)}ms</span>
                  </div>
                  <div className={styles.stat}>
                    <span className={styles.statLabel}>Context</span>
                    <span className={styles.statValue}>{(deployment.deployment.max_context_length / 1000).toFixed(0)}K</span>
                  </div>
                </div>

                <div className={styles.pricing}>
                  <div className={styles.priceItem}>
                    <span className={styles.priceLabel}>Input:</span>
                    <span className={styles.priceValue}>${deployment.pricing.input_cost_per_million.toFixed(2)}/1M</span>
                  </div>
                  <div className={styles.priceItem}>
                    <span className={styles.priceLabel}>Output:</span>
                    <span className={styles.priceValue}>${deployment.pricing.output_cost_per_million.toFixed(2)}/1M</span>
                  </div>
                </div>

                <div className={styles.cardFooter}>
                  <span className={`${styles.statusIndicator} ${styles[deployment.availability.status]}`}>
                    {deployment.availability.status}
                  </span>
                  <span className={styles.throughput}>
                    {deployment.performance.tokens_per_second.toFixed(0)} tok/s
                  </span>
                </div>

                <SketchButton
                  variant="secondary"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedDeployment(deployment)
                  }}
                >
                  View Details
                </SketchButton>
              </SketchCard>
            ))}
          </div>
        )
      ) : (
        /* Provider Comparison View */
        <ProviderComparisonView
          deployments={deployments}
          selectedBaseModel={selectedBaseModel}
          onSelectDeployment={setSelectedDeployment}
        />
      )}

      {/* Detail Modal */}
      {selectedDeployment && (
        <DeploymentDetailModal
          deployment={selectedDeployment}
          onClose={() => setSelectedDeployment(null)}
        />
      )}
    </div>
  )
}

export default Deployments
