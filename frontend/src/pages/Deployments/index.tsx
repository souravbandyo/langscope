/**
 * Deployments Page
 * 
 * Lists all model deployments across providers with pricing and performance data.
 */

import { useState } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useDeployments } from '@/api/hooks'
import type { DeploymentResponse } from '@/api/types'
import styles from './Deployments.module.css'

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

export function Deployments() {
  const [providerFilter, setProviderFilter] = useState('')
  const [maxPriceFilter, setMaxPriceFilter] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedDeployment, setSelectedDeployment] = useState<DeploymentResponse | null>(null)

  const { data, isLoading, error, refetch } = useDeployments({
    provider: providerFilter || undefined,
    max_price: maxPriceFilter ? parseFloat(maxPriceFilter) : undefined,
  })

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

  const deployments = data?.deployments || []

  // Get unique providers
  const providers = [...new Set(deployments.map(d => d.provider.name))].sort()

  // Filter deployments
  const filteredDeployments = deployments.filter(deployment => {
    const matchesSearch = !searchQuery ||
      deployment.deployment.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      deployment.base_model_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      deployment.provider.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesSearch
  })

  // Sort by rating
  const sortedDeployments = [...filteredDeployments].sort(
    (a, b) => b.trueskill.raw.mu - a.trueskill.raw.mu
  )

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Deployments</h1>
        <p className={styles.subtitle}>
          Compare {deployments.length} model deployments across providers
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
        </div>
      </SketchCard>

      {/* Deployments Grid */}
      {sortedDeployments.length === 0 ? (
        <EmptyState
          title="No deployments found"
          message="Try adjusting your filters"
          icon="🔍"
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
