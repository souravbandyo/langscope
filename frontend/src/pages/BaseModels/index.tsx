/**
 * Base Models Page
 * 
 * Lists all base LLM models with architecture details and allows provider comparison.
 */

import { useState } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useBaseModels, useProviderComparison } from '@/api/hooks'
import type { BaseModelResponse } from '@/api/types'
import styles from './BaseModels.module.css'

function formatParams(params: number): string {
  if (params >= 1e12) return `${(params / 1e12).toFixed(1)}T`
  if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`
  if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`
  return params.toLocaleString()
}

interface ModelDetailModalProps {
  model: BaseModelResponse
  onClose: () => void
}

function ModelDetailModal({ model, onClose }: ModelDetailModalProps) {
  const { data: comparison, isLoading } = useProviderComparison(model.id)

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>{model.name}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalContent}>
          {/* Architecture Section */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Architecture</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Type</span>
                <span className={styles.detailValue}>{model.architecture.type}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Parameters</span>
                <span className={styles.detailValue}>{model.architecture.parameters_display}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Hidden Size</span>
                <span className={styles.detailValue}>{model.architecture.hidden_size.toLocaleString()}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Layers</span>
                <span className={styles.detailValue}>{model.architecture.num_layers}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Attention Heads</span>
                <span className={styles.detailValue}>{model.architecture.num_attention_heads}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>KV Heads</span>
                <span className={styles.detailValue}>{model.architecture.num_kv_heads}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Vocab Size</span>
                <span className={styles.detailValue}>{model.architecture.vocab_size.toLocaleString()}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Native Precision</span>
                <span className={styles.detailValue}>{model.architecture.native_precision}</span>
              </div>
            </div>
          </section>

          {/* Capabilities Section */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Capabilities</h3>
            <div className={styles.capabilitiesList}>
              <div className={styles.capabilityTags}>
                {model.capabilities.modalities.map(mod => (
                  <span key={mod} className={styles.capabilityTag}>{mod}</span>
                ))}
              </div>
              <div className={styles.capabilityFlags}>
                {model.capabilities.supports_function_calling && <span className={styles.flag}>Function Calling</span>}
                {model.capabilities.supports_json_mode && <span className={styles.flag}>JSON Mode</span>}
                {model.capabilities.supports_vision && <span className={styles.flag}>Vision</span>}
                {model.capabilities.supports_audio && <span className={styles.flag}>Audio</span>}
                {model.capabilities.supports_streaming && <span className={styles.flag}>Streaming</span>}
              </div>
              {model.capabilities.languages.length > 0 && (
                <div className={styles.languages}>
                  <span className={styles.detailLabel}>Languages:</span>
                  <span>{model.capabilities.languages.slice(0, 5).join(', ')}{model.capabilities.languages.length > 5 ? ` +${model.capabilities.languages.length - 5} more` : ''}</span>
                </div>
              )}
            </div>
          </section>

          {/* Context Window Section */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Context Window</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Max Context</span>
                <span className={styles.detailValue}>{model.context.max_context_length.toLocaleString()} tokens</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Recommended</span>
                <span className={styles.detailValue}>{model.context.recommended_context.toLocaleString()} tokens</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Max Output</span>
                <span className={styles.detailValue}>{model.context.max_output_tokens.toLocaleString()} tokens</span>
              </div>
            </div>
          </section>

          {/* License Section */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>License</h3>
            <div className={styles.licenseInfo}>
              <span className={styles.licenseType}>{model.license.type}</span>
              {model.license.commercial_use && <span className={styles.licenseBadge}>Commercial Use</span>}
              {model.license.requires_agreement && <span className={styles.licenseBadgeWarning}>Agreement Required</span>}
            </div>
          </section>

          {/* Provider Comparison Section */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Provider Comparison</h3>
            {isLoading ? (
              <div className={styles.loadingSmall}>Loading providers...</div>
            ) : comparison?.providers && comparison.providers.length > 0 ? (
              <div className={styles.providerTable}>
                <div className={styles.providerHeader}>
                  <span>Provider</span>
                  <span>Input $/1M</span>
                  <span>Output $/1M</span>
                  <span>Latency</span>
                  <span>Rating</span>
                </div>
                {comparison.providers.map(p => (
                  <div key={p.deployment_id} className={styles.providerRow}>
                    <span className={styles.providerName}>{p.provider.name}</span>
                    <span>${p.pricing.input_cost_per_million.toFixed(2)}</span>
                    <span>${p.pricing.output_cost_per_million.toFixed(2)}</span>
                    <span>{p.performance.avg_latency_ms.toFixed(0)}ms</span>
                    <span>{p.trueskill.raw.mu.toFixed(0)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.noProviders}>No provider deployments available</p>
            )}
          </section>

          {/* Quantizations Section */}
          {Object.keys(model.quantizations).length > 0 && (
            <section className={styles.section}>
              <h3 className={styles.sectionTitle}>Quantization Options</h3>
              <div className={styles.quantTable}>
                <div className={styles.quantHeader}>
                  <span>Name</span>
                  <span>Bits</span>
                  <span>VRAM</span>
                  <span>Quality</span>
                </div>
                {Object.entries(model.quantizations).map(([name, q]) => (
                  <div key={name} className={styles.quantRow}>
                    <span>{name}</span>
                    <span>{q.bits}-bit</span>
                    <span>{q.vram_gb} GB</span>
                    <span>{(q.quality_retention * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Benchmarks Section */}
          {Object.keys(model.benchmarks).length > 0 && (
            <section className={styles.section}>
              <h3 className={styles.sectionTitle}>Benchmark Scores</h3>
              <div className={styles.benchmarkGrid}>
                {Object.entries(model.benchmarks).slice(0, 8).map(([name, score]) => (
                  <div key={name} className={styles.benchmarkItem}>
                    <span className={styles.benchmarkName}>{name}</span>
                    <span className={styles.benchmarkScore}>{score.score.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

export function BaseModels() {
  const [familyFilter, setFamilyFilter] = useState('')
  const [orgFilter, setOrgFilter] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedModel, setSelectedModel] = useState<BaseModelResponse | null>(null)

  const { data, isLoading, error, refetch } = useBaseModels({
    family: familyFilter || undefined,
    organization: orgFilter || undefined,
  })

  if (isLoading) {
    return <LoadingState message="Loading base models..." />
  }

  if (error) {
    return (
      <ErrorState
        title="Failed to load base models"
        error={error as Error}
        onRetry={() => refetch()}
      />
    )
  }

  const models = data?.models || []

  // Get unique families and organizations
  const families = [...new Set(models.map(m => m.family))].filter(Boolean).sort()
  const organizations = [...new Set(models.map(m => m.organization))].filter(Boolean).sort()

  // Filter models
  const filteredModels = models.filter(model => {
    const matchesSearch = !searchQuery ||
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.family.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesSearch
  })

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Base Models</h1>
        <p className={styles.subtitle}>
          Browse {models.length} base LLM architectures and compare provider deployments
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
              placeholder="Search by name, ID, or family..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Family</label>
            <select
              className={styles.select}
              value={familyFilter}
              onChange={(e) => setFamilyFilter(e.target.value)}
            >
              <option value="">All Families</option>
              {families.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Organization</label>
            <select
              className={styles.select}
              value={orgFilter}
              onChange={(e) => setOrgFilter(e.target.value)}
            >
              <option value="">All Organizations</option>
              {organizations.map(o => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </div>
        </div>
      </SketchCard>

      {/* Models Grid */}
      {filteredModels.length === 0 ? (
        <EmptyState
          title="No base models found"
          message="Try adjusting your filters"
          icon="🔍"
        />
      ) : (
        <div className={styles.modelsGrid}>
          {filteredModels.map(model => (
            <SketchCard
              key={model.id}
              padding="md"
              className={styles.modelCard}
              onClick={() => setSelectedModel(model)}
            >
              <div className={styles.modelHeader}>
                <h3 className={styles.modelName}>{model.name}</h3>
                <span className={styles.orgBadge}>{model.organization}</span>
              </div>

              <div className={styles.modelMeta}>
                <span className={styles.family}>{model.family}</span>
                {model.version && <span className={styles.version}>v{model.version}</span>}
              </div>

              <div className={styles.modelStats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Parameters</span>
                  <span className={styles.statValue}>{formatParams(model.architecture.parameters)}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Context</span>
                  <span className={styles.statValue}>{(model.context.max_context_length / 1000).toFixed(0)}K</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Deployments</span>
                  <span className={styles.statValue}>{model.deployment_count}</span>
                </div>
              </div>

              <div className={styles.capabilityIcons}>
                {model.capabilities.supports_vision && <span title="Vision">👁️</span>}
                {model.capabilities.supports_audio && <span title="Audio">🔊</span>}
                {model.capabilities.supports_function_calling && <span title="Function Calling">⚡</span>}
                {model.capabilities.supports_json_mode && <span title="JSON Mode">📋</span>}
              </div>

              <div className={styles.architectureInfo}>
                <span className={styles.archType}>{model.architecture.type}</span>
                <span className={styles.precision}>{model.architecture.native_precision}</span>
              </div>

              <SketchButton
                variant="secondary"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedModel(model)
                }}
              >
                View Details
              </SketchButton>
            </SketchCard>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedModel && (
        <ModelDetailModal
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}
    </div>
  )
}

export default BaseModels
