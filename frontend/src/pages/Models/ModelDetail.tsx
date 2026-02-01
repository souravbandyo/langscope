/**
 * Model Detail Page
 * 
 * Shows detailed information about a specific model including
 * ratings across domains, benchmark results, and performance metrics.
 */

import { useParams, useNavigate } from 'react-router-dom'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState } from '@/components/common'
import { useModel } from '@/api/hooks'
import { useBenchmarkResults } from '@/api/hooks/useBenchmarks'
import styles from './ModelDetail.module.css'

export function ModelDetail() {
  const { '*': modelPath } = useParams()
  const navigate = useNavigate()
  // Model IDs can contain slashes (e.g., "openai/gpt-4o"), so we use wildcard routing
  const decodedModelId = modelPath ? decodeURIComponent(modelPath) : ''
  
  const { data: model, isLoading, error, refetch } = useModel(decodedModelId)
  const { data: benchmarks } = useBenchmarkResults(decodedModelId)

  if (isLoading) {
    return <LoadingState message="Loading model details..." />
  }

  if (error || !model) {
    const errorMessage = (error as Error)?.message || ''
    const isAuthError = errorMessage.includes('401') || errorMessage.includes('AUTH')
    
    return (
      <div className={styles.container}>
        <button className={styles.backButton} onClick={() => navigate('/models')}>
          ← Back to Models
        </button>
        <ErrorState 
          title={isAuthError ? "Authentication required" : "Model not found"}
          message={isAuthError ? "Please sign in to view model details" : `The model "${decodedModelId}" could not be found`}
          error={error as Error}
          onRetry={() => refetch()}
        />
      </div>
    )
  }

  const domainRatings = Object.entries(model.trueskill_by_domain || {})

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <button className={styles.backButton} onClick={() => navigate('/models')}>
          ← Back to Models
        </button>
        <div className={styles.headerContent}>
          <div>
            <h1 className={styles.title}>{model.name}</h1>
            <p className={styles.modelId}>{model.model_id}</p>
          </div>
          <span className={styles.providerBadge}>{model.provider}</span>
        </div>
      </header>

      <div className={styles.grid}>
        {/* Overview Card */}
        <SketchCard padding="lg" className={styles.overviewCard}>
          <h2 className={styles.sectionTitle}>Overview</h2>
          
          <div className={styles.overviewStats}>
            <div className={styles.overviewStat}>
              <span className={styles.overviewLabel}>Global Rating</span>
              <span className={styles.overviewValue}>
                {model.trueskill?.raw?.mu?.toFixed(1) || 'N/A'}
                <span className={styles.sigma}>±{model.trueskill?.raw?.sigma?.toFixed(1) || '?'}</span>
              </span>
            </div>
            <div className={styles.overviewStat}>
              <span className={styles.overviewLabel}>Cost-Adjusted</span>
              <span className={styles.overviewValue}>
                {model.trueskill?.cost_adjusted?.mu?.toFixed(1) || 'N/A'}
                <span className={styles.sigma}>±{model.trueskill?.cost_adjusted?.sigma?.toFixed(1) || '?'}</span>
              </span>
            </div>
            <div className={styles.overviewStat}>
              <span className={styles.overviewLabel}>Matches Played</span>
              <span className={styles.overviewValue}>{model.total_matches_played || 0}</span>
            </div>
            <div className={styles.overviewStat}>
              <span className={styles.overviewLabel}>Domains</span>
              <span className={styles.overviewValue}>{model.domains_evaluated?.length || 0}</span>
            </div>
          </div>
        </SketchCard>

        {/* Pricing Card */}
        <SketchCard padding="lg" className={styles.pricingCard}>
          <h2 className={styles.sectionTitle}>Pricing</h2>
          
          <div className={styles.pricingGrid}>
            <div className={styles.pricingItem}>
              <span className={styles.pricingLabel}>Input</span>
              <span className={styles.pricingValue}>
                ${model.input_cost_per_million?.toFixed(2) || '?'}
                <span className={styles.pricingUnit}>/1M tokens</span>
              </span>
            </div>
            <div className={styles.pricingItem}>
              <span className={styles.pricingLabel}>Output</span>
              <span className={styles.pricingValue}>
                ${model.output_cost_per_million?.toFixed(2) || '?'}
                <span className={styles.pricingUnit}>/1M tokens</span>
              </span>
            </div>
          </div>
          
          {model.pricing_source && (
            <p className={styles.pricingSource}>Source: {model.pricing_source}</p>
          )}
        </SketchCard>

        {/* Performance Card */}
        <SketchCard padding="lg" className={styles.performanceCard}>
          <h2 className={styles.sectionTitle}>Performance</h2>
          
          <div className={styles.performanceGrid}>
            <div className={styles.performanceItem}>
              <span className={styles.performanceLabel}>Avg Latency</span>
              <span className={styles.performanceValue}>
                {model.avg_latency_ms ? `${model.avg_latency_ms.toFixed(0)}ms` : 'N/A'}
              </span>
            </div>
            <div className={styles.performanceItem}>
              <span className={styles.performanceLabel}>Avg TTFT</span>
              <span className={styles.performanceValue}>
                {model.avg_ttft_ms ? `${model.avg_ttft_ms.toFixed(0)}ms` : 'N/A'}
              </span>
            </div>
          </div>
        </SketchCard>

        {/* Domain Ratings Card */}
        <SketchCard padding="lg" className={styles.domainCard}>
          <h2 className={styles.sectionTitle}>Domain Ratings</h2>
          
          {domainRatings.length === 0 ? (
            <p className={styles.emptyText}>No domain ratings yet</p>
          ) : (
            <div className={styles.domainList}>
              {domainRatings.map(([domain, rating]) => (
                <div key={domain} className={styles.domainItem}>
                  <span className={styles.domainName}>{domain}</span>
                  <div className={styles.domainRatings}>
                    <span className={styles.domainRating}>
                      Raw: <strong>{rating.raw?.mu?.toFixed(1) || 'N/A'}</strong>
                    </span>
                    <span className={styles.domainRating}>
                      Cost: <strong>{rating.cost_adjusted?.mu?.toFixed(1) || 'N/A'}</strong>
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </SketchCard>

        {/* Benchmarks Card */}
        <SketchCard padding="lg" className={styles.benchmarkCard}>
          <h2 className={styles.sectionTitle}>Benchmark Results</h2>
          
          {!benchmarks || benchmarks.length === 0 ? (
            <p className={styles.emptyText}>No benchmark results available</p>
          ) : (
            <div className={styles.benchmarkList}>
              {benchmarks.map((result, idx) => (
                <div key={idx} className={styles.benchmarkItem}>
                  <span className={styles.benchmarkName}>{result.benchmark_id}</span>
                  <span className={styles.benchmarkScore}>
                    {result.score?.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </SketchCard>
      </div>
    </div>
  )
}

export default ModelDetail
