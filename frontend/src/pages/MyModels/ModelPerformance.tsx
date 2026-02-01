/**
 * Model Performance View
 * 
 * Unified performance view that adapts based on model type:
 * - LLM: TrueSkill charts, domain breakdown, 10-dimensional scores
 * - ASR: WER heatmap by language/difficulty, latency charts
 * - TTS: Quality scores, round-trip intelligibility
 * - VLM: Accuracy by task type, visual examples
 */

import { useState, useMemo } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { LoadingState, ErrorState } from '@/components/common'
import { useMyModelPerformance, useModelComparison, useRunEvaluation, useEvaluationStatus } from '@/api/hooks'
import {
  MODEL_TYPE_CONFIGS,
  getModelTypeIcon,
  getPrimaryMetric,
  supportsGroundTruth,
  supportsSubjective,
} from '@/types/modelTypes'
import type { UserModel, ModelType } from '@/api/types'
import styles from './ModelPerformance.module.css'

interface Props {
  model: UserModel
  onClose: () => void
}

export function ModelPerformance({ model, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<'overview' | 'comparison' | 'history'>('overview')
  const [selectedDomain, setSelectedDomain] = useState<string>('')
  const [runningEvalId, setRunningEvalId] = useState<string | null>(null)

  const { data: performance, isLoading, error, refetch } = useMyModelPerformance(model.id)
  const runEvaluationMutation = useRunEvaluation()
  const { data: evalStatus } = useEvaluationStatus(runningEvalId || '', {
    refetchInterval: runningEvalId ? 3000 : false
  })

  const typeConfig = MODEL_TYPE_CONFIGS[model.modelType]
  const primaryMetric = getPrimaryMetric(model.modelType)

  // Get available domains for this model type
  const availableDomains = useMemo(() => {
    const domains: { id: string; name: string; type: 'ground_truth' | 'subjective' }[] = []
    
    typeConfig.groundTruthDomains.forEach(d => {
      domains.push({ id: d, name: d.replace(/_/g, ' '), type: 'ground_truth' })
    })
    typeConfig.subjectiveDomains.forEach(d => {
      domains.push({ id: d, name: d.replace(/_/g, ' '), type: 'subjective' })
    })
    
    return domains
  }, [typeConfig])

  // Handle running a new evaluation
  const handleRunEvaluation = async (domain: string, evalType: 'subjective' | 'ground_truth') => {
    try {
      const result = await runEvaluationMutation.mutateAsync({
        modelId: model.id,
        domain,
        evaluationType: evalType,
        sampleCount: 10,
      })
      setRunningEvalId(result.evaluationId)
    } catch (err) {
      console.error('Failed to start evaluation:', err)
    }
  }

  // Check if evaluation completed
  if (evalStatus?.status === 'completed' && runningEvalId) {
    setRunningEvalId(null)
    refetch()
  }

  if (isLoading) {
    return (
      <div className={styles.overlay} onClick={onClose}>
        <div className={styles.modal} onClick={e => e.stopPropagation()}>
          <LoadingState message="Loading performance data..." />
        </div>
      </div>
    )
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerInfo}>
            <span className={styles.modelIcon}>{typeConfig.icon}</span>
            <div>
              <h2 className={styles.modelName}>{model.name}</h2>
              <span className={styles.modelType}>{typeConfig.displayName} • v{model.version}</span>
            </div>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        {/* Tabs */}
        <div className={styles.tabs}>
          <button 
            className={`${styles.tab} ${activeTab === 'overview' ? styles.active : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`${styles.tab} ${activeTab === 'comparison' ? styles.active : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            Compare
          </button>
          <button 
            className={`${styles.tab} ${activeTab === 'history' ? styles.active : ''}`}
            onClick={() => setActiveTab('history')}
          >
            History
          </button>
        </div>

        {/* Content */}
        <div className={styles.content}>
          {error ? (
            <ErrorState
              title="Failed to load performance"
              error={error as Error}
              onRetry={() => refetch()}
            />
          ) : (
            <>
              {/* Overview Tab */}
              {activeTab === 'overview' && (
                <div className={styles.overviewContent}>
                  {/* Summary Stats */}
                  <div className={styles.summaryRow}>
                    <StickyNote title="Primary Metric" color="blue" rotation={-1}>
                      <div className={styles.primaryMetric}>
                        <span className={styles.metricValue}>
                          {performance?.trueskill?.raw_quality?.mu.toFixed(0) ||
                           performance?.groundTruthMetrics?.[primaryMetric?.id || '']?.toFixed(2) ||
                           'N/A'}
                        </span>
                        <span className={styles.metricName}>{primaryMetric?.name}</span>
                      </div>
                    </StickyNote>

                    <StickyNote title="Evaluations" color="yellow" rotation={1}>
                      <span className={styles.statValue}>{model.totalEvaluations}</span>
                    </StickyNote>

                    <StickyNote title="Domains" color="green" rotation={-1}>
                      <span className={styles.statValue}>{model.domainsEvaluated.length}</span>
                    </StickyNote>

                    {performance?.publicRank && (
                      <StickyNote title="Public Rank" color="pink" rotation={1}>
                        <span className={styles.statValue}>
                          #{performance.publicRank} / {performance.publicTotal}
                        </span>
                      </StickyNote>
                    )}
                  </div>

                  {/* Type-Specific Performance View */}
                  {model.modelType === 'LLM' || model.modelType === 'VLM' ? (
                    <TrueSkillView performance={performance} modelType={model.modelType} />
                  ) : model.modelType === 'ASR' || model.modelType === 'STT' ? (
                    <ASRView performance={performance} />
                  ) : model.modelType === 'TTS' ? (
                    <TTSView performance={performance} />
                  ) : (
                    <GenericView performance={performance} modelType={model.modelType} />
                  )}

                  {/* Run Evaluation Section */}
                  <SketchCard padding="lg" className={styles.evalSection}>
                    <h3 className={styles.sectionTitle}>Run Evaluation</h3>
                    <p className={styles.sectionDesc}>
                      Select a domain to evaluate your model's performance.
                    </p>

                    <div className={styles.domainGrid}>
                      {availableDomains.map(domain => (
                        <div key={domain.id} className={styles.domainCard}>
                          <div className={styles.domainInfo}>
                            <span className={styles.domainName}>{domain.name}</span>
                            <span className={`${styles.domainType} ${styles[domain.type]}`}>
                              {domain.type === 'ground_truth' ? 'Ground Truth' : 'Arena Battle'}
                            </span>
                          </div>
                          <SketchButton
                            variant="secondary"
                            size="sm"
                            onClick={() => handleRunEvaluation(domain.id, domain.type)}
                            disabled={runEvaluationMutation.isPending || !!runningEvalId}
                          >
                            {runningEvalId ? 'Running...' : 'Evaluate'}
                          </SketchButton>
                        </div>
                      ))}
                    </div>

                    {/* Evaluation Progress */}
                    {evalStatus && runningEvalId && (
                      <div className={styles.evalProgress}>
                        <div className={styles.progressBar}>
                          <div 
                            className={styles.progressFill}
                            style={{ width: `${evalStatus.progress || 0}%` }}
                          />
                        </div>
                        <span className={styles.progressText}>
                          {evalStatus.currentStep || 'Running evaluation...'}
                        </span>
                      </div>
                    )}
                  </SketchCard>
                </div>
              )}

              {/* Comparison Tab */}
              {activeTab === 'comparison' && (
                <ComparisonView model={model} />
              )}

              {/* History Tab */}
              {activeTab === 'history' && (
                <HistoryView performance={performance} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// TrueSkill View (LLM/VLM)
// =============================================================================

interface TrueSkillViewProps {
  performance: any
  modelType: ModelType
}

function TrueSkillView({ performance, modelType }: TrueSkillViewProps) {
  const dimensions = [
    { id: 'raw_quality', name: 'Raw Quality', icon: 'ph ph-star' },
    { id: 'cost_adjusted', name: 'Cost-Adjusted', icon: 'ph ph-currency-dollar' },
    { id: 'latency', name: 'Latency', icon: 'ph ph-lightning' },
    { id: 'ttft', name: 'TTFT', icon: 'ph ph-rocket' },
    { id: 'consistency', name: 'Consistency', icon: 'ph ph-target' },
    { id: 'token_efficiency', name: 'Token Efficiency', icon: 'ph ph-textbox' },
    { id: 'instruction_following', name: 'Instructions', icon: 'ph ph-clipboard-text' },
    { id: 'hallucination_resistance', name: 'Hallucination', icon: 'ph ph-magnifying-glass' },
    { id: 'long_context', name: 'Long Context', icon: 'ph ph-books' },
  ]

  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>TrueSkill Ratings</h3>
      <p className={styles.sectionDesc}>
        Multi-dimensional performance scores across 9 evaluation dimensions.
      </p>

      <div className={styles.dimensionGrid}>
        {dimensions.map(dim => {
          const rating = performance?.trueskill?.[dim.id]
          return (
            <div key={dim.id} className={styles.dimensionCard}>
              <i className={`${dim.icon} ${styles.dimIcon}`}></i>
              <div className={styles.dimInfo}>
                <span className={styles.dimName}>{dim.name}</span>
                <div className={styles.dimRating}>
                  <span className={styles.dimMu}>
                    {rating?.mu?.toFixed(0) || 'N/A'}
                  </span>
                  {rating?.sigma && (
                    <span className={styles.dimSigma}>
                      ±{rating.sigma.toFixed(0)}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Domain Breakdown */}
      {performance?.trueskillByDomain && Object.keys(performance.trueskillByDomain).length > 0 && (
        <div className={styles.domainBreakdown}>
          <h4 className={styles.subsectionTitle}>Performance by Domain</h4>
          <div className={styles.domainList}>
            {Object.entries(performance.trueskillByDomain).map(([domain, ratings]: [string, any]) => (
              <div key={domain} className={styles.domainRow}>
                <span className={styles.domainLabel}>{domain.replace(/_/g, ' ')}</span>
                <span className={styles.domainScore}>
                  {ratings.raw_quality?.mu?.toFixed(0) || 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </SketchCard>
  )
}

// =============================================================================
// ASR View
// =============================================================================

interface ASRViewProps {
  performance: any
}

function ASRView({ performance }: ASRViewProps) {
  const metrics = performance?.groundTruthMetrics || {}
  
  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>ASR Performance</h3>
      
      <div className={styles.asrMetrics}>
        <div className={styles.metricCard}>
          <i className={`ph ph-chart-bar ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Word Error Rate</span>
            <span className={styles.metricValue}>
              {metrics.wer !== undefined ? `${(metrics.wer * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <i className={`ph ph-text-aa ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Character Error Rate</span>
            <span className={styles.metricValue}>
              {metrics.cer !== undefined ? `${(metrics.cer * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <i className={`ph ph-timer ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Latency</span>
            <span className={styles.metricValue}>
              {metrics.latency !== undefined ? `${metrics.latency.toFixed(0)}ms` : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <span className={styles.metricIcon}>📈</span>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Real-Time Factor</span>
            <span className={styles.metricValue}>
              {metrics.rtf !== undefined ? `${metrics.rtf.toFixed(2)}x` : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Performance by Language */}
      {performance?.groundTruthByDomain?.asr && (
        <div className={styles.languageBreakdown}>
          <h4 className={styles.subsectionTitle}>Performance by Language</h4>
          <div className={styles.languageGrid}>
            {Object.entries(performance.groundTruthByDomain.asr).map(([lang, metrics]: [string, any]) => (
              <div key={lang} className={styles.languageCard}>
                <span className={styles.langCode}>{lang.toUpperCase()}</span>
                <span className={styles.langWer}>
                  WER: {metrics.wer !== undefined ? `${(metrics.wer * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </SketchCard>
  )
}

// =============================================================================
// TTS View
// =============================================================================

interface TTSViewProps {
  performance: any
}

function TTSView({ performance }: TTSViewProps) {
  const metrics = performance?.groundTruthMetrics || {}
  
  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>TTS Performance</h3>
      
      <div className={styles.ttsMetrics}>
        <div className={styles.metricCard}>
          <i className={`ph ph-target ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Mean Opinion Score</span>
            <span className={styles.metricValue}>
              {metrics.mos !== undefined ? metrics.mos.toFixed(2) : 'N/A'}
            </span>
            <span className={styles.metricScale}>/5.0</span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <i className={`ph ph-speaker-high ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Intelligibility</span>
            <span className={styles.metricValue}>
              {metrics.intelligibility !== undefined ? `${(metrics.intelligibility * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <i className={`ph ph-masks-theater ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Naturalness</span>
            <span className={styles.metricValue}>
              {metrics.naturalness !== undefined ? metrics.naturalness.toFixed(2) : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.metricCard}>
          <i className={`ph ph-lightning ${styles.metricIcon}`}></i>
          <div className={styles.metricInfo}>
            <span className={styles.metricLabel}>Generation Latency</span>
            <span className={styles.metricValue}>
              {metrics.latency !== undefined ? `${metrics.latency.toFixed(0)}ms` : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </SketchCard>
  )
}

// =============================================================================
// Generic View (for other model types)
// =============================================================================

interface GenericViewProps {
  performance: any
  modelType: ModelType
}

function GenericView({ performance, modelType }: GenericViewProps) {
  const config = MODEL_TYPE_CONFIGS[modelType]
  const metrics = performance?.groundTruthMetrics || {}
  
  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>{config.displayName} Performance</h3>
      
      <div className={styles.genericMetrics}>
        {config.metrics.map(metric => (
          <div key={metric.id} className={styles.metricCard}>
            <div className={styles.metricInfo}>
              <span className={styles.metricLabel}>{metric.name}</span>
              <span className={styles.metricValue}>
                {metrics[metric.id] !== undefined 
                  ? `${metrics[metric.id].toFixed(2)}${metric.unit || ''}`
                  : 'N/A'
                }
              </span>
              <span className={styles.metricDesc}>{metric.description}</span>
            </div>
          </div>
        ))}
      </div>
    </SketchCard>
  )
}

// =============================================================================
// Comparison View
// =============================================================================

interface ComparisonViewProps {
  model: UserModel
}

function ComparisonView({ model }: ComparisonViewProps) {
  const primaryMetric = getPrimaryMetric(model.modelType)
  const { data: comparison, isLoading } = useModelComparison({
    domain: model.domainsEvaluated[0] || 'general',
    modelType: model.modelType,
    metric: primaryMetric?.id,
    includePublic: true,
    limit: 20,
  })

  if (isLoading) {
    return <LoadingState message="Loading comparison data..." />
  }

  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>Comparison with Public Models</h3>
      
      {comparison?.entries?.length ? (
        <table className={styles.comparisonTable}>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Type</th>
              <th>{primaryMetric?.name || 'Score'}</th>
            </tr>
          </thead>
          <tbody>
            {comparison.entries.map(entry => (
              <tr 
                key={entry.modelId}
                className={entry.isUserModel ? styles.userModelRow : ''}
              >
                <td className={styles.rankCell}>
                  {entry.rank <= 3 ? (
                    <span className={`${styles.medal} ${styles[`rank${entry.rank}`]}`}>
                      <i className="ph-fill ph-medal"></i>
                    </span>
                  ) : (
                    `#${entry.rank}`
                  )}
                </td>
                <td className={styles.modelCell}>
                  {entry.name}
                  {entry.isUserModel && <span className={styles.yourModel}>(Your Model)</span>}
                </td>
                <td className={styles.typeCell}>{entry.modelType}</td>
                <td className={styles.scoreCell}>
                  {entry.metrics[primaryMetric?.id || '']?.toFixed(2) || 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className={styles.noData}>No comparison data available yet. Run an evaluation first.</p>
      )}
    </SketchCard>
  )
}

// =============================================================================
// History View
// =============================================================================

interface HistoryViewProps {
  performance: any
}

function HistoryView({ performance }: HistoryViewProps) {
  const history = performance?.evaluationHistory || []

  return (
    <SketchCard padding="lg">
      <h3 className={styles.sectionTitle}>Evaluation History</h3>
      
      {history.length > 0 ? (
        <div className={styles.historyList}>
          {history.map((entry: any, index: number) => (
            <div key={index} className={styles.historyItem}>
              <div className={styles.historyDate}>
                {new Date(entry.timestamp).toLocaleDateString()}
              </div>
              <div className={styles.historyInfo}>
                <span className={styles.historyDomain}>{entry.domain.replace(/_/g, ' ')}</span>
                <span className={`${styles.historyType} ${styles[entry.metricType]}`}>
                  {entry.metricType === 'ground_truth' ? 'Ground Truth' : 'Arena'}
                </span>
              </div>
              <div className={styles.historyScores}>
                {Object.entries(entry.scores).slice(0, 3).map(([key, value]: [string, any]) => (
                  <span key={key} className={styles.historyScore}>
                    {key}: {value.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className={styles.noData}>No evaluation history yet.</p>
      )}
    </SketchCard>
  )
}

export default ModelPerformance
