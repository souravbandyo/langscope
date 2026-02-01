import { useState, useMemo } from 'react'
import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { ErrorState, LoadingState, EmptyState } from '@/components/common'
import { useGroundTruthDomains, useGroundTruthLeaderboard, useNeedleHeatmap, useGroundTruthCoverage } from '@/api/hooks'
import type { GroundTruthDomainInfo } from '@/api/types'
import styles from './GroundTruth.module.css'

// Domain category definitions with icons and descriptions
const domainCategories = {
  multimodal: {
    name: 'Multimodal',
    icon: 'ph ph-images',
    description: 'Audio, image, and document processing tasks',
  },
  long_context: {
    name: 'Long Context',
    icon: 'ph ph-books',
    description: 'Tasks requiring large context windows',
  },
}

// Domain-specific configurations
const domainConfigs: Record<string, { icon: string; metricLabel: string; metricDirection: 'higher' | 'lower' }> = {
  asr: { icon: 'ph ph-microphone', metricLabel: 'WER', metricDirection: 'lower' },
  tts: { icon: 'ph ph-speaker-high', metricLabel: 'TTS Score', metricDirection: 'higher' },
  visual_qa: { icon: 'ph ph-eye', metricLabel: 'Accuracy', metricDirection: 'higher' },
  document_extraction: { icon: 'ph ph-file-text', metricLabel: 'Field Accuracy', metricDirection: 'higher' },
  image_captioning: { icon: 'ph ph-image', metricLabel: 'CIDEr', metricDirection: 'higher' },
  ocr: { icon: 'ph ph-textbox', metricLabel: 'Character Accuracy', metricDirection: 'higher' },
  needle_in_haystack: { icon: 'ph ph-magnifying-glass', metricLabel: 'Retrieval Accuracy', metricDirection: 'higher' },
  long_document_qa: { icon: 'ph ph-book-open', metricLabel: 'Accuracy', metricDirection: 'higher' },
  multi_document_reasoning: { icon: 'ph ph-files', metricLabel: 'Answer Accuracy', metricDirection: 'higher' },
  long_context_code_completion: { icon: 'ph ph-code', metricLabel: 'Tests Pass', metricDirection: 'higher' },
  long_summarization: { icon: 'ph ph-clipboard-text', metricLabel: 'ROUGE-L', metricDirection: 'higher' },
}

// Context lengths for needle-in-haystack visualization
const contextLengths = ['4K', '8K', '16K', '32K', '64K', '128K', '200K']
const needlePositions = ['0%', '25%', '50%', '75%', '100%']

/**
 * Ground Truth page for objective evaluation results
 */
export function GroundTruth() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedDomain, setSelectedDomain] = useState('')
  const [selectedModel, setSelectedModel] = useState('')

  const { data: domains, isLoading: domainsLoading, error: domainsError, refetch: refetchDomains } = useGroundTruthDomains()
  const { data: leaderboard, isLoading: leaderboardLoading } = useGroundTruthLeaderboard(
    selectedDomain,
    { limit: 20 }
  )
  const { data: coverage } = useGroundTruthCoverage(selectedDomain)
  const { data: heatmap } = useNeedleHeatmap(selectedModel)

  const selectedDomainInfo = domains?.find((d) => d.name === selectedDomain)

  // Filter domains by category
  const filteredDomains = useMemo(() => {
    if (!domains) return []
    if (selectedCategory === 'all') return domains
    return domains.filter((d) => d.category === selectedCategory)
  }, [domains, selectedCategory])

  // Group domains by category for display
  const domainsByCategory = useMemo(() => {
    if (!domains) return {}
    return domains.reduce((acc, domain) => {
      const cat = domain.category || 'other'
      if (!acc[cat]) acc[cat] = []
      acc[cat].push(domain)
      return acc
    }, {} as Record<string, GroundTruthDomainInfo[]>)
  }, [domains])

  // Get domain config
  const getDomainConfig = (domainName: string) => {
    const normalized = domainName.toLowerCase().replace(/ /g, '_')
    return domainConfigs[normalized] || { icon: 'ph ph-clipboard-text', metricLabel: 'Score', metricDirection: 'higher' as const }
  }

  return (
    <div className={styles.groundTruth}>
      <div className={styles.header}>
        <h1 className={styles.title}>Ground Truth Evaluation</h1>
        <p className={styles.subtitle}>
          Objective model performance with verifiable answers
        </p>
      </div>

      {/* Category Tabs */}
      <div className={styles.categoryTabs}>
        <button
          className={`${styles.categoryTab} ${selectedCategory === 'all' ? styles.active : ''}`}
          onClick={() => setSelectedCategory('all')}
        >
          <i className={`ph ph-chart-bar ${styles.tabIcon}`}></i>
          <span className={styles.tabName}>All Domains</span>
        </button>
        <button
          className={`${styles.categoryTab} ${selectedCategory === 'multimodal' ? styles.active : ''}`}
          onClick={() => setSelectedCategory('multimodal')}
        >
          <i className={`ph ph-images ${styles.tabIcon}`}></i>
          <span className={styles.tabName}>Multimodal</span>
          <span className={styles.tabDesc}>ASR, TTS, Visual QA</span>
        </button>
        <button
          className={`${styles.categoryTab} ${selectedCategory === 'long_context' ? styles.active : ''}`}
          onClick={() => setSelectedCategory('long_context')}
        >
          <i className={`ph ph-books ${styles.tabIcon}`}></i>
          <span className={styles.tabName}>Long Context</span>
          <span className={styles.tabDesc}>Needle, Doc QA, Code</span>
        </button>
      </div>

      {/* Domain Cards */}
      <div className={styles.domainSection}>
        {domainsLoading ? (
          <LoadingState message="Loading domains..." />
        ) : domainsError ? (
          <ErrorState
            title="Failed to load domains"
            error={domainsError as Error}
            onRetry={() => refetchDomains()}
          />
        ) : !filteredDomains?.length ? (
          <EmptyState
            title="No ground truth domains"
            message="Ground truth evaluation domains will appear here when configured."
            icon="ph ph-target"
          />
        ) : (
          <div className={styles.domainGrid}>
            {filteredDomains.map((domain) => {
              const config = getDomainConfig(domain.name)
              return (
                <button
                  key={domain.name}
                  className={`${styles.domainCard} ${selectedDomain === domain.name ? styles.selected : ''}`}
                  onClick={() => {
                    setSelectedDomain(domain.name)
                    setSelectedModel('')
                  }}
                >
                  <i className={`${config.icon} ${styles.domainIcon}`}></i>
                  <div className={styles.domainInfo}>
                    <h3 className={styles.domainName}>
                      {domain.name.replace(/_/g, ' ')}
                    </h3>
                    <div className={styles.domainMeta}>
                      <span className={styles.domainCategory}>
                        {domain.category?.replace(/_/g, ' ')}
                      </span>
                      <span className={styles.domainMetric}>
                        {config.metricLabel}
                      </span>
                    </div>
                    <span className={styles.sampleCount}>
                      {domain.sample_count.toLocaleString()} samples
                    </span>
                  </div>
                  {domain.difficulty_distribution && (
                    <div className={styles.difficultyBar}>
                      {Object.entries(domain.difficulty_distribution).map(([level, count]) => (
                        <div
                          key={level}
                          className={`${styles.difficultySegment} ${styles[level]}`}
                          style={{ flex: count }}
                          title={`${level}: ${count}`}
                        />
                      ))}
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Selected Domain Info */}
      {selectedDomainInfo && (
        <div className={styles.domainDetails}>
          <StickyNote title="Domain Info" color="blue" rotation={-1}>
            <div className={styles.infoContent}>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Primary Metric:</span>
                <span className={styles.infoValue}>
                  {getDomainConfig(selectedDomainInfo.name).metricLabel}
                  <span className={styles.metricDirection}>
                    ({getDomainConfig(selectedDomainInfo.name).metricDirection} is better)
                  </span>
                </span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Languages:</span>
                <span className={styles.infoValue}>
                  {selectedDomainInfo.supported_languages?.join(', ') || 'All'}
                </span>
              </div>
              {selectedDomainInfo.description && (
                <p className={styles.infoDesc}>{selectedDomainInfo.description}</p>
              )}
            </div>
          </StickyNote>

          {coverage && (
            <StickyNote title="Coverage Stats" color="green" rotation={1}>
              <div className={styles.coverageContent}>
                <div className={styles.coverageMain}>
                  <span className={styles.coveragePercent}>
                    {coverage.coverage_percentage.toFixed(1)}%
                  </span>
                  <span className={styles.coverageLabel}>coverage</span>
                </div>
                <div className={styles.coverageDetails}>
                  <div className={styles.coverageRow}>
                    <span>Total:</span>
                    <strong>{coverage.total_samples.toLocaleString()}</strong>
                  </div>
                  <div className={styles.coverageRow}>
                    <span>Used:</span>
                    <strong>{coverage.used_samples.toLocaleString()}</strong>
                  </div>
                </div>
              </div>
            </StickyNote>
          )}
        </div>
      )}

      {/* Leaderboard */}
      {selectedDomain && (
        <SketchCard padding="lg">
          <div className={styles.leaderboardHeader}>
            <h2 className={styles.sectionTitle}>
              <i className={getDomainConfig(selectedDomain).icon}></i> {selectedDomain.replace(/_/g, ' ')} Leaderboard
            </h2>
            <span className={styles.metricInfo}>
              Ranked by {getDomainConfig(selectedDomain).metricLabel}
            </span>
          </div>

          {leaderboardLoading ? (
            <div className={styles.loading}>Loading leaderboard...</div>
          ) : !leaderboard?.length ? (
            <div className={styles.empty}>No results for this domain yet</div>
          ) : (
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.thRank}>Rank</th>
                  <th className={styles.thModel}>Model</th>
                  <th className={styles.thRating}>TrueSkill (μ)</th>
                  <th className={styles.thScore}>{getDomainConfig(selectedDomain).metricLabel}</th>
                  <th className={styles.thEvals}>Evaluations</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry) => (
                  <tr
                    key={entry.deployment_id}
                    className={`${styles.tableRow} ${selectedModel === entry.deployment_id ? styles.selectedRow : ''}`}
                    onClick={() => setSelectedModel(entry.deployment_id)}
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
                    <td className={styles.modelName}>{entry.deployment_id}</td>
                    <td className={styles.rating}>{Math.round(entry.trueskill_mu)}</td>
                    <td className={styles.score}>
                      <span className={styles.scoreValue}>
                        {entry.primary_metric_avg.toFixed(2)}
                      </span>
                    </td>
                    <td className={styles.evaluations}>{entry.total_evaluations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </SketchCard>
      )}

      {/* Needle-in-Haystack Heatmap */}
      {selectedModel && selectedDomain === 'needle_in_haystack' && heatmap && (
        <SketchCard padding="lg">
          <div className={styles.heatmapHeader}>
            <h2 className={styles.sectionTitle}>
              <i className="ph ph-magnifying-glass"></i> Needle-in-Haystack Performance
            </h2>
            <span className={styles.modelLabel}>{selectedModel}</span>
          </div>
          
          <div className={styles.heatmapStats}>
            <div className={styles.statBox}>
              <span className={styles.statValue}>
                {(heatmap.overall_accuracy * 100).toFixed(1)}%
              </span>
              <span className={styles.statLabel}>Overall Accuracy</span>
            </div>
          </div>

          <div className={styles.heatmapContainer}>
            {/* Y-axis labels (Needle Position) */}
            <div className={styles.heatmapYAxis}>
              <span className={styles.axisTitle}>Position</span>
              {needlePositions.map((pos) => (
                <span key={pos} className={styles.axisLabel}>{pos}</span>
              ))}
            </div>

            <div className={styles.heatmapMain}>
              {/* X-axis labels (Context Length) */}
              <div className={styles.heatmapXAxis}>
                {contextLengths.map((len) => (
                  <span key={len} className={styles.axisLabel}>{len}</span>
                ))}
              </div>
              <span className={styles.axisTitle}>Context Length</span>

              {/* Heatmap Grid */}
              <div className={styles.heatmapGrid}>
                {heatmap.heatmap.map((cell, index) => {
                  const accuracy = cell.accuracy
                  // Color scale from red (0%) to green (100%)
                  const hue = accuracy * 120 // 0 = red, 120 = green
                  const backgroundColor = `hsl(${hue}, 70%, ${40 + accuracy * 20}%)`
                  
                  return (
                    <div
                      key={index}
                      className={styles.heatmapCell}
                      style={{ backgroundColor }}
                      title={`Context: ${cell.context_length}K tokens, Position: ${(cell.needle_position * 100).toFixed(0)}%, Accuracy: ${(accuracy * 100).toFixed(1)}%, Samples: ${cell.sample_count}`}
                    >
                      <span className={styles.cellValue}>
                        {(accuracy * 100).toFixed(0)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Color Legend */}
          <div className={styles.heatmapLegend}>
            <span className={styles.legendLabel}>Accuracy:</span>
            <div className={styles.legendGradient}>
              <span className={styles.legendMin}>0%</span>
              <div className={styles.gradientBar} />
              <span className={styles.legendMax}>100%</span>
            </div>
          </div>

          <p className={styles.heatmapNote}>
            Click on cells to see detailed results. The heatmap shows retrieval accuracy 
            at different context lengths and needle positions.
          </p>
        </SketchCard>
      )}

      {/* Help Section */}
      <div className={styles.helpSection}>
        <h3 className={styles.helpTitle}>About Ground Truth Evaluation</h3>
        <div className={styles.helpGrid}>
          <div className={styles.helpCard}>
            <i className={`ph ph-images ${styles.helpIcon}`}></i>
            <h4>Multimodal Domains</h4>
            <p>ASR (speech-to-text), TTS (text-to-speech), Visual QA, OCR, and document extraction with objective metrics like WER and accuracy.</p>
          </div>
          <div className={styles.helpCard}>
            <i className={`ph ph-books ${styles.helpIcon}`}></i>
            <h4>Long Context Domains</h4>
            <p>Needle-in-Haystack, Document QA, Code Completion - testing how well models maintain quality at large context lengths.</p>
          </div>
          <div className={styles.helpCard}>
            <i className={`ph ph-ruler ${styles.helpIcon}`}></i>
            <h4>Objective Metrics</h4>
            <p>Unlike subjective LLM judging, ground truth evaluation uses deterministic metrics computed against verified correct answers.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export { GroundTruth as default }
