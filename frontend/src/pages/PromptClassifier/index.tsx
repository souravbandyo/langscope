/**
 * Prompt Classifier Page
 * 
 * Interactive tool for testing prompt domain classification.
 */

import { useState } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState } from '@/components/common'
import { 
  useClassifyPrompt, 
  useProcessPrompt, 
  usePromptMetrics, 
  usePromptDomains,
  usePromptLanguages 
} from '@/api/hooks'
import type { ClassifyResponse, ProcessResponse } from '@/api/types'
import styles from './PromptClassifier.module.css'

const EXAMPLE_PROMPTS = [
  { label: 'Medical', text: 'What are the symptoms of diabetes and how is it treated?' },
  { label: 'Legal', text: 'Can I break a lease if my landlord refuses to make repairs?' },
  { label: 'Code', text: 'Write a Python function to sort a list using quicksort algorithm.' },
  { label: 'Hindi Medical', text: 'मधुमेह के लक्षण क्या हैं और इसका इलाज कैसे होता है?' },
  { label: 'Financial', text: 'What is the difference between a Roth IRA and traditional IRA?' },
  { label: 'Creative', text: 'Write a short poem about the beauty of autumn leaves.' },
]

export function PromptClassifier() {
  const [prompt, setPrompt] = useState('')
  const [context, setContext] = useState('')
  const [classifyResult, setClassifyResult] = useState<ClassifyResponse | null>(null)
  const [processResult, setProcessResult] = useState<ProcessResponse | null>(null)

  const { data: metricsData, isLoading: metricsLoading } = usePromptMetrics()
  const { data: domainsData } = usePromptDomains()
  const { data: languagesData } = usePromptLanguages()

  const classifyMutation = useClassifyPrompt()
  const processMutation = useProcessPrompt()

  const handleClassify = () => {
    if (!prompt.trim()) return
    
    setClassifyResult(null)
    setProcessResult(null)

    classifyMutation.mutate(
      { prompt: prompt.trim(), context: context.trim() || undefined },
      {
        onSuccess: (data) => {
          setClassifyResult(data)
        }
      }
    )
  }

  const handleProcess = () => {
    if (!prompt.trim()) return

    setProcessResult(null)

    processMutation.mutate(
      { prompt: prompt.trim() },
      {
        onSuccess: (data) => {
          setProcessResult(data)
        }
      }
    )
  }

  const handleExampleClick = (text: string) => {
    setPrompt(text)
    setClassifyResult(null)
    setProcessResult(null)
  }

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return styles.confidenceHigh
    if (confidence >= 0.5) return styles.confidenceMedium
    return styles.confidenceLow
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Prompt Classifier</h1>
        <p className={styles.subtitle}>
          Test domain classification to understand how prompts are routed
        </p>
      </header>

      <div className={styles.mainContent}>
        <div className={styles.inputSection}>
          {/* Example Prompts */}
          <SketchCard padding="md" className={styles.examplesCard}>
            <h3 className={styles.cardTitle}>Example Prompts</h3>
            <div className={styles.exampleButtons}>
              {EXAMPLE_PROMPTS.map((example) => (
                <button
                  key={example.label}
                  className={styles.exampleButton}
                  onClick={() => handleExampleClick(example.text)}
                >
                  {example.label}
                </button>
              ))}
            </div>
          </SketchCard>

          {/* Input Form */}
          <SketchCard padding="md" className={styles.inputCard}>
            <h3 className={styles.cardTitle}>Test Classification</h3>
            
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Prompt</label>
              <textarea
                className={styles.textarea}
                placeholder="Enter a prompt to classify..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
              />
            </div>

            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Context (optional)</label>
              <textarea
                className={styles.textarea}
                placeholder="Add context to improve classification accuracy..."
                value={context}
                onChange={(e) => setContext(e.target.value)}
                rows={2}
              />
            </div>

            <div className={styles.buttonRow}>
              <SketchButton
                variant="primary"
                onClick={handleClassify}
                disabled={!prompt.trim() || classifyMutation.isPending}
              >
                {classifyMutation.isPending ? 'Classifying...' : 'Classify'}
              </SketchButton>
              
              <SketchButton
                variant="secondary"
                onClick={handleProcess}
                disabled={!prompt.trim() || processMutation.isPending}
              >
                {processMutation.isPending ? 'Processing...' : 'Process & Recommend'}
              </SketchButton>
            </div>

            {(classifyMutation.error || processMutation.error) && (
              <div className={styles.error}>
                {(classifyMutation.error as Error)?.message || 
                 (processMutation.error as Error)?.message || 
                 'Classification failed. Please try again.'}
              </div>
            )}
          </SketchCard>
        </div>

        <div className={styles.resultsSection}>
          {/* Classification Result */}
          {classifyResult && (
            <SketchCard padding="md" className={styles.resultCard}>
              <h3 className={styles.cardTitle}>Classification Result</h3>
              
              <div className={styles.resultMain}>
                <div className={styles.domainResult}>
                  <span className={styles.resultLabel}>Detected Domain</span>
                  <span className={styles.domainName}>{classifyResult.domain}</span>
                  <div className={styles.confidenceBar}>
                    <div 
                      className={`${styles.confidenceFill} ${getConfidenceColor(classifyResult.confidence)}`}
                      style={{ width: `${classifyResult.confidence * 100}%` }}
                    />
                  </div>
                  <span className={styles.confidenceText}>
                    {(classifyResult.confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>

                <div className={styles.languageResult}>
                  <span className={styles.resultLabel}>Detected Language</span>
                  <span className={styles.languageName}>{classifyResult.language}</span>
                  <span className={styles.languageConfidence}>
                    {(classifyResult.language_confidence * 100).toFixed(1)}% confidence
                  </span>
                </div>
              </div>

              {classifyResult.secondary_domains && classifyResult.secondary_domains.length > 0 && (
                <div className={styles.secondaryDomains}>
                  <span className={styles.resultLabel}>Alternative Domains</span>
                  <div className={styles.secondaryList}>
                    {classifyResult.secondary_domains.map((sd, i) => (
                      <div key={i} className={styles.secondaryItem}>
                        <span className={styles.secondaryDomain}>{sd.domain}</span>
                        <span className={styles.secondaryConfidence}>
                          {(sd.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </SketchCard>
          )}

          {/* Process Result */}
          {processResult && (
            <SketchCard padding="md" className={styles.resultCard}>
              <h3 className={styles.cardTitle}>Processing Result</h3>
              
              <div className={styles.processDetails}>
                <div className={styles.processItem}>
                  <span className={styles.resultLabel}>Detected Domain</span>
                  <span className={styles.domainName}>{processResult.detected_domain}</span>
                </div>

                {processResult.cache_key && (
                  <div className={styles.processItem}>
                    <span className={styles.resultLabel}>Cache Key</span>
                    <code className={styles.cacheKey}>{processResult.cache_key}</code>
                  </div>
                )}

                {processResult.recommended_models && processResult.recommended_models.length > 0 && (
                  <div className={styles.recommendedModels}>
                    <span className={styles.resultLabel}>Recommended Models</span>
                    <div className={styles.modelList}>
                      {processResult.recommended_models.map((model, i) => (
                        <span key={i} className={styles.modelBadge}>{model}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </SketchCard>
          )}

          {/* Metrics Card */}
          <SketchCard padding="md" className={styles.metricsCard}>
            <h3 className={styles.cardTitle}>Classification Metrics</h3>
            
            {metricsLoading ? (
              <div className={styles.loadingSmall}>Loading metrics...</div>
            ) : metricsData ? (
              <div className={styles.metricsGrid}>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>{metricsData.total_classified}</span>
                  <span className={styles.metricLabel}>Total Classified</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>{metricsData.total_processed}</span>
                  <span className={styles.metricLabel}>Total Processed</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>{metricsData.cache_hits}</span>
                  <span className={styles.metricLabel}>Cache Hits</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>{metricsData.cache_misses}</span>
                  <span className={styles.metricLabel}>Cache Misses</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>
                    {metricsData.avg_classification_time_ms.toFixed(1)}ms
                  </span>
                  <span className={styles.metricLabel}>Avg Classification Time</span>
                </div>
                <div className={styles.metricItem}>
                  <span className={styles.metricValue}>
                    {metricsData.cache_hits + metricsData.cache_misses > 0 
                      ? ((metricsData.cache_hits / (metricsData.cache_hits + metricsData.cache_misses)) * 100).toFixed(1)
                      : 0}%
                  </span>
                  <span className={styles.metricLabel}>Cache Hit Rate</span>
                </div>
              </div>
            ) : (
              <p className={styles.noMetrics}>No metrics available</p>
            )}

            {metricsData?.domain_distribution && Object.keys(metricsData.domain_distribution).length > 0 && (
              <div className={styles.domainDistribution}>
                <h4 className={styles.distributionTitle}>Domain Distribution</h4>
                <div className={styles.distributionBars}>
                  {Object.entries(metricsData.domain_distribution)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 10)
                    .map(([domain, count]) => {
                      const maxCount = Math.max(...Object.values(metricsData.domain_distribution))
                      const percentage = (count / maxCount) * 100
                      return (
                        <div key={domain} className={styles.distributionItem}>
                          <span className={styles.distributionDomain}>{domain}</span>
                          <div className={styles.distributionBarContainer}>
                            <div 
                              className={styles.distributionBar} 
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <span className={styles.distributionCount}>{count}</span>
                        </div>
                      )
                    })}
                </div>
              </div>
            )}
          </SketchCard>

          {/* Available Domains & Languages */}
          <div className={styles.infoCards}>
            {domainsData && (
              <SketchCard padding="md" className={styles.infoCard}>
                <h4 className={styles.infoTitle}>Supported Domains ({domainsData.domains.length})</h4>
                <div className={styles.tagList}>
                  {domainsData.domains.slice(0, 20).map(domain => (
                    <span key={domain} className={styles.domainTag}>{domain}</span>
                  ))}
                  {domainsData.domains.length > 20 && (
                    <span className={styles.moreTag}>+{domainsData.domains.length - 20} more</span>
                  )}
                </div>
              </SketchCard>
            )}

            {languagesData && (
              <SketchCard padding="md" className={styles.infoCard}>
                <h4 className={styles.infoTitle}>Supported Languages ({languagesData.languages.length})</h4>
                <div className={styles.tagList}>
                  {languagesData.languages.map(lang => (
                    <span key={lang} className={styles.languageTag}>{lang}</span>
                  ))}
                </div>
              </SketchCard>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PromptClassifier
