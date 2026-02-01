/**
 * Self-Hosted Deployments Page
 * 
 * Manage self-hosted model deployments with cost estimation calculator.
 */

import { useState } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useSelfHostedDeployments, usePublicSelfHosted, useEstimateCosts, useDeleteSelfHosted } from '@/api/hooks'
import type { SelfHostedResponse, CostEstimateRequest } from '@/api/types'
import styles from './SelfHosted.module.css'

type ViewMode = 'my' | 'public'

interface CostCalculatorProps {
  onClose: () => void
}

function CostCalculator({ onClose }: CostCalculatorProps) {
  const [hourlyComputeCost, setHourlyComputeCost] = useState('')
  const [throughput, setThroughput] = useState('')
  const [utilization, setUtilization] = useState('80')
  
  const estimateMutation = useEstimateCosts()

  const handleEstimate = () => {
    if (!hourlyComputeCost || !throughput) return
    
    const request: CostEstimateRequest = {
      hourly_compute_cost: parseFloat(hourlyComputeCost),
      expected_throughput_tps: parseFloat(throughput),
      utilization: parseFloat(utilization) / 100,
    }
    
    estimateMutation.mutate(request)
  }

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.calculatorModal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>Cost Calculator</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalContent}>
          <p className={styles.calcDescription}>
            Estimate per-token costs for your self-hosted deployment based on compute costs and expected throughput.
          </p>

          <div className={styles.calcForm}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Hourly Compute Cost ($)</label>
              <input
                type="number"
                className={styles.formInput}
                placeholder="e.g. 2.50"
                value={hourlyComputeCost}
                onChange={(e) => setHourlyComputeCost(e.target.value)}
                min="0"
                step="0.01"
              />
              <span className={styles.formHint}>Total hourly cost for your GPU instance</span>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Expected Throughput (tokens/sec)</label>
              <input
                type="number"
                className={styles.formInput}
                placeholder="e.g. 100"
                value={throughput}
                onChange={(e) => setThroughput(e.target.value)}
                min="0"
              />
              <span className={styles.formHint}>Average tokens generated per second</span>
            </div>

            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Utilization (%)</label>
              <input
                type="number"
                className={styles.formInput}
                value={utilization}
                onChange={(e) => setUtilization(e.target.value)}
                min="0"
                max="100"
              />
              <span className={styles.formHint}>Expected GPU utilization during inference</span>
            </div>

            <SketchButton
              variant="primary"
              onClick={handleEstimate}
              disabled={!hourlyComputeCost || !throughput || estimateMutation.isPending}
            >
              {estimateMutation.isPending ? 'Calculating...' : 'Calculate Costs'}
            </SketchButton>
          </div>

          {estimateMutation.data && (
            <div className={styles.calcResults}>
              <h3 className={styles.resultsTitle}>Estimated Costs</h3>
              <div className={styles.resultCards}>
                <div className={styles.resultCard}>
                  <span className={styles.resultLabel}>Input Cost</span>
                  <span className={styles.resultValue}>
                    ${estimateMutation.data.input_cost_per_million.toFixed(4)}
                  </span>
                  <span className={styles.resultUnit}>per 1M tokens</span>
                </div>
                <div className={styles.resultCard}>
                  <span className={styles.resultLabel}>Output Cost</span>
                  <span className={styles.resultValue}>
                    ${estimateMutation.data.output_cost_per_million.toFixed(4)}
                  </span>
                  <span className={styles.resultUnit}>per 1M tokens</span>
                </div>
              </div>
              {estimateMutation.data.assumptions && (
                <div className={styles.assumptions}>
                  <span className={styles.assumptionsLabel}>Assumptions:</span>
                  <ul>
                    {Object.entries(estimateMutation.data.assumptions).map(([key, value]) => (
                      <li key={key}>{key}: {String(value)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {estimateMutation.error && (
            <div className={styles.calcError}>
              Failed to calculate costs. Please try again.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface DeploymentDetailModalProps {
  deployment: SelfHostedResponse
  onClose: () => void
  onDelete?: (id: string) => void
  isOwner: boolean
}

function DeploymentDetailModal({ deployment, onClose, onDelete, isOwner }: DeploymentDetailModalProps) {
  const [confirmDelete, setConfirmDelete] = useState(false)

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>Self-Hosted Deployment</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.modalContent}>
          {/* Basic Info */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Overview</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>ID</span>
                <span className={styles.detailValue}>{deployment.id}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Base Model</span>
                <span className={styles.detailValue}>{deployment.base_model_id}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Visibility</span>
                <span className={styles.detailValue}>{deployment.is_public ? 'Public' : 'Private'}</span>
              </div>
            </div>
          </section>

          {/* Hardware */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Hardware Configuration</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>GPU Type</span>
                <span className={styles.detailValue}>{deployment.hardware.gpu_type}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>GPU Count</span>
                <span className={styles.detailValue}>{deployment.hardware.gpu_count}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>GPU Memory</span>
                <span className={styles.detailValue}>{deployment.hardware.gpu_memory_gb} GB</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>CPU Cores</span>
                <span className={styles.detailValue}>{deployment.hardware.cpu_cores}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>RAM</span>
                <span className={styles.detailValue}>{deployment.hardware.ram_gb} GB</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Cloud Provider</span>
                <span className={styles.detailValue}>{deployment.hardware.cloud_provider}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Instance Type</span>
                <span className={styles.detailValue}>{deployment.hardware.instance_type}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Region</span>
                <span className={styles.detailValue}>{deployment.hardware.region}</span>
              </div>
            </div>
          </section>

          {/* Software */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Software Configuration</h3>
            <div className={styles.detailGrid}>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Framework</span>
                <span className={styles.detailValue}>{deployment.software.serving_framework}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Version</span>
                <span className={styles.detailValue}>{deployment.software.framework_version}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Quantization</span>
                <span className={styles.detailValue}>{deployment.software.quantization}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Tensor Parallel</span>
                <span className={styles.detailValue}>{deployment.software.tensor_parallel_size}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>Max Model Len</span>
                <span className={styles.detailValue}>{deployment.software.max_model_len.toLocaleString()}</span>
              </div>
              <div className={styles.detailItem}>
                <span className={styles.detailLabel}>GPU Utilization</span>
                <span className={styles.detailValue}>{(deployment.software.gpu_memory_utilization * 100).toFixed(0)}%</span>
              </div>
            </div>
          </section>

          {/* Costs */}
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Costs</h3>
            <div className={styles.costCards}>
              <div className={styles.costCard}>
                <span className={styles.costLabel}>Input</span>
                <span className={styles.costValue}>${deployment.costs.input_cost_per_million.toFixed(4)}</span>
                <span className={styles.costUnit}>per 1M tokens</span>
              </div>
              <div className={styles.costCard}>
                <span className={styles.costLabel}>Output</span>
                <span className={styles.costValue}>${deployment.costs.output_cost_per_million.toFixed(4)}</span>
                <span className={styles.costUnit}>per 1M tokens</span>
              </div>
              <div className={styles.costCard}>
                <span className={styles.costLabel}>Compute</span>
                <span className={styles.costValue}>${deployment.costs.hourly_compute_cost.toFixed(2)}</span>
                <span className={styles.costUnit}>per hour</span>
              </div>
            </div>
            {deployment.costs.notes && (
              <p className={styles.costNotes}>{deployment.costs.notes}</p>
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

          {/* Delete Section */}
          {isOwner && onDelete && (
            <section className={styles.section}>
              <h3 className={styles.sectionTitle}>Danger Zone</h3>
              {!confirmDelete ? (
                <SketchButton
                  variant="secondary"
                  onClick={() => setConfirmDelete(true)}
                >
                  Delete Deployment
                </SketchButton>
              ) : (
                <div className={styles.confirmDelete}>
                  <p>Are you sure you want to delete this deployment? This action cannot be undone.</p>
                  <div className={styles.confirmButtons}>
                    <SketchButton
                      variant="secondary"
                      onClick={() => setConfirmDelete(false)}
                    >
                      Cancel
                    </SketchButton>
                    <SketchButton
                      variant="primary"
                      onClick={() => onDelete(deployment.id)}
                    >
                      Yes, Delete
                    </SketchButton>
                  </div>
                </div>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

export function SelfHosted() {
  const [viewMode, setViewMode] = useState<ViewMode>('my')
  const [showCalculator, setShowCalculator] = useState(false)
  const [selectedDeployment, setSelectedDeployment] = useState<SelfHostedResponse | null>(null)

  const { data: myData, isLoading: myLoading, error: myError, refetch: myRefetch } = useSelfHostedDeployments()
  const { data: publicData, isLoading: publicLoading, error: publicError, refetch: publicRefetch } = usePublicSelfHosted()
  
  const deleteMutation = useDeleteSelfHosted()

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => {
        setSelectedDeployment(null)
        myRefetch()
      }
    })
  }

  const isLoading = viewMode === 'my' ? myLoading : publicLoading
  const error = viewMode === 'my' ? myError : publicError
  const data = viewMode === 'my' ? myData : publicData
  const refetch = viewMode === 'my' ? myRefetch : publicRefetch

  if (isLoading) {
    return <LoadingState message="Loading self-hosted deployments..." />
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

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerTop}>
          <div>
            <h1 className={styles.title}>Self-Hosted Deployments</h1>
            <p className={styles.subtitle}>
              Manage your self-hosted model deployments and compare costs
            </p>
          </div>
          <SketchButton variant="primary" onClick={() => setShowCalculator(true)}>
            Cost Calculator
          </SketchButton>
        </div>
      </header>

      {/* View Toggle */}
      <SketchCard padding="md" className={styles.viewToggle}>
        <div className={styles.toggleButtons}>
          <button
            className={`${styles.toggleButton} ${viewMode === 'my' ? styles.active : ''}`}
            onClick={() => setViewMode('my')}
          >
            My Deployments
          </button>
          <button
            className={`${styles.toggleButton} ${viewMode === 'public' ? styles.active : ''}`}
            onClick={() => setViewMode('public')}
          >
            Public Deployments
          </button>
        </div>
      </SketchCard>

      {/* Deployments Grid */}
      {deployments.length === 0 ? (
        <EmptyState
          title={viewMode === 'my' ? "No deployments yet" : "No public deployments"}
          message={viewMode === 'my' ? "Register your self-hosted deployment to track its performance" : "No users have shared public deployments"}
          icon="🖥️"
        />
      ) : (
        <div className={styles.deploymentsGrid}>
          {deployments.map(deployment => (
            <SketchCard
              key={deployment.id}
              padding="md"
              className={styles.deploymentCard}
              onClick={() => setSelectedDeployment(deployment)}
            >
              <div className={styles.deploymentHeader}>
                <h3 className={styles.deploymentName}>{deployment.base_model_id}</h3>
                <span className={deployment.is_public ? styles.publicBadge : styles.privateBadge}>
                  {deployment.is_public ? 'Public' : 'Private'}
                </span>
              </div>

              <div className={styles.hardwareInfo}>
                <span className={styles.gpuInfo}>
                  {deployment.hardware.gpu_count}× {deployment.hardware.gpu_type}
                </span>
                <span className={styles.cloudInfo}>
                  {deployment.hardware.cloud_provider} / {deployment.hardware.region}
                </span>
              </div>

              <div className={styles.deploymentStats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Rating</span>
                  <span className={styles.statValue}>{deployment.trueskill.raw.mu.toFixed(0)}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Cost/Hr</span>
                  <span className={styles.statValue}>${deployment.costs.hourly_compute_cost.toFixed(2)}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Quant</span>
                  <span className={styles.statValue}>{deployment.software.quantization}</span>
                </div>
              </div>

              <div className={styles.costInfo}>
                <span>${deployment.costs.input_cost_per_million.toFixed(4)} / ${deployment.costs.output_cost_per_million.toFixed(4)} per 1M</span>
              </div>

              <div className={styles.softwareInfo}>
                <span className={styles.frameworkBadge}>{deployment.software.serving_framework}</span>
                <span className={styles.contextLength}>
                  {(deployment.software.max_model_len / 1000).toFixed(0)}K ctx
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

      {/* Cost Calculator Modal */}
      {showCalculator && (
        <CostCalculator onClose={() => setShowCalculator(false)} />
      )}

      {/* Deployment Detail Modal */}
      {selectedDeployment && (
        <DeploymentDetailModal
          deployment={selectedDeployment}
          onClose={() => setSelectedDeployment(null)}
          onDelete={viewMode === 'my' ? handleDelete : undefined}
          isOwner={viewMode === 'my'}
        />
      )}
    </div>
  )
}

export default SelfHosted
