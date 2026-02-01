/**
 * Self-Hosted Deployments Page
 * 
 * Manage self-hosted model deployments with cost estimation calculator and registration wizard.
 */

import { useState, useMemo } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useSelfHostedDeployments, usePublicSelfHosted, useEstimateCosts, useDeleteSelfHosted, useBaseModels } from '@/api/hooks'
import type { SelfHostedResponse, CostEstimateRequest } from '@/api/types'
import styles from './SelfHosted.module.css'

type ViewMode = 'my' | 'public'
type WizardStep = 'model' | 'hardware' | 'software' | 'costs' | 'review'

// GPU options with estimated costs
const GPU_OPTIONS = [
  { id: 'nvidia-a100-80gb', name: 'NVIDIA A100 80GB', vram: 80, costPerHour: 3.50 },
  { id: 'nvidia-a100-40gb', name: 'NVIDIA A100 40GB', vram: 40, costPerHour: 2.50 },
  { id: 'nvidia-h100-80gb', name: 'NVIDIA H100 80GB', vram: 80, costPerHour: 4.50 },
  { id: 'nvidia-a10g', name: 'NVIDIA A10G', vram: 24, costPerHour: 1.20 },
  { id: 'nvidia-l4', name: 'NVIDIA L4', vram: 24, costPerHour: 0.80 },
  { id: 'nvidia-t4', name: 'NVIDIA T4', vram: 16, costPerHour: 0.50 },
  { id: 'nvidia-v100', name: 'NVIDIA V100 16GB', vram: 16, costPerHour: 0.90 },
  { id: 'nvidia-rtx4090', name: 'NVIDIA RTX 4090', vram: 24, costPerHour: 0.70 },
]

const CLOUD_PROVIDERS = ['AWS', 'GCP', 'Azure', 'Lambda Labs', 'RunPod', 'Vast.ai', 'On-Premise']
const SERVING_FRAMEWORKS = ['vLLM', 'TGI', 'llama.cpp', 'TensorRT-LLM', 'Triton']
const QUANTIZATION_OPTIONS = ['none', 'fp16', 'bf16', 'int8', 'int4', 'AWQ', 'GPTQ', 'GGUF']

interface WizardData {
  baseModelId: string
  gpuType: string
  gpuCount: number
  cloudProvider: string
  region: string
  framework: string
  quantization: string
  maxModelLen: number
  tensorParallelSize: number
  hourlyComputeCost: number
  isPublic: boolean
}

// Registration Wizard Component
interface RegistrationWizardProps {
  onClose: () => void
  onComplete: () => void
}

function RegistrationWizard({ onClose, onComplete }: RegistrationWizardProps) {
  const [step, setStep] = useState<WizardStep>('model')
  const [data, setData] = useState<WizardData>({
    baseModelId: '',
    gpuType: 'nvidia-a100-80gb',
    gpuCount: 1,
    cloudProvider: 'AWS',
    region: 'us-east-1',
    framework: 'vLLM',
    quantization: 'none',
    maxModelLen: 8192,
    tensorParallelSize: 1,
    hourlyComputeCost: 3.50,
    isPublic: false,
  })

  const { data: baseModelsData } = useBaseModels()
  const estimateMutation = useEstimateCosts()

  const steps: WizardStep[] = ['model', 'hardware', 'software', 'costs', 'review']
  const currentStepIndex = steps.indexOf(step)

  // Calculate estimated costs
  const estimatedCosts = useMemo(() => {
    const gpu = GPU_OPTIONS.find(g => g.id === data.gpuType)
    const hourlyGpuCost = (gpu?.costPerHour || 0) * data.gpuCount
    const totalHourlyCost = data.hourlyComputeCost || hourlyGpuCost
    // Assume ~100 tokens/sec for a typical model
    const tokensPerHour = 100 * 3600
    const costPerMillion = (totalHourlyCost / tokensPerHour) * 1_000_000

    return {
      hourly: totalHourlyCost,
      inputPerMillion: costPerMillion * 0.7, // Input is cheaper
      outputPerMillion: costPerMillion, // Output is more expensive
      vramRequired: gpu ? gpu.vram * data.gpuCount : 0,
    }
  }, [data])

  const handleNext = () => {
    const nextIndex = currentStepIndex + 1
    if (nextIndex < steps.length) {
      setStep(steps[nextIndex])
    }
  }

  const handleBack = () => {
    const prevIndex = currentStepIndex - 1
    if (prevIndex >= 0) {
      setStep(steps[prevIndex])
    }
  }

  const handleSubmit = () => {
    // In a real implementation, this would call the API to create the deployment
    console.log('Submitting deployment:', data)
    onComplete()
  }

  const selectedGpu = GPU_OPTIONS.find(g => g.id === data.gpuType)

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.wizardModal} onClick={e => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>Register Self-Hosted Deployment</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        {/* Progress Steps */}
        <div className={styles.wizardProgress}>
          {steps.map((s, idx) => (
            <div 
              key={s} 
              className={`${styles.progressStep} ${idx <= currentStepIndex ? styles.active : ''} ${idx < currentStepIndex ? styles.completed : ''}`}
            >
              <div className={styles.stepNumber}>{idx + 1}</div>
              <div className={styles.stepName}>
                {s === 'model' && 'Model'}
                {s === 'hardware' && 'Hardware'}
                {s === 'software' && 'Software'}
                {s === 'costs' && 'Costs'}
                {s === 'review' && 'Review'}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.wizardContent}>
          {/* Step 1: Model Selection */}
          {step === 'model' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Select Base Model</h3>
              <p className={styles.stepDesc}>Choose the base model you're hosting</p>

              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Base Model</label>
                <select
                  className={styles.formSelect}
                  value={data.baseModelId}
                  onChange={(e) => setData({ ...data, baseModelId: e.target.value })}
                >
                  <option value="">Select a model...</option>
                  {baseModelsData?.models?.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                  <option value="custom">Custom / Other</option>
                </select>
              </div>

              {data.baseModelId === 'custom' && (
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Custom Model ID</label>
                  <input
                    type="text"
                    className={styles.formInput}
                    placeholder="e.g., meta-llama/Llama-3-70B"
                    value={data.baseModelId}
                    onChange={(e) => setData({ ...data, baseModelId: e.target.value })}
                  />
                </div>
              )}

              <div className={styles.formGroup}>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={data.isPublic}
                    onChange={(e) => setData({ ...data, isPublic: e.target.checked })}
                  />
                  Make deployment public (visible in public leaderboard)
                </label>
              </div>
            </div>
          )}

          {/* Step 2: Hardware Configuration */}
          {step === 'hardware' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Hardware Configuration</h3>
              <p className={styles.stepDesc}>Specify your GPU and infrastructure setup</p>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>GPU Type</label>
                  <select
                    className={styles.formSelect}
                    value={data.gpuType}
                    onChange={(e) => setData({ ...data, gpuType: e.target.value })}
                  >
                    {GPU_OPTIONS.map(gpu => (
                      <option key={gpu.id} value={gpu.id}>
                        {gpu.name} ({gpu.vram}GB) - ~${gpu.costPerHour}/hr
                      </option>
                    ))}
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>GPU Count</label>
                  <input
                    type="number"
                    className={styles.formInput}
                    min={1}
                    max={8}
                    value={data.gpuCount}
                    onChange={(e) => setData({ ...data, gpuCount: parseInt(e.target.value) || 1 })}
                  />
                </div>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Cloud Provider</label>
                  <select
                    className={styles.formSelect}
                    value={data.cloudProvider}
                    onChange={(e) => setData({ ...data, cloudProvider: e.target.value })}
                  >
                    {CLOUD_PROVIDERS.map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Region</label>
                  <input
                    type="text"
                    className={styles.formInput}
                    placeholder="e.g., us-east-1"
                    value={data.region}
                    onChange={(e) => setData({ ...data, region: e.target.value })}
                  />
                </div>
              </div>

              <div className={styles.hardwareSummary}>
                <h4>Configuration Summary</h4>
                <div className={styles.summaryGrid}>
                  <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>Total VRAM</span>
                    <span className={styles.summaryValue}>{(selectedGpu?.vram || 0) * data.gpuCount} GB</span>
                  </div>
                  <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>Est. GPU Cost</span>
                    <span className={styles.summaryValue}>${((selectedGpu?.costPerHour || 0) * data.gpuCount).toFixed(2)}/hr</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Software Configuration */}
          {step === 'software' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Software Configuration</h3>
              <p className={styles.stepDesc}>Configure serving framework and model settings</p>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Serving Framework</label>
                  <select
                    className={styles.formSelect}
                    value={data.framework}
                    onChange={(e) => setData({ ...data, framework: e.target.value })}
                  >
                    {SERVING_FRAMEWORKS.map(f => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Quantization</label>
                  <select
                    className={styles.formSelect}
                    value={data.quantization}
                    onChange={(e) => setData({ ...data, quantization: e.target.value })}
                  >
                    {QUANTIZATION_OPTIONS.map(q => (
                      <option key={q} value={q}>{q === 'none' ? 'None (Full Precision)' : q.toUpperCase()}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Max Context Length</label>
                  <input
                    type="number"
                    className={styles.formInput}
                    min={512}
                    max={200000}
                    step={512}
                    value={data.maxModelLen}
                    onChange={(e) => setData({ ...data, maxModelLen: parseInt(e.target.value) || 8192 })}
                  />
                  <span className={styles.formHint}>Maximum tokens in context window</span>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Tensor Parallel Size</label>
                  <input
                    type="number"
                    className={styles.formInput}
                    min={1}
                    max={data.gpuCount}
                    value={data.tensorParallelSize}
                    onChange={(e) => setData({ ...data, tensorParallelSize: parseInt(e.target.value) || 1 })}
                  />
                  <span className={styles.formHint}>Number of GPUs for tensor parallelism</span>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Cost Configuration */}
          {step === 'costs' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Cost Configuration</h3>
              <p className={styles.stepDesc}>Enter your actual compute costs for accurate per-token pricing</p>

              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Hourly Compute Cost ($)</label>
                <input
                  type="number"
                  className={styles.formInput}
                  min={0}
                  step={0.01}
                  value={data.hourlyComputeCost}
                  onChange={(e) => setData({ ...data, hourlyComputeCost: parseFloat(e.target.value) || 0 })}
                />
                <span className={styles.formHint}>Total hourly cost including all infrastructure</span>
              </div>

              <div className={styles.costEstimation}>
                <h4>Estimated Per-Token Costs</h4>
                <p className={styles.costNote}>Based on ~100 tokens/second throughput</p>
                <div className={styles.costCards}>
                  <div className={styles.costCard}>
                    <span className={styles.costLabel}>Input</span>
                    <span className={styles.costValue}>${estimatedCosts.inputPerMillion.toFixed(4)}</span>
                    <span className={styles.costUnit}>per 1M tokens</span>
                  </div>
                  <div className={styles.costCard}>
                    <span className={styles.costLabel}>Output</span>
                    <span className={styles.costValue}>${estimatedCosts.outputPerMillion.toFixed(4)}</span>
                    <span className={styles.costUnit}>per 1M tokens</span>
                  </div>
                </div>
              </div>

              <StickyNote title="Cloud Comparison" color="yellow" rotation={0}>
                <p>Typical cloud provider costs for similar models:</p>
                <ul className={styles.comparisonList}>
                  <li>OpenAI GPT-4: ~$30/1M input, ~$60/1M output</li>
                  <li>Anthropic Claude: ~$15/1M input, ~$75/1M output</li>
                  <li>Together AI: ~$2/1M input, ~$6/1M output</li>
                </ul>
              </StickyNote>
            </div>
          )}

          {/* Step 5: Review */}
          {step === 'review' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Review Configuration</h3>
              <p className={styles.stepDesc}>Verify your deployment configuration</p>

              <div className={styles.reviewGrid}>
                <div className={styles.reviewSection}>
                  <h4>Model</h4>
                  <p>{data.baseModelId || 'Not selected'}</p>
                  <p>Visibility: {data.isPublic ? 'Public' : 'Private'}</p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>Hardware</h4>
                  <p>{data.gpuCount}× {selectedGpu?.name}</p>
                  <p>{data.cloudProvider} / {data.region}</p>
                  <p>Total VRAM: {(selectedGpu?.vram || 0) * data.gpuCount} GB</p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>Software</h4>
                  <p>Framework: {data.framework}</p>
                  <p>Quantization: {data.quantization}</p>
                  <p>Max Context: {data.maxModelLen.toLocaleString()} tokens</p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>Costs</h4>
                  <p>Hourly: ${data.hourlyComputeCost.toFixed(2)}</p>
                  <p>Input: ${estimatedCosts.inputPerMillion.toFixed(4)}/1M</p>
                  <p>Output: ${estimatedCosts.outputPerMillion.toFixed(4)}/1M</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Navigation Buttons */}
        <div className={styles.wizardFooter}>
          <button
            className={styles.wizardBtn}
            onClick={handleBack}
            disabled={currentStepIndex === 0}
          >
            ← Back
          </button>

          <div className={styles.stepIndicator}>
            Step {currentStepIndex + 1} of {steps.length}
          </div>

          {step !== 'review' ? (
            <button
              className={`${styles.wizardBtn} ${styles.primary}`}
              onClick={handleNext}
              disabled={step === 'model' && !data.baseModelId}
            >
              Next →
            </button>
          ) : (
            <button
              className={`${styles.wizardBtn} ${styles.primary}`}
              onClick={handleSubmit}
            >
              Create Deployment
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

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
  const [showWizard, setShowWizard] = useState(false)
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

  const handleWizardComplete = () => {
    setShowWizard(false)
    myRefetch()
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
              Register, manage, and compare your self-hosted model deployments
            </p>
          </div>
          <div className={styles.headerActions}>
            <SketchButton variant="secondary" onClick={() => setShowCalculator(true)}>
              <i className="ph ph-calculator"></i> Cost Calculator
            </SketchButton>
            <SketchButton variant="primary" onClick={() => setShowWizard(true)}>
              <i className="ph ph-plus"></i> Register Deployment
            </SketchButton>
          </div>
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
          icon="ph ph-desktop"
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

      {/* Registration Wizard */}
      {showWizard && (
        <RegistrationWizard 
          onClose={() => setShowWizard(false)} 
          onComplete={handleWizardComplete}
        />
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
