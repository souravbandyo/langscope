/**
 * Register Model Wizard
 * 
 * Smart wizard that uses the ModelType mapper to dynamically configure
 * registration fields based on the selected model type.
 */

import { useState, useMemo } from 'react'
import { SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useCreateMyModel, useTestModelConnection, useBaseModels } from '@/api/hooks'
import {
  MODEL_TYPE_CONFIGS,
  getModelTypesByCategory,
  getModelTypeIcon,
  getModelTypeDisplayName,
  getPrimaryMetric,
  supportsGroundTruth,
  supportsSubjective,
  type ModelType,
  type APIFieldConfig,
} from '@/types/modelTypes'
import type { UserModelCreate, ModelAPIConfig, ModelTypeSpecificConfig, ModelCosts } from '@/api/types'
import styles from './RegisterModelWizard.module.css'

type WizardStep = 'type' | 'base' | 'api' | 'costs' | 'review'

interface WizardData {
  modelType: ModelType | null
  name: string
  description: string
  version: string
  baseModelId: string
  apiConfig: Partial<ModelAPIConfig>
  typeConfig: Partial<ModelTypeSpecificConfig>
  costs: ModelCosts
  isPublic: boolean
}

const initialData: WizardData = {
  modelType: null,
  name: '',
  description: '',
  version: '1.0',
  baseModelId: '',
  apiConfig: {
    apiFormat: 'openai',
  },
  typeConfig: {},
  costs: {
    inputCostPerMillion: 0,
    outputCostPerMillion: 0,
    currency: 'USD',
    isEstimate: true,
  },
  isPublic: false,
}

interface Props {
  onClose: () => void
  onComplete: () => void
}

export function RegisterModelWizard({ onClose, onComplete }: Props) {
  const [step, setStep] = useState<WizardStep>('type')
  const [data, setData] = useState<WizardData>(initialData)
  const [connectionTest, setConnectionTest] = useState<{
    status: 'idle' | 'testing' | 'success' | 'error'
    latency?: number
    error?: string
  }>({ status: 'idle' })

  const createMutation = useCreateMyModel()
  const testConnectionMutation = useTestModelConnection()
  const { data: baseModelsData } = useBaseModels()

  const steps: WizardStep[] = ['type', 'base', 'api', 'costs', 'review']
  const currentStepIndex = steps.indexOf(step)
  const modelTypeCategories = getModelTypesByCategory()

  // Get the config for selected model type
  const selectedTypeConfig = data.modelType ? MODEL_TYPE_CONFIGS[data.modelType] : null

  // Filter base models by type compatibility (simplified - just filter by capabilities)
  const filteredBaseModels = useMemo(() => {
    if (!baseModelsData?.models || !data.modelType) return []
    // For now, show all base models - in production, filter by type
    return baseModelsData.models
  }, [baseModelsData, data.modelType])

  const handleTypeSelect = (type: ModelType) => {
    setData(prev => ({
      ...prev,
      modelType: type,
      // Reset type-specific fields
      typeConfig: {},
      apiConfig: {
        ...prev.apiConfig,
        apiFormat: type === 'LLM' ? 'openai' : 'custom',
      }
    }))
  }

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

  const handleTestConnection = async () => {
    if (!data.apiConfig.endpoint || !data.apiConfig.apiKey || !data.apiConfig.modelId) {
      setConnectionTest({ status: 'error', error: 'Please fill in all required fields' })
      return
    }

    setConnectionTest({ status: 'testing' })

    try {
      const result = await testConnectionMutation.mutateAsync({
        endpoint: data.apiConfig.endpoint,
        apiKey: data.apiConfig.apiKey,
        modelId: data.apiConfig.modelId,
        apiFormat: data.apiConfig.apiFormat || 'openai',
      })

      if (result.success) {
        setConnectionTest({ status: 'success', latency: result.latencyMs })
      } else {
        setConnectionTest({ status: 'error', error: result.error })
      }
    } catch (err) {
      setConnectionTest({ 
        status: 'error', 
        error: err instanceof Error ? err.message : 'Connection failed' 
      })
    }
  }

  const handleSubmit = async () => {
    if (!data.modelType || !data.name || !data.apiConfig.endpoint) {
      return
    }

    const createData: UserModelCreate = {
      name: data.name,
      description: data.description || undefined,
      modelType: data.modelType,
      version: data.version,
      baseModelId: data.baseModelId || undefined,
      apiConfig: data.apiConfig as ModelAPIConfig,
      typeConfig: data.typeConfig as ModelTypeSpecificConfig,
      costs: data.costs,
      isPublic: data.isPublic,
    }

    try {
      await createMutation.mutateAsync(createData)
      onComplete()
    } catch (err) {
      console.error('Failed to create model:', err)
    }
  }

  const canProceed = useMemo(() => {
    switch (step) {
      case 'type':
        return data.modelType !== null
      case 'base':
        return data.name.trim().length > 0
      case 'api':
        return data.apiConfig.endpoint && data.apiConfig.modelId
      case 'costs':
        return true
      case 'review':
        return true
      default:
        return false
    }
  }, [step, data])

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>Register New Model</h2>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        {/* Progress Steps */}
        <div className={styles.progress}>
          {steps.map((s, idx) => (
            <div 
              key={s}
              className={`${styles.progressStep} ${idx <= currentStepIndex ? styles.active : ''} ${idx < currentStepIndex ? styles.completed : ''}`}
            >
              <div className={styles.stepNumber}>{idx + 1}</div>
              <div className={styles.stepName}>
                {s === 'type' && 'Type'}
                {s === 'base' && 'Details'}
                {s === 'api' && 'API'}
                {s === 'costs' && 'Costs'}
                {s === 'review' && 'Review'}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.content}>
          {/* Step 1: Model Type Selection */}
          {step === 'type' && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Select Model Type</h3>
              <p className={styles.stepDesc}>
                Choose the type of AI model you want to register. This determines the evaluation metrics and domains available.
              </p>

              <div className={styles.typeGrid}>
                {Object.entries(modelTypeCategories).map(([category, types]) => (
                  <div key={category} className={styles.typeCategory}>
                    <h4 className={styles.categoryTitle}>{category}</h4>
                    <div className={styles.typeOptions}>
                      {types.map(type => {
                        const config = MODEL_TYPE_CONFIGS[type]
                        return (
                          <button
                            key={type}
                            className={`${styles.typeOption} ${data.modelType === type ? styles.selected : ''}`}
                            onClick={() => handleTypeSelect(type)}
                          >
                            <i className={`${config.icon} ${styles.typeIcon}`}></i>
                            <div className={styles.typeInfo}>
                              <span className={styles.typeName}>{config.displayName}</span>
                              <span className={styles.typeDesc}>{config.description}</span>
                            </div>
                            {data.modelType === type && (
                              <span className={styles.checkmark}><i className="ph ph-check"></i></span>
                            )}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {selectedTypeConfig && (
                <StickyNote title="Selected Type Info" color="blue" rotation={1}>
                  <div className={styles.typeDetails}>
                    <p><strong>Input:</strong> {selectedTypeConfig.inputFormat}</p>
                    <p><strong>Output:</strong> {selectedTypeConfig.outputFormat}</p>
                    <p><strong>Evaluation:</strong> {
                      supportsGroundTruth(data.modelType!) && supportsSubjective(data.modelType!)
                        ? 'Ground Truth + Arena Battles'
                        : supportsGroundTruth(data.modelType!)
                        ? 'Ground Truth Only'
                        : 'Arena Battles Only'
                    }</p>
                    <p><strong>Primary Metric:</strong> {getPrimaryMetric(data.modelType!)?.name}</p>
                  </div>
                </StickyNote>
              )}
            </div>
          )}

          {/* Step 2: Base Model & Details */}
          {step === 'base' && selectedTypeConfig && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Model Details</h3>
              <p className={styles.stepDesc}>
                Provide basic information about your model.
              </p>

              <div className={styles.formGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.label}>Model Name *</label>
                  <input
                    type="text"
                    className={styles.input}
                    placeholder="My Custom GPT-4 Deployment"
                    value={data.name}
                    onChange={e => setData(prev => ({ ...prev, name: e.target.value }))}
                  />
                  <span className={styles.hint}>A descriptive name for your model</span>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.label}>Version</label>
                  <input
                    type="text"
                    className={styles.input}
                    placeholder="1.0"
                    value={data.version}
                    onChange={e => setData(prev => ({ ...prev, version: e.target.value }))}
                  />
                  <span className={styles.hint}>Version identifier (e.g., 1.0, 2.0-beta)</span>
                </div>

                <div className={`${styles.formGroup} ${styles.fullWidth}`}>
                  <label className={styles.label}>Description</label>
                  <textarea
                    className={styles.textarea}
                    placeholder="Optional description of this model..."
                    value={data.description}
                    onChange={e => setData(prev => ({ ...prev, description: e.target.value }))}
                    rows={3}
                  />
                </div>

                <div className={`${styles.formGroup} ${styles.fullWidth}`}>
                  <label className={styles.label}>Base Model (Optional)</label>
                  <select
                    className={styles.select}
                    value={data.baseModelId}
                    onChange={e => setData(prev => ({ ...prev, baseModelId: e.target.value }))}
                  >
                    <option value="">Select a base model...</option>
                    {filteredBaseModels.map(model => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.organization})
                      </option>
                    ))}
                    <option value="custom">Custom / Not Listed</option>
                  </select>
                  <span className={styles.hint}>
                    Link to an existing base model for benchmark comparison
                  </span>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={data.isPublic}
                      onChange={e => setData(prev => ({ ...prev, isPublic: e.target.checked }))}
                    />
                    Make model public
                  </label>
                  <span className={styles.hint}>
                    Public models appear on leaderboards (others cannot access your API)
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: API Configuration */}
          {step === 'api' && selectedTypeConfig && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>API Configuration</h3>
              <p className={styles.stepDesc}>
                Configure how to connect to your model. API keys are stored securely and never shared.
              </p>

              <div className={styles.formGrid}>
                {selectedTypeConfig.apiFields.map(field => (
                  <APIField
                    key={field.name}
                    field={field}
                    value={data.apiConfig[field.name as keyof ModelAPIConfig] ?? data.typeConfig[field.name as keyof ModelTypeSpecificConfig] ?? ''}
                    onChange={(value) => {
                      if (['endpoint', 'apiKey', 'modelId', 'apiFormat'].includes(field.name)) {
                        setData(prev => ({
                          ...prev,
                          apiConfig: { ...prev.apiConfig, [field.name]: value }
                        }))
                      } else {
                        setData(prev => ({
                          ...prev,
                          typeConfig: { ...prev.typeConfig, [field.name]: value }
                        }))
                      }
                    }}
                  />
                ))}
              </div>

              {/* Connection Test */}
              <div className={styles.connectionTest}>
                <SketchButton
                  variant="secondary"
                  onClick={handleTestConnection}
                  disabled={connectionTest.status === 'testing'}
                >
                  {connectionTest.status === 'testing' ? 'Testing...' : 'Test Connection'}
                </SketchButton>

                {connectionTest.status === 'success' && (
                  <span className={styles.testSuccess}>
                    <i className="ph ph-check-circle"></i> Connected ({connectionTest.latency}ms latency)
                  </span>
                )}
                {connectionTest.status === 'error' && (
                  <span className={styles.testError}>
                    <i className="ph ph-x-circle"></i> {connectionTest.error}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Step 4: Cost Configuration */}
          {step === 'costs' && selectedTypeConfig && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Cost Configuration</h3>
              <p className={styles.stepDesc}>
                Enter pricing information for cost-adjusted rankings and comparisons.
              </p>

              <div className={styles.formGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.label}>Input Cost (per 1M tokens)</label>
                  <div className={styles.inputWithUnit}>
                    <span className={styles.unit}>$</span>
                    <input
                      type="number"
                      className={styles.input}
                      placeholder="0.00"
                      step="0.01"
                      min="0"
                      value={data.costs.inputCostPerMillion || ''}
                      onChange={e => setData(prev => ({
                        ...prev,
                        costs: { ...prev.costs, inputCostPerMillion: parseFloat(e.target.value) || 0 }
                      }))}
                    />
                  </div>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.label}>Output Cost (per 1M tokens)</label>
                  <div className={styles.inputWithUnit}>
                    <span className={styles.unit}>$</span>
                    <input
                      type="number"
                      className={styles.input}
                      placeholder="0.00"
                      step="0.01"
                      min="0"
                      value={data.costs.outputCostPerMillion || ''}
                      onChange={e => setData(prev => ({
                        ...prev,
                        costs: { ...prev.costs, outputCostPerMillion: parseFloat(e.target.value) || 0 }
                      }))}
                    />
                  </div>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={data.costs.isEstimate}
                      onChange={e => setData(prev => ({
                        ...prev,
                        costs: { ...prev.costs, isEstimate: e.target.checked }
                      }))}
                    />
                    These are estimated costs
                  </label>
                </div>

                <div className={`${styles.formGroup} ${styles.fullWidth}`}>
                  <label className={styles.label}>Cost Notes (Optional)</label>
                  <textarea
                    className={styles.textarea}
                    placeholder="Any notes about pricing (e.g., volume discounts, self-hosted costs...)"
                    value={data.costs.notes || ''}
                    onChange={e => setData(prev => ({
                      ...prev,
                      costs: { ...prev.costs, notes: e.target.value }
                    }))}
                    rows={2}
                  />
                </div>
              </div>

              <StickyNote title="Pricing Reference" color="yellow" rotation={-1}>
                <div className={styles.pricingRef}>
                  <p><strong>Typical Cloud Pricing:</strong></p>
                  <ul>
                    <li>GPT-4o: $2.50 / $10.00 per 1M</li>
                    <li>Claude 3.5: $3.00 / $15.00 per 1M</li>
                    <li>Llama 3.1 70B: $0.59 / $0.79 per 1M</li>
                  </ul>
                </div>
              </StickyNote>
            </div>
          )}

          {/* Step 5: Review */}
          {step === 'review' && selectedTypeConfig && (
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Review & Create</h3>
              <p className={styles.stepDesc}>
                Review your model configuration before creating.
              </p>

              <div className={styles.reviewGrid}>
                <div className={styles.reviewSection}>
                  <h4>Model Type</h4>
                  <p>
                    <i className={`${selectedTypeConfig.icon} ${styles.reviewIcon}`}></i>
                    {selectedTypeConfig.displayName}
                  </p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>Details</h4>
                  <p><strong>Name:</strong> {data.name}</p>
                  <p><strong>Version:</strong> {data.version}</p>
                  {data.baseModelId && <p><strong>Base Model:</strong> {data.baseModelId}</p>}
                  <p><strong>Visibility:</strong> {data.isPublic ? 'Public' : 'Private'}</p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>API</h4>
                  <p><strong>Endpoint:</strong> {data.apiConfig.endpoint}</p>
                  <p><strong>Model ID:</strong> {data.apiConfig.modelId}</p>
                  <p><strong>Format:</strong> {data.apiConfig.apiFormat}</p>
                  <p><strong>API Key:</strong> ••••••••{data.apiConfig.apiKey?.slice(-4) || ''}</p>
                </div>

                <div className={styles.reviewSection}>
                  <h4>Costs</h4>
                  <p>
                    <strong>Input:</strong> ${data.costs.inputCostPerMillion.toFixed(2)}/1M
                  </p>
                  <p>
                    <strong>Output:</strong> ${data.costs.outputCostPerMillion.toFixed(2)}/1M
                  </p>
                  {data.costs.isEstimate && <p className={styles.estimate}>(Estimated)</p>}
                </div>

                <div className={styles.reviewSection}>
                  <h4>Evaluation</h4>
                  <p><strong>Primary Metric:</strong> {getPrimaryMetric(data.modelType!)?.name}</p>
                  <p><strong>Available Domains:</strong></p>
                  <div className={styles.domainList}>
                    {selectedTypeConfig.groundTruthDomains.length > 0 && (
                      <span className={styles.domainBadge}>Ground Truth: {selectedTypeConfig.groundTruthDomains.length}</span>
                    )}
                    {selectedTypeConfig.subjectiveDomains.length > 0 && (
                      <span className={styles.domainBadge}>Arena: {selectedTypeConfig.subjectiveDomains.length}</span>
                    )}
                  </div>
                </div>
              </div>

              {createMutation.error && (
                <div className={styles.error}>
                  Failed to create model: {(createMutation.error as Error).message}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className={styles.footer}>
          <button
            className={styles.navBtn}
            onClick={handleBack}
            disabled={currentStepIndex === 0}
          >
            ← Back
          </button>

          <span className={styles.stepIndicator}>
            Step {currentStepIndex + 1} of {steps.length}
          </span>

          {step !== 'review' ? (
            <button
              className={`${styles.navBtn} ${styles.primary}`}
              onClick={handleNext}
              disabled={!canProceed}
            >
              Next →
            </button>
          ) : (
            <button
              className={`${styles.navBtn} ${styles.primary}`}
              onClick={handleSubmit}
              disabled={createMutation.isPending}
            >
              {createMutation.isPending ? 'Creating...' : 'Create Model'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Dynamic API Field Component
// =============================================================================

interface APIFieldProps {
  field: APIFieldConfig
  value: string | number | boolean
  onChange: (value: string | number | boolean) => void
}

function APIField({ field, value, onChange }: APIFieldProps) {
  const renderInput = () => {
    switch (field.type) {
      case 'select':
        return (
          <select
            className={styles.select}
            value={String(value)}
            onChange={e => onChange(e.target.value)}
            required={field.required}
          >
            {!field.required && <option value="">Select...</option>}
            {field.options?.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        )

      case 'checkbox':
        return (
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={e => onChange(e.target.checked)}
            />
            {field.label}
          </label>
        )

      case 'number':
        return (
          <input
            type="number"
            className={styles.input}
            placeholder={field.placeholder}
            value={value as number || ''}
            onChange={e => onChange(parseFloat(e.target.value) || 0)}
            required={field.required}
            min={field.validation?.min}
            max={field.validation?.max}
          />
        )

      case 'password':
        return (
          <input
            type="password"
            className={styles.input}
            placeholder={field.placeholder}
            value={String(value || '')}
            onChange={e => onChange(e.target.value)}
            required={field.required}
          />
        )

      case 'url':
        return (
          <input
            type="url"
            className={styles.input}
            placeholder={field.placeholder}
            value={String(value || '')}
            onChange={e => onChange(e.target.value)}
            required={field.required}
          />
        )

      default:
        return (
          <input
            type="text"
            className={styles.input}
            placeholder={field.placeholder}
            value={String(value || '')}
            onChange={e => onChange(e.target.value)}
            required={field.required}
            minLength={field.validation?.minLength}
            maxLength={field.validation?.maxLength}
          />
        )
    }
  }

  if (field.type === 'checkbox') {
    return (
      <div className={styles.formGroup}>
        {renderInput()}
        {field.helpText && <span className={styles.hint}>{field.helpText}</span>}
      </div>
    )
  }

  return (
    <div className={styles.formGroup}>
      <label className={styles.label}>
        {field.label} {field.required && '*'}
      </label>
      {renderInput()}
      {field.helpText && <span className={styles.hint}>{field.helpText}</span>}
    </div>
  )
}

export default RegisterModelWizard
