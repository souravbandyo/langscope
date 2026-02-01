/**
 * My Models Dashboard
 * 
 * Central hub for managing user's private models across all types (LLM, ASR, TTS, VLM, etc.)
 * with unified registration, performance tracking, and comparison features.
 */

import { useState, useMemo } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { LoadingState, ErrorState, EmptyState } from '@/components/common'
import { useMyModelsByType, useDeleteMyModel } from '@/api/hooks'
import { 
  MODEL_TYPE_CONFIGS, 
  getModelTypesByCategory,
  getModelTypeIcon,
  getModelTypeDisplayName,
  getPrimaryMetric,
  type ModelType 
} from '@/types/modelTypes'
import type { UserModel } from '@/api/types'
import { RegisterModelWizard } from './RegisterModelWizard'
import { ModelPerformance } from './ModelPerformance'
import styles from './MyModels.module.css'

type ViewMode = 'grid' | 'list'
type FilterType = 'all' | ModelType

export function MyModels() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [filterType, setFilterType] = useState<FilterType>('all')
  const [showWizard, setShowWizard] = useState(false)
  const [selectedModel, setSelectedModel] = useState<UserModel | null>(null)
  const [showPerformance, setShowPerformance] = useState(false)

  const { data, isLoading, error, refetch } = useMyModelsByType()
  const deleteMutation = useDeleteMyModel()

  const modelTypeCategories = getModelTypesByCategory()

  // Handle API not available - show empty state instead of error
  const isApiNotAvailable = error && (
    (error as Error)?.message?.includes('Not Found') ||
    (error as Error)?.message?.includes('404') ||
    (error as Error)?.message?.includes('NetworkError')
  )

  // Treat API not available as empty models list
  const models: UserModel[] = isApiNotAvailable ? [] : (data?.models || [])

  // Filter models based on selected type
  const filteredModels = useMemo(() => {
    if (!models || models.length === 0) return []
    if (filterType === 'all') return models
    return models.filter(m => m.modelType === filterType)
  }, [models, filterType])

  // Get stats by type
  const typeStats = useMemo(() => {
    if (!models || models.length === 0) return {}
    const grouped = models.reduce((acc, model) => {
      if (!acc[model.modelType]) acc[model.modelType] = []
      acc[model.modelType].push(model)
      return acc
    }, {} as Record<string, UserModel[]>)
    
    const stats: Record<string, { count: number; activeCount: number }> = {}
    Object.entries(grouped).forEach(([type, typeModels]) => {
      stats[type] = {
        count: typeModels.length,
        activeCount: typeModels.filter(m => m.isActive).length
      }
    })
    return stats
  }, [models])
  
  const totalModels = models?.length || 0

  const handleDeleteModel = (modelId: string) => {
    if (confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      deleteMutation.mutate(modelId, {
        onSuccess: () => {
          setSelectedModel(null)
          refetch()
        }
      })
    }
  }

  const handleViewPerformance = (model: UserModel) => {
    setSelectedModel(model)
    setShowPerformance(true)
  }

  if (isLoading) {
    return <LoadingState message="Loading your models..." />
  }

  // Only show error if it's not a "not found" error (API not implemented yet)
  if (error && !isApiNotAvailable) {
    return (
      <ErrorState
        title="Failed to load models"
        error={error as Error}
        onRetry={() => refetch()}
      />
    )
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerTop}>
          <div>
            <h1 className={styles.title}>My Models</h1>
            <p className={styles.subtitle}>
              Manage and compare your private models across all types
            </p>
          </div>
          <div className={styles.headerActions}>
            <SketchButton 
              variant="primary" 
              onClick={() => setShowWizard(true)}
            >
              + Register Model
            </SketchButton>
          </div>
        </div>
      </header>

      {/* Stats Overview */}
      <div className={styles.statsRow}>
        <StickyNote title="Total Models" color="blue" rotation={-1}>
          <span className={styles.statValue}>{totalModels}</span>
        </StickyNote>
        {Object.entries(modelTypeCategories).map(([category, types]) => {
          const categoryCount = types.reduce((sum, type) => 
            sum + (typeStats[type]?.count || 0), 0
          )
          if (categoryCount === 0) return null
          return (
            <StickyNote 
              key={category} 
              title={category} 
              color="yellow" 
              rotation={1}
            >
              <span className={styles.statValue}>{categoryCount}</span>
            </StickyNote>
          )
        })}
      </div>

      {/* Filter Tabs */}
      <SketchCard padding="md" className={styles.filterCardWrapper}>
        <div className={styles.filterCardInner}>
          <div className={styles.filterTabs}>
            <button
              className={`${styles.filterTab} ${filterType === 'all' ? styles.active : ''}`}
              onClick={() => setFilterType('all')}
            >
              All Types
              <span className={styles.filterCount}>{totalModels}</span>
            </button>
            
            {Object.entries(modelTypeCategories).map(([category, types]) => (
              <div key={category} className={styles.categoryGroup}>
                <span className={styles.categoryLabel}>{category}</span>
                <div className={styles.categoryTabs}>
                  {types.map(type => {
                    const count = typeStats[type]?.count || 0
                    if (count === 0 && filterType !== type) return null
                    return (
                      <button
                        key={type}
                        className={`${styles.filterTab} ${filterType === type ? styles.active : ''}`}
                        onClick={() => setFilterType(type)}
                      >
                        <i className={`${getModelTypeIcon(type)} ${styles.typeIcon}`}></i>
                        {type}
                        {count > 0 && <span className={styles.filterCount}>{count}</span>}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className={styles.viewToggle}>
            <button
              className={`${styles.viewBtn} ${viewMode === 'grid' ? styles.active : ''}`}
              onClick={() => setViewMode('grid')}
              title="Grid view"
            >
              ▦
            </button>
            <button
              className={`${styles.viewBtn} ${viewMode === 'list' ? styles.active : ''}`}
              onClick={() => setViewMode('list')}
              title="List view"
            >
              ≡
            </button>
          </div>
        </div>
      </SketchCard>

      {/* API Notice */}
      {isApiNotAvailable && (
        <StickyNote title="Backend Not Available" color="yellow" rotation={0}>
          <p style={{ margin: 0, fontSize: '0.9rem' }}>
            The My Models API is not yet connected. You can explore the UI and registration wizard.
            Connect a backend with the <code>/my-models</code> endpoint to enable full functionality.
          </p>
        </StickyNote>
      )}

      {/* Models Display */}
      {filteredModels.length === 0 ? (
        <EmptyState
          title={filterType === 'all' ? "No models registered yet" : `No ${filterType} models`}
          message={filterType === 'all' 
            ? "Register your first model to start tracking its performance"
            : `Register a ${getModelTypeDisplayName(filterType as ModelType)} model to get started`
          }
          icon={filterType === 'all' ? 'ph ph-robot' : 'ph ph-cube'}
        />
      ) : viewMode === 'grid' ? (
        <div className={styles.modelsGrid}>
          {filteredModels.map(model => (
            <ModelCard 
              key={model.id} 
              model={model}
              onViewPerformance={() => handleViewPerformance(model)}
              onDelete={() => handleDeleteModel(model.id)}
            />
          ))}
        </div>
      ) : (
        <SketchCard padding="none">
          <table className={styles.modelsTable}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Version</th>
                <th>Status</th>
                <th>Evaluations</th>
                <th>Primary Metric</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map(model => (
                <ModelRow 
                  key={model.id} 
                  model={model}
                  onViewPerformance={() => handleViewPerformance(model)}
                  onDelete={() => handleDeleteModel(model.id)}
                />
              ))}
            </tbody>
          </table>
        </SketchCard>
      )}

      {/* Help Section */}
      <div className={styles.helpSection}>
        <h3 className={styles.helpTitle}>Quick Guide</h3>
        <div className={styles.helpGrid}>
          <SketchCard padding="md" className={styles.helpCard}>
            <span className={styles.helpIcon}>1</span>
            <h4>Register Model</h4>
            <p>Add your model by selecting its type and providing API credentials. The smart mapper configures evaluation settings automatically.</p>
          </SketchCard>
          <SketchCard padding="md" className={styles.helpCard}>
            <span className={styles.helpIcon}>2</span>
            <h4>Run Evaluation</h4>
            <p>Choose a domain and run evaluations to get TrueSkill ratings (LLM) or ground-truth metrics (ASR, TTS, VLM).</p>
          </SketchCard>
          <SketchCard padding="md" className={styles.helpCard}>
            <span className={styles.helpIcon}>3</span>
            <h4>Compare Performance</h4>
            <p>See how your models stack up against public leaderboards and track improvement over time.</p>
          </SketchCard>
        </div>
      </div>

      {/* Modals */}
      {showWizard && (
        <RegisterModelWizard
          onClose={() => setShowWizard(false)}
          onComplete={() => {
            setShowWizard(false)
            refetch()
          }}
        />
      )}

      {showPerformance && selectedModel && (
        <ModelPerformance
          model={selectedModel}
          onClose={() => {
            setShowPerformance(false)
            setSelectedModel(null)
          }}
        />
      )}
    </div>
  )
}

// =============================================================================
// Model Card Component
// =============================================================================

interface ModelCardProps {
  model: UserModel
  onViewPerformance: () => void
  onDelete: () => void
}

function ModelCard({ model, onViewPerformance, onDelete }: ModelCardProps) {
  const config = MODEL_TYPE_CONFIGS[model.modelType]
  const primaryMetric = getPrimaryMetric(model.modelType)
  
  // Get primary metric value
  const primaryValue = useMemo(() => {
    if (model.trueskill?.raw_quality) {
      return model.trueskill.raw_quality.mu.toFixed(0)
    }
    if (model.groundTruthMetrics && primaryMetric) {
      return model.groundTruthMetrics[primaryMetric.id]?.toFixed(2)
    }
    return 'N/A'
  }, [model, primaryMetric])

  return (
    <SketchCard padding="md" className={styles.modelCard}>
      <div className={styles.cardHeader}>
        <div className={styles.cardIcon}>{config.icon}</div>
        <div className={styles.cardInfo}>
          <h3 className={styles.cardName}>{model.name}</h3>
          <span className={styles.cardType}>{config.displayName}</span>
        </div>
        <span className={`${styles.statusBadge} ${model.isActive ? styles.active : styles.inactive}`}>
          {model.isActive ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className={styles.cardMeta}>
        <span className={styles.metaItem}>
          <span className={styles.metaLabel}>Version:</span>
          {model.version}
        </span>
        {model.baseModelId && (
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Base:</span>
            {model.baseModelId}
          </span>
        )}
      </div>

      <div className={styles.cardStats}>
        <div className={styles.cardStat}>
          <span className={styles.statLabel}>
            {primaryMetric?.name || 'Rating'}
          </span>
          <span className={styles.statValue}>{primaryValue}</span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.statLabel}>Evaluations</span>
          <span className={styles.statValue}>{model.totalEvaluations}</span>
        </div>
        <div className={styles.cardStat}>
          <span className={styles.statLabel}>Domains</span>
          <span className={styles.statValue}>{model.domainsEvaluated.length}</span>
        </div>
      </div>

      {model.domainsEvaluated.length > 0 && (
        <div className={styles.cardDomains}>
          {model.domainsEvaluated.slice(0, 3).map(domain => (
            <span key={domain} className={styles.domainTag}>
              {domain.replace(/_/g, ' ')}
            </span>
          ))}
          {model.domainsEvaluated.length > 3 && (
            <span className={styles.domainMore}>
              +{model.domainsEvaluated.length - 3}
            </span>
          )}
        </div>
      )}

      <div className={styles.cardCost}>
        <span className={styles.costLabel}>Cost:</span>
        <span className={styles.costValue}>
          ${model.costs.inputCostPerMillion.toFixed(2)} / ${model.costs.outputCostPerMillion.toFixed(2)} per 1M
        </span>
      </div>

      <div className={styles.cardActions}>
        <SketchButton variant="secondary" size="sm" onClick={onViewPerformance}>
          View Performance
        </SketchButton>
        <button className={styles.deleteBtn} onClick={onDelete} title="Delete model">
          <i className="ph ph-trash"></i>
        </button>
      </div>
    </SketchCard>
  )
}

// =============================================================================
// Model Row Component (List View)
// =============================================================================

interface ModelRowProps {
  model: UserModel
  onViewPerformance: () => void
  onDelete: () => void
}

function ModelRow({ model, onViewPerformance, onDelete }: ModelRowProps) {
  const config = MODEL_TYPE_CONFIGS[model.modelType]
  const primaryMetric = getPrimaryMetric(model.modelType)
  
  const primaryValue = useMemo(() => {
    if (model.trueskill?.raw_quality) {
      return model.trueskill.raw_quality.mu.toFixed(0)
    }
    if (model.groundTruthMetrics && primaryMetric) {
      return model.groundTruthMetrics[primaryMetric.id]?.toFixed(2)
    }
    return 'N/A'
  }, [model, primaryMetric])

  return (
    <tr className={styles.tableRow}>
      <td className={styles.modelCell}>
        <span className={styles.modelIcon}>{config.icon}</span>
        <div className={styles.modelInfo}>
          <span className={styles.modelName}>{model.name}</span>
          {model.baseModelId && (
            <span className={styles.modelBase}>{model.baseModelId}</span>
          )}
        </div>
      </td>
      <td className={styles.typeCell}>
        <span className={styles.typeBadge}>{model.modelType}</span>
      </td>
      <td className={styles.versionCell}>{model.version}</td>
      <td className={styles.statusCell}>
        <span className={`${styles.statusDot} ${model.isActive ? styles.active : styles.inactive}`} />
        {model.isActive ? 'Active' : 'Inactive'}
      </td>
      <td className={styles.evalCell}>{model.totalEvaluations}</td>
      <td className={styles.metricCell}>
        <span className={styles.metricValue}>{primaryValue}</span>
        <span className={styles.metricName}>{primaryMetric?.name || 'Rating'}</span>
      </td>
      <td className={styles.actionsCell}>
        <button className={styles.actionBtn} onClick={onViewPerformance} title="View performance">
          <i className="ph ph-chart-bar"></i>
        </button>
        <button className={styles.actionBtn} onClick={onDelete} title="Delete">
          <i className="ph ph-trash"></i>
        </button>
      </td>
    </tr>
  )
}

export default MyModels
