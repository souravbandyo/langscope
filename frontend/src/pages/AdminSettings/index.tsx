/**
 * Admin Settings Page
 * 
 * Configure TrueSkill parameters, dimension weights, and manage cache.
 */

import { useState } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import { LoadingState, ErrorState } from '@/components/common'
import {
  useParamTypes,
  useParams,
  useUpdateParams,
  useResetParams,
  useCacheStats,
  useInvalidateCategory,
  useInvalidateAllCache,
  useResetCacheStats,
  useParamCacheStats,
} from '@/api/hooks'
import styles from './AdminSettings.module.css'

type TabId = 'trueskill' | 'weights' | 'cache'

interface TrueSkillParamEditorProps {
  domain?: string
}

function TrueSkillParamEditor({ domain }: TrueSkillParamEditorProps) {
  const { data, isLoading, error, refetch } = useParams('trueskill', { domain })
  const updateMutation = useUpdateParams()
  const resetMutation = useResetParams()

  const [localParams, setLocalParams] = useState<Record<string, number | string>>({})
  const [hasChanges, setHasChanges] = useState(false)

  // Initialize local params when data loads
  const params = data?.params as Record<string, number> | undefined
  
  const handleParamChange = (key: string, value: string) => {
    setLocalParams(prev => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const handleSave = () => {
    const paramsToSave: Record<string, number> = {}
    for (const [key, value] of Object.entries(localParams)) {
      const num = parseFloat(value as string)
      if (!isNaN(num)) {
        paramsToSave[key] = num
      }
    }

    updateMutation.mutate(
      { paramType: 'trueskill', data: { params: paramsToSave, domain } },
      {
        onSuccess: () => {
          setHasChanges(false)
          setLocalParams({})
          refetch()
        }
      }
    )
  }

  const handleReset = () => {
    resetMutation.mutate(
      { paramType: 'trueskill', domain },
      {
        onSuccess: () => {
          setHasChanges(false)
          setLocalParams({})
          refetch()
        }
      }
    )
  }

  if (isLoading) {
    return <div className={styles.loadingSmall}>Loading parameters...</div>
  }

  if (error) {
    return <div className={styles.errorSmall}>Failed to load parameters</div>
  }

  const TRUESKILL_PARAMS = [
    { key: 'mu_0', label: 'Initial Mean (μ₀)', description: 'Starting rating for new models', default: 1500 },
    { key: 'sigma_0', label: 'Initial Uncertainty (σ₀)', description: 'Starting uncertainty', default: 166 },
    { key: 'beta', label: 'Performance Variability (β)', description: 'Inherent performance variance', default: 83 },
    { key: 'tau', label: 'Dynamics Factor (τ)', description: 'Rating drift over time', default: 8.3 },
    { key: 'k', label: 'Conservative Multiplier (k)', description: 'For conservative estimate calculation', default: 3 },
  ]

  return (
    <div className={styles.paramEditor}>
      <div className={styles.paramGrid}>
        {TRUESKILL_PARAMS.map(({ key, label, description, default: defaultVal }) => (
          <div key={key} className={styles.paramItem}>
            <label className={styles.paramLabel}>{label}</label>
            <input
              type="number"
              className={styles.paramInput}
              value={localParams[key] !== undefined ? localParams[key] : (params?.[key] ?? defaultVal)}
              onChange={(e) => handleParamChange(key, e.target.value)}
              step="0.1"
            />
            <span className={styles.paramDescription}>{description}</span>
          </div>
        ))}
      </div>

      <div className={styles.paramActions}>
        <SketchButton
          variant="primary"
          onClick={handleSave}
          disabled={!hasChanges || updateMutation.isPending}
        >
          {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
        </SketchButton>
        <SketchButton
          variant="secondary"
          onClick={handleReset}
          disabled={resetMutation.isPending}
        >
          {resetMutation.isPending ? 'Resetting...' : 'Reset to Defaults'}
        </SketchButton>
      </div>

      {data?.is_default === false && (
        <div className={styles.customNote}>
          Using custom parameters{domain ? ` for domain: ${domain}` : ''}
        </div>
      )}
    </div>
  )
}

function DimensionWeightsEditor() {
  const { data, isLoading, error, refetch } = useParams('dimension_weights')
  const updateMutation = useUpdateParams()
  const resetMutation = useResetParams()

  const [localWeights, setLocalWeights] = useState<Record<string, string>>({})
  const [hasChanges, setHasChanges] = useState(false)

  const weights = data?.params as Record<string, number> | undefined

  const DIMENSIONS = [
    { key: 'raw_quality', label: 'Raw Quality', default: 0.20 },
    { key: 'instruction_following', label: 'Instruction Following', default: 0.15 },
    { key: 'hallucination_resistance', label: 'Hallucination Resistance', default: 0.15 },
    { key: 'cost_adjusted', label: 'Cost-Adjusted', default: 0.10 },
    { key: 'latency', label: 'Latency', default: 0.10 },
    { key: 'consistency', label: 'Consistency', default: 0.10 },
    { key: 'token_efficiency', label: 'Token Efficiency', default: 0.10 },
    { key: 'ttft', label: 'Time to First Token', default: 0.05 },
    { key: 'long_context', label: 'Long Context', default: 0.05 },
  ]

  const handleWeightChange = (key: string, value: string) => {
    setLocalWeights(prev => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const getCurrentWeight = (key: string, defaultVal: number): number => {
    if (localWeights[key] !== undefined) {
      const parsed = parseFloat(localWeights[key])
      return isNaN(parsed) ? defaultVal : parsed
    }
    return weights?.[key] ?? defaultVal
  }

  const totalWeight = DIMENSIONS.reduce((sum, d) => sum + getCurrentWeight(d.key, d.default), 0)

  const handleSave = () => {
    const weightsToSave: Record<string, number> = {}
    for (const dim of DIMENSIONS) {
      weightsToSave[dim.key] = getCurrentWeight(dim.key, dim.default)
    }

    updateMutation.mutate(
      { paramType: 'dimension_weights', data: { params: weightsToSave } },
      {
        onSuccess: () => {
          setHasChanges(false)
          setLocalWeights({})
          refetch()
        }
      }
    )
  }

  const handleReset = () => {
    resetMutation.mutate(
      { paramType: 'dimension_weights' },
      {
        onSuccess: () => {
          setHasChanges(false)
          setLocalWeights({})
          refetch()
        }
      }
    )
  }

  if (isLoading) {
    return <div className={styles.loadingSmall}>Loading weights...</div>
  }

  if (error) {
    return <div className={styles.errorSmall}>Failed to load weights</div>
  }

  return (
    <div className={styles.weightsEditor}>
      <p className={styles.weightsDescription}>
        Adjust how each dimension contributes to the combined rating score. Weights should sum to 1.0 (100%).
      </p>

      <div className={styles.weightsGrid}>
        {DIMENSIONS.map(({ key, label, default: defaultVal }) => {
          const currentWeight = getCurrentWeight(key, defaultVal)
          return (
            <div key={key} className={styles.weightItem}>
              <label className={styles.weightLabel}>{label}</label>
              <div className={styles.weightInputRow}>
                <input
                  type="number"
                  className={styles.weightInput}
                  value={localWeights[key] !== undefined ? localWeights[key] : (weights?.[key]?.toFixed(2) ?? defaultVal.toFixed(2))}
                  onChange={(e) => handleWeightChange(key, e.target.value)}
                  step="0.01"
                  min="0"
                  max="1"
                />
                <span className={styles.weightPercent}>{(currentWeight * 100).toFixed(0)}%</span>
              </div>
              <div className={styles.weightBar}>
                <div 
                  className={styles.weightBarFill} 
                  style={{ width: `${currentWeight * 100}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className={`${styles.totalWeight} ${Math.abs(totalWeight - 1) > 0.01 ? styles.totalWarning : styles.totalOk}`}>
        Total: {(totalWeight * 100).toFixed(1)}%
        {Math.abs(totalWeight - 1) > 0.01 && (
          <span className={styles.totalHint}> (should be 100%)</span>
        )}
      </div>

      <div className={styles.paramActions}>
        <SketchButton
          variant="primary"
          onClick={handleSave}
          disabled={!hasChanges || updateMutation.isPending || Math.abs(totalWeight - 1) > 0.01}
        >
          {updateMutation.isPending ? 'Saving...' : 'Save Weights'}
        </SketchButton>
        <SketchButton
          variant="secondary"
          onClick={handleReset}
          disabled={resetMutation.isPending}
        >
          {resetMutation.isPending ? 'Resetting...' : 'Reset to Defaults'}
        </SketchButton>
      </div>
    </div>
  )
}

function CacheManager() {
  const { data: cacheStats, isLoading: cacheLoading, refetch: refetchCache } = useCacheStats()
  const { data: paramCacheStats } = useParamCacheStats()

  const invalidateCategoryMutation = useInvalidateCategory()
  const invalidateAllMutation = useInvalidateAllCache()
  const resetStatsMutation = useResetCacheStats()

  const handleInvalidateCategory = (category: string) => {
    invalidateCategoryMutation.mutate(category, {
      onSuccess: () => refetchCache()
    })
  }

  const handleInvalidateAll = () => {
    if (window.confirm('Are you sure you want to invalidate all cache? This may temporarily impact performance.')) {
      invalidateAllMutation.mutate(undefined, {
        onSuccess: () => refetchCache()
      })
    }
  }

  const handleResetStats = () => {
    resetStatsMutation.mutate(undefined, {
      onSuccess: () => refetchCache()
    })
  }

  if (cacheLoading) {
    return <div className={styles.loadingSmall}>Loading cache stats...</div>
  }

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
  }

  return (
    <div className={styles.cacheManager}>
      {/* Overall Stats */}
      {cacheStats && (
        <div className={styles.cacheOverview}>
          <h4 className={styles.sectionSubtitle}>Overview</h4>
          <div className={styles.statsGrid}>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{cacheStats.total_entries.toLocaleString()}</span>
              <span className={styles.statLabel}>Total Entries</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{formatBytes(cacheStats.total_size_bytes)}</span>
              <span className={styles.statLabel}>Total Size</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{(cacheStats.hit_rate * 100).toFixed(1)}%</span>
              <span className={styles.statLabel}>Hit Rate</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{cacheStats.eviction_count.toLocaleString()}</span>
              <span className={styles.statLabel}>Evictions</span>
            </div>
          </div>
        </div>
      )}

      {/* Category Breakdown */}
      {cacheStats?.categories && Object.keys(cacheStats.categories).length > 0 && (
        <div className={styles.categorySection}>
          <h4 className={styles.sectionSubtitle}>Categories</h4>
          <div className={styles.categoryTable}>
            <div className={styles.categoryHeader}>
              <span>Category</span>
              <span>Entries</span>
              <span>Size</span>
              <span>Action</span>
            </div>
            {Object.entries(cacheStats.categories).map(([category, stats]) => (
              <div key={category} className={styles.categoryRow}>
                <span className={styles.categoryName}>{category}</span>
                <span>{stats.entries.toLocaleString()}</span>
                <span>{formatBytes(stats.size_bytes)}</span>
                <button
                  className={styles.invalidateButton}
                  onClick={() => handleInvalidateCategory(category)}
                  disabled={invalidateCategoryMutation.isPending}
                >
                  Invalidate
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Parameter Cache Stats */}
      {paramCacheStats && (
        <div className={styles.paramCacheSection}>
          <h4 className={styles.sectionSubtitle}>Parameter Cache</h4>
          <div className={styles.statsGrid}>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{paramCacheStats.total_cached}</span>
              <span className={styles.statLabel}>Cached Params</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{paramCacheStats.cache_hits}</span>
              <span className={styles.statLabel}>Cache Hits</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{paramCacheStats.cache_misses}</span>
              <span className={styles.statLabel}>Cache Misses</span>
            </div>
            <div className={styles.statCard}>
              <span className={styles.statValue}>{(paramCacheStats.hit_rate * 100).toFixed(1)}%</span>
              <span className={styles.statLabel}>Hit Rate</span>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className={styles.cacheActions}>
        <SketchButton
          variant="secondary"
          onClick={handleResetStats}
          disabled={resetStatsMutation.isPending}
        >
          Reset Statistics
        </SketchButton>
        <SketchButton
          variant="primary"
          onClick={handleInvalidateAll}
          disabled={invalidateAllMutation.isPending}
        >
          {invalidateAllMutation.isPending ? 'Invalidating...' : 'Invalidate All Cache'}
        </SketchButton>
      </div>

      {invalidateAllMutation.isSuccess && (
        <div className={styles.successMessage}>All cache invalidated successfully</div>
      )}
    </div>
  )
}

export function AdminSettings() {
  const [activeTab, setActiveTab] = useState<TabId>('trueskill')

  const tabs = [
    { id: 'trueskill' as TabId, label: 'TrueSkill Parameters' },
    { id: 'weights' as TabId, label: 'Dimension Weights' },
    { id: 'cache' as TabId, label: 'Cache Management' },
  ]

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Admin Settings</h1>
        <p className={styles.subtitle}>
          Configure system parameters and manage cache
        </p>
      </header>

      {/* Tabs */}
      <div className={styles.tabs}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`${styles.tab} ${activeTab === tab.id ? styles.activeTab : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <SketchCard padding="lg" className={styles.tabContent}>
        {activeTab === 'trueskill' && (
          <>
            <h2 className={styles.sectionTitle}>TrueSkill Parameters</h2>
            <p className={styles.sectionDescription}>
              Configure the TrueSkill rating system parameters. These affect how ratings are calculated and updated.
            </p>
            <TrueSkillParamEditor />
          </>
        )}

        {activeTab === 'weights' && (
          <>
            <h2 className={styles.sectionTitle}>Dimension Weights</h2>
            <p className={styles.sectionDescription}>
              Adjust how each evaluation dimension contributes to the combined rating score.
            </p>
            <DimensionWeightsEditor />
          </>
        )}

        {activeTab === 'cache' && (
          <>
            <h2 className={styles.sectionTitle}>Cache Management</h2>
            <p className={styles.sectionDescription}>
              Monitor cache performance and invalidate entries when needed.
            </p>
            <CacheManager />
          </>
        )}
      </SketchCard>
    </div>
  )
}

export default AdminSettings
