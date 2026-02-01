import { useState } from 'react'
import { SketchButton, SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useDomains, useModels, useArenaSession, useArenaBattle, useCompleteArenaSession } from '@/api/hooks'
import type { ArenaSessionStartResponse, ArenaSessionResult, ModelResponse } from '@/api/types'
import styles from './Arena.module.css'

type ArenaState = 'setup' | 'battle' | 'results'

/**
 * Arena page for interactive model battles
 */
export function Arena() {
  const [state, setState] = useState<ArenaState>('setup')
  const [session, setSession] = useState<ArenaSessionStartResponse | null>(null)
  const [results, setResults] = useState<ArenaSessionResult | null>(null)

  const handleSessionStart = (sessionData: ArenaSessionStartResponse) => {
    setSession(sessionData)
    setState('battle')
  }

  const handleSessionComplete = (resultData: ArenaSessionResult) => {
    setResults(resultData)
    setState('results')
  }

  const handleReset = () => {
    setSession(null)
    setResults(null)
    setState('setup')
  }

  return (
    <div className={styles.arena}>
      <h1 className={styles.title}>⚔️ Arena Mode</h1>
      <p className={styles.subtitle}>
        Compare models head-to-head and provide your own rankings
      </p>

      {state === 'setup' && <ArenaSetup onStart={handleSessionStart} />}
      {state === 'battle' && session && (
        <ArenaBattle session={session} onComplete={handleSessionComplete} />
      )}
      {state === 'results' && <ArenaResults results={results} onReset={handleReset} />}
    </div>
  )
}

function ArenaSetup({ onStart }: { onStart: (session: ArenaSessionStartResponse) => void }) {
  const [domain, setDomain] = useState('')
  const [useCase, setUseCase] = useState('general')
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { data: domainsData, isLoading: domainsLoading, error: domainsError, refetch: refetchDomains } = useDomains()
  const { data: modelsData, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useModels()
  const startSession = useArenaSession()

  // Default domains when API is unavailable
  const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']
  const domains = domainsError ? defaultDomains : (domainsData?.domains || [])
  
  // Get models list
  const models = modelsData?.models || []
  
  // Filter models by selected domain if applicable
  const filteredModels = domain
    ? models.filter((m: ModelResponse) => m.domains_evaluated?.includes(domain) || !m.domains_evaluated?.length)
    : models

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    )
  }

  const handleStart = async () => {
    if (!domain) {
      setError('Please select a domain')
      return
    }
    
    if (selectedModels.length > 0 && selectedModels.length < 2) {
      setError('Please select at least 2 models or none (to use all available)')
      return
    }

    setIsStarting(true)
    setError(null)

    try {
      const session = await startSession.mutateAsync({
        domain,
        use_case: useCase,
        model_ids: selectedModels.length >= 2 ? selectedModels : undefined,
      })
      onStart(session)
    } catch (err) {
      const errorMessage = err instanceof Error && err.message.includes('Failed to fetch')
        ? 'Unable to connect to server. Please check if the API is running.'
        : 'Failed to start session. Please try again.'
      setError(errorMessage)
      console.error('Arena start error:', err)
    } finally {
      setIsStarting(false)
    }
  }

  return (
    <SketchCard padding="lg">
      <div className={styles.setupForm}>
        <h2 className={styles.sectionTitle}>Start a New Session</h2>

        {domainsError && (
          <div className={styles.warning}>
            ⚠️ Could not load domains from API. Using defaults.{' '}
            <button className={styles.retryLink} onClick={() => refetchDomains()}>Retry</button>
          </div>
        )}

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.formGroup}>
          <label className={styles.label}>Select Domain:</label>
          {domainsLoading ? (
            <div className={styles.loadingText}>Loading domains...</div>
          ) : (
            <select
              className={styles.select}
              value={domain}
              onChange={(e) => {
                setDomain(e.target.value)
                setSelectedModels([]) // Reset model selection when domain changes
              }}
            >
              <option value="">Choose a domain...</option>
              {domains.map((d) => (
                <option key={d} value={d}>
                  {d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>Use Case:</label>
          <select
            className={styles.select}
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
          >
            <option value="general">General</option>
            <option value="accuracy">Accuracy-focused</option>
            <option value="speed">Speed-focused</option>
            <option value="cost">Cost-sensitive</option>
          </select>
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Select Models (optional - select 2+ or leave empty for all):
          </label>
          {modelsLoading ? (
            <div className={styles.loadingText}>Loading models...</div>
          ) : modelsError ? (
            <div className={styles.warning}>
              ⚠️ Could not load models.{' '}
              <button className={styles.retryLink} onClick={() => refetchModels()}>Retry</button>
            </div>
          ) : (
            <div className={styles.modelGrid}>
              {filteredModels.length === 0 ? (
                <div className={styles.loadingText}>No models available{domain ? ` for ${domain}` : ''}</div>
              ) : (
                filteredModels.map((model: ModelResponse) => (
                  <label key={model.model_id} className={styles.modelCheckbox}>
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model.model_id)}
                      onChange={() => handleModelToggle(model.model_id)}
                    />
                    <span className={styles.modelName}>{model.name}</span>
                    <span className={styles.modelProvider}>{model.provider}</span>
                  </label>
                ))
              )}
            </div>
          )}
          {selectedModels.length > 0 && (
            <div className={styles.selectedCount}>
              {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''} selected
            </div>
          )}
        </div>

        <SketchButton
          variant="primary"
          size="lg"
          onClick={handleStart}
          disabled={isStarting || domainsLoading}
        >
          {isStarting ? 'Starting...' : 'Start Arena Session'}
        </SketchButton>
      </div>
    </SketchCard>
  )
}

function ArenaBattle({
  session,
  onComplete,
}: {
  session: ArenaSessionStartResponse
  onComplete: (results: ArenaSessionResult) => void
}) {
  const [battleNum, setBattleNum] = useState(1)
  const [rankings, setRankings] = useState<Record<string, number>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const totalBattles = 5

  const submitBattle = useArenaBattle(session.session_id)
  const completeSession = useCompleteArenaSession(session.session_id)

  // Get two random models for this battle
  const models = session.models_available.slice(0, 2)

  const handleRankChange = (modelId: string, rank: number) => {
    setRankings((prev) => ({ ...prev, [modelId]: rank }))
  }

  const handleNextBattle = async () => {
    if (Object.keys(rankings).length < 2) {
      return
    }

    setIsSubmitting(true)

    try {
      await submitBattle.mutateAsync({
        participant_ids: models,
        user_ranking: rankings,
      })

      if (battleNum >= totalBattles) {
        const results = await completeSession.mutateAsync()
        onComplete(results)
      } else {
        setBattleNum(battleNum + 1)
        setRankings({})
      }
    } catch (err) {
      console.error('Battle submission error:', err)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={styles.battleContainer}>
      <div className={styles.battleHeader}>
        <span className={styles.battleProgress}>
          Battle {battleNum} of {totalBattles}
        </span>
        <span className={styles.domain}>
          Domain: {session.domain.replace(/_/g, ' ')}
        </span>
      </div>

      <div className={styles.prompt}>
        <StickyNote title="Compare these models" color="blue" rotation={0}>
          Rank the model responses based on quality. 1 = Best.
        </StickyNote>
      </div>

      <div className={styles.responses}>
        {models.map((modelId, index) => (
          <SketchCard key={modelId} padding="md" className={styles.responseCard}>
            <h3 className={styles.responseTitle}>Response {String.fromCharCode(65 + index)}</h3>
            <div className={styles.responseContent}>
              <p className={styles.modelHint}>Model: {modelId}</p>
              <p className={styles.predictionHint}>
                Predicted rating: {Math.round(session.predictions[modelId]?.mu || 1500)}
              </p>
            </div>
            <div className={styles.rankSelect}>
              <label>Rank:</label>
              <select
                className={styles.rankDropdown}
                value={rankings[modelId] || ''}
                onChange={(e) => handleRankChange(modelId, parseInt(e.target.value))}
              >
                <option value="">Select...</option>
                <option value="1">1st (Best)</option>
                <option value="2">2nd</option>
              </select>
            </div>
          </SketchCard>
        ))}
      </div>

      <div className={styles.battleActions}>
        <SketchButton
          variant="primary"
          onClick={handleNextBattle}
          disabled={Object.keys(rankings).length < 2 || isSubmitting}
        >
          {isSubmitting
            ? 'Submitting...'
            : battleNum >= totalBattles
            ? 'Finish Session'
            : 'Next Battle →'}
        </SketchButton>
      </div>
    </div>
  )
}

function ArenaResults({
  results,
  onReset,
}: {
  results: ArenaSessionResult | null
  onReset: () => void
}) {
  if (!results) {
    return (
      <div className={styles.results}>
        <SketchCard padding="lg">
          <p>No results available</p>
          <SketchButton onClick={onReset}>Start New Session</SketchButton>
        </SketchCard>
      </div>
    )
  }

  const deltas = Object.entries(results.deltas || {})
    .map(([modelId, dims]) => ({
      modelId,
      delta: dims.raw_quality || 0,
    }))
    .sort((a, b) => b.delta - a.delta)

  return (
    <div className={styles.results}>
      <StickyNote title="Session Complete!" color="green" rotation={-1} pinned>
        <div className={styles.resultStats}>
          <p>Battles completed: {results.n_battles}</p>
          <p>Models evaluated: {results.n_models}</p>
          <p>Prediction accuracy: {(results.prediction_accuracy * 100).toFixed(1)}%</p>
        </div>
      </StickyNote>

      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}>Rating Changes</h2>
        <ol className={styles.rankingList}>
          {deltas.map(({ modelId, delta }) => (
            <li key={modelId}>
              {modelId}{' '}
              <span className={delta >= 0 ? styles.delta : styles.deltaNeg}>
                {delta >= 0 ? '+' : ''}{delta.toFixed(1)} rating
              </span>
            </li>
          ))}
        </ol>

        <div className={styles.sessionMeta}>
          <p>Kendall's τ: {results.kendall_tau?.toFixed(3) || 'N/A'}</p>
          <p>Conservation: {results.conservation_satisfied ? '✓ Satisfied' : '✗ Not satisfied'}</p>
        </div>

        <SketchButton onClick={onReset}>Start New Session</SketchButton>
      </SketchCard>
    </div>
  )
}

export { Arena as default }
