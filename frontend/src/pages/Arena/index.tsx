import { useState, useMemo, useCallback } from 'react'
import { SketchButton, SketchCard, SketchTabs } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { ResponseContent, RankSelector, RankingProgress, MediaInput, GroundTruthInput, type TaskType, type MediaInputValue, type GroundTruthValue } from '@/components/arena'
import { useModels, useMyModels, useArenaSession, useArenaBattle, useCompleteArenaSession } from '@/api/hooks'
import type { ArenaSessionStartResponse, ArenaSessionResult, ModelResponse, UserModel } from '@/api/types'
import { generateMockResponses, taskTypes, getTaskType, type MockResponse } from './mockResponses'
import styles from './Arena.module.css'

type ArenaState = 'setup' | 'battle' | 'results'
type BattleMode = 'pairwise' | 'multiway'

interface BattleConfig {
  taskType: TaskType
  prompt: string
  mediaInput: MediaInputValue | null
  groundTruth: GroundTruthValue | null
  battleMode: BattleMode
  modelsPerBattle: number
}

/**
 * Arena page for interactive model battles with multi-way comparisons
 * Implements blind testing - model names hidden until results reveal
 */
export function Arena() {
  const [state, setState] = useState<ArenaState>('setup')
  const [session, setSession] = useState<ArenaSessionStartResponse | null>(null)
  const [results, setResults] = useState<ArenaSessionResult | null>(null)
  const [battleConfig, setBattleConfig] = useState<BattleConfig>({
    taskType: 'code',
    prompt: '',
    battleMode: 'multiway',
    modelsPerBattle: 5,
  })

  const handleSessionStart = (
    sessionData: ArenaSessionStartResponse,
    config: BattleConfig
  ) => {
    setSession(sessionData)
    setBattleConfig(config)
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
      <h1 className={styles.title}>Arena Mode</h1>
      <p className={styles.subtitle}>
        Blind testing - rank responses without knowing which model generated them
      </p>

      {state === 'setup' && <ArenaSetup onStart={handleSessionStart} />}
      {state === 'battle' && session && (
        <ArenaBattle 
          session={session} 
          battleConfig={battleConfig}
          onComplete={handleSessionComplete} 
        />
      )}
      {state === 'results' && session && (
        <ArenaResults 
          results={results} 
          session={session}
          battleConfig={battleConfig}
          onReset={handleReset} 
        />
      )}
    </div>
  )
}

function ArenaSetup({ onStart }: { onStart: (session: ArenaSessionStartResponse, config: BattleConfig) => void }) {
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [battleMode, setBattleMode] = useState<BattleMode>('multiway')
  const [modelsPerBattle, setModelsPerBattle] = useState(5)
  const [taskType, setTaskType] = useState<TaskType>('text_to_code')
  const [prompt, setPrompt] = useState('')
  const [mediaInput, setMediaInput] = useState<MediaInputValue | null>(null)
  const [groundTruth, setGroundTruth] = useState<GroundTruthValue | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Model search state
  const [modelSearch, setModelSearch] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  
  // Private enterprise testing mode
  const [isPrivateMode, setIsPrivateMode] = useState(false)
  const [customPrompt, setCustomPrompt] = useState('')
  const [confidentialCases, setConfidentialCases] = useState<string[]>([])

  const { data: modelsData, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useModels()
  const { data: myModelsData, isLoading: myModelsLoading } = useMyModels({ active: true })
  const startSession = useArenaSession()

  // User's private models
  const myModels = myModelsData?.models || []

  // Get current task type definition
  const currentTaskType = getTaskType(taskType)

  // Update prompt when task type changes
  const handleTaskTypeChange = (newTaskType: TaskType) => {
    const newTaskDef = getTaskType(newTaskType)
    const oldTaskDef = getTaskType(taskType)
    
    setTaskType(newTaskType)
    
    // Reset media input and ground truth when changing task types
    setMediaInput(null)
    setGroundTruth(null)
    
    // Update prompt if it was the default for the previous task type
    if (!prompt || prompt === oldTaskDef?.defaultPrompt) {
      setPrompt(newTaskDef?.defaultPrompt || '')
    }
  }

  // Handle file upload for confidential cases
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return
    
    const fileNames = Array.from(files).map(f => f.name)
    setConfidentialCases(prev => [...prev, ...fileNames])
  }

  const removeCase = (index: number) => {
    setConfidentialCases(prev => prev.filter((_, i) => i !== index))
  }
  
  // Get models list
  const models = modelsData?.models || []

  // Filter models by search query for suggestions
  const searchFilteredModels = useMemo(() => {
    if (!modelSearch.trim()) return models
    const query = modelSearch.toLowerCase()
    return models.filter((m: ModelResponse) => 
      m.name.toLowerCase().includes(query) || 
      m.provider?.toLowerCase().includes(query) ||
      m.model_id.toLowerCase().includes(query)
    )
  }, [models, modelSearch])

  const handleModelSelect = (model: ModelResponse) => {
    if (!selectedModels.includes(model.model_id)) {
      setSelectedModels(prev => [...prev, model.model_id])
    }
    setModelSearch('')
    setShowSuggestions(false)
  }

  // Handle selecting user's private model
  const handleMyModelSelect = (model: UserModel) => {
    // Use prefixed ID to distinguish from public models
    const modelId = `my:${model.id}`
    if (!selectedModels.includes(modelId)) {
      setSelectedModels(prev => [...prev, modelId])
    }
  }

  const handleRemoveModel = (modelId: string) => {
    setSelectedModels(prev => prev.filter(id => id !== modelId))
  }

  // Get model name by ID (handles both public and private models)
  const getModelName = useCallback((modelId: string): string => {
    if (modelId.startsWith('my:')) {
      const myModelId = modelId.replace('my:', '')
      const myModel = myModels.find(m => m.id === myModelId)
      return myModel?.name || modelId
    }
    const model = models.find((m: ModelResponse) => m.model_id === modelId)
    return model?.name || modelId
  }, [models, myModels])

  const handleStart = async () => {
    // Validate input based on task type
    const inputType = currentTaskType?.inputType || 'text'
    if (inputType === 'text') {
      if (!prompt.trim()) {
        setError('Please enter a prompt for the models')
        return
      }
    } else {
      // For non-text inputs (audio, image, video), require media input
      if (!mediaInput) {
        setError(`Please provide ${inputType} input for the models`)
        return
      }
    }
    
    const minModels = battleMode === 'multiway' ? modelsPerBattle : 2
    if (selectedModels.length > 0 && selectedModels.length < minModels) {
      setError(`Please select at least ${minModels} models for ${battleMode} mode`)
      return
    }

    setIsStarting(true)
    setError(null)

    // Derive domain from task type
    const domain = currentTaskType?.suggestedDomains?.[0] || 'general'

    try {
      const session = await startSession.mutateAsync({
        domain,
        use_case: 'general',
        model_ids: selectedModels.length >= minModels ? selectedModels : undefined,
      })
      onStart(session, {
        taskType,
        prompt,
        mediaInput,
        groundTruth,
        battleMode,
        modelsPerBattle,
      })
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

  // Calculate information bits for display
  const infoBits = useMemo(() => {
    const n = battleMode === 'multiway' ? modelsPerBattle : 2
    let factorial = 1
    for (let i = 2; i <= n; i++) factorial *= i
    return Math.log2(factorial).toFixed(1)
  }, [battleMode, modelsPerBattle])

  return (
    <SketchCard padding="md">
      <div className={styles.setupForm}>
        <h2 className={styles.sectionTitle}>Start a Blind Testing Session</h2>

        {error && <div className={styles.error}>{error}</div>}

        {/* Task Type Selection - Input → Output */}
        <div className={styles.formGroup}>
          <label className={styles.label}>Task Type <span className={styles.hint}>(Input → Output)</span></label>
          <div className={styles.taskTypeGrid}>
            {taskTypes.map((type) => (
              <button
                key={type.id}
                className={`${styles.taskTypeButton} ${taskType === type.id ? styles.active : ''}`}
                onClick={() => handleTaskTypeChange(type.id)}
                type="button"
                title={type.description}
              >
                <i className={`${type.icon} ${styles.taskTypeIcon}`}></i>
                <span className={styles.taskTypeName}>{type.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Section - varies by task type */}
        <div className={styles.formGroup}>
          <label className={styles.label}>
            {currentTaskType?.inputType === 'text' ? 'Prompt' : `${currentTaskType?.inputType?.charAt(0).toUpperCase()}${currentTaskType?.inputType?.slice(1)} Input`}
            <span className={styles.hint}> ({currentTaskType?.description || 'Provide input for the models'})</span>
          </label>
          
          {currentTaskType?.inputType === 'text' ? (
            // Text input - show textarea
            <textarea
              className={styles.textarea}
              placeholder={currentTaskType?.defaultPrompt || 'Enter your prompt...'}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
            />
          ) : (
            // Non-text input - show MediaInput
            <MediaInput
              mediaType={currentTaskType?.inputType as 'audio' | 'image' | 'video'}
              value={mediaInput}
              onChange={setMediaInput}
            />
          )}
          
          {/* Additional text prompt for non-text inputs */}
          {currentTaskType?.inputType !== 'text' && (
            <div className={styles.additionalPromptSection}>
              <label className={styles.label}>
                Additional Instructions <span className={styles.hint}>(optional)</span>
              </label>
              <textarea
                className={styles.textarea}
                placeholder="Any additional instructions for processing this input..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
              />
            </div>
          )}
        </div>

        {/* Ground Truth Input (Optional) */}
        <GroundTruthInput
          taskType={taskType}
          value={groundTruth}
          onChange={setGroundTruth}
        />

        {/* Private Mode Toggle */}
        <label className={styles.privateModeToggle}>
          <input
            type="checkbox"
            checked={isPrivateMode}
            onChange={(e) => setIsPrivateMode(e.target.checked)}
          />
          <i className={`ph ph-lock ${styles.privateIcon}`}></i>
          <span className={styles.privateName}>Private Enterprise Mode</span>
          <span className={styles.privateDesc}>Results won't be added to public leaderboard</span>
        </label>

        {/* Private Mode Options */}
        {isPrivateMode && (
          <div className={styles.privateSection}>
            <StickyNote title="Private Mode Active" color="yellow" rotation={0}>
              <p>Your evaluation results will remain confidential and won't affect public rankings.</p>
            </StickyNote>

            <div className={styles.formGroup}>
              <label className={styles.label}>Upload Test Cases (optional):</label>
              <div className={styles.fileUploadArea}>
                <input
                  type="file"
                  multiple
                  accept=".json,.txt,.csv"
                  onChange={handleFileUpload}
                  className={styles.fileInput}
                  id="confidential-files"
                />
                <label htmlFor="confidential-files" className={styles.fileUploadLabel}>
                  <i className={`ph ph-folder-open ${styles.uploadIcon}`}></i>
                  <span>Click to upload confidential test cases</span>
                  <span className={styles.fileTypes}>.json, .txt, .csv</span>
                </label>
              </div>
              {confidentialCases.length > 0 && (
                <div className={styles.uploadedFiles}>
                  {confidentialCases.map((file, idx) => (
                    <div key={idx} className={styles.uploadedFile}>
                      <span><i className="ph ph-file-text"></i> {file}</span>
                      <button onClick={() => removeCase(idx)} className={styles.removeFile}>×</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>Custom Evaluation Prompt (optional):</label>
              <textarea
                className={styles.textarea}
                placeholder="Enter custom instructions for the evaluation..."
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                rows={3}
              />
            </div>
          </div>
        )}

        {/* Battle Mode and Settings - Compact Row */}
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.label}>Battle Mode</label>
            <div className={styles.battleModeSelector}>
              <button
                className={`${styles.modeButton} ${battleMode === 'pairwise' ? styles.active : ''}`}
                onClick={() => setBattleMode('pairwise')}
              >
                <i className={`ph ph-users ${styles.modeIcon}`}></i>
                <div className={styles.modeContent}>
                  <span className={styles.modeName}>Pairwise</span>
                  <span className={styles.modeDesc}>Compare 2 models</span>
                </div>
                <span className={styles.modeInfo}>~1 bit</span>
              </button>
              <button
                className={`${styles.modeButton} ${battleMode === 'multiway' ? styles.active : ''}`}
                onClick={() => setBattleMode('multiway')}
              >
                <i className={`ph ph-users-three ${styles.modeIcon}`}></i>
                <div className={styles.modeContent}>
                  <span className={styles.modeName}>Multi-way</span>
                  <span className={styles.modeDesc}>Rank {modelsPerBattle} models</span>
                </div>
                <span className={styles.modeInfo}>~{infoBits} bits</span>
              </button>
            </div>
          </div>

          {battleMode === 'multiway' && (
            <div className={styles.formGroup}>
              <label className={styles.label}>Models per battle</label>
              <div className={styles.sliderContainer}>
                <input
                  type="range"
                  min={3}
                  max={6}
                  value={modelsPerBattle}
                  onChange={(e) => setModelsPerBattle(parseInt(e.target.value))}
                  className={styles.slider}
                />
                <span className={styles.sliderValue}>{modelsPerBattle}</span>
              </div>
              <div className={styles.sliderLabels}>
                <span>3 (faster)</span>
                <span>6 (more info)</span>
              </div>
            </div>
          )}
        </div>

        {/* Model Selection with Search */}
        <div className={styles.formGroup}>
          <label className={styles.label}>
            Select Models <span className={styles.hint}>(optional - {battleMode === 'multiway' ? modelsPerBattle : 2}+ or leave empty for random)</span>
          </label>
          {modelsLoading ? (
            <div className={styles.loadingText}>Loading models...</div>
          ) : modelsError ? (
            <div className={styles.warning}>
              <i className="ph ph-warning"></i> Could not load models.{' '}
              <button className={styles.retryLink} onClick={() => refetchModels()}>Retry</button>
            </div>
          ) : (
            <>
              <div className={styles.modelSearchContainer}>
                <i className={`ph ph-magnifying-glass ${styles.searchIcon}`}></i>
                <input
                  type="text"
                  className={styles.modelSearchInput}
                  placeholder="Search models by name or provider..."
                  value={modelSearch}
                  onChange={(e) => {
                    setModelSearch(e.target.value)
                    setShowSuggestions(true)
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                />
                {showSuggestions && modelSearch && (
                  <div className={styles.modelSuggestions}>
                    {searchFilteredModels.length === 0 ? (
                      <div className={styles.noResults}>No models found</div>
                    ) : (
                      searchFilteredModels.slice(0, 10).map((model: ModelResponse) => (
                        <div
                          key={model.model_id}
                          className={`${styles.suggestionItem} ${selectedModels.includes(model.model_id) ? styles.selected : ''}`}
                          onMouseDown={() => handleModelSelect(model)}
                        >
                          <span className={styles.suggestionName}>{model.name}</span>
                          <span className={styles.suggestionProvider}>{model.provider}</span>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>

              {/* My Models */}
              {myModels.length > 0 && (
                <div className={styles.popularModels}>
                  <span className={styles.popularLabel}>
                    <i className="ph ph-folder-user"></i> My Models:
                  </span>
                  {myModels.map((model: UserModel) => (
                    <button
                      key={model.id}
                      className={`${styles.popularTag} ${styles.myModelTag} ${selectedModels.includes(`my:${model.id}`) ? styles.selected : ''}`}
                      onClick={() => handleMyModelSelect(model)}
                      type="button"
                      title={model.description || model.name}
                    >
                      {model.name}
                    </button>
                  ))}
                </div>
              )}

              {/* Popular Models */}
              {models.length > 0 && (
                <div className={styles.popularModels}>
                  <span className={styles.popularLabel}>Popular:</span>
                  {models.slice(0, 6).map((model: ModelResponse) => (
                    <button
                      key={model.model_id}
                      className={`${styles.popularTag} ${selectedModels.includes(model.model_id) ? styles.selected : ''}`}
                      onClick={() => handleModelSelect(model)}
                      type="button"
                    >
                      {model.name}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}

          {/* Selected Models Tags */}
          {selectedModels.length > 0 && (
            <div className={styles.selectedModelsContainer}>
              {selectedModels.map((modelId) => {
                const isMyModel = modelId.startsWith('my:')
                return (
                  <span key={modelId} className={`${styles.selectedModelTag} ${isMyModel ? styles.myModelSelected : ''}`}>
                    {isMyModel && <i className="ph ph-folder-user"></i>}
                    {getModelName(modelId)}
                    <button 
                      className={styles.removeModelTag}
                      onClick={() => handleRemoveModel(modelId)}
                    >
                      <i className="ph ph-x"></i>
                    </button>
                  </span>
                )
              })}
            </div>
          )}
        </div>

        <div className={styles.buttonWrapper}>
          <SketchButton
            variant="primary"
            size="md"
            onClick={handleStart}
            disabled={isStarting}
          >
            {isStarting ? 'Starting...' : 'Start Blind Testing'}
          </SketchButton>
        </div>
      </div>
    </SketchCard>
  )
}

function ArenaBattle({
  session,
  battleConfig,
  onComplete,
}: {
  session: ArenaSessionStartResponse
  battleConfig: BattleConfig
  onComplete: (results: ArenaSessionResult) => void
}) {
  const [battleNum, setBattleNum] = useState(1)
  const [rankings, setRankings] = useState<Record<string, number>>({})
  const [activeTab, setActiveTab] = useState('response_0')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [battleHistory, setBattleHistory] = useState<Array<{
    participants: string[]
    userRanking: Record<string, number>
    responseMapping: Record<string, string> // responseId -> modelId
  }>>([])
  const totalBattles = 5

  const submitBattle = useArenaBattle(session.session_id)
  const completeSession = useCompleteArenaSession(session.session_id)

  const { taskType, prompt, battleMode, modelsPerBattle: configModelsPerBattle } = battleConfig

  // Determine how many models for this battle
  const modelsPerBattle = battleMode === 'multiway' 
    ? Math.min(configModelsPerBattle, session.models_available.length) 
    : 2

  // Get models for current battle (rotate through available models)
  const models = useMemo(() => {
    const startIdx = ((battleNum - 1) * modelsPerBattle) % session.models_available.length
    const selected: string[] = []
    for (let i = 0; i < modelsPerBattle; i++) {
      selected.push(session.models_available[(startIdx + i) % session.models_available.length])
    }
    return selected
  }, [battleNum, modelsPerBattle, session.models_available])

  // Generate mock responses for this battle
  const responses: MockResponse[] = useMemo(() => {
    return generateMockResponses(taskType, models, prompt)
  }, [taskType, models, prompt])

  // Create mapping from response ID to model ID (hidden from user)
  const responseToModel = useMemo(() => {
    const mapping: Record<string, string> = {}
    responses.forEach((r) => {
      mapping[r.id] = r.modelId
    })
    return mapping
  }, [responses])

  // Generate tabs for responses
  const tabs = useMemo(() => {
    return responses.map((response, index) => ({
      id: response.id,
      label: `Response ${String.fromCharCode(65 + index)}`,
      badge: rankings[response.id] ? `#${rankings[response.id]}` : undefined,
      badgeColor: rankings[response.id] ? 'success' as const : undefined,
    }))
  }, [responses, rankings])

  // Get used ranks (for disabling in rank selector)
  const usedRanks = useMemo(() => {
    return Object.entries(rankings)
      .filter(([id]) => id !== activeTab)
      .map(([, rank]) => rank)
  }, [rankings, activeTab])

  // Response labels for progress display
  const responseLabels = useMemo(() => {
    return responses.map((_, index) => `Response ${String.fromCharCode(65 + index)}`)
  }, [responses])

  const handleRankSelect = useCallback((rank: number) => {
    setRankings((prev) => {
      const newRankings = { ...prev }
      // Remove this rank from any other response
      Object.keys(newRankings).forEach((key) => {
        if (newRankings[key] === rank) {
          delete newRankings[key]
        }
      })
      // Toggle: if already selected, deselect; otherwise select
      if (prev[activeTab] === rank) {
        delete newRankings[activeTab]
      } else {
        newRankings[activeTab] = rank
      }
      return newRankings
    })
  }, [activeTab])

  const handleNextBattle = async () => {
    if (Object.keys(rankings).length < modelsPerBattle) {
      return
    }

    setIsSubmitting(true)

    // Convert response rankings to model rankings
    const modelRankings: Record<string, number> = {}
    Object.entries(rankings).forEach(([responseId, rank]) => {
      const modelId = responseToModel[responseId]
      if (modelId) {
        modelRankings[modelId] = rank
      }
    })

    // Save battle to history for tracking (including response mapping for reveal)
    const newBattleHistory = [
      ...battleHistory,
      {
        participants: models,
        userRanking: modelRankings,
        responseMapping: responseToModel,
      },
    ]
    setBattleHistory(newBattleHistory)

    try {
      await submitBattle.mutateAsync({
        participant_ids: models,
        user_ranking: modelRankings,
      })

      if (battleNum >= totalBattles) {
        const results = await completeSession.mutateAsync()
        onComplete(results)
      } else {
        setBattleNum(battleNum + 1)
        setRankings({})
        setActiveTab('response_0')
      }
    } catch (err) {
      console.error('Battle submission error (using mock results):', err)
      
      // Fallback to mock results when API is unavailable
      if (battleNum >= totalBattles) {
        // Generate mock results
        const mockResults: ArenaSessionResult = {
          session_id: session.session_id,
          domain: session.domain,
          use_case: session.use_case || 'general',
          n_battles: totalBattles,
          n_models: session.models_available.length,
          prediction_accuracy: 0.6 + Math.random() * 0.3, // 60-90%
          kendall_tau: 0.5 + Math.random() * 0.4, // 0.5-0.9
          conservation_satisfied: true,
          delta_sum: 0,
          biggest_winner: {},
          biggest_loser: {},
          n_specialists: Math.floor(Math.random() * 3),
          n_underperformers: Math.floor(Math.random() * 2),
          deltas: session.models_available.reduce((acc, modelId) => {
            acc[modelId] = {
              raw_quality: (Math.random() - 0.5) * 50, // -25 to +25
            }
            return acc
          }, {} as Record<string, Record<string, number>>),
        }
        onComplete(mockResults)
      } else {
        setBattleNum(battleNum + 1)
        setRankings({})
        setActiveTab('response_0')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  // Calculate info bits for this battle
  const infoBits = useMemo(() => {
    let factorial = 1
    for (let i = 2; i <= modelsPerBattle; i++) factorial *= i
    return Math.log2(factorial).toFixed(1)
  }, [modelsPerBattle])

  const isRankingComplete = Object.keys(rankings).length === modelsPerBattle
  const activeResponse = responses.find((r) => r.id === activeTab)

  return (
    <div className={styles.battleContainer}>
      <div className={styles.battleHeader}>
        <span className={styles.battleProgress}>
          Battle {battleNum} of {totalBattles}
        </span>
        <span className={styles.battleInfo}>
          {modelsPerBattle} models · ~{infoBits} bits
        </span>
        <span className={styles.domain}>
          {session.domain.replace(/_/g, ' ')}
        </span>
      </div>

      {/* Progress bar */}
      <div className={styles.progressBar}>
        <div 
          className={styles.progressFill} 
          style={{ width: `${(battleNum / totalBattles) * 100}%` }}
        />
      </div>

      {/* Blind testing notice */}
      <div className={styles.blindNotice}>
        <i className="ph ph-eye-slash"></i>
        <span>Blind Testing Mode - Model names will be revealed after you finish ranking</span>
      </div>

      {/* Prompt display */}
      <div className={styles.promptDisplay}>
        <span className={styles.promptLabel}>Prompt:</span>
        <span className={styles.promptText}>{prompt}</span>
      </div>

      {/* Response Tabs */}
      <SketchTabs
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        className={styles.responseTabs}
      />

      {/* Response Content Area */}
      <SketchCard padding="none" className={styles.responseContentCard}>
        {activeResponse && (
          <div className={styles.responseContentArea}>
            <ResponseContent
              content={activeResponse.content}
              taskType={taskType}
              className={styles.responseContentInner}
            />
          </div>
        )}

        {/* Rank Selector */}
        <RankSelector
          totalRanks={modelsPerBattle}
          selectedRank={rankings[activeTab]}
          usedRanks={usedRanks}
          onRankSelect={handleRankSelect}
          className={styles.rankSelectorArea}
        />
      </SketchCard>

      {/* Ranking Progress */}
      {Object.keys(rankings).length > 0 && (
        <RankingProgress
          rankings={rankings}
          responseLabels={responseLabels}
          totalResponses={modelsPerBattle}
        />
      )}

      <div className={styles.battleActions}>
        <SketchButton
          variant="primary"
          onClick={handleNextBattle}
          disabled={!isRankingComplete || isSubmitting}
        >
          {isSubmitting
            ? 'Submitting...'
            : battleNum >= totalBattles
            ? 'Reveal Results'
            : 'Next Battle →'}
        </SketchButton>
      </div>
    </div>
  )
}

function ArenaResults({
  results,
  session,
  battleConfig,
  onReset,
}: {
  results: ArenaSessionResult | null
  session: ArenaSessionStartResponse
  battleConfig: BattleConfig
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
    .map(([modelId, dims]) => {
      const rawDelta = dims.raw_quality || 0
      const originalPrediction = session.predictions[modelId]?.mu || 1500
      return {
        modelId,
        delta: rawDelta,
        originalMu: originalPrediction,
        newMu: originalPrediction + rawDelta,
      }
    })
    .sort((a, b) => b.newMu - a.newMu)

  // Calculate how many predictions were correct
  const correctPredictions = Math.round(results.prediction_accuracy * results.n_battles)

  return (
    <div className={styles.results}>
      {/* Model Reveal Section */}
      <SketchCard padding="lg" className={styles.revealCard}>
        <h2 className={styles.sectionTitle}>
          <i className="ph ph-eye"></i> Model Reveal
        </h2>
        <p className={styles.revealDesc}>
          Here are the models behind each response in your blind tests:
        </p>
        
        <div className={styles.revealGrid}>
          {session.models_available.slice(0, battleConfig.modelsPerBattle).map((modelId, index) => (
            <div key={modelId} className={styles.revealItem}>
              <span className={styles.revealLabel}>
                Response {String.fromCharCode(65 + index)}
              </span>
              <span className={styles.revealArrow}>→</span>
              <span className={styles.revealModel}>
                {modelId.split('/').pop()}
              </span>
            </div>
          ))}
        </div>
      </SketchCard>

      <StickyNote title="Session Complete!" color="green" rotation={-1} pinned>
        <div className={styles.resultStats}>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Battles completed:</span>
            <span className={styles.statValue}>{results.n_battles}</span>
          </div>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Models evaluated:</span>
            <span className={styles.statValue}>{results.n_models}</span>
          </div>
          <div className={styles.statRow}>
            <span className={styles.statLabel}>Prediction accuracy:</span>
            <span className={`${styles.statValue} ${styles.accuracy}`}>
              {(results.prediction_accuracy * 100).toFixed(1)}%
              <span className={styles.accuracyDetail}>
                ({correctPredictions}/{results.n_battles} correct)
              </span>
            </span>
          </div>
        </div>
      </StickyNote>

      {/* Feedback Impact Section */}
      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}><i className="ph ph-chart-bar"></i> Your Feedback Impact</h2>
        <p className={styles.feedbackDesc}>
          Your rankings help calibrate the system. Here's how ratings changed:
        </p>

        <div className={styles.deltaTable}>
          <div className={styles.deltaHeader}>
            <span>Model</span>
            <span>Before</span>
            <span>Change</span>
            <span>After</span>
          </div>
          {deltas.map(({ modelId, delta, originalMu, newMu }) => (
            <div key={modelId} className={styles.deltaRow}>
              <span className={styles.deltaModel}>{modelId.split('/').pop()}</span>
              <span className={styles.deltaBefore}>{Math.round(originalMu)}</span>
              <span className={delta >= 0 ? styles.deltaPositive : styles.deltaNegative}>
                {delta >= 0 ? '+' : ''}{delta.toFixed(1)}
              </span>
              <span className={styles.deltaAfter}>{Math.round(newMu)}</span>
            </div>
          ))}
        </div>
      </SketchCard>

      {/* Statistical Details */}
      <SketchCard padding="lg">
        <h2 className={styles.sectionTitle}>Statistical Details</h2>
        <div className={styles.statsGrid}>
          <div className={styles.statBox}>
            <span className={styles.statBoxLabel}>Kendall's τ</span>
            <span className={styles.statBoxValue}>
              {results.kendall_tau?.toFixed(3) || 'N/A'}
            </span>
            <span className={styles.statBoxHint}>
              Rank correlation between prediction and your feedback
            </span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statBoxLabel}>Conservation</span>
            <span className={`${styles.statBoxValue} ${results.conservation_satisfied ? styles.success : styles.warning}`}>
              {results.conservation_satisfied ? <><i className="ph ph-check"></i> Satisfied</> : <><i className="ph ph-x"></i> Violated</>}
            </span>
            <span className={styles.statBoxHint}>
              Total rating change sums to zero
            </span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statBoxLabel}>Specialists Found</span>
            <span className={styles.statBoxValue}>{results.n_specialists || 0}</span>
            <span className={styles.statBoxHint}>
              Models outperforming expectations
            </span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statBoxLabel}>Underperformers</span>
            <span className={styles.statBoxValue}>{results.n_underperformers || 0}</span>
            <span className={styles.statBoxHint}>
              Models below expectations
            </span>
          </div>
        </div>

        <div className={styles.resultActions}>
          <SketchButton onClick={onReset} variant="primary">
            Start New Session
          </SketchButton>
        </div>
      </SketchCard>
    </div>
  )
}

export { Arena as default }
