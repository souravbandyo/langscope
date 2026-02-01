import { useState } from 'react'
import styles from './GroundTruthInput.module.css'

export interface GroundTruthValue {
  text?: string
  testCases?: string
  expectedOutput?: string
  normalizationCode?: string
  useAutoNormalize: boolean
}

export interface GroundTruthInputProps {
  taskType: string
  value: GroundTruthValue | null
  onChange: (value: GroundTruthValue | null) => void
  className?: string
}

// Default normalization code templates
const DEFAULT_NORMALIZERS: Record<string, string> = {
  asr: `def normalize(text):
    """Standard ASR normalization for WER/CER calculation."""
    import re
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\\w\\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text`,
  
  asr_multilingual: `def normalize(text):
    """Multilingual ASR normalization (handles code-switching)."""
    import re
    # Lowercase
    text = text.lower()
    # Remove punctuation but keep Devanagari, Bengali, etc.
    text = re.sub(r'[^\\w\\s\\u0900-\\u097F\\u0980-\\u09FF]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Normalize numbers to words (optional)
    # text = numbers_to_words(text)
    return text`,

  tts: `def normalize(text):
    """TTS round-trip normalization."""
    import re
    text = text.lower()
    text = re.sub(r'[^\\w\\s]', '', text)
    text = ' '.join(text.split())
    return text`,

  code: `def normalize(code):
    """Code normalization - strip whitespace, normalize indentation."""
    import re
    lines = code.strip().split('\\n')
    # Remove empty lines
    lines = [l.rstrip() for l in lines if l.strip()]
    return '\\n'.join(lines)`,

  text: `def normalize(text):
    """General text normalization."""
    import re
    text = text.lower()
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()`,
}

/**
 * Ground truth input component with optional normalization code editor
 */
export function GroundTruthInput({
  taskType,
  value,
  onChange,
  className,
}: GroundTruthInputProps) {
  const [showNormalizer, setShowNormalizer] = useState(false)
  const [isEnabled, setIsEnabled] = useState(!!value)

  // Determine input type based on task
  const inputConfig = getInputConfig(taskType)

  const handleToggle = (enabled: boolean) => {
    setIsEnabled(enabled)
    if (enabled && !value) {
      // Initialize with defaults
      const defaultNormalizer = getDefaultNormalizer(taskType)
      onChange({
        text: '',
        useAutoNormalize: true,
        normalizationCode: defaultNormalizer,
      })
    } else if (!enabled) {
      onChange(null)
    }
  }

  const handleTextChange = (text: string) => {
    onChange({
      ...value,
      text,
      useAutoNormalize: value?.useAutoNormalize ?? true,
      normalizationCode: value?.normalizationCode || getDefaultNormalizer(taskType),
    })
  }

  const handleNormalizerToggle = (useAuto: boolean) => {
    onChange({
      ...value,
      text: value?.text || '',
      useAutoNormalize: useAuto,
      normalizationCode: useAuto ? getDefaultNormalizer(taskType) : (value?.normalizationCode || ''),
    })
  }

  const handleNormalizerCodeChange = (code: string) => {
    onChange({
      ...value,
      text: value?.text || '',
      useAutoNormalize: false,
      normalizationCode: code,
    })
  }

  return (
    <div className={`${styles.groundTruthInput} ${className || ''}`}>
      {/* Enable/Disable Toggle */}
      <div className={styles.toggleRow}>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={isEnabled}
            onChange={(e) => handleToggle(e.target.checked)}
          />
          <span className={styles.toggleLabel}>
            <i className="ph ph-target"></i>
            Include Ground Truth
          </span>
        </label>
        <span className={styles.toggleHint}>
          {isEnabled 
            ? 'Metrics like WER, CER will be calculated'
            : 'Blind testing without metrics'
          }
        </span>
      </div>

      {isEnabled && (
        <div className={styles.groundTruthContent}>
          {/* Ground Truth Text Input */}
          <div className={styles.inputSection}>
            <label className={styles.label}>
              <i className={inputConfig.icon}></i>
              {inputConfig.label}
            </label>
            <textarea
              className={styles.textarea}
              placeholder={inputConfig.placeholder}
              value={value?.text || ''}
              onChange={(e) => handleTextChange(e.target.value)}
              rows={4}
            />
            <span className={styles.inputHint}>{inputConfig.hint}</span>
          </div>

          {/* Normalization Section */}
          <div className={styles.normalizerSection}>
            <div className={styles.normalizerHeader}>
              <button
                type="button"
                className={styles.normalizerToggle}
                onClick={() => setShowNormalizer(!showNormalizer)}
              >
                <i className={`ph ${showNormalizer ? 'ph-caret-down' : 'ph-caret-right'}`}></i>
                <i className="ph ph-code"></i>
                Normalization
                <span className={styles.normalizerBadge}>
                  {value?.useAutoNormalize ? 'Auto' : 'Custom'}
                </span>
              </button>
            </div>

            {showNormalizer && (
              <div className={styles.normalizerContent}>
                {/* Auto/Custom Toggle */}
                <div className={styles.normalizerOptions}>
                  <button
                    type="button"
                    className={`${styles.optionButton} ${value?.useAutoNormalize ? styles.active : ''}`}
                    onClick={() => handleNormalizerToggle(true)}
                  >
                    <i className="ph ph-magic-wand"></i>
                    Auto-generate
                  </button>
                  <button
                    type="button"
                    className={`${styles.optionButton} ${!value?.useAutoNormalize ? styles.active : ''}`}
                    onClick={() => handleNormalizerToggle(false)}
                  >
                    <i className="ph ph-pencil"></i>
                    Custom code
                  </button>
                </div>

                {/* Normalization Code Editor */}
                <div className={styles.codeEditor}>
                  <div className={styles.codeHeader}>
                    <span className={styles.codeLang}>Python</span>
                    {value?.useAutoNormalize && (
                      <span className={styles.codeAutoLabel}>Auto-generated</span>
                    )}
                  </div>
                  <textarea
                    className={styles.codeTextarea}
                    value={value?.normalizationCode || getDefaultNormalizer(taskType)}
                    onChange={(e) => handleNormalizerCodeChange(e.target.value)}
                    readOnly={value?.useAutoNormalize}
                    rows={10}
                    spellCheck={false}
                  />
                </div>

                {/* Preset Templates */}
                <div className={styles.presetSection}>
                  <span className={styles.presetLabel}>Presets:</span>
                  <div className={styles.presetButtons}>
                    <button
                      type="button"
                      className={styles.presetButton}
                      onClick={() => handleNormalizerCodeChange(DEFAULT_NORMALIZERS.asr)}
                    >
                      Standard
                    </button>
                    <button
                      type="button"
                      className={styles.presetButton}
                      onClick={() => handleNormalizerCodeChange(DEFAULT_NORMALIZERS.asr_multilingual)}
                    >
                      Multilingual
                    </button>
                    <button
                      type="button"
                      className={styles.presetButton}
                      onClick={() => handleNormalizerCodeChange(DEFAULT_NORMALIZERS.code)}
                    >
                      Code
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Metrics Preview */}
          <div className={styles.metricsPreview}>
            <span className={styles.metricsLabel}>
              <i className="ph ph-chart-bar"></i>
              Metrics to calculate:
            </span>
            <div className={styles.metricsTags}>
              {getMetricsForTask(taskType).map((metric) => (
                <span key={metric} className={styles.metricTag}>
                  {metric}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Helper functions
interface InputConfig {
  label: string
  placeholder: string
  hint: string
  icon: string
}

function getInputConfig(taskType: string): InputConfig {
  if (taskType.includes('audio_to_text') || taskType === 'asr') {
    return {
      label: 'Reference Transcription',
      placeholder: 'Enter the correct transcription of the audio...',
      hint: 'This will be compared against model outputs to calculate WER, CER',
      icon: 'ph ph-article',
    }
  }
  if (taskType.includes('text_to_audio') || taskType === 'tts') {
    return {
      label: 'Original Text',
      placeholder: 'Enter the text that was synthesized...',
      hint: 'Used for round-trip WER calculation (TTS→ASR→compare)',
      icon: 'ph ph-article',
    }
  }
  if (taskType.includes('image_to_text') || taskType === 'visual_qa') {
    return {
      label: 'Expected Answer',
      placeholder: 'Enter the expected answer or caption...',
      hint: 'Compared using exact match, contains, or semantic similarity',
      icon: 'ph ph-check-circle',
    }
  }
  if (taskType.includes('text_to_code') || taskType === 'code') {
    return {
      label: 'Test Cases / Expected Output',
      placeholder: 'Enter test cases or expected output...',
      hint: 'Used to verify code correctness (pass@k)',
      icon: 'ph ph-test-tube',
    }
  }
  // Default
  return {
    label: 'Ground Truth',
    placeholder: 'Enter the expected output...',
    hint: 'Used to evaluate model responses',
    icon: 'ph ph-target',
  }
}

function getDefaultNormalizer(taskType: string): string {
  if (taskType.includes('audio_to_text') || taskType === 'asr') {
    return DEFAULT_NORMALIZERS.asr
  }
  if (taskType.includes('text_to_audio') || taskType === 'tts') {
    return DEFAULT_NORMALIZERS.tts
  }
  if (taskType.includes('code')) {
    return DEFAULT_NORMALIZERS.code
  }
  return DEFAULT_NORMALIZERS.text
}

function getMetricsForTask(taskType: string): string[] {
  if (taskType.includes('audio_to_text') || taskType === 'asr') {
    return ['WER', 'CER', 'MER', 'Accuracy']
  }
  if (taskType.includes('text_to_audio') || taskType === 'tts') {
    return ['Round-trip WER', 'UTMOS', 'SNR', 'Composite']
  }
  if (taskType.includes('image_to_text') || taskType === 'visual_qa') {
    return ['Exact Match', 'Contains', 'Accuracy']
  }
  if (taskType.includes('code')) {
    return ['Syntax Valid', 'Tests Pass', 'pass@k']
  }
  if (taskType.includes('text_to_text')) {
    return ['BLEU', 'ROUGE-L', 'F1']
  }
  return ['Accuracy', 'F1']
}

export { DEFAULT_NORMALIZERS }
