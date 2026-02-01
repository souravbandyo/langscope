import { useState, useRef, useEffect } from 'react'
import styles from './MediaInput.module.css'

export type MediaType = 'audio' | 'image' | 'video'

export interface MediaInputValue {
  type: 'file' | 'url' | 'recording'
  file?: File
  url?: string
  blob?: Blob
  preview?: string
}

export interface MediaInputProps {
  mediaType: MediaType
  value: MediaInputValue | null
  onChange: (value: MediaInputValue | null) => void
  className?: string
}

/**
 * Media input component supporting file upload, URL input, and recording (for audio)
 */
export function MediaInput({
  mediaType,
  value,
  onChange,
  className,
}: MediaInputProps) {
  const [inputMode, setInputMode] = useState<'upload' | 'url' | 'record'>('upload')
  const [urlInput, setUrlInput] = useState('')
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setError(null)
    
    // Validate file type
    const validTypes = getValidMimeTypes(mediaType)
    if (!validTypes.some(type => file.type.startsWith(type))) {
      setError(`Invalid file type. Please upload ${mediaType} files.`)
      return
    }

    // Create preview URL
    const preview = URL.createObjectURL(file)
    
    onChange({
      type: 'file',
      file,
      preview,
    })
  }

  const handleUrlSubmit = () => {
    if (!urlInput.trim()) {
      setError('Please enter a URL')
      return
    }
    
    setError(null)
    onChange({
      type: 'url',
      url: urlInput.trim(),
      preview: urlInput.trim(),
    })
  }

  const handleClear = () => {
    onChange(null)
    setUrlInput('')
    setError(null)
  }

  const config = getMediaConfig(mediaType)

  return (
    <div className={`${styles.mediaInput} ${className || ''}`}>
      {/* Mode Selector */}
      <div className={styles.modeSelector}>
        <button
          type="button"
          className={`${styles.modeButton} ${inputMode === 'upload' ? styles.active : ''}`}
          onClick={() => setInputMode('upload')}
        >
          <i className="ph ph-upload-simple"></i>
          Upload
        </button>
        <button
          type="button"
          className={`${styles.modeButton} ${inputMode === 'url' ? styles.active : ''}`}
          onClick={() => setInputMode('url')}
        >
          <i className="ph ph-link"></i>
          URL
        </button>
        {mediaType === 'audio' && (
          <button
            type="button"
            className={`${styles.modeButton} ${inputMode === 'record' ? styles.active : ''}`}
            onClick={() => setInputMode('record')}
          >
            <i className="ph ph-microphone"></i>
            Record
          </button>
        )}
      </div>

      {/* Input Area */}
      <div className={styles.inputArea}>
        {inputMode === 'upload' && (
          <FileUploadArea
            mediaType={mediaType}
            config={config}
            onFileChange={handleFileChange}
          />
        )}

        {inputMode === 'url' && (
          <div className={styles.urlInputArea}>
            <input
              type="url"
              className={styles.urlInput}
              placeholder={`Paste ${mediaType} URL...`}
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
            />
            <button
              type="button"
              className={styles.urlSubmitButton}
              onClick={handleUrlSubmit}
            >
              Load
            </button>
          </div>
        )}

        {inputMode === 'record' && mediaType === 'audio' && (
          <AudioRecorder
            onRecordingComplete={(blob, preview) => {
              onChange({
                type: 'recording',
                blob,
                preview,
              })
            }}
          />
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className={styles.error}>
          <i className="ph ph-warning"></i> {error}
        </div>
      )}

      {/* Preview */}
      {value && (
        <div className={styles.preview}>
          <div className={styles.previewHeader}>
            <span className={styles.previewLabel}>
              <i className={config.icon}></i>
              {value.type === 'file' && value.file?.name}
              {value.type === 'url' && 'From URL'}
              {value.type === 'recording' && 'Recording'}
            </span>
            <button
              type="button"
              className={styles.clearButton}
              onClick={handleClear}
            >
              <i className="ph ph-x"></i>
            </button>
          </div>
          <MediaPreview mediaType={mediaType} value={value} />
        </div>
      )}
    </div>
  )
}

// File Upload Area Component
function FileUploadArea({
  mediaType,
  config,
  onFileChange,
}: {
  mediaType: MediaType
  config: MediaConfig
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <div 
      className={styles.uploadArea}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={config.accept}
        onChange={onFileChange}
        className={styles.hiddenInput}
      />
      <i className={`${config.icon} ${styles.uploadIcon}`}></i>
      <p className={styles.uploadText}>
        Click to upload or drag and drop
      </p>
      <p className={styles.uploadHint}>
        {config.hint}
      </p>
    </div>
  )
}

// Audio Recorder Component
function AudioRecorder({
  onRecordingComplete,
}: {
  onRecordingComplete: (blob: Blob, previewUrl: string) => void
}) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number | null>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Setup audio analyser for visual feedback
      const audioContext = new AudioContext()
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // Start visual feedback
      const updateLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
          analyserRef.current.getByteFrequencyData(dataArray)
          const avg = dataArray.reduce((a, b) => a + b) / dataArray.length
          setAudioLevel(avg / 255)
        }
        animationRef.current = requestAnimationFrame(updateLevel)
      }
      updateLevel()

      // Start recording
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const previewUrl = URL.createObjectURL(blob)
        onRecordingComplete(blob, previewUrl)
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop())
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1)
      }, 1000)

    } catch (err) {
      console.error('Failed to start recording:', err)
      alert('Could not access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className={styles.recorderArea}>
      {isRecording ? (
        <>
          <div className={styles.recordingIndicator}>
            <span className={styles.recordingDot}></span>
            Recording... {formatTime(recordingTime)}
          </div>
          <div 
            className={styles.audioLevelBar}
            style={{ '--level': audioLevel } as React.CSSProperties}
          />
          <button
            type="button"
            className={styles.stopButton}
            onClick={stopRecording}
          >
            <i className="ph ph-stop-circle"></i>
            Stop Recording
          </button>
        </>
      ) : (
        <button
          type="button"
          className={styles.recordButton}
          onClick={startRecording}
        >
          <i className="ph ph-microphone"></i>
          Start Recording
        </button>
      )}
    </div>
  )
}

// Media Preview Component
function MediaPreview({
  mediaType,
  value,
}: {
  mediaType: MediaType
  value: MediaInputValue
}) {
  if (!value.preview) return null

  switch (mediaType) {
    case 'audio':
      return (
        <audio
          src={value.preview}
          controls
          className={styles.audioPreview}
        />
      )
    case 'image':
      return (
        <img
          src={value.preview}
          alt="Preview"
          className={styles.imagePreview}
        />
      )
    case 'video':
      return (
        <video
          src={value.preview}
          controls
          className={styles.videoPreview}
        />
      )
    default:
      return null
  }
}

// Helper types and functions
interface MediaConfig {
  accept: string
  icon: string
  hint: string
}

function getMediaConfig(mediaType: MediaType): MediaConfig {
  switch (mediaType) {
    case 'audio':
      return {
        accept: 'audio/*',
        icon: 'ph ph-waveform',
        hint: 'MP3, WAV, WebM, or other audio formats',
      }
    case 'image':
      return {
        accept: 'image/*',
        icon: 'ph ph-image',
        hint: 'PNG, JPG, WebP, or other image formats',
      }
    case 'video':
      return {
        accept: 'video/*',
        icon: 'ph ph-video-camera',
        hint: 'MP4, WebM, or other video formats',
      }
  }
}

function getValidMimeTypes(mediaType: MediaType): string[] {
  switch (mediaType) {
    case 'audio':
      return ['audio/']
    case 'image':
      return ['image/']
    case 'video':
      return ['video/']
  }
}
