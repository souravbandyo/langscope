import { useState } from 'react'
import styles from './ResponseContent.module.css'

// Task type is now Input→Output format (e.g., 'text_to_code', 'audio_to_text')
export type TaskType = 
  | 'text_to_text' 
  | 'text_to_code' 
  | 'text_to_audio' 
  | 'text_to_image' 
  | 'text_to_video'
  | 'audio_to_text'
  | 'audio_to_audio'
  | 'image_to_text'
  | 'image_to_image'
  | 'video_to_text'
  | 'video_to_video'
  // Legacy types for backward compatibility
  | 'code' 
  | 'text' 
  | 'audio' 
  | 'image' 
  | 'video'

export type OutputType = 'text' | 'code' | 'audio' | 'image' | 'video'

export interface ResponseContentProps {
  content: string
  taskType: TaskType
  language?: string // for code highlighting
  className?: string
}

// Map task type to output type for rendering
function getOutputType(taskType: TaskType): OutputType {
  // New Input→Output format
  if (taskType.includes('_to_')) {
    const output = taskType.split('_to_')[1] as OutputType
    return output
  }
  // Legacy format
  return taskType as OutputType
}

/**
 * Renders response content based on task type's output format
 * - code: Syntax highlighted code block
 * - text: Formatted text
 * - audio: Audio player
 * - image: Image viewer
 * - video: Video player
 */
export function ResponseContent({
  content,
  taskType,
  language = 'python',
  className,
}: ResponseContentProps) {
  const outputType = getOutputType(taskType)
  
  switch (outputType) {
    case 'code':
      return <CodeContent content={content} language={language} className={className} />
    case 'audio':
      return <AudioContent content={content} className={className} />
    case 'image':
      return <ImageContent content={content} className={className} />
    case 'video':
      return <VideoContent content={content} className={className} />
    case 'text':
    default:
      return <TextContent content={content} className={className} />
  }
}

// Code content with syntax highlighting and copy button
function CodeContent({
  content,
  language,
  className,
}: {
  content: string
  language: string
  className?: string
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <div className={`${styles.codeContainer} ${className || ''}`}>
      <div className={styles.codeHeader}>
        <span className={styles.codeLanguage}>{language}</span>
        <button
          className={styles.copyButton}
          onClick={handleCopy}
          type="button"
        >
          {copied ? '✓ Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={styles.codeBlock}>
        <code className={styles.code}>{content}</code>
      </pre>
    </div>
  )
}

// Text content renderer
function TextContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  return (
    <div className={`${styles.textContainer} ${className || ''}`}>
      <div className={styles.textContent}>
        {content.split('\n').map((paragraph, idx) => (
          <p key={idx} className={styles.paragraph}>
            {paragraph || '\u00A0'}
          </p>
        ))}
      </div>
    </div>
  )
}

// Audio content with player
function AudioContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(false)

  // For mock data, we'll show a placeholder if the URL doesn't work
  const handleError = () => {
    setError(true)
  }

  return (
    <div className={`${styles.audioContainer} ${className || ''}`}>
      <div className={styles.audioVisual}>
        {/* Placeholder waveform visualization */}
        <div className={styles.waveform}>
          {Array.from({ length: 40 }).map((_, i) => (
            <div
              key={i}
              className={`${styles.waveBar} ${isPlaying ? styles.waveBarActive : ''}`}
              style={{
                height: `${20 + Math.sin(i * 0.5) * 15 + Math.random() * 10}px`,
                animationDelay: `${i * 0.05}s`,
              }}
            />
          ))}
        </div>
      </div>
      {error ? (
        <div className={styles.audioPlaceholder}>
          <span className={styles.audioIcon}>🔊</span>
          <p>Audio preview not available</p>
          <p className={styles.audioNote}>Mock audio: {content}</p>
        </div>
      ) : (
        <audio
          className={styles.audioPlayer}
          controls
          src={content}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          onError={handleError}
        >
          Your browser does not support the audio element.
        </audio>
      )}
    </div>
  )
}

// Image content with viewer
function ImageContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  const [isZoomed, setIsZoomed] = useState(false)
  const [error, setError] = useState(false)

  const handleError = () => {
    setError(true)
  }

  return (
    <div className={`${styles.imageContainer} ${className || ''}`}>
      {error ? (
        <div className={styles.imagePlaceholder}>
          <span className={styles.imageIcon}>🖼️</span>
          <p>Image preview not available</p>
          <p className={styles.imageNote}>Mock image: {content}</p>
        </div>
      ) : (
        <>
          <img
            src={content}
            alt="Generated response"
            className={`${styles.image} ${isZoomed ? styles.imageZoomed : ''}`}
            onClick={() => setIsZoomed(!isZoomed)}
            onError={handleError}
          />
          <button
            className={styles.zoomButton}
            onClick={() => setIsZoomed(!isZoomed)}
            type="button"
          >
            {isZoomed ? 'Shrink' : 'Expand'}
          </button>
        </>
      )}
    </div>
  )
}

// Video content with player
function VideoContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  const [error, setError] = useState(false)

  const handleError = () => {
    setError(true)
  }

  return (
    <div className={`${styles.videoContainer} ${className || ''}`}>
      {error ? (
        <div className={styles.videoPlaceholder}>
          <span className={styles.videoIcon}>🎬</span>
          <p>Video preview not available</p>
          <p className={styles.videoNote}>Mock video: {content}</p>
        </div>
      ) : (
        <video
          className={styles.video}
          controls
          src={content}
          onError={handleError}
        >
          Your browser does not support the video element.
        </video>
      )}
    </div>
  )
}
