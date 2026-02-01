/**
 * ProfileImageUpload Component
 * Circular avatar with hover overlay for uploading profile images
 */

import { useRef, useState } from 'react'
import styles from './ProfileImageUpload.module.css'

interface ProfileImageUploadProps {
  currentImage?: string | null
  initials: string
  onUpload: (file: File) => void
  onRemove?: () => void
  isUploading?: boolean
  size?: 'sm' | 'md' | 'lg'
}

export function ProfileImageUpload({
  currentImage,
  initials,
  onUpload,
  onRemove,
  isUploading = false,
  size = 'md',
}: ProfileImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onUpload(file)
    }
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onUpload(file)
    }
  }

  const sizeClass = {
    sm: styles.sizeSm,
    md: styles.sizeMd,
    lg: styles.sizeLg,
  }[size]

  return (
    <div className={styles.container}>
      <div
        className={`${styles.avatar} ${sizeClass} ${isDragging ? styles.dragging : ''} ${isUploading ? styles.uploading : ''}`}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload profile image"
      >
        {currentImage ? (
          <img src={currentImage} alt="Profile" className={styles.image} />
        ) : (
          <span className={styles.initials}>{initials}</span>
        )}

        <div className={styles.overlay}>
          {isUploading ? (
            <i className="ph ph-spinner-gap"></i>
          ) : (
            <i className="ph ph-camera"></i>
          )}
          <span className={styles.overlayText}>
            {isUploading ? 'Uploading...' : 'Change'}
          </span>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/gif,image/webp"
        onChange={handleFileChange}
        className={styles.fileInput}
        aria-hidden="true"
      />

      {currentImage && onRemove && (
        <button
          className={styles.removeButton}
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          disabled={isUploading}
          aria-label="Remove profile image"
        >
          <i className="ph ph-trash"></i>
        </button>
      )}
    </div>
  )
}
