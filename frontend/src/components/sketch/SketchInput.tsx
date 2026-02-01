import { useRef, useEffect, useState, forwardRef } from 'react'
import rough from 'roughjs'
import { sketchColors, roughStyles } from '@/styles/sketch-theme'
import clsx from 'clsx'
import styles from './SketchInput.module.css'

export interface SketchInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: React.ReactNode
}

/**
 * Hand-drawn input component using Rough.js
 */
export const SketchInput = forwardRef<HTMLInputElement, SketchInputProps>(
  ({ label, error, icon, className, ...props }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const wrapperRef = useRef<HTMLDivElement>(null)
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
    const [isFocused, setIsFocused] = useState(false)

    // Measure wrapper size with ResizeObserver
    useEffect(() => {
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect
          setDimensions({ width, height })
        }
      })

      if (wrapperRef.current) {
        observer.observe(wrapperRef.current)
        // Initial measurement
        const rect = wrapperRef.current.getBoundingClientRect()
        setDimensions({ width: rect.width, height: rect.height })
      }

      return () => observer.disconnect()
    }, [])

    // Draw sketch border
    useEffect(() => {
      const canvas = canvasRef.current
      if (!canvas || dimensions.width === 0) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.clearRect(0, 0, dimensions.width, dimensions.height)

      const rc = rough.canvas(canvas)
      const padding = 2

      let strokeColor = sketchColors.pencil
      if (isFocused) strokeColor = sketchColors.accent
      if (error) strokeColor = sketchColors.error

      rc.rectangle(
        padding,
        padding,
        dimensions.width - padding * 2,
        dimensions.height - padding * 2,
        {
          ...roughStyles.input,
          stroke: strokeColor,
          strokeWidth: isFocused ? 2 : 1.2,
        }
      )
    }, [dimensions, isFocused, error])

    return (
      <div className={clsx(styles.container, className)}>
        {label && <label className={styles.label}>{label}</label>}
        <div ref={wrapperRef} className={styles.inputWrapper}>
          <canvas
            ref={canvasRef}
            width={dimensions.width}
            height={dimensions.height}
            className={styles.canvas}
          />
          {icon && <span className={styles.icon}>{icon}</span>}
          <input
            ref={ref}
            className={clsx(styles.input, icon && styles.hasIcon)}
            onFocus={(e) => {
              setIsFocused(true)
              props.onFocus?.(e)
            }}
            onBlur={(e) => {
              setIsFocused(false)
              props.onBlur?.(e)
            }}
            {...props}
          />
        </div>
        {error && <span className={styles.error}>{error}</span>}
      </div>
    )
  }
)

SketchInput.displayName = 'SketchInput'
