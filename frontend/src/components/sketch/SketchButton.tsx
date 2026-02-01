import { useRef, useEffect, useState } from 'react'
import rough from 'roughjs'
import { sketchColors, roughStyles } from '@/styles/sketch-theme'
import clsx from 'clsx'
import styles from './SketchButton.module.css'

export interface SketchButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
}

/**
 * Hand-drawn button component using Rough.js
 */
export function SketchButton({
  variant = 'default',
  size = 'md',
  children,
  className,
  disabled,
  ...props
}: SketchButtonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Measure button size
  useEffect(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect()
      setDimensions({ width: rect.width, height: rect.height })
    }
  }, [children, size])

  // Draw sketch border
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || dimensions.width === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, dimensions.width, dimensions.height)

    const rc = rough.canvas(canvas)
    const padding = 2

    // Determine colors based on variant
    let strokeColor = sketchColors.ink
    let fillColor = 'transparent'

    if (variant === 'primary') {
      fillColor = sketchColors.accent
      strokeColor = sketchColors.ink
    } else if (variant === 'outline') {
      strokeColor = sketchColors.pencil
    }

    if (disabled) {
      strokeColor = sketchColors.pencilLight
      fillColor = variant === 'primary' ? sketchColors.grid : 'transparent'
    }

    // Draw the sketch rectangle
    rc.rectangle(
      padding,
      padding,
      dimensions.width - padding * 2,
      dimensions.height - padding * 2,
      {
        ...roughStyles.button,
        stroke: strokeColor,
        fill: fillColor,
        fillStyle: variant === 'primary' ? 'solid' : 'hachure',
      }
    )
  }, [dimensions, variant, disabled])

  const sizeClasses = {
    sm: styles.sizeSm,
    md: styles.sizeMd,
    lg: styles.sizeLg,
  }

  return (
    <button
      ref={buttonRef}
      className={clsx(
        styles.sketchButton,
        sizeClasses[size],
        variant === 'primary' && styles.primary,
        disabled && styles.disabled,
        className
      )}
      disabled={disabled}
      {...props}
    >
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className={styles.canvas}
      />
      <span className={styles.content}>{children}</span>
    </button>
  )
}
