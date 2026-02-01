import { useRef, useEffect, useState } from 'react'
import rough from 'roughjs'
import { sketchColors, roughStyles } from '@/styles/sketch-theme'
import clsx from 'clsx'
import styles from './SketchCard.module.css'

export interface SketchCardProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
  hoverable?: boolean
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

/**
 * Hand-drawn card component using Rough.js
 */
export function SketchCard({
  children,
  className,
  onClick,
  hoverable = false,
  padding = 'md',
}: SketchCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Measure card size
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setDimensions({ width, height })
      }
    })

    if (cardRef.current) {
      observer.observe(cardRef.current)
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
    const padding = 3

    rc.rectangle(
      padding,
      padding,
      dimensions.width - padding * 2,
      dimensions.height - padding * 2,
      {
        ...roughStyles.card,
        stroke: sketchColors.ink,
        fill: sketchColors.paper,
        fillStyle: 'solid',
      }
    )
  }, [dimensions])

  const paddingClasses = {
    none: '',
    sm: styles.paddingSm,
    md: styles.paddingMd,
    lg: styles.paddingLg,
  }

  return (
    <div
      ref={cardRef}
      className={clsx(
        styles.sketchCard,
        paddingClasses[padding],
        hoverable && styles.hoverable,
        onClick && styles.clickable,
        className
      )}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className={styles.canvas}
      />
      <div className={styles.content}>{children}</div>
    </div>
  )
}
