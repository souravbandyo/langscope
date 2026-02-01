import { useRef, useEffect, useState } from 'react'
import rough from 'roughjs'
import { sketchColors, roughStyles } from '@/styles/sketch-theme'
import clsx from 'clsx'
import styles from './SketchTabs.module.css'

export interface SketchTab {
  id: string
  label: string
  badge?: string | number
  badgeColor?: 'success' | 'warning' | 'error' | 'default'
}

export interface SketchTabsProps {
  tabs: SketchTab[]
  activeTab: string
  onTabChange: (tabId: string) => void
  className?: string
}

/**
 * Hand-drawn tabs component using Rough.js
 */
export function SketchTabs({
  tabs,
  activeTab,
  onTabChange,
  className,
}: SketchTabsProps) {
  return (
    <div className={clsx(styles.tabsContainer, className)}>
      <div className={styles.tabsList}>
        {tabs.map((tab) => (
          <SketchTabButton
            key={tab.id}
            tab={tab}
            isActive={activeTab === tab.id}
            onClick={() => onTabChange(tab.id)}
          />
        ))}
      </div>
    </div>
  )
}

interface SketchTabButtonProps {
  tab: SketchTab
  isActive: boolean
  onClick: () => void
}

function SketchTabButton({ tab, isActive, onClick }: SketchTabButtonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Measure button size
  useEffect(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect()
      setDimensions({ width: rect.width, height: rect.height })
    }
  }, [tab.label, tab.badge])

  // Draw sketch border
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || dimensions.width === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, dimensions.width, dimensions.height)

    const rc = rough.canvas(canvas)
    const padding = 2

    // Draw tab shape - active tabs have a filled background
    if (isActive) {
      // Draw filled rectangle for active tab
      rc.rectangle(
        padding,
        padding,
        dimensions.width - padding * 2,
        dimensions.height - padding * 2,
        {
          ...roughStyles.button,
          stroke: sketchColors.ink,
          fill: sketchColors.paper,
          fillStyle: 'solid',
        }
      )
      // Draw bottom line to "connect" to content
      rc.line(
        padding,
        dimensions.height - padding,
        dimensions.width - padding,
        dimensions.height - padding,
        {
          stroke: sketchColors.paper,
          strokeWidth: 3,
        }
      )
    } else {
      // Just draw top and sides for inactive tabs
      rc.rectangle(
        padding,
        padding,
        dimensions.width - padding * 2,
        dimensions.height - padding * 2,
        {
          ...roughStyles.button,
          stroke: sketchColors.pencilLight,
          fill: sketchColors.grid,
          fillStyle: 'solid',
        }
      )
    }
  }, [dimensions, isActive])

  const badgeColorClass = tab.badgeColor ? styles[`badge${tab.badgeColor.charAt(0).toUpperCase() + tab.badgeColor.slice(1)}`] : ''

  return (
    <button
      ref={buttonRef}
      className={clsx(
        styles.tab,
        isActive && styles.tabActive
      )}
      onClick={onClick}
      type="button"
    >
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className={styles.canvas}
      />
      <span className={styles.tabContent}>
        <span className={styles.tabLabel}>{tab.label}</span>
        {tab.badge !== undefined && (
          <span className={clsx(styles.tabBadge, badgeColorClass)}>
            {tab.badge}
          </span>
        )}
      </span>
    </button>
  )
}
