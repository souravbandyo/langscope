import { useRef, useEffect, useState } from 'react'
import rough from 'roughjs'
import { sketchColors, roughStyles } from '@/styles/sketch-theme'
import styles from './RankSelector.module.css'

export interface RankSelectorProps {
  totalRanks: number
  selectedRank?: number
  usedRanks: number[] // ranks already used by other responses
  onRankSelect: (rank: number) => void
  className?: string
}

/**
 * Visual rank selector with buttons for each rank
 * Disables ranks that are already used by other responses
 */
export function RankSelector({
  totalRanks,
  selectedRank,
  usedRanks,
  onRankSelect,
  className,
}: RankSelectorProps) {
  const ranks = Array.from({ length: totalRanks }, (_, i) => i + 1)

  return (
    <div className={`${styles.rankSelector} ${className || ''}`}>
      <div className={styles.rankLabel}>Your Rank:</div>
      <div className={styles.rankButtons}>
        {ranks.map((rank) => {
          const isSelected = selectedRank === rank
          const isUsed = usedRanks.includes(rank) && !isSelected

          return (
            <RankButton
              key={rank}
              rank={rank}
              isSelected={isSelected}
              isDisabled={isUsed}
              totalRanks={totalRanks}
              onClick={() => !isUsed && onRankSelect(rank)}
            />
          )
        })}
      </div>
      <div className={styles.rankHint}>
        <span className={styles.bestLabel}>Best</span>
        <span className={styles.worstLabel}>Worst</span>
      </div>
    </div>
  )
}

interface RankButtonProps {
  rank: number
  isSelected: boolean
  isDisabled: boolean
  totalRanks: number
  onClick: () => void
}

function RankButton({
  rank,
  isSelected,
  isDisabled,
  totalRanks,
  onClick,
}: RankButtonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Measure button size
  useEffect(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect()
      setDimensions({ width: rect.width, height: rect.height })
    }
  }, [rank])

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
    let fillColor = 'transparent'

    if (isSelected) {
      strokeColor = sketchColors.success
      fillColor = sketchColors.success
    } else if (isDisabled) {
      strokeColor = sketchColors.pencilLight
      fillColor = sketchColors.grid
    }

    rc.rectangle(
      padding,
      padding,
      dimensions.width - padding * 2,
      dimensions.height - padding * 2,
      {
        ...roughStyles.button,
        stroke: strokeColor,
        fill: fillColor,
        fillStyle: isSelected ? 'solid' : isDisabled ? 'solid' : 'hachure',
      }
    )
  }, [dimensions, isSelected, isDisabled])

  // Get ordinal suffix
  const getOrdinal = (n: number) => {
    const s = ['th', 'st', 'nd', 'rd']
    const v = n % 100
    return n + (s[(v - 20) % 10] || s[v] || s[0])
  }

  return (
    <button
      ref={buttonRef}
      className={`
        ${styles.rankButton}
        ${isSelected ? styles.rankButtonSelected : ''}
        ${isDisabled ? styles.rankButtonDisabled : ''}
      `}
      onClick={onClick}
      disabled={isDisabled}
      type="button"
      title={`Rank ${getOrdinal(rank)}${rank === 1 ? ' (Best)' : rank === totalRanks ? ' (Worst)' : ''}`}
    >
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        className={styles.canvas}
      />
      <span className={styles.rankNumber}>{rank}</span>
    </button>
  )
}

// Export a simpler version for showing ranking progress
export interface RankingProgressProps {
  rankings: Record<string, number> // responseId -> rank
  responseLabels: string[] // ['Response A', 'Response B', ...]
  totalResponses: number
}

export function RankingProgress({
  rankings,
  responseLabels,
  totalResponses,
}: RankingProgressProps) {
  const rankedCount = Object.keys(rankings).length
  const remaining = totalResponses - rankedCount

  // Sort by rank for display
  const sortedRankings = Object.entries(rankings)
    .sort(([, a], [, b]) => a - b)
    .map(([responseId, rank]) => ({
      responseId,
      rank,
      label: responseLabels[parseInt(responseId.replace('response_', ''))] || responseId,
    }))

  return (
    <div className={styles.rankingProgress}>
      <div className={styles.progressLabel}>
        Ranking Progress:
      </div>
      <div className={styles.progressItems}>
        {sortedRankings.map(({ responseId, rank, label }) => (
          <span key={responseId} className={styles.progressItem}>
            #{rank} {label}
          </span>
        ))}
        {remaining > 0 && (
          <span className={styles.progressRemaining}>
            ({remaining} remaining)
          </span>
        )}
      </div>
    </div>
  )
}
