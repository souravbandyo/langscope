import { useEffect, useRef } from 'react'
import rough from 'roughjs'
import type { RoughCanvas } from 'roughjs/bin/canvas'
import { roughDefaults } from '@/styles/sketch-theme'

export interface RoughCanvasProps {
  width: number
  height: number
  draw: (rc: RoughCanvas, ctx: CanvasRenderingContext2D) => void
  className?: string
  style?: React.CSSProperties
}

/**
 * Base canvas component for Rough.js drawings
 * Provides a canvas with Rough.js instance for hand-drawn graphics
 */
export function RoughCanvasComponent({
  width,
  height,
  draw,
  className = '',
  style,
}: RoughCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Create Rough.js canvas
    const rc = rough.canvas(canvas, { options: roughDefaults })

    // Call the draw function
    draw(rc, ctx)
  }, [width, height, draw])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={{ display: 'block', ...style }}
    />
  )
}

/**
 * Hook to use Rough.js on a canvas element
 */
export function useRoughCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement>
): RoughCanvas | null {
  const rcRef = useRef<RoughCanvas | null>(null)

  useEffect(() => {
    if (canvasRef.current) {
      rcRef.current = rough.canvas(canvasRef.current, { options: roughDefaults })
    }
  }, [canvasRef])

  return rcRef.current
}

/**
 * Draw a hand-drawn rectangle
 */
export function drawSketchRect(
  rc: RoughCanvas,
  x: number,
  y: number,
  width: number,
  height: number,
  options?: Parameters<RoughCanvas['rectangle']>[4]
) {
  return rc.rectangle(x, y, width, height, {
    ...roughDefaults,
    ...options,
  })
}

/**
 * Draw a hand-drawn line
 */
export function drawSketchLine(
  rc: RoughCanvas,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  options?: Parameters<RoughCanvas['line']>[4]
) {
  return rc.line(x1, y1, x2, y2, {
    ...roughDefaults,
    ...options,
  })
}

/**
 * Draw a hand-drawn circle
 */
export function drawSketchCircle(
  rc: RoughCanvas,
  x: number,
  y: number,
  diameter: number,
  options?: Parameters<RoughCanvas['circle']>[3]
) {
  return rc.circle(x, y, diameter, {
    ...roughDefaults,
    ...options,
  })
}

export { RoughCanvasComponent as RoughCanvas }
