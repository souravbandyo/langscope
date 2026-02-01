import { Outlet, useLocation } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import rough from 'roughjs'
import { SketchSidebar } from './SketchSidebar'
import styles from './PageLayout.module.css'

interface PageLayoutProps {
  children?: React.ReactNode
}

/**
 * Main page layout - matches test.html app-container exactly
 */
export function PageLayout({ children }: PageLayoutProps) {
  const lineLeftRef = useRef<HTMLCanvasElement>(null)
  const location = useLocation()

  useEffect(() => {
    // Draw left divider line
    if (lineLeftRef.current) {
      const canvas = lineLeftRef.current
      canvas.width = 20
      canvas.height = window.innerHeight
      const rc = rough.canvas(canvas)
      rc.line(10, 0, 10, canvas.height, {
        roughness: 1.5,
        stroke: '#ccc',
        strokeWidth: 2
      })
    }
  }, [location.pathname])

  return (
    <div className={styles.appContainer}>
      {/* Skip to main content link for accessibility */}
      <a href="#main-content" className={styles.skipLink}>
        Skip to main content
      </a>
      <SketchSidebar />
      <canvas ref={lineLeftRef} className={styles.separatorLeft}></canvas>
      <main className={styles.main}>
        <div id="main-content" className={styles.content} tabIndex={-1}>
          {children || <Outlet />}
        </div>
      </main>
    </div>
  )
}
