import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import rough from 'roughjs'
import styles from './SearchBar.module.css'

/**
 * Use-case search bar for finding the best model
 * Matches test.html exactly with rough.js hand-drawn effects
 */
export function SearchBar() {
  const [query, setQuery] = useState('')
  const navigate = useNavigate()
  const wrapperRef = useRef<HTMLDivElement>(null)
  const searchCanvasRef = useRef<HTMLCanvasElement>(null)
  const btnCanvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const drawEffects = () => {
      // Draw search box border around entire wrapper
      if (searchCanvasRef.current && wrapperRef.current) {
        const canvas = searchCanvasRef.current
        const wrapper = wrapperRef.current
        const rect = wrapper.getBoundingClientRect()
        canvas.width = rect.width + 10
        canvas.height = rect.height + 10
        const rc = rough.canvas(canvas)
        rc.rectangle(5, 5, canvas.width - 10, canvas.height - 10, {
          roughness: 2,
          stroke: '#555',
          strokeWidth: 2
        })
      }

      // Draw button background with hachure fill
      if (btnCanvasRef.current) {
        const canvas = btnCanvasRef.current
        const rect = canvas.getBoundingClientRect()
        canvas.width = rect.width
        canvas.height = 50
        const rc = rough.canvas(canvas)
        rc.rectangle(2, 2, canvas.width - 4, canvas.height - 4, {
          roughness: 2,
          stroke: 'transparent',
          fill: '#333333',
          fillStyle: 'hachure',
          fillWeight: 3,
          hachureGap: 2,
          hachureAngle: -41
        })
      }
    }

    // Draw on mount and after layout
    drawEffects()
    window.addEventListener('resize', drawEffects)
    return () => window.removeEventListener('resize', drawEffects)
  }, [])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      navigate(`/recommendations?q=${encodeURIComponent(query)}`)
    }
  }

  return (
    <form onSubmit={handleSearch} className={styles.searchWrapper} ref={wrapperRef}>
      <canvas ref={searchCanvasRef} className={styles.searchBoxCanvas}></canvas>
      <input
        type="text"
        className={styles.searchInput}
        placeholder="Describe your usecase..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button type="submit" className={styles.searchBtn}>
        Search
        <canvas ref={btnCanvasRef} className={styles.btnCanvas}></canvas>
      </button>
    </form>
  )
}
