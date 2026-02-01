import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@/test/test-utils'
import { StickyNote, AlertSticky, LeaderboardSticky } from './StickyNote'

describe('StickyNote', () => {
  it('renders children content', () => {
    render(<StickyNote>Test content</StickyNote>)
    expect(screen.getByText('Test content')).toBeInTheDocument()
  })

  it('renders title when provided', () => {
    render(<StickyNote title="My Title">Content</StickyNote>)
    expect(screen.getByText('My Title')).toBeInTheDocument()
  })

  it('renders pin when pinned prop is true', () => {
    const { container } = render(<StickyNote pinned>Content</StickyNote>)
    expect(container.querySelector('[class*="pin"]')).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn()
    render(<StickyNote onClick={handleClick}>Click me</StickyNote>)
    
    fireEvent.click(screen.getByText('Click me'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('renders different colors', () => {
    const colors = ['yellow', 'pink', 'blue', 'green', 'orange'] as const
    
    colors.forEach((color) => {
      const { container, unmount } = render(
        <StickyNote color={color}>Content</StickyNote>
      )
      expect(container.firstChild).toBeInTheDocument()
      unmount()
    })
  })
})

describe('AlertSticky', () => {
  it('renders alert content', () => {
    render(<AlertSticky>Alert message</AlertSticky>)
    expect(screen.getByText('Alert message')).toBeInTheDocument()
  })

  it('renders with different types', () => {
    const types = ['info', 'warning', 'success', 'error'] as const
    
    types.forEach((type) => {
      const { unmount } = render(<AlertSticky type={type}>Alert</AlertSticky>)
      expect(screen.getByText('Alert')).toBeInTheDocument()
      unmount()
    })
  })
})

describe('LeaderboardSticky', () => {
  const mockEntries = [
    { rank: 1, name: 'GPT-4', score: 1625 },
    { rank: 2, name: 'Claude 3', score: 1598 },
    { rank: 3, name: 'Gemini', score: 1567 },
  ]

  it('renders leaderboard entries', () => {
    render(
      <LeaderboardSticky
        title="Top Models"
        domain="code"
        entries={mockEntries}
      />
    )
    
    expect(screen.getByText('Top Models')).toBeInTheDocument()
    expect(screen.getByText('GPT-4')).toBeInTheDocument()
    expect(screen.getByText('Claude 3')).toBeInTheDocument()
  })

  it('displays domain tag', () => {
    render(
      <LeaderboardSticky
        title="Top Models"
        domain="code"
        entries={mockEntries}
      />
    )
    
    expect(screen.getByText('#code')).toBeInTheDocument()
  })
})
