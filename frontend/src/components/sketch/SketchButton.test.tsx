import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@/test/test-utils'
import { SketchButton } from './SketchButton'

describe('SketchButton', () => {
  it('renders with children text', () => {
    render(<SketchButton>Click me</SketchButton>)
    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn()
    render(<SketchButton onClick={handleClick}>Click me</SketchButton>)
    
    fireEvent.click(screen.getByRole('button'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('is disabled when disabled prop is true', () => {
    render(<SketchButton disabled>Disabled</SketchButton>)
    
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
  })

  it('renders different sizes', () => {
    const { rerender } = render(<SketchButton size="sm">Small</SketchButton>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<SketchButton size="md">Medium</SketchButton>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<SketchButton size="lg">Large</SketchButton>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders primary variant', () => {
    render(<SketchButton variant="primary">Primary</SketchButton>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })
})
