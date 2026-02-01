/**
 * LangScope Sketch Theme
 * Color palette and styling constants for hand-drawn UI
 */

export const sketchColors: Record<string, string> = {
  // Paper colors
  paper: '#ffffff',
  paperAlt: '#fafafa',
  grid: '#e8e8e0',
  gridDark: '#d0d0c8',

  // Ink colors
  ink: '#2d2d2d',
  inkLight: '#555555',
  pencil: '#666666',
  pencilLight: '#999999',

  // Sticky note colors
  stickyYellow: '#fff740',
  stickyYellowDark: '#e6de3a',
  stickyPink: '#ff7eb9',
  stickyBlue: '#7afcff',
  stickyGreen: '#7cff7c',
  stickyOrange: '#ffb347',

  // Accent colors
  highlight: '#ffeb3b',
  accent: '#4a90d9',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
}

export const sketchFonts = {
  handwriting: "'Patrick Hand', cursive",
  title: "'Architects Daughter', cursive",
  accent: "'Caveat', cursive",
} as const

/**
 * Default Rough.js options for consistent sketch styling
 */
export const roughDefaults = {
  roughness: 1.5,
  bowing: 1,
  stroke: sketchColors.ink,
  strokeWidth: 1.5,
  fill: 'transparent',
  fillStyle: 'hachure',
  fillWeight: 1,
  hachureAngle: -41,
  hachureGap: 6,
} as const

/**
 * Rough.js options for different component styles
 */
export const roughStyles = {
  button: {
    roughness: 1.2,
    bowing: 0.8,
    strokeWidth: 1.5,
  },
  card: {
    roughness: 1.5,
    bowing: 1,
    strokeWidth: 1.5,
  },
  input: {
    roughness: 1,
    bowing: 0.5,
    strokeWidth: 1.2,
  },
  table: {
    roughness: 0.8,
    bowing: 0.5,
    strokeWidth: 1,
  },
} as const

/**
 * Sticky note rotation presets (in degrees)
 */
export const stickyRotations = [-3, -2, -1, 0, 1, 2, 3] as const

/**
 * Get a random sticky note rotation
 */
export function getRandomRotation(): number {
  return stickyRotations[Math.floor(Math.random() * stickyRotations.length)]
}

/**
 * Domain icon mapping
 */
export const domainIcons: Record<string, string> = {
  code: '{ }',
  coding: '{ }',
  code_generation: '{ }',
  mathematical_reasoning: '∫x',
  math: '∫x',
  finance: '📊',
  content_moderation: '📋',
  medical: '⚕️',
  medical_assistance: '⚕️',
  legal: '⚖️',
  multilingual: '🌐',
  vision: '👁️',
  speech: '🎤',
  default: '📁',
}

/**
 * Get icon for a domain
 */
export function getDomainIcon(domain: string): string {
  const normalizedDomain = domain.toLowerCase().replace(/[- ]/g, '_')
  return domainIcons[normalizedDomain] || domainIcons.default
}
