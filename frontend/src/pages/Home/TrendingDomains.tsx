import { useNavigate } from 'react-router-dom'
import { useDomains, useLeaderboard } from '@/api/hooks'
import styles from './TrendingDomains.module.css'

// Domain icons mapping - using Phosphor icon class names
// Each domain has a unique, visually distinct icon
const domainIcons: Record<string, string> = {
  // Code-related domains
  code_generation: 'ph ph-code',
  code: 'ph ph-code',
  code_completion: 'ph ph-terminal',
  code_review: 'ph ph-git-branch',
  
  // Math & Reasoning domains
  mathematical_reasoning: 'ph ph-function',
  reasoning: 'ph ph-brain',
  math: 'ph ph-calculator',
  mathematics: 'ph ph-sigma',
  
  // Business & Professional domains
  finance: 'ph ph-chart-line-up',
  legal: 'ph ph-scales',
  business: 'ph ph-briefcase',
  
  // Content domains
  content_moderation: 'ph ph-shield-check',
  creative_writing: 'ph ph-pencil-line',
  writing: 'ph ph-article',
  
  // Healthcare
  medical_assistance: 'ph ph-first-aid',
  healthcare: 'ph ph-heartbeat',
  
  // Language domains
  translation: 'ph ph-translate',
  summarization: 'ph ph-file-text',
  
  // Q&A and Knowledge
  qa: 'ph ph-question',
  knowledge: 'ph ph-book-open',
  
  // Vision & Multimodal
  vision: 'ph ph-eye',
  multimodal: 'ph ph-images',
  
  // Other domains
  chat: 'ph ph-chat-dots',
  general: 'ph ph-sparkle',
}

// Fallback icons for domains not in the mapping - cycle through these
const fallbackIcons = [
  'ph ph-graph',
  'ph ph-lightbulb',
  'ph ph-rocket',
  'ph ph-cube',
  'ph ph-star',
  'ph ph-gear',
  'ph ph-target',
  'ph ph-puzzle-piece',
]

// Get icon for domain - ensures each domain gets a unique icon
function getDomainIcon(domainName: string, index: number): string {
  // Check for exact match
  if (domainIcons[domainName]) {
    return domainIcons[domainName]
  }
  
  // Check for partial matches
  const lowerName = domainName.toLowerCase()
  for (const [key, icon] of Object.entries(domainIcons)) {
    if (lowerName.includes(key) || key.includes(lowerName)) {
      return icon
    }
  }
  
  // Use fallback icons based on index to ensure variety
  return fallbackIcons[index % fallbackIcons.length]
}

// Default domains when API is not available
const defaultDomains = ['code', 'reasoning', 'writing', 'translation', 'summarization']

/**
 * Single trending domain item that fetches its own top model
 */
function TrendingDomainItem({ 
  domainName, 
  index, 
  onClick 
}: { 
  domainName: string
  index: number
  onClick: () => void 
}) {
  const { data, isLoading } = useLeaderboard({ domain: domainName, limit: 1 })
  
  // Get the top model name
  const topModel = data?.entries?.[0]?.name
  const topModelDisplay = isLoading 
    ? 'Loading...' 
    : topModel 
      ? `${topModel.length > 15 ? topModel.substring(0, 15) + '...' : topModel}`
      : 'No data yet'

  return (
    <button
      className={styles.domainCard}
      onClick={onClick}
    >
      <div className={styles.domainTitle}>
        #{domainName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
      </div>
      <div className={styles.domainStats}>
        <div className={styles.domainIcon}>
          <i className={getDomainIcon(domainName, index)}></i>
        </div>
        <div className={styles.statsText}>
          <div>Statistics at: 4.7K</div>
          <div>Models: {topModelDisplay}</div>
        </div>
        <i className={`ph ph-arrow-right ${styles.arrow}`}></i>
      </div>
    </button>
  )
}

/**
 * Trending domains sidebar with statistics
 * Matches test.html styling with Phosphor icons
 */
export function TrendingDomains() {
  const navigate = useNavigate()
  const { data, isLoading, error, refetch } = useDomains()

  const handleDomainClick = (domainId: string) => {
    navigate(`/rankings/${domainId}`)
  }

  if (isLoading) {
    return (
      <div className={styles.container}>
        <h2 className={styles.title}>Trending Domains</h2>
        <div className={styles.loading}>Loading domains...</div>
      </div>
    )
  }

  // Use default domains if error or no data
  const domains = (error || !data?.domains?.length) ? defaultDomains : data.domains

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Trending Domains</h2>

      {error && (
        <button className={styles.retryButton} onClick={() => refetch()}>
          ⚠️ Connection issue - Click to retry
        </button>
      )}

      <div className={styles.domainList}>
        {domains.slice(0, 5).map((domainName, index) => (
          <TrendingDomainItem
            key={domainName}
            domainName={domainName}
            index={index}
            onClick={() => handleDomainClick(domainName)}
          />
        ))}
      </div>
    </div>
  )
}
