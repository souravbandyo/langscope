import { useNavigate } from 'react-router-dom'
import { useMemo } from 'react'
import { LeaderboardSticky } from '@/components/sticky'
import { ErrorState, LoadingState } from '@/components/common'
import { useDomains, useLeaderboard } from '@/api/hooks'
import styles from './DomainLeaderboards.module.css'

const stickyColors = ['yellow', 'blue', 'green', 'pink', 'orange'] as const
const rotations = [-2, 1, -1, 2, -1.5]

/** Shuffle array using Fisher-Yates algorithm */
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

/**
 * Single domain leaderboard sticky note
 */
function DomainLeaderboardSticky({ 
  domain, 
  color, 
  rotation, 
  onClick 
}: { 
  domain: string
  color: typeof stickyColors[number]
  rotation: number
  onClick: () => void 
}) {
  const { data, isLoading, error } = useLeaderboard({ domain, limit: 3 })

  const entries = data?.entries?.slice(0, 3).map((entry) => ({
    rank: entry.rank,
    name: entry.name,
    score: Math.round(entry.mu),
  })) || []

  const displayName = domain.replace(/_/g, ' ')

  if (isLoading) {
    return (
      <LeaderboardSticky
        title={`#${displayName} Leaderboard`}
        domain={displayName}
        entries={[
          { rank: 1, name: 'Loading...', score: 0 },
        ]}
        color={color}
        rotation={rotation}
        onClick={onClick}
      />
    )
  }

  // Show error state in sticky
  if (error) {
    return (
      <LeaderboardSticky
        title={`#${displayName} Leaderboard`}
        domain={displayName}
        entries={[{ rank: 1, name: '⚠️ Connection error', score: 0 }]}
        color={color}
        rotation={rotation}
        onClick={onClick}
      />
    )
  }

  return (
    <LeaderboardSticky
      title={`#${displayName} Leaderboard`}
      domain={displayName}
      entries={entries.length > 0 ? entries : [{ rank: 1, name: 'No data yet', score: 0 }]}
      color={color}
      rotation={rotation}
      onClick={onClick}
    />
  )
}

/**
 * Domain leaderboards displayed as sticky notes
 */
export function DomainLeaderboards() {
  const navigate = useNavigate()
  const { data: domainsData, isLoading, error, refetch } = useDomains()

  // Randomly shuffle domains on component mount and pick 8
  // Must be called before any conditional returns (React Rules of Hooks)
  const randomDomains = useMemo(() => {
    const allDomains = domainsData?.domains || []
    return shuffleArray(allDomains).slice(0, 8)
  }, [domainsData?.domains])

  const handleLeaderboardClick = (domainId: string) => {
    navigate(`/rankings/${domainId}`)
  }

  if (isLoading) {
    return (
      <section className={styles.section}>
        <h2 className={styles.title}>Domain Leaderboards</h2>
        <LoadingState message="Loading domains..." />
      </section>
    )
  }

  if (error) {
    return (
      <section className={styles.section}>
        <h2 className={styles.title}>Domain Leaderboards</h2>
        <ErrorState
          title="Failed to load domains"
          error={error as Error}
          onRetry={() => refetch()}
        />
      </section>
    )
  }

  // Show placeholder stickies if no domains
  if (randomDomains.length === 0) {
    const placeholderDomains = ['Code', 'Reasoning', 'Writing', 'Translation', 'Math', 'Vision', 'QA', 'Summarization']
    return (
      <section className={styles.section}>
        <h2 className={styles.title}>Domain Leaderboards</h2>
        <div className={styles.stickyGrid}>
          {placeholderDomains.map((name, index) => (
            <LeaderboardSticky
              key={name}
              title={`#${name} Leaderboard`}
              domain={name}
              entries={[{ rank: 1, name: 'No data yet', score: 0 }]}
              color={stickyColors[index % stickyColors.length]}
              rotation={rotations[index % rotations.length]}
              onClick={() => navigate('/rankings')}
            />
          ))}
        </div>
      </section>
    )
  }

  return (
    <section className={styles.section}>
      <h2 className={styles.title}>Domain Leaderboards</h2>

      <div className={styles.stickyGrid}>
        {randomDomains.map((domain, index) => (
          <DomainLeaderboardSticky
            key={domain}
            domain={domain}
            color={stickyColors[index % stickyColors.length]}
            rotation={rotations[index % rotations.length]}
            onClick={() => handleLeaderboardClick(domain)}
          />
        ))}
      </div>
    </section>
  )
}
