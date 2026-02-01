import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SketchCard, SketchButton } from '@/components/sketch'
import { useAuthStore } from '@/store/authStore'
import { useMatches, useDomains, usePromptMetrics } from '@/api/hooks'
import styles from './User.module.css'

/**
 * User profile page showing account information, activity, and preferences
 */
export function User() {
  const navigate = useNavigate()
  const { user, signOut, isLoading } = useAuthStore()
  const [activeTab, setActiveTab] = useState<'profile' | 'activity' | 'preferences'>('profile')

  // Fetch user activity data
  const { data: matchesData } = useMatches({ limit: 10 })
  const { data: domainsData } = useDomains()
  const { data: promptMetrics } = usePromptMetrics()

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  // Get user info
  const userEmail = user?.email || 'Not signed in'
  const username = userEmail.split('@')[0]
  const domain = userEmail.includes('@') ? userEmail.split('@')[1] : ''
  const initials = username.substring(0, 2).toUpperCase()

  if (!user) {
    return (
      <div className={styles.user}>
        <h1 className={styles.title}>User Profile</h1>
        <SketchCard padding="lg">
          <div className={styles.notSignedIn}>
            <i className="ph ph-user-circle"></i>
            <h2>Not Signed In</h2>
            <p>Please sign in to view your profile.</p>
            <SketchButton onClick={() => navigate('/auth')}>
              Sign In
            </SketchButton>
          </div>
        </SketchCard>
      </div>
    )
  }

  const tabs = [
    { id: 'profile' as const, label: 'Profile' },
    { id: 'activity' as const, label: 'Activity' },
    { id: 'preferences' as const, label: 'Preferences' },
  ]

  return (
    <div className={styles.user}>
      <h1 className={styles.title}>User Profile</h1>

      {/* Profile Header */}
      <SketchCard padding="lg">
        <div className={styles.profileHeader}>
          <div className={styles.avatar}>{initials}</div>
          <div className={styles.profileInfo}>
            <h2 className={styles.username}>{username}</h2>
            <p className={styles.email}>{userEmail}</p>
          </div>
        </div>
      </SketchCard>

      {/* Tabs */}
      <div className={styles.tabs}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`${styles.tab} ${activeTab === tab.id ? styles.activeTab : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <>
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Account Details</h2>
            <div className={styles.detailsList}>
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>Username</span>
                <span className={styles.detailValue}>{username}</span>
              </div>
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>Email</span>
                <span className={styles.detailValue}>{userEmail}</span>
              </div>
              {domain && (
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Domain</span>
                  <span className={styles.detailValue}>{domain}</span>
                </div>
              )}
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>User ID</span>
                <span className={styles.detailValue}>{user.id || 'N/A'}</span>
              </div>
              {user.created_at && (
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Member Since</span>
                  <span className={styles.detailValue}>
                    {new Date(user.created_at).toLocaleDateString()}
                  </span>
                </div>
              )}
            </div>
          </SketchCard>

          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Account Actions</h2>
            <div className={styles.actions}>
              <SketchButton 
                variant="outline"
                onClick={handleSignOut}
                disabled={isLoading}
              >
                {isLoading ? 'Signing out...' : 'Sign Out'}
              </SketchButton>
            </div>
          </SketchCard>
        </>
      )}

      {/* Activity Tab */}
      {activeTab === 'activity' && (
        <>
          {/* Activity Stats */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Activity Summary</h2>
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <span className={styles.statValue}>{domainsData?.total || 0}</span>
                <span className={styles.statLabel}>Domains Active</span>
              </div>
              <div className={styles.statCard}>
                <span className={styles.statValue}>{matchesData?.total || 0}</span>
                <span className={styles.statLabel}>Total Matches</span>
              </div>
              <div className={styles.statCard}>
                <span className={styles.statValue}>{promptMetrics?.total_classified || 0}</span>
                <span className={styles.statLabel}>Prompts Classified</span>
              </div>
              <div className={styles.statCard}>
                <span className={styles.statValue}>
                  {promptMetrics?.cache_hits && promptMetrics?.cache_misses 
                    ? ((promptMetrics.cache_hits / (promptMetrics.cache_hits + promptMetrics.cache_misses)) * 100).toFixed(0)
                    : 0}%
                </span>
                <span className={styles.statLabel}>Cache Hit Rate</span>
              </div>
            </div>
          </SketchCard>

          {/* Recent Matches */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Recent Matches</h2>
            {matchesData?.matches && matchesData.matches.length > 0 ? (
              <div className={styles.matchesList}>
                {matchesData.matches.slice(0, 5).map(match => (
                  <div key={match.match_id} className={styles.matchItem}>
                    <div className={styles.matchInfo}>
                      <span className={styles.matchDomain}>{match.domain}</span>
                      <span className={styles.matchDate}>
                        {new Date(match.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <div className={styles.matchParticipants}>
                      {match.participants.slice(0, 3).map((p, i) => (
                        <span key={i} className={styles.participantBadge}>{p}</span>
                      ))}
                      {match.participants.length > 3 && (
                        <span className={styles.moreBadge}>+{match.participants.length - 3}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.emptyState}>No matches recorded yet</p>
            )}
          </SketchCard>

          {/* Domain Usage */}
          {promptMetrics?.domain_distribution && Object.keys(promptMetrics.domain_distribution).length > 0 && (
            <SketchCard padding="lg">
              <h2 className={styles.sectionTitle}>Domain Usage</h2>
              <div className={styles.domainUsage}>
                {Object.entries(promptMetrics.domain_distribution)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 8)
                  .map(([domainName, count]) => (
                    <div key={domainName} className={styles.domainItem}>
                      <span className={styles.domainName}>{domainName}</span>
                      <span className={styles.domainCount}>{count}</span>
                    </div>
                  ))}
              </div>
            </SketchCard>
          )}
        </>
      )}

      {/* Preferences Tab */}
      {activeTab === 'preferences' && (
        <>
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Display Preferences</h2>
            <div className={styles.preferencesList}>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Default Dimension</span>
                  <span className={styles.preferenceDescription}>
                    Primary dimension for leaderboard rankings
                  </span>
                </div>
                <select className={styles.preferenceSelect}>
                  <option value="raw_quality">Raw Quality</option>
                  <option value="cost_adjusted">Cost-Adjusted</option>
                  <option value="latency">Latency</option>
                  <option value="combined">Combined</option>
                </select>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Results per Page</span>
                  <span className={styles.preferenceDescription}>
                    Number of items to display in lists
                  </span>
                </div>
                <select className={styles.preferenceSelect}>
                  <option value="10">10</option>
                  <option value="25">25</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </select>
              </div>
            </div>
          </SketchCard>

          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Notification Preferences</h2>
            <div className={styles.preferencesList}>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Rating Updates</span>
                  <span className={styles.preferenceDescription}>
                    Get notified when model ratings change significantly
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input type="checkbox" defaultChecked />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>New Models</span>
                  <span className={styles.preferenceDescription}>
                    Get notified when new models are added
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input type="checkbox" />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>System Alerts</span>
                  <span className={styles.preferenceDescription}>
                    Important system notifications and updates
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input type="checkbox" defaultChecked />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
            </div>
          </SketchCard>

          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Quick Links</h2>
            <div className={styles.quickLinks}>
              <SketchButton variant="secondary" onClick={() => navigate('/arena')}>
                Start Arena Session
              </SketchButton>
              <SketchButton variant="secondary" onClick={() => navigate('/rankings')}>
                View Rankings
              </SketchButton>
              <SketchButton variant="secondary" onClick={() => navigate('/recommendations')}>
                Get Recommendations
              </SketchButton>
            </div>
          </SketchCard>
        </>
      )}
    </div>
  )
}

export { User as default }
