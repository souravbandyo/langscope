import { SearchBar } from './SearchBar'
import { TrendingDomains } from './TrendingDomains'
import { DomainLeaderboards } from './DomainLeaderboards'
import styles from './Home.module.css'

/**
 * Home page with search, trending domains, and leaderboard stickies
 * Layout matches wireframe: centered logo + search, right sidebar for trending
 */
export function Home() {
  return (
    <div className={styles.home}>
      {/* Main Content */}
      <div className={styles.mainContent}>
        {/* Hero Section with Logo and Search */}
        <div className={styles.heroSection}>
          <div className={styles.logo}>
            <div className={styles.logoFrame}>
              <img 
                src="/images/langscope-logo.png" 
                alt="LangScope" 
                className={styles.logoImage}
              />
            </div>
          </div>

          <p className={styles.tagline}>Find the best model for your needs</p>

          <SearchBar />
        </div>

        {/* Domain Leaderboards as Sticky Notes */}
        <DomainLeaderboards />
      </div>

      {/* Trending Domains Sidebar */}
      <aside className={styles.trendingSidebar}>
        <TrendingDomains />
      </aside>
    </div>
  )
}

export { Home as default }
