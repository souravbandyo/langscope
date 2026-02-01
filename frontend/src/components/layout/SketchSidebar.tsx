import { useState, useEffect } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { useAuthStore } from '@/store/authStore'
import { useProfile } from '@/api/hooks'
import styles from './SketchSidebar.module.css'

interface NavItem {
  path: string
  label: string
  iconClass: string
}

// Main navigation - discovery-first flow
const mainNavItems: NavItem[] = [
  { path: '/', label: 'Home', iconClass: 'ph ph-house' },
  { path: '/rankings', label: 'Rankings', iconClass: 'ph ph-trophy' },
  { path: '/models', label: 'Models', iconClass: 'ph ph-cube' },
  { path: '/arena', label: 'Arena', iconClass: 'ph ph-sword' },
  { path: '/about', label: 'About', iconClass: 'ph ph-info' },
]

// User-specific navigation items
const userNavItems: NavItem[] = [
  { path: '/my-models', label: 'My Models', iconClass: 'ph ph-folder-user' },
]

/**
 * Hand-drawn sidebar navigation with mobile hamburger menu
 */
export function SketchSidebar() {
  const [isOpen, setIsOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { user, signOut, isLoading } = useAuthStore()
  
  // Fetch profile data to get display name
  const { data: profile } = useProfile()

  // Close menu when route changes
  useEffect(() => {
    setIsOpen(false)
  }, [location.pathname])

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (isOpen && !target.closest(`.${styles.sidebar}`) && !target.closest(`.${styles.hamburger}`)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [isOpen])

  const handleLogout = async () => {
    await signOut()
    navigate('/')
  }

  // Get user display name - prefer profile display_name, fallback to email prefix
  const userEmail = user?.email || ''
  const displayName = profile?.display_name || (userEmail ? userEmail.split('@')[0] : 'User')

  return (
    <>
      {/* Mobile Header - Shows on mobile only */}
      <header className={styles.mobileHeader}>
        <button 
          className={styles.hamburger}
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle menu"
        >
          <i className={clsx('ph', isOpen ? 'ph-x' : 'ph-list')}></i>
        </button>
        <img 
          src="/images/science-project-logo.png" 
          alt="Science Project" 
          className={styles.mobileLogoImage}
        />
        <div className={styles.headerSpacer}></div>
      </header>

      {/* Overlay - Mobile Only */}
      {isOpen && <div className={styles.overlay} onClick={() => setIsOpen(false)} />}

      {/* Sidebar */}
      <aside className={clsx(styles.sidebar, isOpen && styles.sidebarOpen)}>
        {/* Logo Section - Keep original logo image */}
        <div className={styles.topSection}>
          <div className={styles.logoTop}>
            <img 
              src="/images/science-project-logo.png" 
              alt="Science Project" 
              className={styles.logoImage}
            />
          </div>

          {/* Main Navigation */}
          <nav className={styles.navMenu}>
            <ul>
              {mainNavItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    clsx(styles.navItem, isActive && styles.active)
                  }
                  end={item.path === '/'}
                >
                  <li>
                    <i className={item.iconClass}></i>
                    {item.label}
                  </li>
                </NavLink>
              ))}
            </ul>
          </nav>
        </div>

        {/* Bottom Navigation - User Menu */}
        <nav className={clsx(styles.navMenu, styles.userMenu)}>
          <ul>
            {/* My Models - only show when logged in */}
            {user && userNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  clsx(styles.navItem, isActive && styles.active)
                }
              >
                <li>
                  <i className={item.iconClass}></i>
                  {item.label}
                </li>
              </NavLink>
            ))}
            {/* Logout - functional button styled as nav item */}
            <li>
              <button 
                className={styles.navButton}
                onClick={handleLogout}
                disabled={isLoading}
              >
                <i className="ph ph-sign-out"></i>
                {isLoading ? 'Logging out...' : 'Logout'}
              </button>
            </li>
            {/* User - link to user page, shows login name when logged in */}
            <NavLink
              to="/user"
              className={({ isActive }) =>
                clsx(styles.navItem, isActive && styles.active)
              }
            >
              <li>
                <i className="ph ph-user"></i>
                {user ? displayName : 'User'}
              </li>
            </NavLink>
          </ul>
        </nav>
      </aside>
    </>
  )
}
