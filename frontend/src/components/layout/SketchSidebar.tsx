import { useState, useEffect } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { useAuthStore } from '@/store/authStore'
import styles from './SketchSidebar.module.css'

interface NavItem {
  path: string
  label: string
  iconClass: string
}

const mainNavItems: NavItem[] = [
  { path: '/', label: 'Home', iconClass: 'ph ph-house' },
  { path: '/rankings', label: 'Rankings', iconClass: 'ph ph-trophy' },
  { path: '/models', label: 'Models', iconClass: 'ph ph-robot' },
  { path: '/arena', label: 'Arena', iconClass: 'ph ph-sword' },
  { path: '/about', label: 'About', iconClass: 'ph ph-info' },
]

/**
 * Hand-drawn sidebar navigation with mobile hamburger menu
 */
export function SketchSidebar() {
  const [isOpen, setIsOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { user, signOut, isLoading } = useAuthStore()

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

  // Get user display name (username part of email or full email)
  const userEmail = user?.email || ''
  const displayName = userEmail ? userEmail.split('@')[0] : 'User'

  return (
    <>
      {/* Hamburger Button - Mobile Only */}
      <button 
        className={styles.hamburger}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
      >
        <i className={clsx('ph', isOpen ? 'ph-x' : 'ph-list')}></i>
      </button>

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
