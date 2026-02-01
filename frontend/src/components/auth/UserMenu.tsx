/**
 * User Menu Component
 * 
 * Displays user info and logout button in the header.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { SketchButton } from '@/components/sketch'
import styles from './UserMenu.module.css'

export function UserMenu() {
  const navigate = useNavigate()
  const { user, signOut, isLoading } = useAuthStore()
  const [showDropdown, setShowDropdown] = useState(false)

  if (!user) {
    return (
      <SketchButton
        variant="outline"
        size="sm"
        onClick={() => navigate('/auth')}
      >
        Sign In
      </SketchButton>
    )
  }

  const handleSignOut = async () => {
    await signOut()
    setShowDropdown(false)
    navigate('/')
  }

  const displayEmail = user.email || 'User'
  const initials = displayEmail.substring(0, 2).toUpperCase()

  return (
    <div className={styles.container}>
      <button
        className={styles.trigger}
        onClick={() => setShowDropdown(!showDropdown)}
        aria-expanded={showDropdown}
        aria-haspopup="true"
      >
        <span className={styles.avatar}>{initials}</span>
        <span className={styles.email}>{displayEmail}</span>
        <span className={styles.chevron}>▼</span>
      </button>

      {showDropdown && (
        <>
          <div 
            className={styles.backdrop} 
            onClick={() => setShowDropdown(false)} 
          />
          <div className={styles.dropdown}>
            <div className={styles.userInfo}>
              <span className={styles.label}>Signed in as</span>
              <span className={styles.value}>{displayEmail}</span>
            </div>
            <div className={styles.divider} />
            <button
              className={styles.menuItem}
              onClick={handleSignOut}
              disabled={isLoading}
            >
              {isLoading ? 'Signing out...' : 'Sign Out'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}

export default UserMenu
