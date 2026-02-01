/**
 * InviteMemberModal Component
 * Modal for inviting new team members
 */

import { useState } from 'react'
import styles from './InviteMemberModal.module.css'

interface InviteMemberModalProps {
  isOpen: boolean
  onClose: () => void
  onInvite: (email: string, role: 'admin' | 'member' | 'viewer') => void
  isLoading?: boolean
}

export function InviteMemberModal({
  isOpen,
  onClose,
  onInvite,
  isLoading = false,
}: InviteMemberModalProps) {
  const [email, setEmail] = useState('')
  const [role, setRole] = useState<'admin' | 'member' | 'viewer'>('member')
  const [error, setError] = useState('')

  if (!isOpen) return null

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address')
      return
    }

    onInvite(email, role)
  }

  const handleClose = () => {
    setEmail('')
    setRole('member')
    setError('')
    onClose()
  }

  return (
    <div className={styles.overlay} onClick={handleClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>Invite Team Member</h2>
          <button className={styles.closeButton} onClick={handleClose}>
            <i className="ph ph-x"></i>
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label htmlFor="email" className={styles.label}>
              Email Address
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="colleague@example.com"
              className={styles.input}
              disabled={isLoading}
              autoFocus
            />
            {error && <span className={styles.error}>{error}</span>}
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Role</label>
            <div className={styles.roleOptions}>
              {([
                { value: 'admin', label: 'Admin', desc: 'Manage members & settings' },
                { value: 'member', label: 'Member', desc: 'Use all features' },
                { value: 'viewer', label: 'Viewer', desc: 'Read-only access' },
              ] as const).map((option) => (
                <label
                  key={option.value}
                  className={`${styles.roleOption} ${role === option.value ? styles.selected : ''}`}
                >
                  <input
                    type="radio"
                    name="role"
                    value={option.value}
                    checked={role === option.value}
                    onChange={(e) => setRole(e.target.value as typeof role)}
                    disabled={isLoading}
                  />
                  <div className={styles.roleContent}>
                    <span className={styles.roleName}>{option.label}</span>
                    <span className={styles.roleDesc}>{option.desc}</span>
                  </div>
                  <i className={`ph ph-check ${styles.checkIcon}`}></i>
                </label>
              ))}
            </div>
          </div>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={handleClose}
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={styles.submitButton}
              disabled={isLoading || !email}
            >
              {isLoading ? (
                <>
                  <i className="ph ph-spinner-gap"></i>
                  Sending...
                </>
              ) : (
                <>
                  <i className="ph ph-paper-plane-tilt"></i>
                  Send Invite
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
