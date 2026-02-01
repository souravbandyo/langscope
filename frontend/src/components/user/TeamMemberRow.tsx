/**
 * TeamMemberRow Component
 * Row component for displaying team members in a list
 */

import { useState } from 'react'
import type { TeamMember } from '../../api/types'
import { RoleBadge } from './RoleBadge'
import styles from './TeamMemberRow.module.css'

interface TeamMemberRowProps {
  member: TeamMember
  currentUserId: string
  isOwnerOrAdmin: boolean
  onChangeRole: (memberId: string, role: 'admin' | 'member' | 'viewer') => void
  onRemove: (memberId: string) => void
}

export function TeamMemberRow({
  member,
  currentUserId,
  isOwnerOrAdmin,
  onChangeRole,
  onRemove,
}: TeamMemberRowProps) {
  const [showRoleDropdown, setShowRoleDropdown] = useState(false)
  const isCurrentUser = member.user_id === currentUserId
  const isOwner = member.role === 'owner'
  const canModify = isOwnerOrAdmin && !isOwner && !isCurrentUser

  const initials = member.display_name
    ? member.display_name.charAt(0).toUpperCase()
    : member.email.charAt(0).toUpperCase()

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  return (
    <div className={styles.row}>
      <div className={styles.memberInfo}>
        <div className={styles.avatar}>
          {member.avatar_url ? (
            <img src={member.avatar_url} alt="" />
          ) : (
            <span>{initials}</span>
          )}
        </div>
        <div className={styles.details}>
          <span className={styles.name}>
            {member.display_name || member.email.split('@')[0]}
            {isCurrentUser && <span className={styles.youBadge}>(You)</span>}
          </span>
          <span className={styles.email}>{member.email}</span>
        </div>
      </div>

      <div className={styles.roleCell}>
        {canModify ? (
          <div className={styles.roleSelector}>
            <button
              className={styles.roleButton}
              onClick={() => setShowRoleDropdown(!showRoleDropdown)}
            >
              <RoleBadge role={member.role} size="sm" />
              <i className="ph ph-caret-down"></i>
            </button>
            {showRoleDropdown && (
              <div className={styles.roleDropdown}>
                {(['admin', 'member', 'viewer'] as const).map((role) => (
                  <button
                    key={role}
                    className={`${styles.roleOption} ${member.role === role ? styles.active : ''}`}
                    onClick={() => {
                      onChangeRole(member.id, role)
                      setShowRoleDropdown(false)
                    }}
                  >
                    <RoleBadge role={role} size="sm" />
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <RoleBadge role={member.role} size="sm" />
        )}
      </div>

      <div className={styles.statusCell}>
        <span className={`${styles.status} ${styles[member.status]}`}>
          {member.status}
        </span>
      </div>

      <div className={styles.dateCell}>{formatDate(member.joined_at)}</div>

      <div className={styles.actionsCell}>
        {canModify && (
          <button
            className={styles.removeButton}
            onClick={() => onRemove(member.id)}
            aria-label="Remove member"
          >
            <i className="ph ph-trash"></i>
          </button>
        )}
      </div>
    </div>
  )
}
