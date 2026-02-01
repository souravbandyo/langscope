/**
 * RoleBadge Component
 * Shows user role in organization with appropriate styling
 */

import styles from './Badge.module.css'

type RoleType = 'owner' | 'admin' | 'member' | 'viewer'

interface RoleBadgeProps {
  role: RoleType
  size?: 'sm' | 'md'
}

const roleLabels: Record<RoleType, string> = {
  owner: 'Owner',
  admin: 'Admin',
  member: 'Member',
  viewer: 'Viewer',
}

export function RoleBadge({ role, size = 'md' }: RoleBadgeProps) {
  return (
    <span className={`${styles.badge} ${styles[`role${role.charAt(0).toUpperCase() + role.slice(1)}`]} ${styles[size]}`}>
      {roleLabels[role]}
    </span>
  )
}
