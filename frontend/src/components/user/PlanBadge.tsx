/**
 * PlanBadge Component
 * Shows subscription tier with appropriate styling
 */

import styles from './Badge.module.css'

type PlanType = 'free' | 'pro' | 'enterprise'

interface PlanBadgeProps {
  plan: PlanType
  size?: 'sm' | 'md'
}

const planLabels: Record<PlanType, string> = {
  free: 'Free',
  pro: 'Pro',
  enterprise: 'Enterprise',
}

export function PlanBadge({ plan, size = 'md' }: PlanBadgeProps) {
  return (
    <span className={`${styles.badge} ${styles[`plan${plan.charAt(0).toUpperCase() + plan.slice(1)}`]} ${styles[size]}`}>
      {planLabels[plan]}
    </span>
  )
}
