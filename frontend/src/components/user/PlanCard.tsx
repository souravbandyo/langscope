/**
 * PlanCard Component
 * Displays a subscription plan with features and pricing
 */

import type { Plan } from '../../api/types'
import styles from './PlanCard.module.css'

interface PlanCardProps {
  plan: Plan
  isCurrentPlan: boolean
  onSelect: (planId: string) => void
  isLoading?: boolean
}

export function PlanCard({
  plan,
  isCurrentPlan,
  onSelect,
  isLoading = false,
}: PlanCardProps) {
  const formatPrice = (cents: number) => {
    if (cents === 0) return plan.id === 'enterprise' ? 'Contact Us' : 'Free'
    return `$${(cents / 100).toFixed(0)}`
  }

  const features = [
    {
      label: 'Evaluations/month',
      value:
        plan.features.evaluations_per_month >= 999999
          ? 'Unlimited'
          : plan.features.evaluations_per_month.toLocaleString(),
      included: true,
    },
    {
      label: 'Team members',
      value:
        plan.features.team_members >= 999999
          ? 'Unlimited'
          : plan.features.team_members.toString(),
      included: true,
    },
    { label: 'API access', included: plan.features.api_access },
    { label: 'All domains', included: plan.features.all_domains },
    { label: 'Custom domains', included: plan.features.custom_domains },
    { label: 'Priority support', included: plan.features.priority_support },
    { label: 'Export reports', included: plan.features.export_reports },
    { label: 'SSO/SAML', included: plan.features.sso_saml },
    { label: 'SLA guarantee', included: plan.features.sla_guarantee },
  ]

  return (
    <div
      className={`${styles.card} ${plan.popular ? styles.popular : ''} ${isCurrentPlan ? styles.current : ''}`}
    >
      {plan.popular && <span className={styles.popularBadge}>Most Popular</span>}
      {isCurrentPlan && <span className={styles.currentBadge}>Current Plan</span>}

      <div className={styles.header}>
        <h3 className={styles.name}>{plan.name}</h3>
        <div className={styles.pricing}>
          <span className={styles.price}>{formatPrice(plan.price_monthly)}</span>
          {plan.price_monthly > 0 && <span className={styles.period}>/month</span>}
        </div>
        {plan.price_yearly > 0 && (
          <span className={styles.yearlyPrice}>
            ${(plan.price_yearly / 100).toFixed(0)}/year (save $
            {((plan.price_monthly * 12 - plan.price_yearly) / 100).toFixed(0)})
          </span>
        )}
      </div>

      <ul className={styles.features}>
        {features.map((feature, index) => (
          <li
            key={index}
            className={`${styles.feature} ${feature.included ? styles.included : styles.excluded}`}
          >
            <i className={`ph ${feature.included ? 'ph-check' : 'ph-x'}`}></i>
            <span>
              {feature.value ? `${feature.value} ${feature.label}` : feature.label}
            </span>
          </li>
        ))}
      </ul>

      <button
        className={`${styles.selectButton} ${isCurrentPlan ? styles.disabled : ''}`}
        onClick={() => !isCurrentPlan && onSelect(plan.id)}
        disabled={isCurrentPlan || isLoading}
      >
        {isLoading ? (
          <>
            <i className="ph ph-spinner-gap"></i>
            Processing...
          </>
        ) : isCurrentPlan ? (
          'Current Plan'
        ) : plan.id === 'enterprise' ? (
          'Contact Sales'
        ) : (
          `Upgrade to ${plan.name}`
        )}
      </button>
    </div>
  )
}
