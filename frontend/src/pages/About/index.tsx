import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import styles from './About.module.css'

/**
 * About page with project information
 */
export function About() {
  return (
    <div className={styles.about}>
      <h1 className={styles.title}>About LangScope</h1>

      <div className={styles.content}>
        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>What is LangScope?</h2>
          <p className={styles.text}>
            LangScope is a multi-domain LLM evaluation framework that uses{' '}
            <strong>TrueSkill + Plackett-Luce</strong> algorithms to rank and
            compare language models across different domains.
          </p>
          <p className={styles.text}>
            Our system provides <strong>10-dimensional ratings</strong> including
            raw quality, cost-adjusted performance, latency, consistency, and more.
          </p>
        </SketchCard>

        <div className={styles.features}>
          <StickyNote title="10D Evaluation" color="yellow" rotation={-2} pinned>
            <ul className={styles.featureList}>
              <li>Raw Quality</li>
              <li>Cost Adjusted</li>
              <li>Latency</li>
              <li>Consistency</li>
              <li>Token Efficiency</li>
            </ul>
          </StickyNote>

          <StickyNote title="Multi-Domain" color="blue" rotation={1} pinned>
            <ul className={styles.featureList}>
              <li>Code Generation</li>
              <li>Math Reasoning</li>
              <li>Medical</li>
              <li>Legal</li>
              <li>Finance</li>
            </ul>
          </StickyNote>

          <StickyNote title="Features" color="green" rotation={-1} pinned>
            <ul className={styles.featureList}>
              <li>Arena Mode</li>
              <li>Transfer Learning</li>
              <li>Ground Truth</li>
              <li>Recommendations</li>
            </ul>
          </StickyNote>
        </div>

        <SketchCard padding="lg">
          <h2 className={styles.sectionTitle}>How It Works</h2>
          <ol className={styles.howItWorks}>
            <li>
              <strong>Peer Evaluation:</strong> Models compete in matches judged by
              other LLMs using Swiss-style pairing
            </li>
            <li>
              <strong>TrueSkill Ratings:</strong> Bayesian skill rating system
              provides robust rankings with uncertainty estimates
            </li>
            <li>
              <strong>Domain Specialization:</strong> Track model performance
              across specific domains like code, medical, legal
            </li>
            <li>
              <strong>User Feedback:</strong> Arena mode lets you contribute your
              own rankings to improve recommendations
            </li>
          </ol>
        </SketchCard>
      </div>
    </div>
  )
}

export { About as default }
