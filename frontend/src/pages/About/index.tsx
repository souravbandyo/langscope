import { SketchCard } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import styles from './About.module.css'

/**
 * About page with comprehensive project information
 */
export function About() {
  return (
    <div className={styles.about}>
      {/* Hero Section */}
      <section className={styles.hero}>
        <img 
          src="/images/langscope-logo.png" 
          alt="LangScope" 
          className={styles.heroLogo}
        />
      </section>

      {/* What is LangScope */}
      <section className={styles.section}>
        <SketchCard padding="lg" className={styles.introCard}>
          <h2 className={styles.sectionTitle}>
            <i className="ph ph-question"></i> What is LangScope?
          </h2>
          <p className={styles.text}>
            LangScope is a comprehensive evaluation framework that uses{' '}
            <strong>TrueSkill + Plackett-Luce</strong> algorithms to provide fair, 
            statistically robust rankings of language models across multiple domains.
          </p>
          <p className={styles.text}>
            Unlike single-score benchmarks, we provide <strong>10-dimensional ratings</strong> that 
            capture different aspects of model performance, helping you find the right model for 
            your specific use case.
          </p>
        </SketchCard>
      </section>

      {/* How It Works - Visual Flow */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <i className="ph ph-flow-arrow"></i> How It Works
        </h2>
        <div className={styles.flowDiagram}>
          <div className={styles.flowStep}>
            <div className={styles.flowIcon}>
              <i className="ph ph-robot"></i>
            </div>
            <div className={styles.flowLabel}>Models Enter</div>
            <p className={styles.flowDesc}>LLMs join the evaluation pool</p>
          </div>
          <div className={styles.flowArrow}>
            <i className="ph ph-arrow-right"></i>
          </div>
          <div className={styles.flowStep}>
            <div className={styles.flowIcon}>
              <i className="ph ph-scales"></i>
            </div>
            <div className={styles.flowLabel}>Peer Evaluation</div>
            <p className={styles.flowDesc}>Swiss-style pairing matches</p>
          </div>
          <div className={styles.flowArrow}>
            <i className="ph ph-arrow-right"></i>
          </div>
          <div className={styles.flowStep}>
            <div className={styles.flowIcon}>
              <i className="ph ph-chart-line-up"></i>
            </div>
            <div className={styles.flowLabel}>TrueSkill Rating</div>
            <p className={styles.flowDesc}>Bayesian skill estimation</p>
          </div>
          <div className={styles.flowArrow}>
            <i className="ph ph-arrow-right"></i>
          </div>
          <div className={styles.flowStep}>
            <div className={styles.flowIcon}>
              <i className="ph ph-trophy"></i>
            </div>
            <div className={styles.flowLabel}>Domain Rankings</div>
            <p className={styles.flowDesc}>Specialized leaderboards</p>
          </div>
        </div>
      </section>

      {/* Key Features - 6 Cards */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <i className="ph ph-sparkle"></i> Key Features
        </h2>
        <div className={styles.featuresGrid}>
          <StickyNote title="10D Evaluation" color="yellow" rotation={-2} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-chart-polar"></i>
              <ul className={styles.featureList}>
                <li>Raw Quality</li>
                <li>Cost Adjusted</li>
                <li>Latency & TTFT</li>
                <li>Consistency</li>
                <li>Token Efficiency</li>
              </ul>
            </div>
          </StickyNote>

          <StickyNote title="Multi-Domain" color="blue" rotation={1} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-folders"></i>
              <ul className={styles.featureList}>
                <li>Code Generation</li>
                <li>Math Reasoning</li>
                <li>Medical & Legal</li>
                <li>Finance & Creative</li>
                <li>Custom Domains</li>
              </ul>
            </div>
          </StickyNote>

          <StickyNote title="Arena Mode" color="pink" rotation={-1} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-sword"></i>
              <ul className={styles.featureList}>
                <li>Blind Testing</li>
                <li>Side-by-Side</li>
                <li>Your Rankings</li>
                <li>Real Prompts</li>
                <li>Community Data</li>
              </ul>
            </div>
          </StickyNote>

          <StickyNote title="Transfer Learning" color="green" rotation={2} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-arrows-left-right"></i>
              <ul className={styles.featureList}>
                <li>Domain Correlations</li>
                <li>Cold Start Support</li>
                <li>Cross-Domain Insights</li>
                <li>Skill Transfer</li>
                <li>Prediction Boost</li>
              </ul>
            </div>
          </StickyNote>

          <StickyNote title="Ground Truth" color="orange" rotation={-1} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-check-square"></i>
              <ul className={styles.featureList}>
                <li>Verified Answers</li>
                <li>Objective Metrics</li>
                <li>Benchmark Correlation</li>
                <li>Accuracy Tracking</li>
                <li>Quality Anchors</li>
              </ul>
            </div>
          </StickyNote>

          <StickyNote title="Recommendations" color="yellow" rotation={1} pinned>
            <div className={styles.featureContent}>
              <i className="ph ph-lightbulb"></i>
              <ul className={styles.featureList}>
                <li>Use-Case Matching</li>
                <li>Budget Optimization</li>
                <li>Latency Constraints</li>
                <li>Domain Expertise</li>
                <li>Personalized Picks</li>
              </ul>
            </div>
          </StickyNote>
        </div>
      </section>

      {/* Who Benefits */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <i className="ph ph-users-three"></i> Who Benefits?
        </h2>
        <div className={styles.useCasesGrid}>
          <SketchCard padding="md" className={styles.useCaseCard}>
            <div className={styles.useCaseIcon}>
              <i className="ph ph-code"></i>
            </div>
            <h3 className={styles.useCaseTitle}>Developers</h3>
            <p className={styles.useCaseDesc}>
              Find the best model for your API integration. Compare costs, latency, 
              and quality to make informed decisions for your applications.
            </p>
          </SketchCard>

          <SketchCard padding="md" className={styles.useCaseCard}>
            <div className={styles.useCaseIcon}>
              <i className="ph ph-flask"></i>
            </div>
            <h3 className={styles.useCaseTitle}>Researchers</h3>
            <p className={styles.useCaseDesc}>
              Understand model capabilities across domains. Use our data for papers, 
              track progress over time, and identify gaps in current models.
            </p>
          </SketchCard>

          <SketchCard padding="md" className={styles.useCaseCard}>
            <div className={styles.useCaseIcon}>
              <i className="ph ph-buildings"></i>
            </div>
            <h3 className={styles.useCaseTitle}>Businesses</h3>
            <p className={styles.useCaseDesc}>
              Optimize your AI spend. Get recommendations based on your budget, 
              latency requirements, and domain-specific needs.
            </p>
          </SketchCard>
        </div>
      </section>

      {/* Technology */}
      <section className={styles.section}>
        <SketchCard padding="lg" className={styles.techCard}>
          <h2 className={styles.sectionTitle}>
            <i className="ph ph-cpu"></i> Our Technology
          </h2>
          <div className={styles.techBadges}>
            <span className={styles.techBadge}>
              <i className="ph ph-chart-scatter"></i> TrueSkill Algorithm
            </span>
            <span className={styles.techBadge}>
              <i className="ph ph-ranking"></i> Plackett-Luce Model
            </span>
            <span className={styles.techBadge}>
              <i className="ph ph-brain"></i> Bayesian Inference
            </span>
            <span className={styles.techBadge}>
              <i className="ph ph-brackets-curly"></i> Swiss Pairing
            </span>
          </div>
          <p className={styles.text}>
            Our ranking system combines <strong>Microsoft's TrueSkill</strong> algorithm 
            (originally designed for Xbox matchmaking) with the <strong>Plackett-Luce</strong> 
            model for handling multi-way comparisons, giving you statistically robust rankings 
            with confidence intervals.
          </p>
        </SketchCard>
      </section>

      {/* Contact / Contribute */}
      <section className={styles.section}>
        <SketchCard padding="lg" className={styles.contactCard}>
          <h2 className={styles.sectionTitle}>
            <i className="ph ph-hand-waving"></i> Get Involved
          </h2>
          <div className={styles.contactGrid}>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className={styles.contactLink}>
              <i className="ph ph-github-logo"></i>
              <span>View on GitHub</span>
            </a>
            <a href="mailto:feedback@langscope.dev" className={styles.contactLink}>
              <i className="ph ph-envelope"></i>
              <span>Send Feedback</span>
            </a>
            <a href="/arena" className={styles.contactLink}>
              <i className="ph ph-sword"></i>
              <span>Contribute Rankings</span>
            </a>
          </div>
          <p className={styles.contactText}>
            LangScope is an open community effort. Help us improve by contributing 
            your rankings in Arena mode or sharing feedback!
          </p>
        </SketchCard>
      </section>
    </div>
  )
}

export { About as default }
