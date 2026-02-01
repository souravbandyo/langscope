import { useState, useMemo } from 'react'
import { SketchCard, SketchButton } from '@/components/sketch'
import styles from './LedgerVerification.module.css'

interface LedgerProof {
  id: string
  type: 'arweave' | 'ipfs' | 'polygon'
  timestamp: string
  dataHash: string
  transactionId: string
  status: 'verified' | 'pending' | 'failed'
  blockNumber?: number
  dataType: 'ratings' | 'matches' | 'evaluations'
}

// Simulated ledger proofs - in production these would come from API
const mockProofs: LedgerProof[] = [
  {
    id: '1',
    type: 'arweave',
    timestamp: '2026-02-01T10:30:00Z',
    dataHash: '0x7f8a9b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a',
    transactionId: 'ar_tx_abc123def456',
    status: 'verified',
    dataType: 'ratings',
  },
  {
    id: '2',
    type: 'ipfs',
    timestamp: '2026-02-01T09:15:00Z',
    dataHash: 'QmXoYpZJKm4dBv8hN9XqGfWQbJnL2Rp4cK7dS5vT6wU8xY',
    transactionId: 'ipfs_pin_789xyz',
    status: 'verified',
    dataType: 'matches',
  },
  {
    id: '3',
    type: 'polygon',
    timestamp: '2026-02-01T08:00:00Z',
    dataHash: '0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b',
    transactionId: '0x4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f',
    status: 'verified',
    blockNumber: 52847291,
    dataType: 'evaluations',
  },
  {
    id: '4',
    type: 'arweave',
    timestamp: '2026-01-31T23:45:00Z',
    dataHash: '0x9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b',
    transactionId: 'ar_tx_pending_001',
    status: 'pending',
    dataType: 'ratings',
  },
]

const ledgerConfig = {
  arweave: {
    name: 'Arweave',
    icon: 'ph ph-archive',
    color: '#333',
    explorer: 'https://viewblock.io/arweave/tx/',
    description: 'Permanent, immutable storage',
  },
  ipfs: {
    name: 'IPFS',
    icon: 'ph ph-globe',
    color: '#1976d2',
    explorer: 'https://ipfs.io/ipfs/',
    description: 'Distributed content storage',
  },
  polygon: {
    name: 'Polygon',
    icon: 'ph ph-link-simple',
    color: '#8247e5',
    explorer: 'https://polygonscan.com/tx/',
    description: 'On-chain verification',
  },
}

/**
 * LedgerVerification component showing decentralized storage proofs
 */
export function LedgerVerification() {
  const [selectedLedger, setSelectedLedger] = useState<'all' | 'arweave' | 'ipfs' | 'polygon'>('all')
  const [isExpanded, setIsExpanded] = useState(false)

  const filteredProofs = useMemo(() => {
    const proofs = selectedLedger === 'all' 
      ? mockProofs 
      : mockProofs.filter(p => p.type === selectedLedger)
    return isExpanded ? proofs : proofs.slice(0, 3)
  }, [selectedLedger, isExpanded])

  const stats = useMemo(() => {
    const verified = mockProofs.filter(p => p.status === 'verified').length
    const pending = mockProofs.filter(p => p.status === 'pending').length
    const byType = {
      arweave: mockProofs.filter(p => p.type === 'arweave').length,
      ipfs: mockProofs.filter(p => p.type === 'ipfs').length,
      polygon: mockProofs.filter(p => p.type === 'polygon').length,
    }
    return { verified, pending, byType }
  }, [])

  const truncateHash = (hash: string) => {
    if (hash.length <= 16) return hash
    return `${hash.slice(0, 8)}...${hash.slice(-8)}`
  }

  return (
    <SketchCard padding="lg">
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          <h2 className={styles.title}>🔐 Public Ledger Verification</h2>
          <span className={styles.badge}>{stats.verified} Verified</span>
        </div>
        <p className={styles.subtitle}>
          All ratings and evaluations are cryptographically anchored to decentralized storage
        </p>
      </div>

      {/* Stats Row */}
      <div className={styles.statsRow}>
        <div className={styles.statItem}>
          <i className={`ph ph-check-circle ${styles.statIcon}`}></i>
          <span className={styles.statValue}>{stats.verified}</span>
          <span className={styles.statLabel}>Verified</span>
        </div>
        <div className={styles.statItem}>
          <i className={`ph ph-hourglass ${styles.statIcon}`}></i>
          <span className={styles.statValue}>{stats.pending}</span>
          <span className={styles.statLabel}>Pending</span>
        </div>
        <div className={styles.statDivider} />
        {Object.entries(ledgerConfig).map(([key, config]) => (
          <div key={key} className={styles.statItem}>
            <i className={`${config.icon} ${styles.statIcon}`}></i>
            <span className={styles.statValue}>{stats.byType[key as keyof typeof stats.byType]}</span>
            <span className={styles.statLabel}>{config.name}</span>
          </div>
        ))}
      </div>

      {/* Filter Tabs */}
      <div className={styles.filterTabs}>
        <button
          className={`${styles.filterTab} ${selectedLedger === 'all' ? styles.active : ''}`}
          onClick={() => setSelectedLedger('all')}
        >
          All
        </button>
        {Object.entries(ledgerConfig).map(([key, config]) => (
          <button
            key={key}
            className={`${styles.filterTab} ${selectedLedger === key ? styles.active : ''}`}
            onClick={() => setSelectedLedger(key as typeof selectedLedger)}
          >
            <i className={config.icon}></i> {config.name}
          </button>
        ))}
      </div>

      {/* Proofs List */}
      <div className={styles.proofsList}>
        {filteredProofs.map((proof) => {
          const config = ledgerConfig[proof.type]
          return (
            <div key={proof.id} className={styles.proofItem}>
              <div className={styles.proofHeader}>
                <span 
                  className={styles.ledgerBadge}
                  style={{ background: `${config.color}15`, borderColor: config.color, color: config.color }}
                >
                  <i className={config.icon}></i> {config.name}
                </span>
                <span className={`${styles.statusBadge} ${styles[proof.status]}`}>
                  {proof.status === 'verified' && <><i className="ph ph-check"></i> Verified</>}
                  {proof.status === 'pending' && <><i className="ph ph-hourglass"></i> Pending</>}
                  {proof.status === 'failed' && <><i className="ph ph-x"></i> Failed</>}
                </span>
                <span className={styles.dataType}>{proof.dataType}</span>
              </div>
              
              <div className={styles.proofDetails}>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Data Hash:</span>
                  <code className={styles.hashValue}>{truncateHash(proof.dataHash)}</code>
                  <button 
                    className={styles.copyBtn}
                    onClick={() => navigator.clipboard.writeText(proof.dataHash)}
                    title="Copy hash"
                  >
                    <i className="ph ph-copy"></i>
                  </button>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>TX ID:</span>
                  <a 
                    href={`${config.explorer}${proof.transactionId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.txLink}
                  >
                    {truncateHash(proof.transactionId)} ↗
                  </a>
                </div>
                {proof.blockNumber && (
                  <div className={styles.detailRow}>
                    <span className={styles.detailLabel}>Block:</span>
                    <span className={styles.blockNumber}>#{proof.blockNumber.toLocaleString()}</span>
                  </div>
                )}
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Time:</span>
                  <span className={styles.timestamp}>
                    {new Date(proof.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {mockProofs.length > 3 && (
        <div className={styles.expandSection}>
          <SketchButton 
            variant="secondary" 
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? 'Show Less' : `Show All (${mockProofs.length})`}
          </SketchButton>
        </div>
      )}

      {/* Info Section */}
      <div className={styles.infoSection}>
        <h4 className={styles.infoTitle}>How Verification Works</h4>
        <div className={styles.infoGrid}>
          <div className={styles.infoCard}>
            <i className={`ph ph-archive ${styles.infoIcon}`}></i>
            <span className={styles.infoName}>Arweave</span>
            <p>Permanent storage with endowment model - data persists forever</p>
          </div>
          <div className={styles.infoCard}>
            <i className={`ph ph-globe ${styles.infoIcon}`}></i>
            <span className={styles.infoName}>IPFS</span>
            <p>Content-addressed storage - hash is the identifier</p>
          </div>
          <div className={styles.infoCard}>
            <i className={`ph ph-link-simple ${styles.infoIcon}`}></i>
            <span className={styles.infoName}>Polygon</span>
            <p>Smart contract verification with on-chain timestamps</p>
          </div>
        </div>
      </div>
    </SketchCard>
  )
}
