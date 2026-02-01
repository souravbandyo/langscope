"""
System constants for LangScope.

TrueSkill + Plackett-Luce rating system with multi-player matches.
"""

# =============================================================================
# TrueSkill Parameters
# =============================================================================

# Default mean rating
TRUESKILL_MU_0: float = 1500.0

# Default uncertainty (standard deviation)
TRUESKILL_SIGMA_0: float = 166.0

# Performance variability (skill variance in a single game)
# Typically σ_0 / 2
TRUESKILL_BETA: float = 83.0

# Dynamics factor (small σ increase between matches to allow for skill changes)
# Typically σ_0 / 20
TRUESKILL_TAU: float = 8.3

# Conservative estimate multiplier (μ - k*σ)
TRUESKILL_CONSERVATIVE_K: float = 3.0

# =============================================================================
# Match Configuration
# =============================================================================

# Number of players per match (target)
PLAYERS_PER_MATCH: int = 6

# Minimum players for a valid match
MIN_PLAYERS: int = 5

# Maximum players per match
MAX_PLAYERS: int = 6

# Maximum matches per model (cap for fair distribution)
MAX_MATCHES_PER_MODEL: int = 50

# Number of judge models per match
JUDGE_COUNT: int = 5

# =============================================================================
# Swiss Pairing Parameters
# =============================================================================

# Maximum TrueSkill μ difference for grouping players
SWISS_DELTA: float = 75.0

# =============================================================================
# Temperature Parameters
# =============================================================================

# Cost temperature for efficiency weighting
# Lower τ_c → more weight on cheaper models
COST_TEMP: float = 0.05

# Rating temperature for softmax weighting (judge/creator selection)
RATING_TEMP: float = 300.0

# Latency temperature for scoring (ms)
# S_lat = 1 / (1 + L / τ_L)
LATENCY_TEMP: float = 1000.0

# Time-to-first-token temperature for scoring (ms)
# S_ttft = 1 / (1 + T / τ_T)
TTFT_TEMP: float = 200.0

# =============================================================================
# Default Dimension Weights for Combined Score
# =============================================================================

DEFAULT_DIMENSION_WEIGHTS = {
    "raw_quality": 0.20,
    "cost_adjusted": 0.10,
    "latency": 0.10,
    "ttft": 0.05,
    "consistency": 0.10,
    "token_efficiency": 0.10,
    "instruction_following": 0.15,
    "hallucination_resistance": 0.15,
    "long_context": 0.05,
}

# =============================================================================
# Plackett-Luce Parameters
# =============================================================================

# Maximum iterations for MM algorithm
PLACKETT_LUCE_MAX_ITER: int = 100

# Convergence tolerance
PLACKETT_LUCE_TOL: float = 1e-6

# =============================================================================
# Transfer Learning Parameters
# =============================================================================

# Bayesian smoothing parameter for correlation learning
CORRELATION_TAU: float = 20.0

# Baseline domain uncertainty for transfer
SIGMA_BASE: float = 50.0

# Z-score threshold for specialist detection
SPECIALIST_Z_THRESHOLD: float = 2.0

# =============================================================================
# Strata Thresholds (based on TrueSkill μ)
# =============================================================================

STRATA_THRESHOLDS = {
    "elite": 1520,   # Stratum 4: Elite performers
    "high": 1450,    # Stratum 3: High performers
    "mid": 1400,     # Stratum 2: Mid performers
    "low": 0,        # Stratum 1: Low performers
}

# Stratum names mapping
STRATUM_NAMES = {
    4: "elite",
    3: "high",
    2: "mid",
    1: "low",
}

# =============================================================================
# Penalty System
# =============================================================================

# Judge penalty for outlier ranking (in μ points)
JUDGE_PENALTY_MU: float = 10.0

# Outlier disagreement threshold (>40% disagreement with consensus)
OUTLIER_DISAGREEMENT_THRESHOLD: float = 0.40

# =============================================================================
# Information Theory
# =============================================================================

# Pre-computed log2(n!) values for common match sizes
INFO_BITS = {
    5: 6.906890595,   # log2(5!) ≈ 6.9 bits
    6: 9.491853096,   # log2(6!) ≈ 9.5 bits
}


