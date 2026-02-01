"""
TrueSkill rating system for uncertainty-aware multi-player ranking.

Implements the TrueSkill algorithm for updating player ratings based on
multi-player match outcomes. Uses factor graphs and Gaussian belief propagation.

References:
- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian skill rating system.
"""

import math
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Import constants as fallback defaults
from langscope.core.constants import (
    TRUESKILL_MU_0,
    TRUESKILL_SIGMA_0,
    TRUESKILL_BETA,
    TRUESKILL_TAU,
    TRUESKILL_CONSERVATIVE_K,
)

if TYPE_CHECKING:
    from langscope.config.params import TrueSkillParams

# Try to import scipy for the normal CDF/PDF
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import math as _math
    
    # Fallback implementations
    def _norm_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + _math.erf(x / _math.sqrt(2)))
    
    def _norm_pdf(x: float) -> float:
        """Probability density function for standard normal."""
        return _math.exp(-0.5 * x * x) / _math.sqrt(2 * _math.pi)


@dataclass
class TrueSkillRating:
    """
    TrueSkill rating with mean and uncertainty.
    
    The skill of a player is modeled as a Gaussian distribution:
    θ ~ N(μ, σ²)
    
    Attributes:
        mu: Mean skill estimate
        sigma: Uncertainty (standard deviation)
    """
    mu: float = TRUESKILL_MU_0
    sigma: float = TRUESKILL_SIGMA_0
    
    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """
        95% confidence interval.
        
        Args:
            z: Z-score (1.96 for 95% CI)
        
        Returns:
            Tuple of (lower, upper) bounds
        """
        return (self.mu - z * self.sigma, self.mu + z * self.sigma)
    
    def conservative_estimate(self, k: float = TRUESKILL_CONSERVATIVE_K) -> float:
        """
        Conservative skill estimate (μ - kσ).
        
        Args:
            k: Number of standard deviations to subtract
        
        Returns:
            Conservative estimate
        """
        return self.mu - k * self.sigma
    
    def variance(self) -> float:
        """Get variance σ²."""
        return self.sigma ** 2
    
    def precision(self) -> float:
        """Get precision 1/σ²."""
        if self.sigma <= 0:
            return float('inf')
        return 1.0 / (self.sigma ** 2)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"mu": self.mu, "sigma": self.sigma}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrueSkillRating':
        """Create from dictionary."""
        return cls(
            mu=data.get("mu", TRUESKILL_MU_0),
            sigma=data.get("sigma", TRUESKILL_SIGMA_0)
        )
    
    def __repr__(self) -> str:
        return f"TrueSkillRating(μ={self.mu:.1f}, σ={self.sigma:.1f})"


class MultiPlayerTrueSkillUpdater:
    """
    TrueSkill update rules for multi-player matches (5-6 players).
    
    For a ranking r₁ > r₂ > ... > rₙ:
    - Each player updates based on comparisons with adjacent ranks
    - Uses Gaussian message passing for belief updates
    
    The update equations for adjacent pair (winner, loser):
    
    c² = 2β² + σ_winner² + σ_loser²
    t = (μ_winner - μ_loser) / c
    v = φ(t) / Φ(t)           # Truncated Gaussian update
    w = v(v + t)              # Uncertainty reduction factor
    
    Δμ_winner = (σ_winner² / c) × v
    Δμ_loser = -(σ_loser² / c) × v
    σ_new = σ × √(1 - (σ² / c²) × w)
    """
    
    def __init__(
        self,
        beta: float = TRUESKILL_BETA,
        tau: float = TRUESKILL_TAU,
        draw_probability: float = 0.0
    ):
        """
        Initialize TrueSkill updater.
        
        Args:
            beta: Performance variability (skill variance in a single game)
            tau: Dynamics factor (skill drift between games)
            draw_probability: Probability of draws (0 for no draws)
        """
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
        
        # Compute epsilon for draw range (not used if draw_probability=0)
        if draw_probability > 0:
            self.epsilon = self._compute_draw_margin(draw_probability)
        else:
            self.epsilon = 0.0
    
    def _compute_draw_margin(self, draw_probability: float) -> float:
        """Compute draw margin epsilon from draw probability."""
        if SCIPY_AVAILABLE:
            return norm.ppf((1 + draw_probability) / 2) * math.sqrt(2) * self.beta
        else:
            # Approximate inverse normal CDF
            p = (1 + draw_probability) / 2
            # Simple approximation for p close to 0.5
            return 0.0
    
    def _v_function(self, t: float, epsilon: float = 0.0) -> float:
        """
        V function for win/loss update.
        
        v(t, ε) = φ(t - ε) / Φ(t - ε)
        
        This is the ratio of the PDF to CDF of the standard normal,
        evaluated at t - ε. Represents the expected value adjustment.
        
        Args:
            t: Normalized performance difference
            epsilon: Draw margin (0 for no draws)
        
        Returns:
            Update factor v
        """
        x = t - epsilon
        
        if SCIPY_AVAILABLE:
            denom = norm.cdf(x)
        else:
            denom = _norm_cdf(x)
        
        if denom < 1e-10:
            # For very negative x, use asymptotic approximation
            return -x
        
        if SCIPY_AVAILABLE:
            return norm.pdf(x) / denom
        else:
            return _norm_pdf(x) / denom
    
    def _w_function(self, t: float, v: float, epsilon: float = 0.0) -> float:
        """
        W function for uncertainty update.
        
        w(t, ε) = v(t, ε) × (v(t, ε) + t - ε)
        
        This determines how much to reduce uncertainty.
        
        Args:
            t: Normalized performance difference
            v: Output of v_function
            epsilon: Draw margin
        
        Returns:
            Uncertainty reduction factor w
        """
        return v * (v + t - epsilon)
    
    def update_from_ranking(
        self,
        players: List[TrueSkillRating],
        ranking: List[int]
    ) -> List[TrueSkillRating]:
        """
        Update TrueSkill ratings based on multi-player ranking.
        
        Uses the partial comparison approach:
        - Player at rank k is compared to players at rank k-1 and k+1
        - Updates are accumulated from all adjacent comparisons
        
        Args:
            players: List of TrueSkillRating objects
            ranking: List where ranking[i] is the rank of player i (1-indexed, 1=best)
        
        Returns:
            List of updated TrueSkillRating objects
        
        Example:
            >>> updater = MultiPlayerTrueSkillUpdater()
            >>> players = [TrueSkillRating() for _ in range(6)]
            >>> ranking = [3, 1, 2, 6, 4, 5]  # Player 1 got rank 3, Player 0 got rank 1, etc.
            >>> new_ratings = updater.update_from_ranking(players, ranking)
        """
        n = len(players)
        
        if n != len(ranking):
            raise ValueError(f"Number of players ({n}) must match ranking length ({len(ranking)})")
        
        if n < 2:
            return players.copy() if hasattr(players, 'copy') else list(players)
        
        # Sort player indices by rank (ascending: best first)
        sorted_indices = sorted(range(n), key=lambda i: ranking[i])
        
        # Initialize update accumulators
        mu_updates = [0.0] * n
        sigma_sq_factors = [1.0] * n  # Multiplicative factors for variance
        
        # Process adjacent pairs in rank order
        for pos in range(n - 1):
            winner_idx = sorted_indices[pos]
            loser_idx = sorted_indices[pos + 1]
            
            winner = players[winner_idx]
            loser = players[loser_idx]
            
            # Compute performance difference variance
            c_sq = 2 * self.beta**2 + winner.sigma**2 + loser.sigma**2
            c = math.sqrt(c_sq)
            
            # Normalized performance difference
            t = (winner.mu - loser.mu) / c
            
            # Compute update factors
            v = self._v_function(t, self.epsilon)
            w = self._w_function(t, v, self.epsilon)
            
            # Winner gets positive update
            winner_var = winner.sigma**2
            mu_updates[winner_idx] += (winner_var / c) * v
            sigma_sq_factors[winner_idx] *= max(0, 1 - (winner_var / c_sq) * w)
            
            # Loser gets negative update
            loser_var = loser.sigma**2
            mu_updates[loser_idx] -= (loser_var / c) * v
            sigma_sq_factors[loser_idx] *= max(0, 1 - (loser_var / c_sq) * w)
        
        # Apply updates with dynamics factor
        updated_players = []
        for i in range(n):
            # Add dynamics factor (slight uncertainty increase for skill drift)
            new_var = players[i].sigma**2 * sigma_sq_factors[i] + self.tau**2
            new_sigma = math.sqrt(new_var)
            
            updated_players.append(TrueSkillRating(
                mu=players[i].mu + mu_updates[i],
                sigma=new_sigma
            ))
        
        return updated_players
    
    def update_pairwise(
        self,
        winner: TrueSkillRating,
        loser: TrueSkillRating,
        drawn: bool = False
    ) -> Tuple[TrueSkillRating, TrueSkillRating]:
        """
        Update ratings for a pairwise comparison.
        
        Convenience method for single comparisons.
        
        Args:
            winner: Winner's rating
            loser: Loser's rating
            drawn: Whether the match was a draw
        
        Returns:
            Tuple of (new_winner_rating, new_loser_rating)
        """
        # Use ranking update with 2 players
        players = [winner, loser]
        
        if drawn:
            # For draws, use ranks [1, 1]
            ranking = [1, 1]
        else:
            ranking = [1, 2]  # winner got rank 1, loser got rank 2
        
        updated = self.update_from_ranking(players, ranking)
        return updated[0], updated[1]
    
    def expected_performance(
        self,
        player_a: TrueSkillRating,
        player_b: TrueSkillRating
    ) -> float:
        """
        Calculate expected win probability for player A against player B.
        
        P(A beats B) = Φ((μ_A - μ_B) / √(2β² + σ_A² + σ_B²))
        
        Args:
            player_a: First player's rating
            player_b: Second player's rating
        
        Returns:
            Probability that player A beats player B
        """
        mu_diff = player_a.mu - player_b.mu
        sigma_sq = 2 * self.beta**2 + player_a.sigma**2 + player_b.sigma**2
        
        t = mu_diff / math.sqrt(sigma_sq)
        
        if SCIPY_AVAILABLE:
            return norm.cdf(t)
        else:
            return _norm_cdf(t)
    
    def quality_multiplayer(self, players: List[TrueSkillRating]) -> float:
        """
        Calculate match quality for a multi-player match.
        
        Higher quality means more uncertain outcome (more informative match).
        
        Args:
            players: List of player ratings
        
        Returns:
            Match quality score (0 to 1)
        """
        if len(players) < 2:
            return 0.0
        
        # Average pairwise quality
        total_quality = 0.0
        count = 0
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p_a_wins = self.expected_performance(players[i], players[j])
                # Quality is highest when p ≈ 0.5
                quality = 2 * min(p_a_wins, 1 - p_a_wins)
                total_quality += quality
                count += 1
        
        return total_quality / count if count > 0 else 0.0


def create_updater(
    beta: float = None,
    tau: float = None,
    params: 'TrueSkillParams' = None
) -> MultiPlayerTrueSkillUpdater:
    """
    Create a TrueSkill updater with specified parameters.
    
    Args:
        beta: Performance variability (overrides params if both provided)
        tau: Dynamics factor (overrides params if both provided)
        params: TrueSkillParams from ParameterManager (optional)
    
    Returns:
        Configured MultiPlayerTrueSkillUpdater
    """
    if params is not None:
        beta = beta if beta is not None else params.beta
        tau = tau if tau is not None else params.tau
    else:
        beta = beta if beta is not None else TRUESKILL_BETA
        tau = tau if tau is not None else TRUESKILL_TAU
    
    return MultiPlayerTrueSkillUpdater(beta=beta, tau=tau)


def create_updater_from_manager(domain: str = None) -> MultiPlayerTrueSkillUpdater:
    """
    Create a TrueSkill updater using ParameterManager.
    
    Args:
        domain: Optional domain for domain-specific params
    
    Returns:
        Configured MultiPlayerTrueSkillUpdater
    """
    try:
        from langscope.config.params import get_parameter_manager
        manager = get_parameter_manager()
        params = manager.get_trueskill_params()
        return MultiPlayerTrueSkillUpdater(beta=params.beta, tau=params.tau)
    except ImportError:
        # Fallback to constants if params module not available
        return MultiPlayerTrueSkillUpdater()


