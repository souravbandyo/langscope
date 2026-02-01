"""
Selection mechanisms for multi-player matches (5-6 players).

Implements Swiss-style pairing, content creator selection, and judge selection
using softmax weighting and inverse match-count sampling.
"""

import random
from typing import List, Dict, Optional, Set, TYPE_CHECKING

import numpy as np

# Import constants as fallback defaults
from langscope.core.constants import (
    SWISS_DELTA,
    MAX_MATCHES_PER_MODEL,
    PLAYERS_PER_MATCH,
    MIN_PLAYERS,
    MAX_PLAYERS,
    RATING_TEMP,
    STRATA_THRESHOLDS,
)

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.config.params import MatchParams, TemperatureParams


def _get_match_params(domain: str = None) -> 'MatchParams':
    """Get match params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_match_params(domain)
    except ImportError:
        from langscope.config.params.models import MatchParams
        return MatchParams()


def _get_temperature_params() -> 'TemperatureParams':
    """Get temperature params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_temperature_params()
    except ImportError:
        from langscope.config.params.models import TemperatureParams
        return TemperatureParams()


class MultiPlayerSwissPairing:
    """
    Swiss-style grouping for multi-player matches (5-6 players).
    
    Goals:
    1. Fair opportunity: All models get similar number of matches
    2. Informative matches: Group models with similar ratings
    3. Diversity: Different opponents across matches
    
    Algorithm:
    1. Select anchor player using inverse match-count weighting
    2. Find eligible players within μ ± Δ of anchor
    3. Select additional players with inverse match-count weighting
    4. Ensure diverse grouping (no recent repeats)
    """
    
    def __init__(
        self,
        mu_delta: float = None,
        max_matches: int = None,
        players_per_match: int = None,
        min_players: int = None,
        domain: str = None
    ):
        """
        Initialize Swiss pairing.
        
        Args:
            mu_delta: Maximum TrueSkill μ difference for grouping
            max_matches: Maximum matches per model
            players_per_match: Target number of players (5-6)
            min_players: Minimum required players
            domain: Optional domain for domain-specific params
        """
        # Get params from manager if not explicitly provided
        params = _get_match_params(domain)
        
        self.mu_delta = mu_delta if mu_delta is not None else params.swiss_delta
        self.max_matches = max_matches if max_matches is not None else params.max_matches_per_model
        self.players_per_match = players_per_match if players_per_match is not None else params.players_per_match
        self.min_players = min_players if min_players is not None else params.min_players
    
    def select_match_players(
        self,
        models: List['LLMModel'],
        domain: str,
        recent_groups: Set[frozenset] = None,
        max_attempts: int = 10
    ) -> Optional[List['LLMModel']]:
        """
        Select 5-6 players for a multi-player match.
        
        Args:
            models: List of all models
            domain: Domain name
            recent_groups: Set of recent player groups to avoid
            max_attempts: Maximum selection attempts
        
        Returns:
            List of 5-6 LLMModel objects, or None if not enough eligible
        """
        recent_groups = recent_groups or set()
        
        for attempt in range(max_attempts):
            result = self._try_select_players(models, domain, recent_groups)
            if result:
                return result
        
        return None
    
    def _try_select_players(
        self,
        models: List['LLMModel'],
        domain: str,
        recent_groups: Set[frozenset]
    ) -> Optional[List['LLMModel']]:
        """Single attempt to select players."""
        # Filter eligible models (under match cap)
        eligible = [
            m for m in models
            if self._get_match_count(m, domain) < self.max_matches
        ]
        
        if len(eligible) < self.min_players:
            return None
        
        # Step 1: Select anchor with inverse match-count weighting
        anchor = self._select_by_inverse_match_count(eligible, domain)
        if not anchor:
            return None
        
        anchor_mu = self._get_mu(anchor, domain)
        selected = [anchor]
        remaining = [m for m in eligible if m.model_id != anchor.model_id]
        
        # Step 2: Find models within μ ± Δ
        stratum = [
            m for m in remaining
            if abs(self._get_mu(m, domain) - anchor_mu) <= self.mu_delta
        ]
        
        # If not enough in stratum, expand to all remaining
        if len(stratum) < self.min_players - 1:
            stratum = remaining
        
        # Step 3: Select remaining players
        target_count = min(self.players_per_match, len(stratum) + 1)
        
        while len(selected) < target_count and stratum:
            next_player = self._select_by_inverse_match_count(stratum, domain)
            if next_player:
                selected.append(next_player)
                stratum = [m for m in stratum if m.model_id != next_player.model_id]
            else:
                break
        
        if len(selected) < self.min_players:
            return None
        
        # Step 4: Check for recent repeat
        group_key = frozenset(m.model_id for m in selected)
        if group_key in recent_groups:
            return None
        
        return selected
    
    def _select_by_inverse_match_count(
        self,
        models: List['LLMModel'],
        domain: str
    ) -> Optional['LLMModel']:
        """
        Select model with inverse match-count weighting.
        
        P(m_i) = (1/(1+n_i)) / Σ_j(1/(1+n_j))
        
        This gives higher probability to models with fewer matches.
        """
        if not models:
            return None
        
        weights = []
        for m in models:
            n = self._get_match_count(m, domain)
            weights.append(1.0 / (1.0 + n))
        
        total = sum(weights)
        if total <= 0:
            return random.choice(models)
        
        probabilities = [w / total for w in weights]
        return random.choices(models, weights=probabilities, k=1)[0]
    
    def _get_match_count(self, model: 'LLMModel', domain: str) -> int:
        """Get match count for model in domain."""
        if domain in model.performance_by_domain:
            return model.performance_by_domain[domain].total_matches_played
        return model.performance.total_matches_played
    
    def _get_mu(self, model: 'LLMModel', domain: str) -> float:
        """Get raw TrueSkill μ for model in domain."""
        if domain in model.trueskill_by_domain:
            return model.trueskill_by_domain[domain].raw.mu
        return model.trueskill.raw.mu


class ContentCreatorSelector:
    """
    Select content creators using softmax weighting.
    
    P(m_i) = e^(μ_i/τ) / Σ_j e^(μ_j/τ)
    
    Higher-rated models are more likely to be selected.
    """
    
    def __init__(
        self,
        tau: float = None,
        elite_threshold: float = None,
        domain: str = None
    ):
        """
        Initialize content creator selector.
        
        Args:
            tau: Softmax temperature (higher = more uniform)
            elite_threshold: Minimum μ for content creation
            domain: Optional domain for domain-specific params
        """
        # Get params from manager if not explicitly provided
        temp_params = _get_temperature_params()
        
        self.tau = tau if tau is not None else temp_params.rating_temp
        
        if elite_threshold is None:
            try:
                from langscope.config.params import get_parameter_manager
                strata_params = get_parameter_manager().get_strata_params(domain)
                self.elite_threshold = strata_params.elite_threshold
            except ImportError:
                self.elite_threshold = float(STRATA_THRESHOLDS.get("elite", 1520))
        else:
            self.elite_threshold = elite_threshold
    
    def select(
        self,
        models: List['LLMModel'],
        domain: str,
        exclude: List[str] = None,
        min_stratum: int = 3
    ) -> Optional['LLMModel']:
        """
        Select content creator from high-rated models.
        
        Args:
            models: List of all models
            domain: Domain for evaluation
            exclude: Model names to exclude
            min_stratum: Minimum stratum requirement
        
        Returns:
            Selected model or None
        """
        exclude = exclude or []
        
        # Filter to eligible models
        eligible = []
        for m in models:
            if m.name in exclude or m.model_id in exclude:
                continue
            if m.get_stratum(domain) >= min_stratum:
                eligible.append(m)
        
        if not eligible:
            return None
        
        # Softmax weighting
        mus = [self._get_mu(m, domain) for m in eligible]
        weights = self._softmax(mus)
        
        return random.choices(eligible, weights=weights, k=1)[0]
    
    def _get_mu(self, model: 'LLMModel', domain: str) -> float:
        """Get raw TrueSkill μ for model."""
        if domain in model.trueskill_by_domain:
            return model.trueskill_by_domain[domain].raw.mu
        return model.trueskill.raw.mu
    
    def _softmax(self, values: List[float]) -> List[float]:
        """Compute softmax probabilities."""
        arr = np.array(values)
        scaled = arr / self.tau
        scaled = scaled - np.max(scaled)  # Numerical stability
        exp_vals = np.exp(scaled)
        return (exp_vals / np.sum(exp_vals)).tolist()


class JudgeSelector(ContentCreatorSelector):
    """
    Select judges using softmax weighting.
    
    Judges provide rankings of all responses.
    Higher-rated judges get more weight in aggregation.
    """
    
    def __init__(self, tau: float = None, elite_threshold: float = None, domain: str = None):
        """
        Initialize judge selector.
        
        Args:
            tau: Softmax temperature
            elite_threshold: Minimum μ for judges
            domain: Optional domain for domain-specific params
        """
        super().__init__(tau=tau, elite_threshold=elite_threshold, domain=domain)
    
    def select_judges(
        self,
        models: List['LLMModel'],
        domain: str,
        n_judges: int = 5,
        exclude: List[str] = None,
        min_stratum: int = 2
    ) -> List['LLMModel']:
        """
        Select n judges independently.
        
        Args:
            models: List of all models
            domain: Domain for evaluation
            n_judges: Number of judges to select
            exclude: Model names/IDs to exclude
            min_stratum: Minimum stratum for judges
        
        Returns:
            List of selected judge models
        """
        exclude = exclude or []
        judges = []
        current_exclude = list(exclude)
        
        for _ in range(n_judges):
            judge = self.select(models, domain, current_exclude, min_stratum)
            if judge:
                judges.append(judge)
                current_exclude.append(judge.name)
                current_exclude.append(judge.model_id)
        
        return judges
    
    def get_judge_weights(
        self,
        judges: List['LLMModel'],
        domain: str
    ) -> List[float]:
        """
        Get softmax weights for judges based on their TrueSkill μ.
        
        w_k = e^(μ_k/τ) / Σ_j e^(μ_j/τ)
        
        Args:
            judges: List of judge models
            domain: Domain for evaluation
        
        Returns:
            List of weights (sums to 1)
        """
        if not judges:
            return []
        
        mus = [self._get_mu(j, domain) for j in judges]
        return self._softmax(mus)


def select_match_participants(
    models: List['LLMModel'],
    domain: str,
    n_players: int = None,
    n_judges: int = None,
    recent_groups: Set[frozenset] = None
) -> Optional[Dict]:
    """
    Convenience function to select all match participants.
    
    Args:
        models: List of all models
        domain: Domain for evaluation
        n_players: Number of competing players (from params if None)
        n_judges: Number of judges (from params if None)
        recent_groups: Recent player groups to avoid
    
    Returns:
        Dictionary with players, judges, case_creator, question_creator
        or None if selection failed
    """
    # Get params for defaults
    match_params = _get_match_params(domain)
    
    if n_players is None:
        n_players = match_params.players_per_match
    if n_judges is None:
        n_judges = match_params.judge_count
    
    pairing = MultiPlayerSwissPairing(domain=domain)
    creator_selector = ContentCreatorSelector(domain=domain)
    judge_selector = JudgeSelector(domain=domain)
    
    # Select players
    players = pairing.select_match_players(models, domain, recent_groups)
    if not players:
        return None
    
    # Track exclusions
    exclude = [p.name for p in players] + [p.model_id for p in players]
    
    # Select case creator
    case_creator = creator_selector.select(models, domain, exclude)
    if not case_creator:
        return None
    exclude.extend([case_creator.name, case_creator.model_id])
    
    # Select question creator
    question_creator = creator_selector.select(models, domain, exclude)
    if not question_creator:
        return None
    exclude.extend([question_creator.name, question_creator.model_id])
    
    # Select judges
    judges = judge_selector.select_judges(models, domain, n_judges, exclude)
    if len(judges) < 1:
        return None
    
    return {
        "players": players,
        "case_creator": case_creator,
        "question_creator": question_creator,
        "judges": judges,
        "judge_weights": judge_selector.get_judge_weights(judges, domain)
    }


