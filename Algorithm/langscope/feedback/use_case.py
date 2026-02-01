"""
Use-case specific ranking adjustments learned from user feedback.

This module enables personalized recommendations by learning from
aggregate user feedback patterns. Users with similar use cases
(e.g., "patient education" vs "clinical diagnosis") have different
preferences that can be learned over time.

Key Formula:
    μᵢ^use-case = μᵢ^global + β_use-case × Δ̄ᵢ^use-case

Where:
    - Δ̄ᵢ^use-case = average delta for model i across users with this use case
    - β_use-case = n_users / (n_users + τ_use-case) = smoothing factor

The smoothing factor (β) increases with more user data:
    - 1 user: β = 1/11 ≈ 0.09 (weak influence)
    - 10 users: β = 10/20 = 0.5 (moderate influence)
    - 50 users: β = 50/60 ≈ 0.83 (strong influence)
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import math

# Default smoothing parameter
USE_CASE_TAU: float = 10.0


@dataclass
class UseCaseProfile:
    """
    Profile of user feedback for a specific use case.
    
    Tracks aggregate delta statistics to learn use-case specific
    model preferences.
    """
    use_case: str
    n_users: int = 0
    model_deltas: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_feedback(self, deltas: Dict[str, float]):
        """Add delta values from a user session."""
        self.n_users += 1
        for model_id, delta in deltas.items():
            self.model_deltas[model_id].append(delta)
    
    def get_average_delta(self, model_id: str) -> float:
        """Get average delta for a model in this use case."""
        if model_id not in self.model_deltas:
            return 0.0
        deltas = self.model_deltas[model_id]
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)
    
    def get_beta(self, tau: float = USE_CASE_TAU) -> float:
        """Get current smoothing factor."""
        return self.n_users / (self.n_users + tau)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "use_case": self.use_case,
            "n_users": self.n_users,
            "model_deltas": dict(self.model_deltas),
            "beta": self.get_beta()
        }


class UseCaseAdjustmentManager:
    """
    Learn and apply use-case specific adjustments from aggregate user feedback.
    
    When users specify their use case and provide feedback, we track the
    delta patterns to learn which models perform better for specific needs.
    
    Formula:
        μᵢ^use-case = μᵢ^global + β_use-case × Δ̄ᵢ^use-case
    
    Where:
        - Δ̄ᵢ^use-case = average delta for model i across users with this use case
        - β_use-case = n_users / (n_users + τ_use-case), smoothing factor
    
    Example:
        >>> manager = UseCaseAdjustmentManager(tau_use_case=10.0)
        >>> manager.add_user_feedback("patient_education", {"llama": +50, "gpt-4": -50})
        >>> manager.add_user_feedback("patient_education", {"llama": +60, "gpt-4": -60})
        >>> # Now Llama gets boosted for patient education use case
        >>> adjusted_mu = manager.get_adjusted_mu("patient_education", "llama", 1500)
        >>> print(adjusted_mu)
        1509.17  # 1500 + (2/12) * 55
    """
    
    def __init__(self, tau_use_case: float = USE_CASE_TAU):
        """
        Initialize the manager.
        
        Args:
            tau_use_case: Smoothing parameter (higher = slower adaptation)
        """
        self.tau = tau_use_case
        self.profiles: Dict[str, UseCaseProfile] = {}
    
    def _get_or_create_profile(self, use_case: str) -> UseCaseProfile:
        """Get or create a use case profile."""
        if use_case not in self.profiles:
            self.profiles[use_case] = UseCaseProfile(use_case=use_case)
        return self.profiles[use_case]
    
    def add_user_feedback(
        self,
        use_case: str,
        deltas: Dict[str, float]
    ):
        """
        Record deltas from a user session.
        
        Args:
            use_case: The user's use case category
            deltas: {model_id: delta_value} from the session
        """
        profile = self._get_or_create_profile(use_case)
        profile.add_feedback(deltas)
    
    def get_adjustment(
        self,
        use_case: str,
        model_id: str
    ) -> float:
        """
        Get use-case specific adjustment for a model.
        
        This is the value to ADD to the global μ to get the
        use-case specific rating.
        
        Formula:
            adjustment = β × Δ̄
        
        Args:
            use_case: The use case category
            model_id: The model to get adjustment for
        
        Returns:
            Adjustment value (positive = boost, negative = penalty)
        """
        if use_case not in self.profiles:
            return 0.0
        
        profile = self.profiles[use_case]
        avg_delta = profile.get_average_delta(model_id)
        beta = profile.get_beta(self.tau)
        
        return beta * avg_delta
    
    def get_adjusted_mu(
        self,
        use_case: str,
        model_id: str,
        global_mu: float
    ) -> float:
        """
        Get use-case adjusted μ for a model.
        
        Formula:
            μᵢ^use-case = μᵢ^global + β × Δ̄ᵢ^use-case
        
        Args:
            use_case: The use case category
            model_id: The model identifier
            global_mu: The model's global mean rating
        
        Returns:
            Adjusted mean rating for this use case
        """
        adjustment = self.get_adjustment(use_case, model_id)
        return global_mu + adjustment
    
    def get_beta(self, use_case: str) -> float:
        """
        Get current smoothing factor for a use case.
        
        The beta value indicates how much weight we give to
        user feedback vs. global ratings:
            - β ≈ 0: Little user data, stick to global
            - β ≈ 1: Lots of user data, trust use-case specific
        
        Args:
            use_case: The use case category
        
        Returns:
            Smoothing factor β ∈ [0, 1]
        """
        if use_case not in self.profiles:
            return 0.0
        return self.profiles[use_case].get_beta(self.tau)
    
    def get_user_count(self, use_case: str) -> int:
        """Get number of users who have provided feedback for this use case."""
        if use_case not in self.profiles:
            return 0
        return self.profiles[use_case].n_users
    
    def get_adjusted_ranking(
        self,
        use_case: str,
        models: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Get a ranked list of models adjusted for use case.
        
        Args:
            use_case: The use case category
            models: {model_id: global_mu}
        
        Returns:
            List of (model_id, adjusted_mu) sorted by adjusted_mu descending
        """
        adjusted = []
        for model_id, global_mu in models.items():
            adjusted_mu = self.get_adjusted_mu(use_case, model_id, global_mu)
            adjusted.append((model_id, adjusted_mu))
        
        return sorted(adjusted, key=lambda x: -x[1])
    
    def list_use_cases(self) -> List[str]:
        """List all known use cases."""
        return list(self.profiles.keys())
    
    def get_profile_summary(self, use_case: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a use case.
        
        Args:
            use_case: The use case category
        
        Returns:
            Summary dictionary or None if use case not found
        """
        if use_case not in self.profiles:
            return None
        
        profile = self.profiles[use_case]
        
        # Find top boosted and penalized models
        adjustments = []
        for model_id in profile.model_deltas:
            adj = self.get_adjustment(use_case, model_id)
            adjustments.append((model_id, adj))
        
        adjustments.sort(key=lambda x: -x[1])
        
        return {
            "use_case": use_case,
            "n_users": profile.n_users,
            "beta": profile.get_beta(self.tau),
            "n_models": len(profile.model_deltas),
            "top_boosted": adjustments[:3] if adjustments else [],
            "top_penalized": adjustments[-3:] if adjustments else []
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manager state to dictionary for serialization."""
        return {
            "tau": self.tau,
            "profiles": {
                use_case: profile.to_dict()
                for use_case, profile in self.profiles.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UseCaseAdjustmentManager':
        """Create manager from dictionary."""
        manager = cls(tau_use_case=data.get("tau", USE_CASE_TAU))
        
        for use_case, profile_data in data.get("profiles", {}).items():
            profile = UseCaseProfile(use_case=use_case)
            profile.n_users = profile_data.get("n_users", 0)
            profile.model_deltas = defaultdict(
                list,
                {k: list(v) for k, v in profile_data.get("model_deltas", {}).items()}
            )
            manager.profiles[use_case] = profile
        
        return manager
