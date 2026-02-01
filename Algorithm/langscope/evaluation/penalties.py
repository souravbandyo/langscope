"""
Penalty system for LangScope.

Applies penalties for various infractions:
- Invalid ranking format
- Outlier/inconsistent judging
- Poor content quality
- Gaming attempts
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import constants as fallback defaults
from langscope.core.constants import (
    JUDGE_PENALTY_MU,
    OUTLIER_DISAGREEMENT_THRESHOLD,
)

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.config.params import PenaltyParams


def _get_penalty_params(domain: str = None) -> 'PenaltyParams':
    """Get penalty params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_penalty_params(domain)
    except ImportError:
        from langscope.config.params.models import PenaltyParams
        return PenaltyParams()


class PenaltyType(Enum):
    """Types of penalties."""
    INVALID_FORMAT = "invalid_format"
    OUTLIER_JUDGE = "outlier_judge"
    POOR_CONTENT = "poor_content"
    INCONSISTENT = "inconsistent"
    GAMING_ATTEMPT = "gaming_attempt"


@dataclass
class Penalty:
    """A penalty applied to a model."""
    penalty_id: str
    model_id: str
    penalty_type: PenaltyType
    domain: str
    mu_penalty: float  # Amount subtracted from μ
    description: str
    timestamp: str = ""
    match_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "penalty_id": self.penalty_id,
            "model_id": self.model_id,
            "penalty_type": self.penalty_type.value,
            "domain": self.domain,
            "mu_penalty": self.mu_penalty,
            "description": self.description,
            "timestamp": self.timestamp,
            "match_id": self.match_id,
            "metadata": self.metadata,
        }


class PenaltySystem:
    """
    Manages penalties for models.
    
    Tracks penalty history and applies penalties to model ratings.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize penalty system.
        
        Args:
            domain: Optional domain for domain-specific penalty params
        """
        self.penalties: List[Penalty] = []
        self.domain = domain
        
        # Get penalty params
        params = _get_penalty_params(domain)
        
        # Penalty amounts by type
        self.penalty_amounts = {
            PenaltyType.INVALID_FORMAT: 5.0,
            PenaltyType.OUTLIER_JUDGE: params.judge_penalty_mu,
            PenaltyType.POOR_CONTENT: params.content_rejection_penalty,
            PenaltyType.INCONSISTENT: 3.0,
            PenaltyType.GAMING_ATTEMPT: 20.0,
        }
    
    def apply_penalty(
        self,
        model: 'LLMModel',
        penalty_type: PenaltyType,
        domain: str,
        description: str = "",
        match_id: str = "",
        custom_amount: float = None
    ) -> Penalty:
        """
        Apply a penalty to a model.
        
        Args:
            model: Model to penalize
            penalty_type: Type of penalty
            domain: Domain where penalty applies
            description: Penalty description
            match_id: Related match ID
            custom_amount: Custom penalty amount (overrides default)
        
        Returns:
            Applied Penalty
        """
        import uuid
        
        # Get penalty amount
        amount = custom_amount if custom_amount is not None else self.penalty_amounts.get(
            penalty_type, 5.0
        )
        
        # Create penalty record
        penalty = Penalty(
            penalty_id=f"penalty_{uuid.uuid4().hex[:12]}",
            model_id=model.model_id,
            penalty_type=penalty_type,
            domain=domain,
            mu_penalty=amount,
            description=description or f"Penalty for {penalty_type.value}",
            match_id=match_id,
        )
        
        # Apply to model rating
        if domain in model.trueskill_by_domain:
            model.trueskill_by_domain[domain].raw.mu -= amount
            model.trueskill_by_domain[domain].cost_adjusted.mu -= amount
        else:
            model.trueskill.raw.mu -= amount
            model.trueskill.cost_adjusted.mu -= amount
        
        # Record penalty
        self.penalties.append(penalty)
        
        return penalty
    
    def get_model_penalties(
        self,
        model_id: str,
        domain: str = None
    ) -> List[Penalty]:
        """Get penalties for a model."""
        penalties = [p for p in self.penalties if p.model_id == model_id]
        if domain:
            penalties = [p for p in penalties if p.domain == domain]
        return penalties
    
    def get_total_penalty(
        self,
        model_id: str,
        domain: str = None
    ) -> float:
        """Get total penalty amount for a model."""
        penalties = self.get_model_penalties(model_id, domain)
        return sum(p.mu_penalty for p in penalties)
    
    def check_gaming_attempt(
        self,
        model: 'LLMModel',
        rankings: List[Dict[str, int]],
        consensus: Dict[str, int],
        match_id: str,
        domain: str
    ) -> Optional[Penalty]:
        """
        Check for gaming attempts (consistently ranking self high).
        
        This is only applicable if the model serves as both judge and competitor,
        which is generally avoided but checked for safety.
        
        Args:
            model: Model to check
            rankings: Rankings provided by this model as judge
            consensus: Consensus ranking
            match_id: Match ID
            domain: Domain
        
        Returns:
            Penalty if gaming detected, None otherwise
        """
        # Check if model consistently ranks itself higher than consensus
        self_improvements = 0
        total_rankings = 0
        
        for ranking in rankings:
            if model.model_id in ranking and model.model_id in consensus:
                total_rankings += 1
                if ranking[model.model_id] < consensus[model.model_id]:
                    self_improvements += 1
        
        # If consistently self-promoting (>80% of cases)
        if total_rankings > 2 and self_improvements / total_rankings > 0.8:
            return self.apply_penalty(
                model,
                PenaltyType.GAMING_ATTEMPT,
                domain,
                "Detected self-promotion in judging",
                match_id,
            )
        
        return None


def apply_judge_penalty(
    model: 'LLMModel',
    domain: str,
    reason: str = "Outlier ranking",
    match_id: str = "",
    penalty_amount: float = JUDGE_PENALTY_MU
):
    """
    Convenience function to apply judge penalty.
    
    Args:
        model: Model to penalize
        domain: Domain
        reason: Penalty reason
        match_id: Related match ID
        penalty_amount: Amount to subtract from μ
    """
    # Direct penalty application without PenaltySystem
    if domain in model.trueskill_by_domain:
        model.trueskill_by_domain[domain].raw.mu -= penalty_amount
        model.trueskill_by_domain[domain].cost_adjusted.mu -= penalty_amount
    else:
        model.trueskill.raw.mu -= penalty_amount
        model.trueskill.cost_adjusted.mu -= penalty_amount


def apply_content_penalty(
    model: 'LLMModel',
    domain: str,
    reason: str = "Poor content quality",
    penalty_amount: float = 5.0
):
    """
    Convenience function to apply content quality penalty.
    
    Args:
        model: Model to penalize
        domain: Domain
        reason: Penalty reason
        penalty_amount: Amount to subtract from μ
    """
    if domain in model.trueskill_by_domain:
        model.trueskill_by_domain[domain].raw.mu -= penalty_amount
        model.trueskill_by_domain[domain].cost_adjusted.mu -= penalty_amount
    else:
        model.trueskill.raw.mu -= penalty_amount
        model.trueskill.cost_adjusted.mu -= penalty_amount


def detect_inconsistent_judge(
    judge_rankings: List[Dict[str, int]],
    consensus: Dict[str, int],
    threshold: float = None,
    domain: str = None
) -> List[int]:
    """
    Detect judges with inconsistent rankings.
    
    Args:
        judge_rankings: List of judge rankings
        consensus: Consensus ranking
        threshold: Disagreement threshold (from ParameterManager if None)
        domain: Optional domain for domain-specific threshold
    
    Returns:
        Indices of inconsistent judges
    """
    from langscope.ranking.cost_adjustment import ranking_distance
    
    if threshold is None:
        params = _get_penalty_params(domain)
        threshold = params.outlier_disagreement_threshold
    
    inconsistent = []
    
    for i, ranking in enumerate(judge_rankings):
        distance = ranking_distance(ranking, consensus, method="kendall")
        if distance > threshold:
            inconsistent.append(i)
    
    return inconsistent


def compute_judge_reliability(
    judge_rankings: List[Dict[str, int]],
    consensus: Dict[str, int]
) -> List[float]:
    """
    Compute reliability score for each judge.
    
    Higher score = more agreement with consensus.
    
    Args:
        judge_rankings: List of judge rankings
        consensus: Consensus ranking
    
    Returns:
        List of reliability scores (0 to 1)
    """
    from langscope.ranking.cost_adjustment import ranking_distance
    
    reliabilities = []
    
    for ranking in judge_rankings:
        distance = ranking_distance(ranking, consensus, method="kendall")
        # Convert distance to reliability (1 - distance)
        reliability = max(0, 1 - distance)
        reliabilities.append(reliability)
    
    return reliabilities


