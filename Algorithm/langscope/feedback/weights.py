"""
User feedback weighting for TrueSkill updates.

User feedback receives elevated weight compared to LLM judges because:
1. Users have domain-specific knowledge (e.g., medical requirements)
2. Users have deployment context awareness (target audience, use case)
3. Users represent ground truth for their specific needs
4. Users can apply implicit criteria not captured in LLM judge prompts

Key Formula:
    w_user = α_u × w_base
    
Where:
    α_u ∈ [1, 3] = user credibility multiplier (default = 2)
    w_base = base weight for a top-quartile LLM judge
"""

import math
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from langscope.config.params import FeedbackParams

# Legacy constants (for backward compatibility)
USER_WEIGHT_MULTIPLIER: float = 2.0  # α_u: user weight multiplier
USER_WEIGHT_BASE: float = 1.0        # Base weight for top LLM judge


def _get_feedback_params() -> 'FeedbackParams':
    """Get feedback params from ParameterManager or use defaults."""
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_feedback_params()
    except ImportError:
        from langscope.config.params.models import FeedbackParams
        return FeedbackParams()


def get_user_feedback_weight(
    alpha_u: float = None,
    base_weight: float = None
) -> float:
    """
    Calculate user feedback weight.
    
    The user weight multiplies the base weight to give user judgments
    more influence on rating updates.
    
    Formula:
        w_user = α_u × w_base
    
    Args:
        alpha_u: User credibility multiplier (default from ParameterManager)
        base_weight: Base weight for comparison (default from ParameterManager)
    
    Returns:
        User feedback weight
    
    Example:
        >>> get_user_feedback_weight()
        2.0
        >>> get_user_feedback_weight(alpha_u=3.0)
        3.0
    """
    params = _get_feedback_params()
    if alpha_u is None:
        alpha_u = params.user_weight_multiplier
    if base_weight is None:
        base_weight = params.user_weight_base
    return alpha_u * base_weight


def apply_user_weighted_trueskill_update(
    mu_pred: float,
    sigma_pred: float,
    update_v: float,
    update_w: float,
    c: float,
    user_weight: float = None
) -> Tuple[float, float]:
    """
    Apply TrueSkill update with user weight.
    
    Modifies the standard TrueSkill update equations to incorporate
    the elevated user weight, resulting in larger rating changes
    and faster uncertainty reduction.
    
    Formulas:
        μ_post = μ_pred + w_user × (σ²/c) × v
        σ_post = σ_pred × √(1 - w_user × (σ²/c²) × w)
    
    Args:
        mu_pred: Pre-update mean rating
        sigma_pred: Pre-update uncertainty
        update_v: TrueSkill v factor (mean update direction)
        update_w: TrueSkill w factor (variance reduction)
        c: Combined performance variance √(2β² + σ₁² + σ₂²)
        user_weight: User weight (default: get_user_feedback_weight())
    
    Returns:
        Tuple of (mu_post, sigma_post)
    
    Note:
        With user_weight=2, rating changes are doubled and uncertainty
        is reduced more aggressively compared to LLM judge feedback.
    """
    if user_weight is None:
        user_weight = get_user_feedback_weight()
    
    # Precompute values
    sigma_sq = sigma_pred ** 2
    c_sq = c ** 2
    
    # Mean update (scaled by user weight)
    mu_post = mu_pred + user_weight * (sigma_sq / c) * update_v
    
    # Variance update factor
    # Clamp to non-negative to handle edge cases
    sigma_factor = 1 - user_weight * (sigma_sq / c_sq) * update_w
    sigma_factor = max(0, sigma_factor)
    
    # Sigma update
    sigma_post = sigma_pred * math.sqrt(sigma_factor)
    
    return mu_post, sigma_post


def scale_rating_delta(
    base_delta: float,
    user_weight: float = None
) -> float:
    """
    Scale a rating delta by user weight.
    
    This is a simplified method when you already have the base
    rating change and just want to scale it.
    
    Args:
        base_delta: Base rating change (from standard TrueSkill)
        user_weight: User weight multiplier
    
    Returns:
        Scaled rating delta
    """
    if user_weight is None:
        user_weight = get_user_feedback_weight()
    
    return base_delta * user_weight


def compute_effective_match_count(
    n_battles: int,
    user_weight: float = None
) -> float:
    """
    Compute effective match count accounting for user weight.
    
    User battles count for more than automated battles due to
    the elevated weight, accelerating convergence.
    
    Args:
        n_battles: Number of user battles
        user_weight: User weight multiplier
    
    Returns:
        Effective match count (equivalent automated battles)
    
    Example:
        >>> compute_effective_match_count(10)
        20.0  # 10 user battles = 20 equivalent automated battles
    """
    if user_weight is None:
        user_weight = get_user_feedback_weight()
    
    return n_battles * user_weight
