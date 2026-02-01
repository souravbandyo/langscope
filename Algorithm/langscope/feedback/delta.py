"""
Prediction-feedback delta tracking with zero-sum conservation.

This module implements the core delta calculations that compare
system predictions with post-feedback ratings. The key invariant
is the conservation law: the sum of all deltas must equal zero.

Key Formulas:
    Δᵢ = μᵢ^post - μᵢ^pred  (individual delta)
    Σᵢ Δᵢ = 0               (conservation law)
    zᵢ = Δᵢ / √(σ_pred² + σ_post²)  (user surprise score)
"""

from typing import Dict, List, Tuple
import math

from langscope.feedback.user_feedback import FeedbackDelta


class ZeroSumViolationError(Exception):
    """Raised when the zero-sum conservation law is violated."""
    
    def __init__(self, delta_sum: float, tolerance: float = 1e-6):
        self.delta_sum = delta_sum
        self.tolerance = tolerance
        super().__init__(
            f"Conservation violation: Σ Δᵢ = {delta_sum:.10f} ≠ 0. "
            f"Rating updates must be zero-sum (tolerance: {tolerance})."
        )


def compute_session_deltas(
    predictions: Dict[str, Tuple[float, float]],
    post_ratings: Dict[str, Tuple[float, float]],
    validate_conservation: bool = True,
    tolerance: float = 1e-6
) -> Tuple[Dict[str, FeedbackDelta], float]:
    """
    Compute all deltas for a user session.
    
    For each model that appears in both predictions and post_ratings,
    computes the FeedbackDelta including the z-score (user surprise).
    
    INVARIANT: Σᵢ Δᵢ = 0 (conservation law)
    
    This invariant ensures that rating points are conserved - 
    what some models gain, others must lose. This is a fundamental
    property of the TrueSkill update rules.
    
    Args:
        predictions: {model_id: (mu_pred, sigma_pred)}
        post_ratings: {model_id: (mu_post, sigma_post)}
        validate_conservation: If True, raise error on violation
        tolerance: Maximum allowed deviation from zero-sum
    
    Returns:
        Tuple of:
        - deltas: {model_id: FeedbackDelta}
        - delta_sum: Sum of all deltas (should be ~0)
    
    Raises:
        ZeroSumViolationError: If conservation violated and validation enabled
    
    Example:
        >>> predictions = {"gpt-4": (1600, 50), "claude": (1580, 55)}
        >>> post_ratings = {"gpt-4": (1620, 48), "claude": (1560, 52)}
        >>> deltas, delta_sum = compute_session_deltas(predictions, post_ratings)
        >>> print(delta_sum)
        0.0  # Conservation satisfied: +20 + (-20) = 0
    """
    deltas = {}
    delta_sum = 0.0
    
    for model_id in predictions:
        if model_id in post_ratings:
            mu_pred, sigma_pred = predictions[model_id]
            mu_post, sigma_post = post_ratings[model_id]
            
            delta = FeedbackDelta.compute(
                model_id=model_id,
                mu_pred=mu_pred,
                sigma_pred=sigma_pred,
                mu_post=mu_post,
                sigma_post=sigma_post
            )
            deltas[model_id] = delta
            delta_sum += delta.delta
    
    # Validate conservation law
    if validate_conservation and abs(delta_sum) > tolerance:
        raise ZeroSumViolationError(delta_sum, tolerance)
    
    return deltas, delta_sum


def validate_zero_sum(
    deltas: Dict[str, float],
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that deltas satisfy zero-sum conservation.
    
    Args:
        deltas: {model_id: delta_value}
        tolerance: Maximum allowed deviation from zero
    
    Returns:
        True if conservation is satisfied
    
    Raises:
        ZeroSumViolationError: If conservation is violated
    """
    delta_sum = sum(deltas.values())
    
    if abs(delta_sum) > tolerance:
        raise ZeroSumViolationError(delta_sum, tolerance)
    
    return True


def compute_user_surprise_score(delta: FeedbackDelta) -> str:
    """
    Interpret z-score for user surprise level.
    
    The z-score measures how surprising the user's feedback was
    relative to the system's prediction, normalized by uncertainty.
    
    Interpretation:
        |z| < 1:     User feedback aligns with system prediction
        1 ≤ |z| < 2: Moderate surprise (user has different preferences)
        |z| ≥ 2:     Significant surprise (potential specialist or miscalibration)
    
    Args:
        delta: FeedbackDelta with computed z-score
    
    Returns:
        String description of surprise level:
        - "aligned": |z| < 1
        - "moderate_surprise": 1 ≤ |z| < 2
        - "significant_surprise": |z| ≥ 2
    
    Example:
        >>> delta = FeedbackDelta.compute("model", 1500, 50, 1580, 45)
        >>> compute_user_surprise_score(delta)
        'moderate_surprise'  # Large positive change
    """
    z = abs(delta.z_score)
    
    if z < 1.0:
        return "aligned"
    elif z < 2.0:
        return "moderate_surprise"
    else:
        return "significant_surprise"


def detect_user_specialists(
    deltas: Dict[str, FeedbackDelta],
    z_threshold: float = 2.0
) -> List[str]:
    """
    Identify models with significant positive surprise.
    
    A model is considered a potential specialist for the user's use case
    if it received a significantly higher rating than predicted (z > threshold).
    
    This indicates the user values something about this model that the
    automated evaluation system didn't fully capture.
    
    Args:
        deltas: {model_id: FeedbackDelta}
        z_threshold: Z-score threshold for specialist detection (default: 2.0)
    
    Returns:
        List of model IDs that are potential specialists
    
    Example:
        >>> # Llama unexpectedly outperformed predictions
        >>> specialists = detect_user_specialists(deltas, z_threshold=2.0)
        >>> print(specialists)
        ['llama-3-70b']  # User values Llama's specific qualities
    """
    specialists = []
    
    for model_id, delta in deltas.items():
        if delta.z_score > z_threshold:
            specialists.append(model_id)
    
    return specialists


def detect_user_underperformers(
    deltas: Dict[str, FeedbackDelta],
    z_threshold: float = -2.0
) -> List[str]:
    """
    Identify models with significant negative surprise.
    
    These are models that performed worse than the system predicted,
    which may indicate the user has criteria not captured by automated evaluation.
    
    Args:
        deltas: {model_id: FeedbackDelta}
        z_threshold: Z-score threshold (default: -2.0, should be negative)
    
    Returns:
        List of model IDs that underperformed relative to predictions
    """
    underperformers = []
    
    for model_id, delta in deltas.items():
        if delta.z_score < z_threshold:
            underperformers.append(model_id)
    
    return underperformers


def summarize_session_deltas(
    deltas: Dict[str, FeedbackDelta]
) -> Dict[str, any]:
    """
    Create a summary of delta statistics for a session.
    
    Args:
        deltas: {model_id: FeedbackDelta}
    
    Returns:
        Summary dictionary with statistics
    """
    if not deltas:
        return {
            "n_models": 0,
            "delta_sum": 0.0,
            "max_delta": 0.0,
            "min_delta": 0.0,
            "max_z_score": 0.0,
            "min_z_score": 0.0,
            "n_specialists": 0,
            "n_underperformers": 0,
            "conservation_satisfied": True
        }
    
    delta_values = [d.delta for d in deltas.values()]
    z_scores = [d.z_score for d in deltas.values()]
    
    delta_sum = sum(delta_values)
    specialists = detect_user_specialists(deltas)
    underperformers = detect_user_underperformers(deltas)
    
    return {
        "n_models": len(deltas),
        "delta_sum": delta_sum,
        "max_delta": max(delta_values),
        "min_delta": min(delta_values),
        "max_z_score": max(z_scores),
        "min_z_score": min(z_scores),
        "n_specialists": len(specialists),
        "n_underperformers": len(underperformers),
        "conservation_satisfied": abs(delta_sum) < 1e-6
    }
