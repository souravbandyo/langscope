"""
User feedback integration with prediction-feedback delta tracking.

This module provides dataclasses for tracking:
- Pre-testing predictions (PredictionState)
- Post-feedback deltas (FeedbackDelta)
- Complete user sessions (UserSession)

The key insight is that we can learn from the difference between what
the system predicted and how the user actually ranked the models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import math


@dataclass
class PredictionState:
    """
    Snapshot of a model's TrueSkill rating before user testing.
    
    Captures the system's prediction for a model's performance,
    which will be compared against the post-feedback rating.
    
    Attributes:
        model_id: Unique identifier for the model
        mu_pred: Mean skill estimate (μ) before user feedback
        sigma_pred: Uncertainty (σ) before user feedback
        timestamp: When the prediction was recorded
    """
    model_id: str
    mu_pred: float
    sigma_pred: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "mu_pred": self.mu_pred,
            "sigma_pred": self.sigma_pred,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionState':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            mu_pred=float(data["mu_pred"]),
            sigma_pred=float(data["sigma_pred"]),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        )
    
    def __repr__(self) -> str:
        return f"PredictionState({self.model_id}, μ={self.mu_pred:.1f}, σ={self.sigma_pred:.1f})"


@dataclass
class FeedbackDelta:
    """
    Delta between prediction and post-feedback rating.
    
    This captures how much the user's feedback changed the model's rating,
    enabling us to learn from discrepancies between automated and user evaluation.
    
    Key Formula:
        Δᵢ = μᵢ^post - μᵢ^pred
        
    The z-score normalizes this to account for uncertainty:
        zᵢ = Δᵢ / √(σ_pred² + σ_post²)
    
    Attributes:
        model_id: Unique identifier for the model
        mu_pred: Mean skill estimate before feedback
        mu_post: Mean skill estimate after feedback
        sigma_pred: Uncertainty before feedback
        sigma_post: Uncertainty after feedback
        delta: Rating change (μ_post - μ_pred)
        z_score: Normalized delta (user surprise score)
    """
    model_id: str
    mu_pred: float
    mu_post: float
    sigma_pred: float
    sigma_post: float
    delta: float
    z_score: float
    
    @classmethod
    def compute(
        cls,
        model_id: str,
        mu_pred: float,
        sigma_pred: float,
        mu_post: float,
        sigma_post: float
    ) -> 'FeedbackDelta':
        """
        Compute delta and z-score from pre/post ratings.
        
        Args:
            model_id: Model identifier
            mu_pred: Pre-feedback mean
            sigma_pred: Pre-feedback uncertainty
            mu_post: Post-feedback mean
            sigma_post: Post-feedback uncertainty
        
        Returns:
            FeedbackDelta instance with computed values
        """
        delta = mu_post - mu_pred
        
        # Z-score denominator: combined uncertainty
        denominator = math.sqrt(sigma_pred**2 + sigma_post**2)
        z_score = delta / denominator if denominator > 0 else 0.0
        
        return cls(
            model_id=model_id,
            mu_pred=mu_pred,
            mu_post=mu_post,
            sigma_pred=sigma_pred,
            sigma_post=sigma_post,
            delta=delta,
            z_score=z_score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "mu_pred": self.mu_pred,
            "mu_post": self.mu_post,
            "sigma_pred": self.sigma_pred,
            "sigma_post": self.sigma_post,
            "delta": self.delta,
            "z_score": self.z_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackDelta':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            mu_pred=float(data["mu_pred"]),
            mu_post=float(data["mu_post"]),
            sigma_pred=float(data["sigma_pred"]),
            sigma_post=float(data["sigma_post"]),
            delta=float(data["delta"]),
            z_score=float(data["z_score"])
        )
    
    def __repr__(self) -> str:
        return f"FeedbackDelta({self.model_id}, Δ={self.delta:+.1f}, z={self.z_score:+.2f})"


@dataclass
class UserSession:
    """
    Complete user feedback session tracking arena testing.
    
    A session captures everything about a user's evaluation experience:
    - Which models they tested
    - What the system predicted
    - How the ratings changed
    - Whether predictions were accurate
    
    Key Invariant:
        Σᵢ Δᵢ = 0 (delta_sum must equal zero)
        
    This zero-sum property ensures rating points are conserved -
    what some models gain, others must lose.
    
    Attributes:
        session_id: Unique session identifier
        user_id: Optional user identifier (for personalization)
        domain: Evaluation domain
        use_case: User's specific use case (e.g., "patient_education")
        models_tested: List of model IDs evaluated
        n_battles: Number of battles/comparisons conducted
        predictions: Pre-testing predictions for each model
        deltas: Post-testing deltas for each model
        delta_sum: Sum of all deltas (must be ~0)
        prediction_accuracy: Pairwise ordering accuracy
        kendall_tau: Kendall's tau correlation coefficient
        timestamp_start: Session start time
        timestamp_end: Session end time
    """
    session_id: str
    user_id: Optional[str]
    domain: str
    use_case: str
    models_tested: List[str]
    n_battles: int
    predictions: Dict[str, PredictionState] = field(default_factory=dict)
    deltas: Dict[str, FeedbackDelta] = field(default_factory=dict)
    delta_sum: float = 0.0
    prediction_accuracy: float = 0.0
    kendall_tau: float = 0.0
    timestamp_start: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    timestamp_end: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "_id": self.session_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "domain": self.domain,
            "use_case": self.use_case,
            "models_tested": self.models_tested,
            "n_battles": self.n_battles,
            "predictions": {
                model_id: pred.to_dict()
                for model_id, pred in self.predictions.items()
            },
            "deltas": {
                model_id: delta.to_dict()
                for model_id, delta in self.deltas.items()
            },
            "delta_sum": self.delta_sum,
            "prediction_accuracy": self.prediction_accuracy,
            "kendall_tau": self.kendall_tau,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create from dictionary."""
        predictions = {
            model_id: PredictionState.from_dict(pred_data)
            for model_id, pred_data in data.get("predictions", {}).items()
        }
        deltas = {
            model_id: FeedbackDelta.from_dict(delta_data)
            for model_id, delta_data in data.get("deltas", {}).items()
        }
        
        return cls(
            session_id=data.get("session_id", data.get("_id", "")),
            user_id=data.get("user_id"),
            domain=data["domain"],
            use_case=data["use_case"],
            models_tested=data.get("models_tested", []),
            n_battles=data.get("n_battles", 0),
            predictions=predictions,
            deltas=deltas,
            delta_sum=float(data.get("delta_sum", 0.0)),
            prediction_accuracy=float(data.get("prediction_accuracy", 0.0)),
            kendall_tau=float(data.get("kendall_tau", 0.0)),
            timestamp_start=data.get("timestamp_start", ""),
            timestamp_end=data.get("timestamp_end", "")
        )
    
    def is_conservation_satisfied(self, tolerance: float = 1e-6) -> bool:
        """Check if zero-sum conservation is satisfied."""
        return abs(self.delta_sum) < tolerance
    
    def __repr__(self) -> str:
        status = "✓" if self.is_conservation_satisfied() else "✗"
        return (
            f"UserSession({self.session_id}, domain={self.domain}, "
            f"battles={self.n_battles}, Σ Δ={self.delta_sum:.6f} {status})"
        )


# MongoDB schema for user sessions
USER_SESSION_SCHEMA: Dict[str, Any] = {
    "_id": str,                    # session_<uuid>
    "session_id": str,             # Same as _id
    "user_id": str,                # Optional user identifier
    "domain": str,                 # Domain evaluated
    "use_case": str,               # Use case category
    "models_tested": List[str],    # Model IDs tested
    "n_battles": int,              # Number of battles
    
    # Pre-testing predictions
    "predictions": {
        # "<model_id>": {
        #     "model_id": str,
        #     "mu_pred": float,
        #     "sigma_pred": float,
        #     "timestamp": str
        # }
    },
    
    # Post-testing deltas
    "deltas": {
        # "<model_id>": {
        #     "model_id": str,
        #     "mu_pred": float,
        #     "mu_post": float,
        #     "sigma_pred": float,
        #     "sigma_post": float,
        #     "delta": float,
        #     "z_score": float
        # }
    },
    
    # Conservation check (must be ~0)
    "delta_sum": float,
    
    # Accuracy metrics
    "prediction_accuracy": float,
    "kendall_tau": float,
    
    # Timestamps
    "timestamp_start": str,
    "timestamp_end": str
}
