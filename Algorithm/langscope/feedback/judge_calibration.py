"""
Calibrate LLM judges against user ground truth.

User feedback provides ground truth for calibrating LLM judges.
This module tracks how well each judge's preferences align with
actual user preferences, enabling us to adjust judge weights.

Key Formulas:
    Calibration_J = (# user-aligned judgments) / (# total overlapping judgments)
    
    w_J^new = w_J^old × (1 + γ × (Calibration_J - 0.5))

Where γ = 0.2 controls the calibration influence.

A judge with Calibration=0.5 gets no weight adjustment (neutral).
A judge with Calibration=0.8 gets weight increased by 6%.
A judge with Calibration=0.3 gets weight decreased by 4%.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

# Default calibration influence factor
CALIBRATION_GAMMA: float = 0.2


@dataclass
class JudgeCalibrationData:
    """Calibration data for a single judge."""
    judge_id: str
    agreed: int = 0        # Number of user-aligned judgments
    total: int = 0         # Total overlapping judgments
    
    @property
    def calibration(self) -> float:
        """Compute calibration score."""
        if self.total == 0:
            return 0.5  # Default: no information
        return self.agreed / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "agreed": self.agreed,
            "total": self.total,
            "calibration": self.calibration
        }


class JudgeCalibrator:
    """
    Track and update judge calibration based on agreement with user feedback.
    
    When a user provides feedback, we compare their pairwise preferences
    to what each LLM judge predicted. Judges that agree with users more
    often get higher calibration scores and increased weight.
    
    Key Formulas:
        Calibration_J = (# user-aligned) / (# total comparisons)
        w_J^new = w_J^old × (1 + γ × (Calibration_J - 0.5))
    
    Example:
        >>> calibrator = JudgeCalibrator(gamma=0.2)
        >>> # Record that GPT-4 judge agreed with user on this comparison
        >>> calibrator.record_comparison(
        ...     judge_id="gpt-4-judge",
        ...     domain="medical",
        ...     judge_preference=("llama", "claude"),  # llama > claude
        ...     user_preference=("llama", "claude")     # user agrees
        ... )
        >>> calibrator.get_calibration("gpt-4-judge")
        1.0  # Perfect agreement so far
    """
    
    def __init__(self, gamma: float = CALIBRATION_GAMMA):
        """
        Initialize the calibrator.
        
        Args:
            gamma: Weight adjustment factor (default: 0.2)
        """
        self.gamma = gamma
        
        # Global calibration data per judge
        self._global_data: Dict[str, JudgeCalibrationData] = {}
        
        # Domain-specific calibration data
        self._domain_data: Dict[str, Dict[str, JudgeCalibrationData]] = defaultdict(dict)
    
    def _get_global_data(self, judge_id: str) -> JudgeCalibrationData:
        """Get or create global calibration data for a judge."""
        if judge_id not in self._global_data:
            self._global_data[judge_id] = JudgeCalibrationData(judge_id=judge_id)
        return self._global_data[judge_id]
    
    def _get_domain_data(self, judge_id: str, domain: str) -> JudgeCalibrationData:
        """Get or create domain-specific calibration data for a judge."""
        if judge_id not in self._domain_data[domain]:
            self._domain_data[domain][judge_id] = JudgeCalibrationData(judge_id=judge_id)
        return self._domain_data[domain][judge_id]
    
    def record_comparison(
        self,
        judge_id: str,
        domain: str,
        judge_preference: Tuple[str, str],
        user_preference: Tuple[str, str]
    ):
        """
        Record a single pairwise comparison between judge and user.
        
        Args:
            judge_id: Identifier for the LLM judge
            domain: Domain where comparison occurred
            judge_preference: (winner, loser) as judged by LLM
            user_preference: (winner, loser) as judged by user
        """
        # Update global data
        global_data = self._get_global_data(judge_id)
        global_data.total += 1
        if judge_preference == user_preference:
            global_data.agreed += 1
        
        # Update domain-specific data
        domain_data = self._get_domain_data(judge_id, domain)
        domain_data.total += 1
        if judge_preference == user_preference:
            domain_data.agreed += 1
    
    def record_rankings(
        self,
        judge_id: str,
        domain: str,
        judge_ranking: Dict[str, int],
        user_ranking: Dict[str, int]
    ):
        """
        Record all pairwise comparisons from two rankings.
        
        Extracts all pairs and compares judge vs user preferences.
        
        Args:
            judge_id: Identifier for the LLM judge
            domain: Domain where ranking occurred
            judge_ranking: {model_id: rank} from judge (1=best)
            user_ranking: {model_id: rank} from user (1=best)
        """
        common_models = set(judge_ranking.keys()) & set(user_ranking.keys())
        models = list(common_models)
        
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                # Determine preferences
                if judge_ranking[model_a] < judge_ranking[model_b]:
                    judge_pref = (model_a, model_b)
                else:
                    judge_pref = (model_b, model_a)
                
                if user_ranking[model_a] < user_ranking[model_b]:
                    user_pref = (model_a, model_b)
                else:
                    user_pref = (model_b, model_a)
                
                self.record_comparison(judge_id, domain, judge_pref, user_pref)
    
    def get_calibration(
        self,
        judge_id: str,
        domain: Optional[str] = None
    ) -> float:
        """
        Get calibration score for a judge.
        
        Args:
            judge_id: The judge identifier
            domain: Optional domain for domain-specific calibration
        
        Returns:
            Calibration score ∈ [0, 1] (0.5 = neutral)
        """
        if domain and domain in self._domain_data:
            data = self._domain_data[domain].get(judge_id)
            if data and data.total > 0:
                return data.calibration
        
        # Fall back to global calibration
        if judge_id in self._global_data:
            data = self._global_data[judge_id]
            if data.total > 0:
                return data.calibration
        
        return 0.5  # Default: no information
    
    def adjust_judge_weight(
        self,
        current_weight: float,
        judge_id: str,
        domain: Optional[str] = None
    ) -> float:
        """
        Adjust judge weight based on calibration.
        
        Formula:
            w_new = w_old × (1 + γ × (calibration - 0.5))
        
        Where:
            - calibration = 0.5: neutral (no adjustment)
            - calibration > 0.5: weight increases
            - calibration < 0.5: weight decreases
        
        Args:
            current_weight: Judge's current weight
            judge_id: Judge identifier
            domain: Optional domain for domain-specific calibration
        
        Returns:
            Adjusted weight
        """
        calibration = self.get_calibration(judge_id, domain)
        adjustment = 1.0 + self.gamma * (calibration - 0.5)
        return current_weight * adjustment
    
    def get_comparison_count(
        self,
        judge_id: str,
        domain: Optional[str] = None
    ) -> int:
        """Get number of recorded comparisons for a judge."""
        if domain and domain in self._domain_data:
            data = self._domain_data[domain].get(judge_id)
            if data:
                return data.total
        
        if judge_id in self._global_data:
            return self._global_data[judge_id].total
        
        return 0
    
    def get_all_calibrations(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, float]:
        """Get calibration scores for all tracked judges."""
        calibrations = {}
        
        if domain and domain in self._domain_data:
            for judge_id, data in self._domain_data[domain].items():
                if data.total > 0:
                    calibrations[judge_id] = data.calibration
        else:
            for judge_id, data in self._global_data.items():
                if data.total > 0:
                    calibrations[judge_id] = data.calibration
        
        return calibrations
    
    def get_judge_stats(
        self,
        judge_id: str,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a judge."""
        if domain and domain in self._domain_data:
            data = self._domain_data[domain].get(judge_id)
            if data:
                return {
                    "judge_id": judge_id,
                    "domain": domain,
                    "agreed": data.agreed,
                    "total": data.total,
                    "calibration": data.calibration,
                    "weight_adjustment": 1.0 + self.gamma * (data.calibration - 0.5)
                }
        
        if judge_id in self._global_data:
            data = self._global_data[judge_id]
            return {
                "judge_id": judge_id,
                "domain": "global",
                "agreed": data.agreed,
                "total": data.total,
                "calibration": data.calibration,
                "weight_adjustment": 1.0 + self.gamma * (data.calibration - 0.5)
            }
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibrator state to dictionary for serialization."""
        return {
            "gamma": self.gamma,
            "global_data": {
                judge_id: data.to_dict()
                for judge_id, data in self._global_data.items()
            },
            "domain_data": {
                domain: {
                    judge_id: data.to_dict()
                    for judge_id, data in domain_judges.items()
                }
                for domain, domain_judges in self._domain_data.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JudgeCalibrator':
        """Create calibrator from dictionary."""
        calibrator = cls(gamma=data.get("gamma", CALIBRATION_GAMMA))
        
        # Load global data
        for judge_id, judge_data in data.get("global_data", {}).items():
            calibrator._global_data[judge_id] = JudgeCalibrationData(
                judge_id=judge_id,
                agreed=judge_data.get("agreed", 0),
                total=judge_data.get("total", 0)
            )
        
        # Load domain data
        for domain, domain_judges in data.get("domain_data", {}).items():
            for judge_id, judge_data in domain_judges.items():
                calibrator._domain_data[domain][judge_id] = JudgeCalibrationData(
                    judge_id=judge_id,
                    agreed=judge_data.get("agreed", 0),
                    total=judge_data.get("total", 0)
                )
        
        return calibrator
