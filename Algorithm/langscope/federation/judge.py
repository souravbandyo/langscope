"""
Judge layer for peer-federated evaluation.

Handles judge ranking validation, aggregation, and outlier detection.
Judges provide complete rankings of all competitor responses.
"""

import re
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from langscope.core.constants import (
    OUTLIER_DISAGREEMENT_THRESHOLD,
    JUDGE_PENALTY_MU,
)
from langscope.ranking.cost_adjustment import (
    aggregate_judge_rankings,
    ranking_distance,
)

if TYPE_CHECKING:
    from langscope.core.model import LLMModel


class JudgeRankingValidator:
    """
    Validates judge rankings for completeness and format.
    
    A valid ranking must:
    1. Include all competitors
    2. Use ranks 1 to N without gaps or duplicates
    3. Follow the expected format
    """
    
    def __init__(self, expected_participants: List[str]):
        """
        Initialize validator.
        
        Args:
            expected_participants: List of expected model IDs
        """
        self.expected_participants = set(expected_participants)
        self.n_participants = len(expected_participants)
    
    def validate(self, ranking: Dict[str, int]) -> Tuple[bool, str]:
        """
        Validate a ranking.
        
        Args:
            ranking: Dictionary {model_id: rank}
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all participants are present
        ranked_participants = set(ranking.keys())
        
        if ranked_participants != self.expected_participants:
            missing = self.expected_participants - ranked_participants
            extra = ranked_participants - self.expected_participants
            
            if missing:
                return False, f"Missing participants: {missing}"
            if extra:
                return False, f"Unexpected participants: {extra}"
        
        # Check ranks are valid (1 to N, no duplicates)
        ranks = list(ranking.values())
        expected_ranks = set(range(1, self.n_participants + 1))
        actual_ranks = set(ranks)
        
        if actual_ranks != expected_ranks:
            return False, f"Invalid ranks: expected {expected_ranks}, got {actual_ranks}"
        
        if len(ranks) != len(set(ranks)):
            return False, "Duplicate ranks found"
        
        return True, ""
    
    def parse_ranking_from_text(
        self,
        text: str,
        participant_labels: Dict[str, str] = None
    ) -> Optional[Dict[str, int]]:
        """
        Parse ranking from judge's text response.
        
        Expected formats:
        - "1. Model A, 2. Model B, ..."
        - "Response A: 1, Response B: 2, ..."
        - JSON format
        
        Args:
            text: Judge's text response
            participant_labels: Mapping from labels to model IDs
        
        Returns:
            Parsed ranking or None if parsing failed
        """
        # Try different parsing strategies
        ranking = self._try_parse_numbered_list(text, participant_labels)
        if ranking:
            return ranking
        
        ranking = self._try_parse_colon_format(text, participant_labels)
        if ranking:
            return ranking
        
        ranking = self._try_parse_json(text, participant_labels)
        if ranking:
            return ranking
        
        return None
    
    def _try_parse_numbered_list(
        self,
        text: str,
        labels: Dict[str, str] = None
    ) -> Optional[Dict[str, int]]:
        """Parse format: 1. Label A, 2. Label B, ..."""
        pattern = r'(\d+)\.\s*([A-Za-z0-9_\s]+)'
        matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        ranking = {}
        for rank_str, label in matches:
            rank = int(rank_str)
            label = label.strip()
            
            if labels and label in labels:
                model_id = labels[label]
            else:
                model_id = label
            
            ranking[model_id] = rank
        
        return ranking if len(ranking) == self.n_participants else None
    
    def _try_parse_colon_format(
        self,
        text: str,
        labels: Dict[str, str] = None
    ) -> Optional[Dict[str, int]]:
        """Parse format: Label A: 1, Label B: 2, ..."""
        pattern = r'([A-Za-z0-9_\s]+):\s*(\d+)'
        matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        ranking = {}
        for label, rank_str in matches:
            rank = int(rank_str)
            label = label.strip()
            
            if labels and label in labels:
                model_id = labels[label]
            else:
                model_id = label
            
            ranking[model_id] = rank
        
        return ranking if len(ranking) == self.n_participants else None
    
    def _try_parse_json(
        self,
        text: str,
        labels: Dict[str, str] = None
    ) -> Optional[Dict[str, int]]:
        """Try to parse JSON format."""
        import json
        
        # Try to find JSON in text
        try:
            # Look for JSON object
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                
                ranking = {}
                for key, value in data.items():
                    if labels and key in labels:
                        model_id = labels[key]
                    else:
                        model_id = key
                    ranking[model_id] = int(value)
                
                return ranking if len(ranking) == self.n_participants else None
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None


class JudgeAggregator:
    """
    Aggregates rankings from multiple judges.
    
    Uses weighted Borda count where judge weights are based
    on their TrueSkill ratings.
    """
    
    def __init__(
        self,
        outlier_threshold: float = OUTLIER_DISAGREEMENT_THRESHOLD
    ):
        """
        Initialize aggregator.
        
        Args:
            outlier_threshold: Maximum disagreement before marking as outlier
        """
        self.outlier_threshold = outlier_threshold
    
    def aggregate(
        self,
        judge_rankings: List[Dict[str, int]],
        judge_weights: List[float] = None
    ) -> Dict[str, int]:
        """
        Aggregate judge rankings into consensus.
        
        Args:
            judge_rankings: List of rankings from each judge
            judge_weights: Weights for each judge (uniform if None)
        
        Returns:
            Aggregated ranking
        """
        return aggregate_judge_rankings(judge_rankings, judge_weights)
    
    def aggregate_with_outlier_detection(
        self,
        judge_rankings: List[Dict[str, int]],
        judge_weights: List[float] = None,
        judge_ids: List[str] = None
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Aggregate with outlier detection.
        
        Args:
            judge_rankings: List of rankings from each judge
            judge_weights: Weights for each judge
            judge_ids: IDs of judges for outlier reporting
        
        Returns:
            Tuple of (aggregated_ranking, outlier_judge_ids)
        """
        if not judge_rankings:
            return {}, []
        
        judge_ids = judge_ids or [str(i) for i in range(len(judge_rankings))]
        
        # First pass: aggregate all rankings
        consensus = self.aggregate(judge_rankings, judge_weights)
        
        # Detect outliers
        outlier_ids = []
        for i, ranking in enumerate(judge_rankings):
            if detect_outlier_judge(ranking, consensus, self.outlier_threshold):
                outlier_ids.append(judge_ids[i])
        
        # If outliers detected, re-aggregate without them
        if outlier_ids and len(outlier_ids) < len(judge_rankings):
            filtered_rankings = []
            filtered_weights = []
            
            for i, (ranking, weight) in enumerate(
                zip(judge_rankings, judge_weights or [1.0] * len(judge_rankings))
            ):
                if judge_ids[i] not in outlier_ids:
                    filtered_rankings.append(ranking)
                    filtered_weights.append(weight)
            
            consensus = self.aggregate(filtered_rankings, filtered_weights)
        
        return consensus, outlier_ids


def detect_outlier_judge(
    judge_ranking: Dict[str, int],
    consensus_ranking: Dict[str, int],
    threshold: float = OUTLIER_DISAGREEMENT_THRESHOLD
) -> bool:
    """
    Detect if a judge's ranking is an outlier.
    
    A judge is considered an outlier if their ranking disagrees
    with the consensus by more than the threshold.
    
    Args:
        judge_ranking: Judge's ranking
        consensus_ranking: Consensus ranking
        threshold: Maximum acceptable disagreement
    
    Returns:
        True if judge is an outlier
    """
    distance = ranking_distance(judge_ranking, consensus_ranking, method="kendall")
    return distance > threshold


def calculate_judge_penalty(
    judge_ranking: Dict[str, int],
    consensus_ranking: Dict[str, int],
    penalty_per_position: float = JUDGE_PENALTY_MU
) -> float:
    """
    Calculate penalty for a judge's disagreement with consensus.
    
    Args:
        judge_ranking: Judge's ranking
        consensus_ranking: Consensus ranking
        penalty_per_position: Penalty per position of disagreement
    
    Returns:
        Total penalty (negative μ adjustment)
    """
    distance = ranking_distance(judge_ranking, consensus_ranking, method="kendall")
    
    if distance > OUTLIER_DISAGREEMENT_THRESHOLD:
        return penalty_per_position
    
    return 0.0


def create_judge_prompt(
    case_text: str,
    question_text: str,
    responses: Dict[str, str],
    response_labels: Dict[str, str] = None
) -> str:
    """
    Create prompt for judge to rank responses.
    
    Args:
        case_text: The case/scenario text
        question_text: The question being answered
        responses: Dictionary {model_id: response_text}
        response_labels: Optional labels for anonymization
    
    Returns:
        Judge prompt string
    """
    # Create labels if not provided
    if response_labels is None:
        labels = list("ABCDEFGHIJ")[:len(responses)]
        response_labels = {
            model_id: labels[i]
            for i, model_id in enumerate(responses.keys())
        }
    
    # Build prompt
    prompt_parts = [
        "You are an expert evaluator. Please rank the following responses to the given case and question.",
        "",
        "## Case",
        case_text,
        "",
        "## Question",
        question_text,
        "",
        "## Responses to Rank",
        ""
    ]
    
    for model_id, response in responses.items():
        label = response_labels.get(model_id, model_id)
        prompt_parts.extend([
            f"### Response {label}",
            response,
            ""
        ])
    
    prompt_parts.extend([
        "## Instructions",
        f"Rank all {len(responses)} responses from best (1) to worst ({len(responses)}).",
        "Consider accuracy, completeness, clarity, and relevance.",
        "",
        "Provide your ranking in the format:",
        "1. [Response Label]",
        "2. [Response Label]",
        "...",
        "",
        "Do not provide any other text or explanation, only the ranking."
    ])
    
    return "\n".join(prompt_parts)


def shuffle_response_order(
    responses: Dict[str, str],
    seed: int = None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Shuffle response order and create label mapping.
    
    This prevents position bias in judging.
    
    Args:
        responses: Dictionary {model_id: response_text}
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (shuffled_responses, label_to_model_id_mapping)
    """
    import random as rnd
    
    if seed is not None:
        rnd.seed(seed)
    
    model_ids = list(responses.keys())
    rnd.shuffle(model_ids)
    
    labels = list("ABCDEFGHIJ")[:len(model_ids)]
    
    shuffled = {}
    label_mapping = {}
    
    for label, model_id in zip(labels, model_ids):
        shuffled[label] = responses[model_id]
        label_mapping[label] = model_id
    
    return shuffled, label_mapping


