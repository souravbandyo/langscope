"""
Multi-player match management for LangScope.

Handles match creation, result storage, and match metadata.
"""

import uuid
import math
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from langscope.core.constants import INFO_BITS

if TYPE_CHECKING:
    from langscope.core.model import LLMModel


@dataclass
class MatchParticipant:
    """A participant in a match."""
    model_id: str
    model_name: str
    role: str  # "competitor", "judge", "case_creator", "question_creator"
    mu_before: float = 0.0
    sigma_before: float = 0.0
    mu_after: float = 0.0
    sigma_after: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "role": self.role,
            "mu_before": self.mu_before,
            "sigma_before": self.sigma_before,
            "mu_after": self.mu_after,
            "sigma_after": self.sigma_after,
        }


@dataclass
class MatchResponse:
    """A response from a competitor."""
    model_id: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw_rank: int = 0
    cost_rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "raw_rank": self.raw_rank,
            "cost_rank": self.cost_rank,
        }


@dataclass
class Match:
    """
    A multi-player evaluation match.
    
    Contains all metadata, responses, rankings, and results
    for a single match evaluation.
    """
    match_id: str
    domain: str
    timestamp: str
    
    # Participants
    competitors: List[MatchParticipant] = field(default_factory=list)
    judges: List[MatchParticipant] = field(default_factory=list)
    case_creator: Optional[MatchParticipant] = None
    question_creator: Optional[MatchParticipant] = None
    
    # Content
    case_text: str = ""
    question_text: str = ""
    
    # Responses
    responses: List[MatchResponse] = field(default_factory=list)
    
    # Rankings
    raw_ranking: Dict[str, int] = field(default_factory=dict)
    cost_ranking: Dict[str, int] = field(default_factory=dict)
    judge_rankings: List[Dict[str, int]] = field(default_factory=list)
    judge_weights: List[float] = field(default_factory=list)
    
    # Plackett-Luce
    pl_strengths: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    info_bits: float = 0.0
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: str = ""
    
    def __post_init__(self):
        if not self.match_id:
            self.match_id = f"match_{uuid.uuid4().hex[:16]}"
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if self.competitors and not self.info_bits:
            n = len(self.competitors)
            self.info_bits = INFO_BITS.get(n, math.log2(math.factorial(n)))
    
    @property
    def participant_count(self) -> int:
        """Get number of competitors."""
        return len(self.competitors)
    
    def get_competitor_ids(self) -> List[str]:
        """Get list of competitor model IDs."""
        return [c.model_id for c in self.competitors]
    
    def get_response(self, model_id: str) -> Optional[MatchResponse]:
        """Get response for a model."""
        for r in self.responses:
            if r.model_id == model_id:
                return r
        return None
    
    def set_rankings(
        self,
        raw_ranking: Dict[str, int],
        cost_ranking: Dict[str, int]
    ):
        """Set final rankings."""
        self.raw_ranking = raw_ranking
        self.cost_ranking = cost_ranking
        
        # Update response ranks
        for response in self.responses:
            response.raw_rank = raw_ranking.get(response.model_id, 0)
            response.cost_rank = cost_ranking.get(response.model_id, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "_id": self.match_id,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "participants": [c.model_id for c in self.competitors],
            "participant_count": self.participant_count,
            "prompt": {
                "case_text": self.case_text,
                "case_generator_id": self.case_creator.model_id if self.case_creator else None,
                "case_generator_mu": self.case_creator.mu_before if self.case_creator else None,
                "question_text": self.question_text,
                "question_generator_id": self.question_creator.model_id if self.question_creator else None,
                "question_generator_mu": self.question_creator.mu_before if self.question_creator else None,
            },
            "responses": {r.model_id: r.to_dict() for r in self.responses},
            "judgment": {
                "raw_ranking": self.raw_ranking,
                "cost_adjusted_ranking": self.cost_ranking,
                "judges": [
                    {
                        "judge_id": j.model_id,
                        "judge_name": j.model_name,
                        "mu_at_judgment": j.mu_before,
                        "raw_ranking": self.judge_rankings[i] if i < len(self.judge_rankings) else {},
                        "weight": self.judge_weights[i] if i < len(self.judge_weights) else 0.0,
                    }
                    for i, j in enumerate(self.judges)
                ],
            },
            "plackett_luce": {
                "raw_strengths": self.pl_strengths,
            },
            "meta": {
                "info_bits": self.info_bits,
                "status": self.status,
                "judgment_method": "weighted_borda",
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Match':
        """Create from dictionary."""
        match = cls(
            match_id=data.get("_id", ""),
            domain=data.get("domain", ""),
            timestamp=data.get("timestamp", ""),
        )
        
        # Reconstruct from stored data
        prompt = data.get("prompt", {})
        match.case_text = prompt.get("case_text", "")
        match.question_text = prompt.get("question_text", "")
        
        judgment = data.get("judgment", {})
        match.raw_ranking = judgment.get("raw_ranking", {})
        match.cost_ranking = judgment.get("cost_adjusted_ranking", {})
        
        pl = data.get("plackett_luce", {})
        match.pl_strengths = pl.get("raw_strengths", {})
        
        meta = data.get("meta", {})
        match.info_bits = meta.get("info_bits", 0.0)
        match.status = meta.get("status", "completed")
        
        return match


def create_match(
    domain: str,
    competitors: List['LLMModel'],
    judges: List['LLMModel'],
    case_creator: 'LLMModel',
    question_creator: 'LLMModel'
) -> Match:
    """
    Create a new match with participants.
    
    Args:
        domain: Domain for evaluation
        competitors: List of competing models
        judges: List of judge models
        case_creator: Model creating the case
        question_creator: Model creating the question
    
    Returns:
        New Match instance
    """
    match = Match(
        match_id=f"match_{uuid.uuid4().hex[:16]}",
        domain=domain,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
    
    # Add competitors
    for model in competitors:
        ts = model.get_domain_trueskill(domain)
        match.competitors.append(MatchParticipant(
            model_id=model.model_id,
            model_name=model.name,
            role="competitor",
            mu_before=ts.raw.mu,
            sigma_before=ts.raw.sigma,
        ))
    
    # Add judges
    for model in judges:
        ts = model.get_domain_trueskill(domain)
        match.judges.append(MatchParticipant(
            model_id=model.model_id,
            model_name=model.name,
            role="judge",
            mu_before=ts.raw.mu,
            sigma_before=ts.raw.sigma,
        ))
    
    # Add content creators
    ts = case_creator.get_domain_trueskill(domain)
    match.case_creator = MatchParticipant(
        model_id=case_creator.model_id,
        model_name=case_creator.name,
        role="case_creator",
        mu_before=ts.raw.mu,
        sigma_before=ts.raw.sigma,
    )
    
    ts = question_creator.get_domain_trueskill(domain)
    match.question_creator = MatchParticipant(
        model_id=question_creator.model_id,
        model_name=question_creator.name,
        role="question_creator",
        mu_before=ts.raw.mu,
        sigma_before=ts.raw.sigma,
    )
    
    # Compute info bits
    n = len(competitors)
    match.info_bits = INFO_BITS.get(n, math.log2(math.factorial(n)))
    
    match.status = "pending"
    
    return match


