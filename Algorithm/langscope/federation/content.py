"""
Content generation layer for peer-federated evaluation.

Handles case generation, question generation, and peer validation.
Content creators are selected from high-stratum models.
"""

from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from langscope.core.model import LLMModel


@dataclass
class GeneratedContent:
    """Container for generated content (case or question)."""
    content_id: str
    content_type: str  # "case" or "question"
    text: str
    generator_id: str
    generator_name: str
    generator_mu: float
    domain: str
    timestamp: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "text": self.text,
            "generator_id": self.generator_id,
            "generator_name": self.generator_name,
            "generator_mu": self.generator_mu,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Result of peer validation."""
    is_valid: bool
    votes: List[Tuple[str, bool]]  # (validator_id, approved)
    issues: List[str] = field(default_factory=list)
    
    @property
    def approval_rate(self) -> float:
        """Get approval rate."""
        if not self.votes:
            return 0.0
        approvals = sum(1 for _, approved in self.votes if approved)
        return approvals / len(self.votes)


class ContentGenerator:
    """
    Manages content generation for matches.
    
    Content includes:
    - Cases: Scenarios or problems to evaluate
    - Questions: Specific questions about the case
    """
    
    def __init__(self, domain: str):
        """
        Initialize content generator.
        
        Args:
            domain: Domain for content generation
        """
        self.domain = domain
    
    def create_case_prompt(
        self,
        difficulty: str = "medium",
        specific_topic: str = None
    ) -> str:
        """
        Create prompt for case generation.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            specific_topic: Optional specific topic focus
        
        Returns:
            Case generation prompt
        """
        base_prompt = f"""Generate a challenging and realistic case or scenario for evaluation in the domain of {self.domain}.

Requirements:
- Difficulty level: {difficulty}
- The case should be detailed enough to allow meaningful evaluation
- Include relevant context and constraints
- The case should have a clear objective or problem to solve
"""
        
        if specific_topic:
            base_prompt += f"\nFocus on the topic: {specific_topic}"
        
        base_prompt += """

Provide only the case text, without any additional commentary.
"""
        return base_prompt
    
    def create_question_prompt(
        self,
        case_text: str,
        n_questions: int = 1
    ) -> str:
        """
        Create prompt for question generation.
        
        Args:
            case_text: The case to generate questions for
            n_questions: Number of questions to generate
        
        Returns:
            Question generation prompt
        """
        return f"""Based on the following case, generate {n_questions} specific question(s) that will effectively evaluate a model's understanding and reasoning.

## Case
{case_text}

## Requirements
- Questions should be specific and answerable based on the case
- Questions should test deep understanding, not just surface recall
- Questions should have clear criteria for evaluation
- Generate exactly {n_questions} question(s)

Provide only the question(s), numbered if multiple, without any additional commentary.
"""
    
    def wrap_generation(
        self,
        text: str,
        content_type: str,
        generator: 'LLMModel',
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        metadata: Dict = None
    ) -> GeneratedContent:
        """
        Wrap generated text in a GeneratedContent object.
        
        Args:
            text: Generated text
            content_type: "case" or "question"
            generator: Model that generated the content
            tokens_used: Number of tokens used
            cost_usd: Cost of generation
            metadata: Additional metadata
        
        Returns:
            GeneratedContent object
        """
        return GeneratedContent(
            content_id=f"{content_type}_{uuid.uuid4().hex[:12]}",
            content_type=content_type,
            text=text,
            generator_id=generator.model_id,
            generator_name=generator.name,
            generator_mu=generator.get_domain_trueskill(self.domain).raw.mu,
            domain=self.domain,
            timestamp=datetime.utcnow().isoformat() + "Z",
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            metadata=metadata or {}
        )


class ContentValidator:
    """
    Validates generated content through peer review.
    
    Uses high-rated models to validate content quality.
    Content is approved if it passes the approval threshold.
    """
    
    def __init__(
        self,
        approval_threshold: float = 0.67,  # 2/3 majority
        min_validators: int = 3
    ):
        """
        Initialize validator.
        
        Args:
            approval_threshold: Minimum approval rate (0 to 1)
            min_validators: Minimum number of validators
        """
        self.approval_threshold = approval_threshold
        self.min_validators = min_validators
    
    def create_validation_prompt(
        self,
        content: GeneratedContent,
        content_type: str = "case"
    ) -> str:
        """
        Create prompt for content validation.
        
        Args:
            content: Content to validate
            content_type: Type of content
        
        Returns:
            Validation prompt
        """
        return f"""Evaluate the following {content_type} for quality and suitability for LLM evaluation.

## {content_type.title()}
{content.text}

## Evaluation Criteria
1. Is the content clear and well-written?
2. Is it appropriate for the domain ({content.domain})?
3. Is it sufficiently challenging but not unreasonable?
4. Does it allow for meaningful evaluation?

## Response Format
Respond with ONLY one of:
- APPROVE: If the content meets all criteria
- REJECT: [brief reason] - If the content has significant issues

Your response:"""
    
    def parse_validation_response(
        self,
        response: str
    ) -> Tuple[bool, str]:
        """
        Parse validator response.
        
        Args:
            response: Validator's response text
        
        Returns:
            Tuple of (approved, reason)
        """
        response = response.strip().upper()
        
        if response.startswith("APPROVE"):
            return True, ""
        
        if response.startswith("REJECT"):
            # Extract reason if provided
            parts = response.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
            return False, reason
        
        # Default to rejection if unclear
        return False, "Unclear response"
    
    def evaluate_validation(
        self,
        votes: List[Tuple[str, bool, str]]  # (validator_id, approved, reason)
    ) -> ValidationResult:
        """
        Evaluate validation votes.
        
        Args:
            votes: List of validation votes
        
        Returns:
            ValidationResult
        """
        vote_tuples = [(v_id, approved) for v_id, approved, _ in votes]
        issues = [reason for _, approved, reason in votes if not approved and reason]
        
        approval_count = sum(1 for _, approved, _ in votes if approved)
        total_votes = len(votes)
        
        if total_votes < self.min_validators:
            return ValidationResult(
                is_valid=False,
                votes=vote_tuples,
                issues=["Insufficient validators"]
            )
        
        is_valid = (approval_count / total_votes) >= self.approval_threshold
        
        return ValidationResult(
            is_valid=is_valid,
            votes=vote_tuples,
            issues=issues
        )


def apply_content_quality_penalty(
    model: 'LLMModel',
    domain: str,
    penalty_mu: float = 5.0
) -> None:
    """
    Apply penalty for poor content quality.
    
    Args:
        model: Model to penalize
        domain: Domain where penalty applies
        penalty_mu: Amount to subtract from μ
    """
    if domain in model.trueskill_by_domain:
        ts = model.trueskill_by_domain[domain]
        ts.raw.mu -= penalty_mu
        ts.cost_adjusted.mu -= penalty_mu
    else:
        model.trueskill.raw.mu -= penalty_mu
        model.trueskill.cost_adjusted.mu -= penalty_mu


