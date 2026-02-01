"""
Domain configuration dataclasses and templates.

Defines domain-specific settings, prompts, and pre-defined templates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from langscope.core.constants import (
    STRATA_THRESHOLDS,
    JUDGE_COUNT,
    PLAYERS_PER_MATCH,
    MIN_PLAYERS,
    SWISS_DELTA,
    COST_TEMP,
    RATING_TEMP,
)


@dataclass
class DomainPrompts:
    """Prompts for a domain."""
    
    case_generation: str = """Generate a challenging and realistic case or scenario for evaluation.

Requirements:
- The case should be detailed and realistic
- Include relevant context and constraints
- The case should have a clear objective or problem to solve
- Difficulty: {difficulty}

Provide only the case text, without any additional commentary."""

    question_generation: str = """Based on the following case, generate a specific question that will effectively evaluate a model's understanding and reasoning.

## Case
{case_text}

## Requirements
- The question should be specific and answerable based on the case
- It should test deep understanding, not just surface recall
- It should have clear criteria for evaluation

Provide only the question, without any additional commentary."""

    answer_generation: str = """## Case
{case_text}

## Question
{question_text}

Please provide a comprehensive and accurate answer to the question based on the case provided."""

    judge_ranking: str = """You are an expert evaluator. Please rank the following responses to the given case and question.

## Case
{case_text}

## Question
{question_text}

## Responses to Rank
{responses}

## Instructions
Rank all {n_responses} responses from best (1) to worst ({n_responses}).
Consider accuracy, completeness, clarity, and relevance.

Provide your ranking in the format:
1. [Response Label]
2. [Response Label]
...

Do not provide any other text or explanation, only the ranking."""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "case_generation": self.case_generation,
            "question_generation": self.question_generation,
            "answer_generation": self.answer_generation,
            "judge_ranking": self.judge_ranking,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainPrompts':
        """Create from dictionary."""
        return cls(
            case_generation=data.get("case_generation", cls.case_generation),
            question_generation=data.get("question_generation", cls.question_generation),
            answer_generation=data.get("answer_generation", cls.answer_generation),
            judge_ranking=data.get("judge_ranking", cls.judge_ranking),
        )


@dataclass
class DomainSettings:
    """Settings for a domain."""
    
    # Strata thresholds
    strata_thresholds: Dict[str, float] = field(default_factory=lambda: dict(STRATA_THRESHOLDS))
    
    # Match configuration
    judge_count: int = JUDGE_COUNT
    players_per_match: int = PLAYERS_PER_MATCH
    min_players: int = MIN_PLAYERS
    swiss_delta: float = SWISS_DELTA
    
    # Temperature parameters
    cost_temperature: float = COST_TEMP
    rating_temperature: float = RATING_TEMP
    
    # Validation settings
    min_stratum_for_judge: int = 2
    min_stratum_for_creator: int = 3
    
    # Match limits
    max_matches_per_model: int = 50
    
    # Evaluation type: "subjective" (LLM-as-judge) or "ground_truth" (metrics-based)
    evaluation_type: str = "subjective"
    
    # Ground truth specific settings (only used when evaluation_type == "ground_truth")
    ground_truth_domain: Optional[str] = None  # Maps to GT domain (e.g., "asr", "needle_in_haystack")
    primary_metric: Optional[str] = None  # Primary metric for GT evaluation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strata_thresholds": self.strata_thresholds,
            "judge_count": self.judge_count,
            "players_per_match": self.players_per_match,
            "min_players": self.min_players,
            "swiss_delta": self.swiss_delta,
            "cost_temperature": self.cost_temperature,
            "rating_temperature": self.rating_temperature,
            "min_stratum_for_judge": self.min_stratum_for_judge,
            "min_stratum_for_creator": self.min_stratum_for_creator,
            "max_matches_per_model": self.max_matches_per_model,
            "evaluation_type": self.evaluation_type,
            "ground_truth_domain": self.ground_truth_domain,
            "primary_metric": self.primary_metric,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainSettings':
        """Create from dictionary."""
        return cls(
            strata_thresholds=data.get("strata_thresholds", dict(STRATA_THRESHOLDS)),
            judge_count=data.get("judge_count", JUDGE_COUNT),
            players_per_match=data.get("players_per_match", PLAYERS_PER_MATCH),
            min_players=data.get("min_players", MIN_PLAYERS),
            swiss_delta=data.get("swiss_delta", SWISS_DELTA),
            cost_temperature=data.get("cost_temperature", COST_TEMP),
            rating_temperature=data.get("rating_temperature", RATING_TEMP),
            min_stratum_for_judge=data.get("min_stratum_for_judge", 2),
            min_stratum_for_creator=data.get("min_stratum_for_creator", 3),
            max_matches_per_model=data.get("max_matches_per_model", 50),
            evaluation_type=data.get("evaluation_type", "subjective"),
            ground_truth_domain=data.get("ground_truth_domain"),
            primary_metric=data.get("primary_metric"),
        )


@dataclass
class DomainStatistics:
    """Statistics for a domain."""
    total_matches: int = 0
    total_models_evaluated: int = 0
    avg_info_bits_per_match: float = 0.0
    top_model_raw: str = ""
    top_model_cost: str = ""
    last_match_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_matches": self.total_matches,
            "total_models_evaluated": self.total_models_evaluated,
            "avg_info_bits_per_match": self.avg_info_bits_per_match,
            "top_model_raw": self.top_model_raw,
            "top_model_cost": self.top_model_cost,
            "last_match_timestamp": self.last_match_timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainStatistics':
        """Create from dictionary."""
        return cls(
            total_matches=data.get("total_matches", 0),
            total_models_evaluated=data.get("total_models_evaluated", 0),
            avg_info_bits_per_match=data.get("avg_info_bits_per_match", 0.0),
            top_model_raw=data.get("top_model_raw", ""),
            top_model_cost=data.get("top_model_cost", ""),
            last_match_timestamp=data.get("last_match_timestamp", ""),
        )


@dataclass
class Domain:
    """Complete domain configuration."""
    
    name: str
    display_name: str = ""
    description: str = ""
    parent_domain: Optional[str] = None
    prompts: DomainPrompts = field(default_factory=DomainPrompts)
    settings: DomainSettings = field(default_factory=DomainSettings)
    statistics: DomainStatistics = field(default_factory=DomainStatistics)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "_id": self.name,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "parent_domain": self.parent_domain,
            "prompts": self.prompts.to_dict(),
            "settings": self.settings.to_dict(),
            "statistics": self.statistics.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Domain':
        """Create from dictionary."""
        return cls(
            name=data.get("name", data.get("_id", "")),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            parent_domain=data.get("parent_domain"),
            prompts=DomainPrompts.from_dict(data.get("prompts", {})),
            settings=DomainSettings.from_dict(data.get("settings", {})),
            statistics=DomainStatistics.from_dict(data.get("statistics", {})),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Pre-defined Domain Templates
# =============================================================================

DOMAIN_TEMPLATES: Dict[str, Domain] = {
    "general": Domain(
        name="general",
        display_name="General Knowledge",
        description="General knowledge and reasoning evaluation",
    ),
    
    "clinical_reasoning": Domain(
        name="clinical_reasoning",
        display_name="Clinical Reasoning",
        description="Medical clinical reasoning and diagnosis evaluation",
        parent_domain="general",
        prompts=DomainPrompts(
            case_generation="""Generate a realistic clinical case for medical reasoning evaluation.

Include:
- Patient demographics and chief complaint
- Relevant medical history
- Physical examination findings
- Laboratory/imaging results (if applicable)
- Difficulty level: {difficulty}

The case should be challenging but solvable with proper clinical reasoning.
Provide only the case, without diagnosis or solution.""",

            question_generation="""Based on the clinical case below, generate a question that tests clinical reasoning.

## Case
{case_text}

The question should:
- Require synthesis of information from the case
- Test differential diagnosis or treatment planning
- Have evidence-based criteria for evaluation

Provide only the question.""",

            judge_ranking="""You are an expert medical evaluator. Rank the following clinical responses.

## Clinical Case
{case_text}

## Question
{question_text}

## Responses
{responses}

Rank all {n_responses} responses from best (1) to worst ({n_responses}).
Consider: clinical accuracy, completeness, evidence-based reasoning, safety.

Format:
1. [Response Label]
2. [Response Label]
...""",
        ),
    ),
    
    "hindi": Domain(
        name="hindi",
        display_name="Hindi Language",
        description="Hindi language understanding and generation",
        parent_domain="general",
        prompts=DomainPrompts(
            case_generation="""एक चुनौतीपूर्ण हिंदी भाषा का परिदृश्य बनाएं।

आवश्यकताएं:
- परिदृश्य विस्तृत और यथार्थवादी होना चाहिए
- हिंदी भाषा की समझ का परीक्षण करे
- कठिनाई स्तर: {difficulty}

केवल परिदृश्य प्रदान करें।""",

            question_generation="""निम्नलिखित परिदृश्य के आधार पर एक प्रश्न बनाएं।

## परिदृश्य
{case_text}

प्रश्न:
- विशिष्ट और उत्तर देने योग्य हो
- गहरी समझ का परीक्षण करे

केवल प्रश्न प्रदान करें।""",
        ),
    ),
    
    "hindi_medical": Domain(
        name="hindi_medical",
        display_name="Hindi Medical",
        description="Medical evaluation in Hindi language",
        parent_domain="hindi",
        prompts=DomainPrompts(
            case_generation="""एक यथार्थवादी चिकित्सा केस हिंदी में बनाएं।

शामिल करें:
- रोगी का विवरण और मुख्य शिकायत
- चिकित्सा इतिहास
- शारीरिक परीक्षण के निष्कर्ष
- कठिनाई: {difficulty}

केवल केस प्रदान करें, निदान नहीं।""",
        ),
    ),
    
    "coding_python": Domain(
        name="coding_python",
        display_name="Python Programming",
        description="Python programming evaluation",
        parent_domain="general",
        prompts=DomainPrompts(
            case_generation="""Create a Python programming challenge.

Requirements:
- Clear problem statement
- Input/output specifications
- Example test cases
- Difficulty: {difficulty}

The problem should test algorithmic thinking and Python proficiency.
Provide only the problem description.""",

            question_generation="""Based on the problem below, create a specific coding question.

## Problem
{case_text}

The question should require implementing a solution.
Provide only the question.""",

            judge_ranking="""You are an expert code reviewer. Rank the following Python solutions.

## Problem
{case_text}

## Question
{question_text}

## Solutions
{responses}

Rank all {n_responses} solutions from best (1) to worst ({n_responses}).
Consider: correctness, efficiency, code quality, edge case handling.

Format:
1. [Response Label]
...""",
        ),
    ),
    
    "math_reasoning": Domain(
        name="math_reasoning",
        display_name="Mathematical Reasoning",
        description="Mathematical problem solving evaluation",
        parent_domain="general",
        prompts=DomainPrompts(
            case_generation="""Create a mathematical reasoning problem.

Requirements:
- Clear problem statement
- All necessary information provided
- Requires multi-step reasoning
- Difficulty: {difficulty}

Provide only the problem, not the solution.""",
        ),
    ),
    
    # ==========================================================================
    # Ground Truth Domains
    # ==========================================================================
    
    "asr": Domain(
        name="asr",
        display_name="Speech Recognition",
        description="Automatic Speech Recognition (ASR) evaluation using ground truth transcripts",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="asr",
            primary_metric="wer",
        ),
    ),
    
    "tts": Domain(
        name="tts",
        display_name="Text-to-Speech",
        description="Text-to-Speech evaluation with composite quality scoring",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="tts",
            primary_metric="composite",
        ),
    ),
    
    "visual_qa": Domain(
        name="visual_qa",
        display_name="Visual Question Answering",
        description="Visual understanding and question answering evaluation",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="visual_qa",
            primary_metric="accuracy",
        ),
    ),
    
    "needle_in_haystack": Domain(
        name="needle_in_haystack",
        display_name="Needle in Haystack",
        description="Long context retrieval evaluation at various depths and lengths",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="needle_in_haystack",
            primary_metric="retrieval_accuracy",
        ),
    ),
    
    "code_execution": Domain(
        name="code_execution",
        display_name="Code Execution",
        description="Code generation evaluation with sandbox execution",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="code_completion",
            primary_metric="tests_pass",
        ),
    ),
    
    "long_document_qa": Domain(
        name="long_document_qa",
        display_name="Long Document QA",
        description="Question answering over long documents",
        settings=DomainSettings(
            evaluation_type="ground_truth",
            ground_truth_domain="long_document_qa",
            primary_metric="accuracy",
        ),
    ),
}


def get_template(template_name: str) -> Optional[Domain]:
    """Get a domain template by name."""
    return DOMAIN_TEMPLATES.get(template_name)


def list_templates() -> List[str]:
    """List available domain templates."""
    return list(DOMAIN_TEMPLATES.keys())


