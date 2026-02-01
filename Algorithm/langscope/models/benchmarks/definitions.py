"""
Benchmark definitions for external evaluations.

Defines what each benchmark measures, how it's scored, and where to get results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class BenchmarkCategory(str, Enum):
    """Benchmark categories."""
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    CHAT = "chat"
    SAFETY = "safety"
    INSTRUCTION = "instruction"


@dataclass
class BenchmarkScoring:
    """Scoring configuration for a benchmark."""
    metric: str  # "accuracy", "elo", "pass@1", "score"
    range: Tuple[float, float]  # (min, max)
    higher_is_better: bool = True
    
    # Reference thresholds
    random_baseline: float = 0.0
    human_average: float = 0.0
    human_expert: float = 0.0
    state_of_art: float = 0.0
    
    # Evaluation variants
    variants: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "range": list(self.range),
            "higher_is_better": self.higher_is_better,
            "thresholds": {
                "random_baseline": self.random_baseline,
                "human_average": self.human_average,
                "human_expert": self.human_expert,
                "state_of_art": self.state_of_art,
            },
            "variants": self.variants,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkScoring':
        """Create from dictionary."""
        thresholds = data.get("thresholds", {})
        return cls(
            metric=data.get("metric", "accuracy"),
            range=tuple(data.get("range", [0, 100])),
            higher_is_better=data.get("higher_is_better", True),
            random_baseline=thresholds.get("random_baseline", 0),
            human_average=thresholds.get("human_average", 0),
            human_expert=thresholds.get("human_expert", 0),
            state_of_art=thresholds.get("state_of_art", 0),
            variants=data.get("variants", {}),
        )


@dataclass
class BenchmarkSource:
    """Source information for a benchmark."""
    paper: str = ""
    dataset: str = ""
    leaderboard: str = ""
    github: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "paper": self.paper,
            "dataset": self.dataset,
            "leaderboard": self.leaderboard,
            "github": self.github,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'BenchmarkSource':
        """Create from dictionary."""
        return cls(
            paper=data.get("paper", ""),
            dataset=data.get("dataset", ""),
            leaderboard=data.get("leaderboard", ""),
            github=data.get("github", ""),
        )


@dataclass
class BenchmarkAutomation:
    """Automation configuration for fetching benchmark scores."""
    can_auto_fetch: bool = False
    source_id: str = ""
    update_frequency: str = "weekly"  # "daily", "weekly", "monthly"
    last_synced: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "can_auto_fetch": self.can_auto_fetch,
            "source_id": self.source_id,
            "update_frequency": self.update_frequency,
            "last_synced": self.last_synced,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkAutomation':
        """Create from dictionary."""
        return cls(
            can_auto_fetch=data.get("can_auto_fetch", False),
            source_id=data.get("source_id", ""),
            update_frequency=data.get("update_frequency", "weekly"),
            last_synced=data.get("last_synced", ""),
        )


@dataclass
class LangScopeCorrelation:
    """Correlation with LangScope ratings."""
    overall_mu: float = 0.0  # Correlation with overall TrueSkill μ
    by_domain: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    last_computed: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_mu": self.overall_mu,
            "by_domain": self.by_domain,
            "sample_size": self.sample_size,
            "last_computed": self.last_computed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LangScopeCorrelation':
        """Create from dictionary."""
        return cls(
            overall_mu=data.get("overall_mu", 0.0),
            by_domain=data.get("by_domain", {}),
            sample_size=data.get("sample_size", 0),
            last_computed=data.get("last_computed", ""),
        )


class BenchmarkDefinition:
    """
    Definition of an external benchmark.
    
    Stores what the benchmark measures, how it's scored, and where
    to get updated results.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        full_name: str = "",
        category: BenchmarkCategory = BenchmarkCategory.KNOWLEDGE,
        subcategories: List[str] = None,
        description: str = "",
        initialize_new: bool = True
    ):
        """
        Initialize a benchmark definition.
        
        Args:
            id: Unique identifier (e.g., "mmlu")
            name: Short name (e.g., "MMLU")
            full_name: Full name
            category: Primary category
            subcategories: Additional subcategories
            description: Description of what it tests
            initialize_new: Whether to initialize with defaults
        """
        self.id = id
        self.name = name
        self.full_name = full_name or name
        self.category = category
        self.subcategories = subcategories or []
        self.description = description
        
        if initialize_new:
            self.scoring = BenchmarkScoring(
                metric="accuracy",
                range=(0, 100),
            )
            self.source = BenchmarkSource()
            self.automation = BenchmarkAutomation()
            self.langscope_correlation = LangScopeCorrelation()
            
            # Task details
            self.tasks: Dict[str, Any] = {}
            
            # Metadata
            self.created_at = datetime.utcnow().isoformat() + "Z"
            self.updated_at = datetime.utcnow().isoformat() + "Z"
    
    def get_percentile(self, score: float) -> int:
        """
        Estimate percentile for a given score.
        
        Uses a simple linear interpolation between thresholds.
        
        Args:
            score: The score to convert
        
        Returns:
            Estimated percentile (0-100)
        """
        # Simple interpolation based on thresholds
        random = self.scoring.random_baseline
        sota = self.scoring.state_of_art
        
        if sota <= random:
            return 50  # Can't estimate
        
        # Normalize score to 0-100 range
        normalized = (score - random) / (sota - random) * 100
        return max(0, min(100, int(normalized)))
    
    def is_score_good(self, score: float) -> bool:
        """Check if a score is above human average."""
        return score >= self.scoring.human_average
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "category": self.category.value if isinstance(self.category, Enum) else self.category,
            "subcategories": self.subcategories,
            "description": self.description,
            "scoring": self.scoring.to_dict(),
            "source": self.source.to_dict(),
            "automation": self.automation.to_dict(),
            "langscope_correlation": self.langscope_correlation.to_dict(),
            "tasks": self.tasks,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkDefinition':
        """Create from MongoDB document."""
        category = data.get("category", "knowledge")
        if isinstance(category, str):
            try:
                category = BenchmarkCategory(category)
            except ValueError:
                category = BenchmarkCategory.KNOWLEDGE
        
        benchmark = cls(
            id=data.get("_id", ""),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            category=category,
            subcategories=data.get("subcategories", []),
            description=data.get("description", ""),
            initialize_new=False,
        )
        
        # Restore nested objects
        benchmark.scoring = BenchmarkScoring.from_dict(data.get("scoring", {}))
        benchmark.source = BenchmarkSource.from_dict(data.get("source", {}))
        benchmark.automation = BenchmarkAutomation.from_dict(data.get("automation", {}))
        benchmark.langscope_correlation = LangScopeCorrelation.from_dict(
            data.get("langscope_correlation", {})
        )
        benchmark.tasks = data.get("tasks", {})
        benchmark.created_at = data.get("created_at", "")
        benchmark.updated_at = data.get("updated_at", "")
        
        return benchmark
    
    def __repr__(self) -> str:
        return f"BenchmarkDefinition(id='{self.id}', name='{self.name}')"


# =============================================================================
# Predefined Benchmarks
# =============================================================================

def _create_mmlu() -> BenchmarkDefinition:
    """Create MMLU benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="mmlu",
        name="MMLU",
        full_name="Massive Multitask Language Understanding",
        category=BenchmarkCategory.KNOWLEDGE,
        subcategories=["academic", "multiple-choice"],
        description="Tests knowledge across 57 subjects from STEM to humanities.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="accuracy",
        range=(0, 100),
        random_baseline=25.0,
        human_average=70.0,
        human_expert=89.8,
        state_of_art=90.1,
        variants={
            "0-shot": "No examples provided",
            "5-shot": "5 examples in context",
            "cot": "Chain-of-thought prompting",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2009.03300",
        dataset="https://huggingface.co/datasets/cais/mmlu",
        leaderboard="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="open_llm_leaderboard",
        update_frequency="weekly",
    )
    benchmark.tasks = {
        "total_subjects": 57,
        "total_questions": 15908,
        "format": "multiple-choice-4",
    }
    return benchmark


def _create_chatbot_arena() -> BenchmarkDefinition:
    """Create Chatbot Arena benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="chatbot_arena",
        name="Chatbot Arena",
        full_name="LMSYS Chatbot Arena Elo Rating",
        category=BenchmarkCategory.CHAT,
        subcategories=["human-preference", "elo"],
        description="Human preference Elo from pairwise comparisons. Highest correlation with LangScope.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="elo",
        range=(800, 1400),
        random_baseline=1000.0,
        human_average=1100.0,
        human_expert=1200.0,
        state_of_art=1300.0,
        variants={
            "overall": "Overall ranking",
            "coding": "Coding-focused prompts",
            "hard_prompts": "Challenging prompts",
            "english": "English-only",
            "multi_turn": "Multi-turn conversations",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2306.05685",
        leaderboard="https://chat.lmsys.org/",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="lmsys_arena",
        update_frequency="daily",
    )
    return benchmark


def _create_humaneval() -> BenchmarkDefinition:
    """Create HumanEval benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="humaneval",
        name="HumanEval",
        full_name="OpenAI HumanEval",
        category=BenchmarkCategory.CODING,
        subcategories=["python", "function-completion"],
        description="Python function completion. Tests code generation ability.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="pass@1",
        range=(0, 100),
        random_baseline=0.0,
        human_average=60.0,
        human_expert=90.0,
        state_of_art=95.0,
        variants={
            "pass@1": "Single attempt success",
            "pass@10": "Success in 10 attempts",
            "pass@100": "Success in 100 attempts",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2107.03374",
        dataset="https://github.com/openai/human-eval",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="open_llm_leaderboard",
        update_frequency="weekly",
    )
    benchmark.tasks = {
        "total_problems": 164,
        "language": "python",
    }
    return benchmark


def _create_gsm8k() -> BenchmarkDefinition:
    """Create GSM8K benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="gsm8k",
        name="GSM8K",
        full_name="Grade School Math 8K",
        category=BenchmarkCategory.MATH,
        subcategories=["word-problems", "arithmetic"],
        description="Grade school math word problems. Tests basic reasoning.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="accuracy",
        range=(0, 100),
        random_baseline=0.0,
        human_average=85.0,
        human_expert=95.0,
        state_of_art=97.0,
        variants={
            "0-shot": "No examples",
            "8-shot-cot": "8 examples with chain-of-thought",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2110.14168",
        dataset="https://huggingface.co/datasets/gsm8k",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="open_llm_leaderboard",
        update_frequency="weekly",
    )
    benchmark.tasks = {
        "total_problems": 8500,
        "test_problems": 1319,
    }
    return benchmark


def _create_ifeval() -> BenchmarkDefinition:
    """Create IFEval benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="ifeval",
        name="IFEval",
        full_name="Instruction Following Evaluation",
        category=BenchmarkCategory.INSTRUCTION,
        subcategories=["format-following", "constraints"],
        description="Instruction following with verifiable constraints.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="accuracy",
        range=(0, 100),
        random_baseline=0.0,
        human_average=75.0,
        human_expert=95.0,
        state_of_art=90.0,
        variants={
            "strict": "Strict matching",
            "loose": "Loose matching",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2311.07911",
        dataset="https://huggingface.co/datasets/wis-k/instruction-following-eval",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="open_llm_leaderboard",
        update_frequency="weekly",
    )
    return benchmark


def _create_truthfulqa() -> BenchmarkDefinition:
    """Create TruthfulQA benchmark definition."""
    benchmark = BenchmarkDefinition(
        id="truthfulqa",
        name="TruthfulQA",
        full_name="TruthfulQA",
        category=BenchmarkCategory.SAFETY,
        subcategories=["truthfulness", "factuality"],
        description="Tests for truthfulness vs common misconceptions.",
    )
    benchmark.scoring = BenchmarkScoring(
        metric="accuracy",
        range=(0, 100),
        random_baseline=25.0,
        human_average=85.0,
        human_expert=95.0,
        state_of_art=80.0,
        variants={
            "mc1": "Multiple choice (single answer)",
            "mc2": "Multiple choice (multiple answers)",
        },
    )
    benchmark.source = BenchmarkSource(
        paper="https://arxiv.org/abs/2109.07958",
        dataset="https://huggingface.co/datasets/truthful_qa",
    )
    benchmark.automation = BenchmarkAutomation(
        can_auto_fetch=True,
        source_id="open_llm_leaderboard",
        update_frequency="weekly",
    )
    return benchmark


# Create predefined benchmarks
PREDEFINED_BENCHMARKS: Dict[str, BenchmarkDefinition] = {
    "mmlu": _create_mmlu(),
    "chatbot_arena": _create_chatbot_arena(),
    "humaneval": _create_humaneval(),
    "gsm8k": _create_gsm8k(),
    "ifeval": _create_ifeval(),
    "truthfulqa": _create_truthfulqa(),
}


def get_benchmark_definition(benchmark_id: str) -> Optional[BenchmarkDefinition]:
    """Get a benchmark definition by ID."""
    return PREDEFINED_BENCHMARKS.get(benchmark_id)


def list_benchmarks_by_category(
    category: BenchmarkCategory
) -> List[BenchmarkDefinition]:
    """List all benchmarks in a category."""
    return [
        b for b in PREDEFINED_BENCHMARKS.values()
        if b.category == category
    ]

