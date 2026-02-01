"""
LLMModel class with 10-dimensional TrueSkill tracking.

Supports:
- DualTrueSkill (raw + cost-adjusted) for backward compatibility
- MultiDimensionalTrueSkill (10 dimensions) for full evaluation

No ELO - using TrueSkill + Plackett-Luce exclusively for multi-player ranking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from langscope.core.rating import (
    TrueSkillRating,
    DualTrueSkill,
    MultiDimensionalTrueSkill,
    DIMENSION_NAMES,
)
from langscope.core.constants import (
    TRUESKILL_MU_0,
    STRATA_THRESHOLDS,
    MAX_MATCHES_PER_MODEL,
)


@dataclass
class PerformanceMetrics:
    """Performance tracking for a model."""
    total_matches_played: int = 0
    total_races_participated: int = 0  # Multi-player races
    avg_rank_raw: float = 0.0
    avg_rank_cost: float = 0.0
    rank_history_raw: List[int] = field(default_factory=list)
    rank_history_cost: List[int] = field(default_factory=list)
    # 10D rank tracking
    rank_history_by_dimension: Dict[str, List[int]] = field(default_factory=dict)
    avg_rank_by_dimension: Dict[str, float] = field(default_factory=dict)
    total_tokens_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    # Additional metrics for 10D
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    latency_history: List[float] = field(default_factory=list)
    ttft_history: List[float] = field(default_factory=list)
    
    def update_ranks(self, raw_rank: int, cost_rank: int):
        """Update rank history and averages."""
        self.rank_history_raw.append(raw_rank)
        self.rank_history_cost.append(cost_rank)
        
        n = len(self.rank_history_raw)
        self.avg_rank_raw = sum(self.rank_history_raw) / n
        self.avg_rank_cost = sum(self.rank_history_cost) / n
    
    def update_dimension_rank(self, dimension: str, rank: int):
        """Update rank history for a specific dimension."""
        if dimension not in self.rank_history_by_dimension:
            self.rank_history_by_dimension[dimension] = []
        
        self.rank_history_by_dimension[dimension].append(rank)
        n = len(self.rank_history_by_dimension[dimension])
        self.avg_rank_by_dimension[dimension] = (
            sum(self.rank_history_by_dimension[dimension]) / n
        )
    
    def update_latency(self, latency_ms: float, ttft_ms: float = None):
        """Update latency metrics."""
        self.latency_history.append(latency_ms)
        self.avg_latency_ms = sum(self.latency_history) / len(self.latency_history)
        
        if ttft_ms is not None:
            self.ttft_history.append(ttft_ms)
            self.avg_ttft_ms = sum(self.ttft_history) / len(self.ttft_history)
    
    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float
    ):
        """Add token usage and cost."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens_used += input_tokens + output_tokens
        self.total_cost_usd += cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_matches_played": self.total_matches_played,
            "total_races_participated": self.total_races_participated,
            "avg_rank_raw": self.avg_rank_raw,
            "avg_rank_cost": self.avg_rank_cost,
            "rank_history_raw": self.rank_history_raw,
            "rank_history_cost": self.rank_history_cost,
            "rank_history_by_dimension": self.rank_history_by_dimension,
            "avg_rank_by_dimension": self.avg_rank_by_dimension,
            "total_tokens_used": self.total_tokens_used,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_ttft_ms": self.avg_ttft_ms,
            "latency_history": self.latency_history,
            "ttft_history": self.ttft_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(
            total_matches_played=data.get("total_matches_played", 0),
            total_races_participated=data.get("total_races_participated", 0),
            avg_rank_raw=data.get("avg_rank_raw", 0.0),
            avg_rank_cost=data.get("avg_rank_cost", 0.0),
            rank_history_raw=data.get("rank_history_raw", []),
            rank_history_cost=data.get("rank_history_cost", []),
            rank_history_by_dimension=data.get("rank_history_by_dimension", {}),
            avg_rank_by_dimension=data.get("avg_rank_by_dimension", {}),
            total_tokens_used=data.get("total_tokens_used", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            avg_ttft_ms=data.get("avg_ttft_ms", 0.0),
            latency_history=data.get("latency_history", []),
            ttft_history=data.get("ttft_history", []),
        )


@dataclass
class MatchIds:
    """Track match participation across roles."""
    played: List[str] = field(default_factory=list)
    judged: List[str] = field(default_factory=list)
    cases_generated: List[str] = field(default_factory=list)
    questions_generated: List[str] = field(default_factory=list)
    
    def add_played(self, match_id: str):
        """Record a match played."""
        if match_id not in self.played:
            self.played.append(match_id)
    
    def add_judged(self, match_id: str):
        """Record a match judged."""
        if match_id not in self.judged:
            self.judged.append(match_id)
    
    def add_case_generated(self, match_id: str):
        """Record a case generated."""
        if match_id not in self.cases_generated:
            self.cases_generated.append(match_id)
    
    def add_question_generated(self, match_id: str):
        """Record a question generated."""
        if match_id not in self.questions_generated:
            self.questions_generated.append(match_id)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary."""
        return {
            "played": self.played,
            "judged": self.judged,
            "cases_generated": self.cases_generated,
            "questions_generated": self.questions_generated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchIds':
        """Create from dictionary."""
        return cls(
            played=data.get("played", []),
            judged=data.get("judged", []),
            cases_generated=data.get("cases_generated", []),
            questions_generated=data.get("questions_generated", []),
        )


class LLMModel:
    """
    LLM Model with dual TrueSkill rating and multi-domain support.
    
    Uses TrueSkill + Plackett-Luce for multi-player ranking (5-6 players).
    Maintains separate raw and cost-adjusted ratings for fair comparison.
    """
    
    def __init__(
        self, 
        name: str, 
        model_id: str, 
        provider: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
        pricing_source: str = "",
        max_matches: int = MAX_MATCHES_PER_MODEL,
        api_key: Optional[str] = None,
        initialize_new: bool = True
    ):
        """
        Initialize an LLM model.
        
        Args:
            name: Human-readable model name
            model_id: Provider-specific model identifier
            provider: Provider name (openai, anthropic, etc.)
            input_cost_per_million: Cost per million input tokens
            output_cost_per_million: Cost per million output tokens
            pricing_source: Source of pricing information
            max_matches: Maximum matches this model can participate in
            api_key: Optional API key for this model
            initialize_new: Whether to initialize with default values
        """
        # Basic model info
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.pricing_source = pricing_source
        self.api_key = api_key
        self.max_matches = max_matches
        
        if initialize_new:
            # Dual TrueSkill ratings per domain (backward compatibility)
            self.trueskill_by_domain: Dict[str, DualTrueSkill] = {}
            
            # Default domain TrueSkill (backward compatibility)
            self.trueskill = DualTrueSkill()
            
            # 10-Dimensional TrueSkill ratings per domain
            self.multi_trueskill_by_domain: Dict[str, MultiDimensionalTrueSkill] = {}
            
            # Default 10D TrueSkill
            self.multi_trueskill = MultiDimensionalTrueSkill()
            
            # Performance metrics per domain
            self.performance_by_domain: Dict[str, PerformanceMetrics] = {}
            self.performance = PerformanceMetrics()
            
            # Match tracking per domain
            self.match_ids_by_domain: Dict[str, MatchIds] = {}
            self.match_ids = MatchIds()
            
            # Metadata
            self.metadata: Dict[str, Any] = {
                "notes": "",
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "domains_evaluated": [],
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
    
    def get_domain_trueskill(self, domain: str) -> DualTrueSkill:
        """
        Get TrueSkill ratings for a specific domain.
        
        Creates a new default rating if domain not seen before.
        
        Args:
            domain: Domain name
        
        Returns:
            DualTrueSkill for the domain
        """
        if domain not in self.trueskill_by_domain:
            self.trueskill_by_domain[domain] = DualTrueSkill()
            if domain not in self.metadata.get("domains_evaluated", []):
                self.metadata.setdefault("domains_evaluated", []).append(domain)
        return self.trueskill_by_domain[domain]
    
    def set_domain_trueskill(
        self,
        domain: str, 
        raw_mu: float,
        raw_sigma: float,
        cost_mu: float,
        cost_sigma: float
    ):
        """
        Set TrueSkill ratings for a domain.
        
        Args:
            domain: Domain name
            raw_mu: Raw rating mean
            raw_sigma: Raw rating uncertainty
            cost_mu: Cost-adjusted rating mean
            cost_sigma: Cost-adjusted rating uncertainty
        """
        self.trueskill_by_domain[domain] = DualTrueSkill(
            raw=TrueSkillRating(raw_mu, raw_sigma),
            cost_adjusted=TrueSkillRating(cost_mu, cost_sigma)
        )
        if domain not in self.metadata.get("domains_evaluated", []):
            self.metadata.setdefault("domains_evaluated", []).append(domain)
        self.metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # =========================================================================
    # 10-Dimensional TrueSkill Methods
    # =========================================================================
    
    def get_domain_multi_trueskill(self, domain: str) -> MultiDimensionalTrueSkill:
        """
        Get 10-dimensional TrueSkill ratings for a specific domain.
        
        Creates a new default rating if domain not seen before.
        
        Args:
            domain: Domain name
        
        Returns:
            MultiDimensionalTrueSkill for the domain
        """
        if domain not in self.multi_trueskill_by_domain:
            self.multi_trueskill_by_domain[domain] = MultiDimensionalTrueSkill()
            if domain not in self.metadata.get("domains_evaluated", []):
                self.metadata.setdefault("domains_evaluated", []).append(domain)
        return self.multi_trueskill_by_domain[domain]
    
    def set_domain_multi_trueskill(
        self,
        domain: str,
        multi_ts: MultiDimensionalTrueSkill
    ):
        """
        Set 10-dimensional TrueSkill ratings for a domain.
        
        Args:
            domain: Domain name
            multi_ts: MultiDimensionalTrueSkill to set
        """
        self.multi_trueskill_by_domain[domain] = multi_ts
        if domain not in self.metadata.get("domains_evaluated", []):
            self.metadata.setdefault("domains_evaluated", []).append(domain)
        self.metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    def get_dimension_rating(
        self,
        dimension: str,
        domain: str = None
    ) -> TrueSkillRating:
        """
        Get TrueSkill rating for a specific dimension.
        
        Args:
            dimension: Dimension name
            domain: Optional domain name (uses global if None)
        
        Returns:
            TrueSkillRating for the dimension
        """
        if domain and domain in self.multi_trueskill_by_domain:
            return self.multi_trueskill_by_domain[domain].get_dimension(dimension)
        return self.multi_trueskill.get_dimension(dimension)
    
    def set_dimension_rating(
        self,
        dimension: str,
        mu: float,
        sigma: float,
        domain: str = None
    ):
        """
        Set TrueSkill rating for a specific dimension.
        
        Args:
            dimension: Dimension name
            mu: New mu value
            sigma: New sigma value
            domain: Optional domain name (uses global if None)
        """
        if domain:
            multi_ts = self.get_domain_multi_trueskill(domain)
            multi_ts.set_dimension(dimension, mu=mu, sigma=sigma)
        else:
            self.multi_trueskill.set_dimension(dimension, mu=mu, sigma=sigma)
        self.metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    def get_all_dimension_ratings(
        self,
        domain: str = None
    ) -> Dict[str, TrueSkillRating]:
        """
        Get all dimension ratings.
        
        Args:
            domain: Optional domain name
        
        Returns:
            Dictionary of dimension name -> TrueSkillRating
        """
        if domain and domain in self.multi_trueskill_by_domain:
            multi_ts = self.multi_trueskill_by_domain[domain]
        else:
            multi_ts = self.multi_trueskill
        
        return {dim: multi_ts.get_dimension(dim) for dim in DIMENSION_NAMES}
    
    def sync_dual_to_multi(self, domain: str = None):
        """
        Sync DualTrueSkill to MultiDimensionalTrueSkill.
        
        Copies raw and cost_adjusted from dual to multi ratings.
        Useful after updating dual ratings.
        
        Args:
            domain: Optional domain name
        """
        if domain:
            if domain in self.trueskill_by_domain:
                dual = self.trueskill_by_domain[domain]
                multi = self.get_domain_multi_trueskill(domain)
                multi.raw_quality = TrueSkillRating(mu=dual.raw.mu, sigma=dual.raw.sigma)
                multi.cost_adjusted = TrueSkillRating(
                    mu=dual.cost_adjusted.mu,
                    sigma=dual.cost_adjusted.sigma
                )
                multi.update_combined()
        else:
            self.multi_trueskill.raw_quality = TrueSkillRating(
                mu=self.trueskill.raw.mu,
                sigma=self.trueskill.raw.sigma
            )
            self.multi_trueskill.cost_adjusted = TrueSkillRating(
                mu=self.trueskill.cost_adjusted.mu,
                sigma=self.trueskill.cost_adjusted.sigma
            )
            self.multi_trueskill.update_combined()
    
    def sync_multi_to_dual(self, domain: str = None):
        """
        Sync MultiDimensionalTrueSkill to DualTrueSkill.
        
        Copies raw_quality and cost_adjusted from multi to dual ratings.
        Useful for backward compatibility.
        
        Args:
            domain: Optional domain name
        """
        if domain:
            if domain in self.multi_trueskill_by_domain:
                multi = self.multi_trueskill_by_domain[domain]
                self.trueskill_by_domain[domain] = multi.to_dual()
        else:
            self.trueskill = self.multi_trueskill.to_dual()
    
    def get_domain_performance(self, domain: str) -> PerformanceMetrics:
        """Get performance metrics for a domain."""
        if domain not in self.performance_by_domain:
            self.performance_by_domain[domain] = PerformanceMetrics()
        return self.performance_by_domain[domain]
    
    def get_domain_match_ids(self, domain: str) -> MatchIds:
        """Get match IDs for a domain."""
        if domain not in self.match_ids_by_domain:
            self.match_ids_by_domain[domain] = MatchIds()
        return self.match_ids_by_domain[domain]
    
    def get_stratum(self, domain: str = None) -> int:
        """
        Get performance stratum (1-4) based on raw TrueSkill μ.
        
        Used for role assignment (judging, content creation).
        Higher strata have more weight in evaluation roles.
        
        Strata:
            4 (Elite): μ >= 1520
            3 (High): 1450 <= μ < 1520
            2 (Mid): 1400 <= μ < 1450
            1 (Low): μ < 1400
        
        Args:
            domain: Optional domain name (uses global if None)
        
        Returns:
            Stratum number (1-4)
        """
        if domain and domain in self.trueskill_by_domain:
            mu = self.trueskill_by_domain[domain].raw.mu
        else:
            mu = self.trueskill.raw.mu
        
        if mu >= STRATA_THRESHOLDS["elite"]:
            return 4  # Elite
        elif mu >= STRATA_THRESHOLDS["high"]:
            return 3  # High
        elif mu >= STRATA_THRESHOLDS["mid"]:
            return 2  # Mid
        else:
            return 1  # Low
    
    def get_match_count(self, domain: str = None) -> int:
        """
        Get number of matches played.
        
        Args:
            domain: Optional domain name (uses global if None)
        
        Returns:
            Number of matches played
        """
        if domain and domain in self.performance_by_domain:
            return self.performance_by_domain[domain].total_matches_played
        return self.performance.total_matches_played
    
    def can_participate(self, domain: str = None) -> bool:
        """
        Check if model can participate in more matches.
        
        Args:
            domain: Optional domain name
        
        Returns:
            True if under match cap
        """
        return self.get_match_count(domain) < self.max_matches
    
    def calculate_response_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a response.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "provider": self.provider,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "pricing_source": self.pricing_source,
            "max_matches": self.max_matches,
            # Dual TrueSkill (backward compatibility)
            "trueskill": self.trueskill.to_dict(),
            "trueskill_by_domain": {
                domain: ts.to_dict()
                for domain, ts in self.trueskill_by_domain.items()
            },
            # 10-Dimensional TrueSkill
            "multi_trueskill": self.multi_trueskill.to_dict(),
            "multi_trueskill_by_domain": {
                domain: mts.to_dict()
                for domain, mts in self.multi_trueskill_by_domain.items()
            },
            "performance": self.performance.to_dict(),
            "performance_by_domain": {
                domain: perf.to_dict() 
                for domain, perf in self.performance_by_domain.items()
            },
            "match_ids": self.match_ids.to_dict(),
            "match_ids_by_domain": {
                domain: ids.to_dict() 
                for domain, ids in self.match_ids_by_domain.items()
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMModel':
        """Create from MongoDB document."""
        model = cls(
            name=data["name"],
            model_id=data["model_id"],
            provider=data["provider"],
            input_cost_per_million=data.get("input_cost_per_million", 0),
            output_cost_per_million=data.get("output_cost_per_million", 0),
            pricing_source=data.get("pricing_source", ""),
            max_matches=data.get("max_matches", MAX_MATCHES_PER_MODEL),
            initialize_new=False
        )
        
        # Restore TrueSkill (dual - backward compatibility)
        model.trueskill = DualTrueSkill.from_dict(data.get("trueskill", {}))
        
        # Restore domain-specific TrueSkill (dual)
        model.trueskill_by_domain = {}
        for domain, ts_dict in data.get("trueskill_by_domain", {}).items():
            model.trueskill_by_domain[domain] = DualTrueSkill.from_dict(ts_dict)
        
        # Restore 10D TrueSkill
        if "multi_trueskill" in data:
            model.multi_trueskill = MultiDimensionalTrueSkill.from_dict(
                data.get("multi_trueskill", {})
            )
        else:
            # Migrate from dual if no 10D data
            model.multi_trueskill = MultiDimensionalTrueSkill.from_dual(model.trueskill)
        
        # Restore domain-specific 10D TrueSkill
        model.multi_trueskill_by_domain = {}
        for domain, mts_dict in data.get("multi_trueskill_by_domain", {}).items():
            model.multi_trueskill_by_domain[domain] = MultiDimensionalTrueSkill.from_dict(mts_dict)
        
        # Migrate domains that only have dual ratings
        for domain in model.trueskill_by_domain:
            if domain not in model.multi_trueskill_by_domain:
                dual = model.trueskill_by_domain[domain]
                model.multi_trueskill_by_domain[domain] = MultiDimensionalTrueSkill.from_dual(dual)
        
        # Restore performance
        model.performance = PerformanceMetrics.from_dict(
            data.get("performance", {})
        )
        model.performance_by_domain = {}
        for domain, perf_dict in data.get("performance_by_domain", {}).items():
            model.performance_by_domain[domain] = PerformanceMetrics.from_dict(perf_dict)
        
        # Restore match IDs
        model.match_ids = MatchIds.from_dict(data.get("match_ids", {}))
        model.match_ids_by_domain = {}
        for domain, ids_dict in data.get("match_ids_by_domain", {}).items():
            model.match_ids_by_domain[domain] = MatchIds.from_dict(ids_dict)
        
        # Restore metadata
        model.metadata = data.get("metadata", {
            "notes": "",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "domains_evaluated": []
        })
        
        return model
    
    def __repr__(self) -> str:
        return (
            f"LLMModel(name='{self.name}', provider='{self.provider}', "
            f"μ_raw={self.trueskill.raw.mu:.1f})"
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LLMModel):
            return False
        return self.model_id == other.model_id
    
    def __hash__(self) -> int:
        return hash(self.model_id)


