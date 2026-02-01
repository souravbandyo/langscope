"""
Match Router for LangScope.

Routes matches to the appropriate workflow based on domain evaluation type:
- Subjective domains use MultiPlayerMatchWorkflow (LLM-as-judge)
- Ground truth domains use GroundTruthMatchWorkflow (metrics-based)
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
from dataclasses import dataclass

from langscope.domain.domain_config import Domain, DOMAIN_TEMPLATES
from langscope.domain.domain_manager import get_domain

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.database.mongodb import MongoDB
    from langscope.federation.workflow import MatchResult
    from langscope.ground_truth.workflow import GroundTruthMatchResult

logger = logging.getLogger(__name__)


@dataclass
class MatchRouterConfig:
    """Configuration for the match router."""
    
    # Whether to fallback to subjective evaluation if GT fails
    fallback_to_subjective: bool = False
    
    # Enable 10D ratings for subjective matches
    enable_10d: bool = True
    
    # Ground truth evaluation mode
    gt_evaluation_mode: str = "metrics_only"


class MatchRouter:
    """
    Routes matches to the appropriate evaluation workflow.
    
    Determines whether to use:
    1. MultiPlayerMatchWorkflow - For subjective LLM-as-judge evaluation
    2. GroundTruthMatchWorkflow - For ground truth metrics-based evaluation
    """
    
    def __init__(
        self,
        db: 'MongoDB' = None,
        llm_caller: Any = None,
        config: MatchRouterConfig = None
    ):
        """
        Initialize the match router.
        
        Args:
            db: Database instance
            llm_caller: LLM API caller (e.g., LiteLLM)
            config: Router configuration
        """
        self.db = db
        self.llm_caller = llm_caller
        self.config = config or MatchRouterConfig()
        
        # Cache for domain configurations
        self._domain_cache: Dict[str, Domain] = {}
    
    def get_evaluation_type(self, domain_name: str) -> str:
        """
        Get the evaluation type for a domain.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            "subjective" or "ground_truth"
        """
        domain = self._get_domain_config(domain_name)
        if domain:
            return domain.settings.evaluation_type
        return "subjective"
    
    def _get_domain_config(self, domain_name: str) -> Optional[Domain]:
        """Get domain configuration, with caching."""
        if domain_name in self._domain_cache:
            return self._domain_cache[domain_name]
        
        # Check templates first
        if domain_name in DOMAIN_TEMPLATES:
            domain = DOMAIN_TEMPLATES[domain_name]
            self._domain_cache[domain_name] = domain
            return domain
        
        # Try to get from domain manager
        domain = get_domain(domain_name)
        if domain:
            self._domain_cache[domain_name] = domain
            return domain
        
        return None
    
    async def route_match(
        self,
        domain: str,
        models: List['LLMModel'],
        **kwargs
    ) -> Optional[Union['MatchResult', 'GroundTruthMatchResult']]:
        """
        Route a match to the appropriate workflow and execute.
        
        Args:
            domain: Domain for evaluation
            models: List of models to evaluate
            **kwargs: Additional arguments passed to workflow
            
        Returns:
            MatchResult or GroundTruthMatchResult
        """
        evaluation_type = self.get_evaluation_type(domain)
        
        logger.info(f"Routing match for domain '{domain}' to {evaluation_type} workflow")
        
        if evaluation_type == "ground_truth":
            return await self._run_ground_truth_match(domain, models, **kwargs)
        else:
            return await self._run_subjective_match(domain, models, **kwargs)
    
    async def _run_subjective_match(
        self,
        domain: str,
        models: List['LLMModel'],
        **kwargs
    ) -> Optional['MatchResult']:
        """Run a subjective (LLM-as-judge) match."""
        from langscope.federation.workflow import MultiPlayerMatchWorkflow
        
        workflow = MultiPlayerMatchWorkflow(
            domain=domain,
            models=models,
            db=self.db,
            llm_caller=self.llm_caller,
            enable_10d=self.config.enable_10d
        )
        
        return await workflow.run_single_match()
    
    async def _run_ground_truth_match(
        self,
        domain: str,
        models: List['LLMModel'],
        sample_id: str = None,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> Optional['GroundTruthMatchResult']:
        """Run a ground truth (metrics-based) match."""
        from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
        from langscope.ground_truth.judge import EvaluationMode
        
        # Get GT domain from domain config
        domain_config = self._get_domain_config(domain)
        gt_domain = domain
        if domain_config and domain_config.settings.ground_truth_domain:
            gt_domain = domain_config.settings.ground_truth_domain
        
        # Determine evaluation mode
        eval_mode = EvaluationMode.METRICS_ONLY
        if self.config.gt_evaluation_mode == "hybrid":
            eval_mode = EvaluationMode.HYBRID
        elif self.config.gt_evaluation_mode == "llm_only":
            eval_mode = EvaluationMode.LLM_ONLY
        
        workflow = GroundTruthMatchWorkflow(
            domain=gt_domain,
            models=models,
            db=self.db,
            llm_caller=self.llm_caller,
            evaluation_mode=eval_mode
        )
        
        result = await workflow.run_single_match(
            model_ids=[m.model_id for m in models],
            sample_id=sample_id,
            filters=filters
        )
        
        # If GT match failed and fallback is enabled, try subjective
        if result is None and self.config.fallback_to_subjective:
            logger.warning(f"GT match failed for {domain}, falling back to subjective")
            return await self._run_subjective_match(domain, models, **kwargs)
        
        return result
    
    async def run_tournament(
        self,
        domain: str,
        models: List['LLMModel'],
        n_rounds: int = 10
    ) -> List[Union['MatchResult', 'GroundTruthMatchResult']]:
        """
        Run a tournament of matches.
        
        Args:
            domain: Domain for evaluation
            models: Models to evaluate
            n_rounds: Number of rounds
            
        Returns:
            List of match results
        """
        results = []
        
        for round_num in range(n_rounds):
            logger.info(f"Tournament round {round_num + 1}/{n_rounds} for {domain}")
            result = await self.route_match(domain, models)
            if result:
                results.append(result)
            else:
                logger.warning(f"Round {round_num + 1} produced no result")
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

_default_router: Optional[MatchRouter] = None


def get_default_router() -> MatchRouter:
    """Get or create the default match router."""
    global _default_router
    if _default_router is None:
        _default_router = MatchRouter()
    return _default_router


def set_default_router(router: MatchRouter):
    """Set the default match router."""
    global _default_router
    _default_router = router


async def route_match(
    domain: str,
    models: List['LLMModel'],
    **kwargs
) -> Optional[Union['MatchResult', 'GroundTruthMatchResult']]:
    """Route a match using the default router."""
    return await get_default_router().route_match(domain, models, **kwargs)


def get_evaluation_type(domain: str) -> str:
    """Get evaluation type for a domain using the default router."""
    return get_default_router().get_evaluation_type(domain)


def is_ground_truth_domain(domain: str) -> bool:
    """Check if a domain uses ground truth evaluation."""
    return get_evaluation_type(domain) == "ground_truth"


def is_subjective_domain(domain: str) -> bool:
    """Check if a domain uses subjective evaluation."""
    return get_evaluation_type(domain) == "subjective"


def list_ground_truth_domains() -> List[str]:
    """List all domains that use ground truth evaluation."""
    gt_domains = []
    for name, domain in DOMAIN_TEMPLATES.items():
        if domain.settings.evaluation_type == "ground_truth":
            gt_domains.append(name)
    return gt_domains


def list_subjective_domains() -> List[str]:
    """List all domains that use subjective evaluation."""
    subjective_domains = []
    for name, domain in DOMAIN_TEMPLATES.items():
        if domain.settings.evaluation_type == "subjective":
            subjective_domains.append(name)
    return subjective_domains


