"""
Strata system for peer-federated evaluation.

Models are assigned to strata based on their TrueSkill μ rating.
Higher-stratum models have more weight in evaluation roles.

Strata:
- 4 (Elite): μ >= 1520 - Can judge, create content, high weight
- 3 (High): μ >= 1450 - Can judge, create content
- 2 (Mid): μ >= 1400 - Can judge with lower weight
- 1 (Low): μ < 1400 - Competitors only
"""

from typing import List, Optional, TYPE_CHECKING

# Import constants as fallback defaults
from langscope.core.constants import STRATA_THRESHOLDS, STRATUM_NAMES

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.config.params import StrataParams


def _get_strata_params(domain: str = None) -> 'StrataParams':
    """
    Get strata params from ParameterManager or use defaults.
    
    Args:
        domain: Optional domain for domain-specific params
    
    Returns:
        StrataParams instance
    """
    try:
        from langscope.config.params import get_parameter_manager
        return get_parameter_manager().get_strata_params(domain)
    except ImportError:
        # Fallback to constants
        from langscope.config.params.models import StrataParams
        return StrataParams(
            elite_threshold=float(STRATA_THRESHOLDS.get("elite", 1520)),
            high_threshold=float(STRATA_THRESHOLDS.get("high", 1450)),
            mid_threshold=float(STRATA_THRESHOLDS.get("mid", 1400)),
            strata_names=STRATUM_NAMES
        )


def get_stratum(mu: float, domain: str = None) -> int:
    """
    Get stratum number based on TrueSkill μ.
    
    Args:
        mu: TrueSkill mean rating
        domain: Optional domain for domain-specific thresholds
    
    Returns:
        Stratum number (1-4)
    """
    params = _get_strata_params(domain)
    return params.get_stratum(mu)


def get_stratum_name(stratum: int, domain: str = None) -> str:
    """
    Get human-readable stratum name.
    
    Args:
        stratum: Stratum number (1-4)
        domain: Optional domain for domain-specific names
    
    Returns:
        Stratum name ("low", "mid", "high", "elite")
    """
    params = _get_strata_params(domain)
    return params.strata_names.get(stratum, "unknown")


def get_stratum_threshold(stratum: int, domain: str = None) -> float:
    """
    Get minimum μ for a stratum.
    
    Args:
        stratum: Stratum number (1-4)
        domain: Optional domain for domain-specific thresholds
    
    Returns:
        Minimum μ value
    """
    params = _get_strata_params(domain)
    if stratum == 4:
        return params.elite_threshold
    elif stratum == 3:
        return params.high_threshold
    elif stratum == 2:
        return params.mid_threshold
    return 0.0


def get_eligible_judges(
    models: List['LLMModel'],
    domain: str = None,
    min_stratum: int = 2,
    exclude: List[str] = None
) -> List['LLMModel']:
    """
    Get models eligible to serve as judges.
    
    Args:
        models: List of all models
        domain: Domain for evaluation (None for global)
        min_stratum: Minimum stratum to be a judge
        exclude: Model names to exclude (e.g., competitors in match)
    
    Returns:
        List of eligible judge models
    """
    exclude = exclude or []
    eligible = []
    
    for model in models:
        if model.name in exclude:
            continue
        
        stratum = model.get_stratum(domain)
        if stratum >= min_stratum:
            eligible.append(model)
    
    return eligible


def get_eligible_creators(
    models: List['LLMModel'],
    domain: str = None,
    min_stratum: int = 3,
    exclude: List[str] = None
) -> List['LLMModel']:
    """
    Get models eligible to create content (cases, questions).
    
    Content creation requires higher capability, so we typically
    require stratum 3+ (High or Elite).
    
    Args:
        models: List of all models
        domain: Domain for evaluation
        min_stratum: Minimum stratum to create content
        exclude: Model names to exclude
    
    Returns:
        List of eligible content creator models
    """
    exclude = exclude or []
    eligible = []
    
    for model in models:
        if model.name in exclude:
            continue
        
        stratum = model.get_stratum(domain)
        if stratum >= min_stratum:
            eligible.append(model)
    
    return eligible


def get_stratum_distribution(
    models: List['LLMModel'],
    domain: str = None
) -> dict:
    """
    Get distribution of models across strata.
    
    Args:
        models: List of models
        domain: Domain for evaluation
    
    Returns:
        Dictionary {stratum: count}
    """
    distribution = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for model in models:
        stratum = model.get_stratum(domain)
        distribution[stratum] = distribution.get(stratum, 0) + 1
    
    return distribution


def can_serve_as_judge(
    model: 'LLMModel',
    domain: str = None,
    min_stratum: int = 2
) -> bool:
    """
    Check if a model can serve as a judge.
    
    Args:
        model: Model to check
        domain: Domain for evaluation
        min_stratum: Minimum required stratum
    
    Returns:
        True if model can judge
    """
    return model.get_stratum(domain) >= min_stratum


def can_create_content(
    model: 'LLMModel',
    domain: str = None,
    min_stratum: int = 3
) -> bool:
    """
    Check if a model can create content.
    
    Args:
        model: Model to check
        domain: Domain for evaluation
        min_stratum: Minimum required stratum
    
    Returns:
        True if model can create content
    """
    return model.get_stratum(domain) >= min_stratum


def calculate_stratum_weight(stratum: int) -> float:
    """
    Calculate weight factor for a stratum.
    
    Higher strata get more weight in aggregation.
    
    Args:
        stratum: Stratum number (1-4)
    
    Returns:
        Weight factor
    """
    # Exponential weighting: stratum 4 has 4x weight of stratum 1
    weights = {
        1: 0.5,
        2: 1.0,
        3: 2.0,
        4: 4.0
    }
    return weights.get(stratum, 1.0)


