"""
Domain module for LangScope.

Manages evaluation domains with custom configurations, prompts, and settings.
"""

from langscope.domain.domain_config import (
    DomainPrompts,
    DomainSettings,
    Domain,
)
from langscope.domain.domain_manager import (
    DomainManager,
    create_domain,
    get_domain,
    list_domains,
)

__all__ = [
    "DomainPrompts",
    "DomainSettings",
    "Domain",
    "DomainManager",
    "create_domain",
    "get_domain",
    "list_domains",
]


