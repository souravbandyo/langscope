"""
Database module for LangScope.

Provides MongoDB integration for persistent storage of models, matches,
domains, and correlations.
"""

from langscope.database.mongodb import MongoDB, get_database
from langscope.database.schemas import (
    MODEL_SCHEMA,
    MATCH_SCHEMA,
    DOMAIN_SCHEMA,
    CORRELATION_SCHEMA,
)

__all__ = [
    "MongoDB",
    "get_database",
    "MODEL_SCHEMA",
    "MATCH_SCHEMA",
    "DOMAIN_SCHEMA",
    "CORRELATION_SCHEMA",
]


