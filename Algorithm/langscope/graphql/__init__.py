"""
GraphQL API module for LangScope.

Provides:
- Type definitions for all entities
- Query resolvers
- Mutation resolvers
- Subscription support for real-time updates
"""

from langscope.graphql.schema import schema, get_graphql_router

__all__ = [
    "schema",
    "get_graphql_router",
]


