"""
GraphQL schema definition for LangScope.

Combines queries, mutations, and subscriptions into a unified schema.
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import Optional, Any

from langscope.graphql.queries import Query
from langscope.graphql.mutations import Mutation
from langscope.graphql.subscriptions import Subscription, pubsub


# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)


async def get_context(db=None, user=None, match_service=None) -> dict:
    """
    Build GraphQL context with dependencies.
    
    Args:
        db: MongoDB instance
        user: Current authenticated user (if any)
        match_service: Match service for triggering matches
    
    Returns:
        Context dict for resolvers
    """
    return {
        "db": db,
        "user": user or {},
        "match_service": match_service,
        "pubsub": pubsub,
    }


def get_graphql_router(
    db=None,
    get_db=None,
    get_user=None,
    get_match_service=None
) -> GraphQLRouter:
    """
    Create a FastAPI-compatible GraphQL router.
    
    Args:
        db: Static MongoDB instance (for simple setups)
        get_db: Async function to get DB from request (for DI)
        get_user: Async function to get user from request
        get_match_service: Async function to get match service
    
    Returns:
        GraphQLRouter to mount in FastAPI app
    """
    async def context_getter(request=None) -> dict:
        """Build context from request."""
        ctx_db = db
        ctx_user = None
        ctx_match_service = None
        
        if get_db:
            ctx_db = await get_db(request)
        if get_user and request:
            ctx_user = await get_user(request)
        if get_match_service:
            ctx_match_service = await get_match_service()
        
        return await get_context(
            db=ctx_db,
            user=ctx_user,
            match_service=ctx_match_service,
        )
    
    return GraphQLRouter(
        schema=schema,
        context_getter=context_getter,
        graphiql=True,  # Enable GraphiQL interface
    )


# Example queries for documentation
EXAMPLE_QUERIES = """
# Get all base models
query GetBaseModels {
  baseModels(limit: 10) {
    id
    name
    organization
    capabilities {
      modalities
      supportsVision
    }
  }
}

# Get deployments for a base model
query GetDeployments($baseModelId: String!) {
  deployments(baseModelId: $baseModelId) {
    edges {
      node {
        id
        name
        provider {
          name
          region
        }
        pricing {
          inputCostPerMillion
          outputCostPerMillion
        }
      }
    }
    pageInfo {
      hasNextPage
    }
  }
}

# Get leaderboard
query GetLeaderboard($domain: String) {
  leaderboard(domain: $domain, dimension: "raw_quality", limit: 10) {
    rank
    name
    provider
    ratingMu
    conservativeRating
    matchesPlayed
  }
}

# Create a deployment
mutation CreateDeployment {
  createDeployment(input: {
    baseModelId: "llama-3-70b"
    name: "Llama 3 70B on Together"
    provider: {
      name: "together"
      region: "us"
    }
    pricing: {
      inputCostPerMillion: 0.88
      outputCostPerMillion: 0.88
    }
  }) {
    success
    deploymentId
    message
  }
}

# Subscribe to rating updates
subscription RatingUpdates {
  ratingUpdates(domain: "legal") {
    deploymentId
    oldMu
    newMu
    matchId
  }
}
"""


