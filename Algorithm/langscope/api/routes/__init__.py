"""
API Routes for LangScope.

Organized by resource type:
- auth: Authentication and user info
- users: User profile management
- organizations: Organization and team management
- billing: Subscription and billing
- models: Model CRUD operations
- domains: Domain CRUD operations
- matches: Match execution and results
- leaderboard: Rankings and leaderboards
- transfer: Transfer learning and correlations
- specialists: Specialist detection
- arena: User feedback sessions
- recommendations: Use-case recommendations
- params: Parameter management
- base_models: Base model CRUD (Phase 11)
- deployments: Model deployment CRUD (Phase 11)
- self_hosted: Self-hosted deployment CRUD (Phase 11)
- benchmarks: External benchmarks (Phase 12)
- prompts: Prompt classification and processing (Phase PM)
- cache: Cache management and sessions (Phase C)
"""

from langscope.api.routes.auth import router as auth_router
from langscope.api.routes.users import router as users_router
from langscope.api.routes.organizations import router as organizations_router
from langscope.api.routes.billing import router as billing_router
from langscope.api.routes.models import router as models_router
from langscope.api.routes.domains import router as domains_router
from langscope.api.routes.matches import router as matches_router
from langscope.api.routes.leaderboard import router as leaderboard_router
from langscope.api.routes.transfer import router as transfer_router
from langscope.api.routes.specialists import router as specialists_router
from langscope.api.routes.arena import router as arena_router
from langscope.api.routes.recommendations import router as recommendations_router
from langscope.api.routes.params import router as params_router
from langscope.api.routes.base_models import router as base_models_router
from langscope.api.routes.deployments import router as deployments_router
from langscope.api.routes.self_hosted import router as self_hosted_router
from langscope.api.routes.benchmarks import router as benchmarks_router
from langscope.api.routes.ground_truth import router as ground_truth_router
from langscope.api.routes.monitoring import router as monitoring_router
from langscope.api.routes.prompts import router as prompts_router
from langscope.api.routes.cache import router as cache_router
from langscope.api.routes.my_models import router as my_models_router

__all__ = [
    "auth_router",
    "users_router",
    "organizations_router",
    "billing_router",
    "models_router",
    "domains_router", 
    "matches_router",
    "leaderboard_router",
    "transfer_router",
    "specialists_router",
    "arena_router",
    "recommendations_router",
    "params_router",
    "base_models_router",
    "deployments_router",
    "self_hosted_router",
    "benchmarks_router",
    "ground_truth_router",
    "monitoring_router",
    "prompts_router",
    "cache_router",
    "my_models_router",
]
