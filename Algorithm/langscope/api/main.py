"""
LangScope API - Main FastAPI Application.

Multi-domain LLM Evaluation Framework REST API.

Provides endpoints for:
- Model management (CRUD)
- Domain management (CRUD)
- Match execution and results
- Leaderboard retrieval (raw and cost-adjusted)
- Transfer learning operations
- Specialist detection
- Arena mode (user feedback sessions)
- Use-case recommendations
"""

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langscope.api.schemas import HealthResponse, ErrorResponse
from langscope.api.middleware import (
    AuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from langscope.api.dependencies import get_db, cleanup, cleanup_async
from langscope.api.routes import (
    auth_router,
    users_router,
    organizations_router,
    billing_router,
    models_router,
    domains_router,
    matches_router,
    leaderboard_router,
    transfer_router,
    specialists_router,
    arena_router,
    recommendations_router,
    params_router,
    base_models_router,
    deployments_router,
    self_hosted_router,
    benchmarks_router,
    prompts_router,
    cache_router,
    ground_truth_router,
    monitoring_router,
    my_models_router,
)
from langscope.graphql import get_graphql_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting LangScope API...")
    
    # Try to connect to database
    try:
        db = get_db()
        logger.info(f"Database connected: {db.db_name}")
    except RuntimeError as e:
        logger.warning(f"Database not available: {e}")
    
    # Initialize cache manager
    try:
        from langscope.api.dependencies import get_cache_manager
        cache = await get_cache_manager()
        logger.info("Cache manager initialized")
    except Exception as e:
        logger.warning(f"Cache manager not available: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LangScope API...")
    await cleanup_async()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="LangScope API",
        description="""
# LangScope - Multi-domain LLM Evaluation Framework

LangScope provides a comprehensive system for evaluating and comparing LLM models
across multiple domains using TrueSkill + Plackett-Luce ranking algorithms.

## Key Features

- **Multi-player Matches**: 5-6 models compete in each match
- **Dual Rankings**: Both raw performance and cost-adjusted rankings
- **Transfer Learning**: Predict model performance in new domains
- **Specialist Detection**: Identify models that excel in specific domains
- **Arena Mode**: User feedback integration for personalized recommendations
- **Use-case Recommendations**: Get model suggestions based on your use case

## Authentication

LangScope uses stateless JWT authentication via Supabase.

**Primary**: Include a Bearer token in the `Authorization` header:
```
Authorization: Bearer <supabase_access_token>
```

**Legacy**: Include an API key in the `X-API-Key` header.

## Rate Limiting

Default: 100 requests per 60 seconds per API key.
Rate limit headers are included in all responses.
        """,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    # Note: allow_origins with credentials cannot use "*" - must specify origins
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:3001,http://127.0.0.1:5173")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (order matters - first added is outermost)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Include routers
    app.include_router(auth_router)
    
    # User Profile & Organization
    app.include_router(users_router)
    app.include_router(organizations_router)
    app.include_router(billing_router)
    
    app.include_router(models_router)
    app.include_router(domains_router)
    app.include_router(matches_router)
    app.include_router(leaderboard_router)
    app.include_router(transfer_router)
    app.include_router(specialists_router)
    app.include_router(arena_router)
    app.include_router(recommendations_router)
    app.include_router(params_router)
    # Phase 11: Multi-provider architecture
    app.include_router(base_models_router)
    app.include_router(deployments_router)
    app.include_router(self_hosted_router)
    # Phase 12: External benchmarks
    app.include_router(benchmarks_router)
    
    # Phase PM: Prompt Management
    app.include_router(prompts_router)
    
    # Phase C: Cache Management
    app.include_router(cache_router)
    
    # Ground Truth & Monitoring
    app.include_router(ground_truth_router)
    app.include_router(monitoring_router)
    
    # My Models (Private Testing)
    app.include_router(my_models_router)
    
    # Phase 15: GraphQL API
    graphql_router = get_graphql_router()
    app.include_router(graphql_router, prefix="/graphql")
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """API root - returns basic info."""
        return {
            "name": "LangScope API",
            "version": API_VERSION,
            "description": "Multi-domain LLM Evaluation Framework",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"]
    )
    async def health_check():
        """
        Health check endpoint.
        
        Returns the API status and database connection state.
        """
        try:
            db = get_db()
            db_connected = db.connected
        except RuntimeError:
            db_connected = False
        
        return HealthResponse(
            status="healthy",
            version=API_VERSION,
            database_connected=db_connected,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if os.getenv("DEBUG") else None,
                "code": "INTERNAL_ERROR"
            }
        )
    
    return app


# Create the app instance
app = create_app()


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "langscope.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
