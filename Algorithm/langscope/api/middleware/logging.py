"""
API middleware for request logging.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Request Logging
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.1f}ms"
        )
        
        return response

