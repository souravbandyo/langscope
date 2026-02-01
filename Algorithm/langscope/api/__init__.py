"""
LangScope API Layer.

Provides REST API endpoints for:
- Model management (CRUD)
- Domain management (CRUD)
- Match execution and results
- Leaderboard retrieval
- Transfer learning operations
- Specialist detection
- Arena mode (user feedback)
- Use-case recommendations
"""

from langscope.api.main import create_app, app

__all__ = ["create_app", "app"]
