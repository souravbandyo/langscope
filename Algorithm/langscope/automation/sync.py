"""
Sync engine for automated data fetching.

Handles syncing from external data sources with change detection.

DEPRECATED: This module is deprecated. Import from langscope.models.sources instead.
This file provides backward compatibility re-exports.
"""

# Re-export from new location for backward compatibility
from langscope.models.sources import (
    SyncEngine,
    SyncResult,
    SyncStatus,
)

__all__ = [
    "SyncEngine",
    "SyncResult",
    "SyncStatus",
]
