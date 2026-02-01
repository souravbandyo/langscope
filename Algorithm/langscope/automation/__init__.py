"""
Automation module for LangScope.

Provides:
- Data source configuration (pricing, models, benchmarks)
- Sync engine for automated data fetching
- Validation and approval workflow for changes
"""

from langscope.automation.data_sources import (
    DataSourceType,
    DataSource,
    SourceConfig,
    AutomationConfig,
    ValidationConfig,
    PREDEFINED_SOURCES,
)
from langscope.automation.sync import (
    SyncEngine,
    SyncResult,
    SyncStatus,
)
from langscope.automation.validation import (
    ChangeType,
    PendingChange,
    validate_price_change,
    check_requires_approval,
)

__all__ = [
    # Data Sources
    "DataSourceType",
    "DataSource",
    "SourceConfig",
    "AutomationConfig",
    "ValidationConfig",
    "PREDEFINED_SOURCES",
    # Sync
    "SyncEngine",
    "SyncResult",
    "SyncStatus",
    # Validation
    "ChangeType",
    "PendingChange",
    "validate_price_change",
    "check_requires_approval",
]


