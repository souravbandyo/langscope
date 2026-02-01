"""
Monitoring module for LangScope.

Provides:
- Dashboard data aggregation
- Sample coverage monitoring
- Error rate tracking
- Leaderboard freshness alerts
"""

from langscope.monitoring.dashboard import (
    DashboardData,
    DashboardAggregator,
    get_dashboard_data,
)
from langscope.monitoring.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    LeaderboardFreshnessAlert,
    ErrorRateAlert,
    CoverageAlert,
)

__all__ = [
    "DashboardData",
    "DashboardAggregator",
    "get_dashboard_data",
    "Alert",
    "AlertLevel",
    "AlertManager",
    "LeaderboardFreshnessAlert",
    "ErrorRateAlert",
    "CoverageAlert",
]


