"""
Analytics module for LangScope.

Provides:
- Rating trend analysis
- Price impact analysis
- Time series queries
"""

from langscope.analytics.trends import (
    RatingTrend,
    TrendPoint,
    get_rating_trend,
    get_top_improvers,
    compute_volatility,
)
from langscope.analytics.price_analysis import (
    PriceChange,
    PriceImpact,
    get_price_trend,
    analyze_price_change_impact,
)

__all__ = [
    # Trends
    "RatingTrend",
    "TrendPoint",
    "get_rating_trend",
    "get_top_improvers",
    "compute_volatility",
    # Price Analysis
    "PriceChange",
    "PriceImpact",
    "get_price_trend",
    "analyze_price_change_impact",
]


