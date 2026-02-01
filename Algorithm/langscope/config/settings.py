"""
Global configuration settings for LangScope.

Loads configuration from environment variables and provides
typed access to all system settings.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    """Global settings for LangScope."""
    
    # Database
    mongodb_uri: str = ""
    db_name: str = "langscope"
    
    # Supabase Authentication
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    
    # Collections
    models_collection: str = "models"
    matches_collection: str = "matches"
    domains_collection: str = "domains"
    correlations_collection: str = "domain_correlations"
    
    # TrueSkill parameters
    trueskill_mu_0: float = 1500.0
    trueskill_sigma_0: float = 166.0
    trueskill_beta: float = 83.0
    trueskill_tau: float = 8.3
    
    # Match configuration
    players_per_match: int = 6
    min_players: int = 5
    max_players: int = 6
    judge_count: int = 5
    max_matches_per_model: int = 50
    
    # Swiss pairing
    swiss_delta: float = 75.0
    
    # Temperature parameters
    cost_temp: float = 0.05
    rating_temp: float = 300.0
    
    # Plackett-Luce
    plackett_luce_max_iter: int = 100
    plackett_luce_tol: float = 1e-6
    
    # Transfer learning
    correlation_tau: float = 20.0
    sigma_base: float = 50.0
    specialist_z_threshold: float = 2.0
    
    # Strata thresholds
    strata_elite: float = 1520.0
    strata_high: float = 1450.0
    strata_mid: float = 1400.0
    
    # Penalties
    judge_penalty_mu: float = 10.0
    outlier_threshold: float = 0.40
    
    # User Feedback Integration
    user_weight_multiplier: float = 2.0  # α_u: user credibility multiplier
    user_weight_base: float = 1.0        # Base weight for comparison
    use_case_tau: float = 10.0           # Smoothing parameter for use-case adjustments
    judge_calibration_gamma: float = 0.2 # Weight adjustment factor for calibration
    user_surprise_threshold: float = 2.0 # Z-score threshold for specialist detection
    
    # User Feedback Collections
    user_sessions_collection: str = "user_sessions"
    use_case_adjustments_collection: str = "use_case_adjustments"
    judge_calibrations_collection: str = "judge_calibrations"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Load settings from environment variables."""
        self.mongodb_uri = os.getenv("MONGODB_URI", self.mongodb_uri)
        self.db_name = os.getenv("LANGSCOPE_DB_NAME", self.db_name)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # Supabase Authentication
        self.supabase_url = os.getenv("SUPABASE_URL", self.supabase_url)
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", self.supabase_anon_key)
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", self.supabase_service_role_key)
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET", self.supabase_jwt_secret)
        
        # Load numeric settings if provided
        if os.getenv("TRUESKILL_MU_0"):
            self.trueskill_mu_0 = float(os.getenv("TRUESKILL_MU_0"))
        if os.getenv("TRUESKILL_SIGMA_0"):
            self.trueskill_sigma_0 = float(os.getenv("TRUESKILL_SIGMA_0"))
        if os.getenv("PLAYERS_PER_MATCH"):
            self.players_per_match = int(os.getenv("PLAYERS_PER_MATCH"))
        if os.getenv("MAX_MATCHES_PER_MODEL"):
            self.max_matches_per_model = int(os.getenv("MAX_MATCHES_PER_MODEL"))
    
    @property
    def strata_thresholds(self) -> Dict[str, float]:
        """Get strata thresholds as dictionary."""
        return {
            "elite": self.strata_elite,
            "high": self.strata_high,
            "mid": self.strata_mid,
            "low": 0.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mongodb_uri": "***" if self.mongodb_uri else "",
            "db_name": self.db_name,
            "trueskill_mu_0": self.trueskill_mu_0,
            "trueskill_sigma_0": self.trueskill_sigma_0,
            "trueskill_beta": self.trueskill_beta,
            "trueskill_tau": self.trueskill_tau,
            "players_per_match": self.players_per_match,
            "min_players": self.min_players,
            "max_players": self.max_players,
            "judge_count": self.judge_count,
            "max_matches_per_model": self.max_matches_per_model,
            "swiss_delta": self.swiss_delta,
            "cost_temp": self.cost_temp,
            "rating_temp": self.rating_temp,
            "strata_thresholds": self.strata_thresholds,
            "log_level": self.log_level,
            # User feedback settings
            "user_weight_multiplier": self.user_weight_multiplier,
            "user_weight_base": self.user_weight_base,
            "use_case_tau": self.use_case_tau,
            "judge_calibration_gamma": self.judge_calibration_gamma,
            "user_surprise_threshold": self.user_surprise_threshold,
            # Supabase (masked)
            "supabase_url": self.supabase_url,
            "supabase_configured": bool(self.supabase_url and self.supabase_jwt_secret),
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(
    mongodb_uri: str = None,
    db_name: str = None,
    **kwargs
) -> Settings:
    """
    Configure global settings.
    
    Args:
        mongodb_uri: MongoDB connection URI
        db_name: Database name
        **kwargs: Additional settings
    
    Returns:
        Configured Settings instance
    """
    global _settings
    
    settings = get_settings()
    
    if mongodb_uri:
        settings.mongodb_uri = mongodb_uri
    if db_name:
        settings.db_name = db_name
    
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return settings


