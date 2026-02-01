"""
MongoDB document schemas for validation.

Updated for multi-player matches with TrueSkill + Plackett-Luce.
These schemas define the structure of documents stored in MongoDB.
"""

from typing import Dict, List, Any

# =============================================================================
# Model Document Schema (with 10D TrueSkill support)
# =============================================================================

MODEL_SCHEMA: Dict[str, Any] = {
    "name": str,                          # Unique model name
    "model_id": str,                       # Provider model ID
    "provider": str,                       # Provider name
    "input_cost_per_million": float,       # Input token cost
    "output_cost_per_million": float,      # Output token cost
    "pricing_source": str,                 # Pricing documentation
    "max_matches": int,                    # Match cap
    
    # Dual TrueSkill (default/global) - Legacy
    "trueskill": {
        "raw": {"mu": float, "sigma": float},
        "cost_adjusted": {"mu": float, "sigma": float}
    },
    
    # Per-domain TrueSkill (legacy)
    "trueskill_by_domain": {
        # "<domain_name>": {
        #     "raw": {"mu": float, "sigma": float},
        #     "cost_adjusted": {"mu": float, "sigma": float}
        # }
    },
    
    # 10-Dimensional TrueSkill (new)
    "multi_trueskill": {
        # "<dimension_name>": {"mu": float, "sigma": float}
        # e.g., "raw_quality": {"mu": 1500, "sigma": 166}
        # "cost_adjusted", "latency", "ttft", "consistency",
        # "token_efficiency", "instruction_following",
        # "hallucination_resistance", "long_context", "combined"
    },
    
    # Per-domain 10D TrueSkill (new)
    "multi_trueskill_by_domain": {
        # "<domain_name>": {
        #     "<dimension_name>": {"mu": float, "sigma": float}
        # }
    },
    
    # Performance metrics
    "performance": {
        "total_matches_played": int,
        "total_races_participated": int,
        "avg_rank_raw": float,
        "avg_rank_cost": float,
        "rank_history_raw": List[int],
        "rank_history_cost": List[int],
        "total_tokens_used": int,
        "total_input_tokens": int,
        "total_output_tokens": int,
        "total_cost_usd": float,
        # 10D average ranks (new)
        "avg_rank_by_dimension": {
            # "<dimension_name>": float
        },
        # Latency stats (new)
        "avg_latency_ms": float,
        "avg_ttft_ms": float,
    },
    
    # Match tracking
    "match_ids": {
        "played": List[str],
        "judged": List[str],
        "cases_generated": List[str],
        "questions_generated": List[str]
    },
    
    "metadata": {
        "notes": str,
        "last_updated": str,  # ISO timestamp
        "created_at": str,
        "domains_evaluated": List[str]
    }
}

# =============================================================================
# Battle Metrics Schema (for 10-Dimensional Evaluation)
# =============================================================================

BATTLE_METRICS_SCHEMA: Dict[str, Any] = {
    "model_id": str,
    "latency_ms": float,               # Total response latency
    "ttft_ms": float,                  # Time to first token
    "cost_usd": float,                 # Response cost
    "input_tokens": int,
    "output_tokens": int,
    
    # Consistency metrics (from repeated runs)
    "consistency_runs": int,           # Number of evaluation runs
    "response_variance": float,        # Variance across runs
    "response_scores": List[float],    # Individual run scores
    
    # Instruction following
    "constraints_satisfied": int,
    "total_constraints": int,
    "constraint_details": Dict[str, bool],  # {constraint_name: passed}
    
    # Hallucination detection
    "hallucination_count": int,
    "verifiable_claims": int,
    "hallucination_details": List[str],
    
    # Long context
    "context_length": int,
    "quality_at_length": float,
    
    "timestamp": str,
}


# =============================================================================
# Multi-player Match Document Schema (with 10D support)
# =============================================================================

MATCH_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # match_<uuid>
    "timestamp": str,                      # ISO timestamp
    "domain": str,                         # Domain name
    
    # List of 5-6 participants
    "participants": List[str],             # Model IDs
    "participant_count": int,              # 5 or 6
    
    "prompt": {
        "case_text": str,
        "case_generator_id": str,
        "case_generator_mu": float,
        "question_text": str,
        "question_generator_id": str,
        "question_generator_mu": float
    },
    
    # Responses from all participants
    "responses": {
        # "<model_id>": {
        #     "text": str,
        #     "tokens": int,
        #     "input_tokens": int,
        #     "output_tokens": int,
        #     "cost_usd": float,
        #     "latency_ms": float,
        #     "ttft_ms": float,
        #     "trueskill_raw_before": {"mu": float, "sigma": float},
        #     "trueskill_raw_after": {"mu": float, "sigma": float},
        #     "trueskill_cost_before": {"mu": float, "sigma": float},
        #     "trueskill_cost_after": {"mu": float, "sigma": float}
        # }
    },
    
    # Rankings from judges (2D legacy)
    "judgment": {
        # Aggregated rankings (1 = best, N = worst)
        "raw_ranking": Dict[str, int],      # {model_id: rank}
        "cost_adjusted_ranking": Dict[str, int],
        
        # Individual judge rankings
        "judges": [
            # {
            #     "judge_id": str,
            #     "judge_name": str,
            #     "mu_at_judgment": float,
            #     "raw_ranking": {"<model_id>": int},  # 1-N ranking
            #     "weight": float,
            #     "prompt": str,
            #     "response": str
            # }
        ]
    },
    
    # 10-Dimensional Rankings (new)
    "dimension_rankings": {
        # "<dimension_name>": {"<model_id>": int, ...}
        # e.g., "raw_quality": {"gpt-4": 1, "claude": 2, ...}
        # "cost_adjusted": {...}
        # "latency": {...}
        # "ttft": {...}
        # "consistency": {...}
        # "token_efficiency": {...}
        # "instruction_following": {...}
        # "hallucination_resistance": {...}
        # "long_context": {...}
        # "combined": {...}
    },
    
    # Battle metrics per participant (new)
    "battle_metrics": {
        # "<model_id>": BATTLE_METRICS_SCHEMA
    },
    
    # Plackett-Luce derived strengths
    "plackett_luce": {
        "raw_strengths": Dict[str, float],      # λ values
        "cost_strengths": Dict[str, float],
        "log_likelihood": float,
        "converged": bool
    },
    
    "meta": {
        "case_id": str,
        "question_id": str,
        "stratum": str,
        "judgment_method": str,
        "cost_temperature": float,
        "info_bits": float  # log₂(n!) information content
    }
}

# =============================================================================
# Domain Document Schema
# =============================================================================

DOMAIN_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # Domain name
    "name": str,                           # Display name
    "description": str,                    # Domain description
    "parent_domain": str,                  # Parent domain (if any)
    "created_at": str,                     # ISO timestamp
    "updated_at": str,
    
    # Domain settings
    "settings": {
        "strata_thresholds": {
            "elite": float,
            "high": float,
            "mid": float,
            "low": float
        },
        "judge_count": int,
        "players_per_match": int,
        "min_players": int,
        "swiss_delta": float,
        "cost_temperature": float,
        "rating_temperature": float
    },
    
    # Domain prompts
    "prompts": {
        "case_generation": str,
        "question_generation": str,
        "answer_generation": str,
        "judge_ranking": str
    },
    
    # Statistics
    "statistics": {
        "total_matches": int,
        "total_models_evaluated": int,
        "avg_info_bits_per_match": float,
        "top_model_raw": str,
        "top_model_cost": str,
        "last_match_timestamp": str
    }
}

# =============================================================================
# Domain Correlation Schema
# =============================================================================

CORRELATION_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # "domain_a|domain_b"
    "domain_a": str,
    "domain_b": str,
    
    # Correlation data
    "prior_correlation": float,            # Expert prior ρ_prior
    "data_correlation": float,             # Observed ρ_data
    "blended_correlation": float,          # Bayesian blend ρ
    "sample_count": int,                   # Number of shared observations
    "alpha": float,                        # Current blending factor
    
    # Metadata
    "created_at": str,
    "updated_at": str,
    "confidence": float                    # Confidence in correlation estimate
}

# =============================================================================
# MongoDB Validators
# =============================================================================

def create_model_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for models collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "model_id", "provider"],
            "properties": {
                "name": {"bsonType": "string"},
                "model_id": {"bsonType": "string"},
                "provider": {"bsonType": "string"},
                "input_cost_per_million": {"bsonType": "double"},
                "output_cost_per_million": {"bsonType": "double"},
                "trueskill": {
                    "bsonType": "object",
                    "properties": {
                        "raw": {
                            "bsonType": "object",
                            "properties": {
                                "mu": {"bsonType": "double"},
                                "sigma": {"bsonType": "double"}
                            }
                        },
                        "cost_adjusted": {
                            "bsonType": "object",
                            "properties": {
                                "mu": {"bsonType": "double"},
                                "sigma": {"bsonType": "double"}
                            }
                        }
                    }
                }
            }
        }
    }


def create_match_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for matches collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "timestamp", "domain", "participants"],
            "properties": {
                "_id": {"bsonType": "string"},
                "timestamp": {"bsonType": "string"},
                "domain": {"bsonType": "string"},
                "participants": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "minItems": 5,
                    "maxItems": 6
                },
                "participant_count": {
                    "bsonType": "int",
                    "minimum": 5,
                    "maximum": 6
                },
                "dimension_rankings": {
                    "bsonType": "object",
                    "description": "10D rankings by dimension"
                },
                "battle_metrics": {
                    "bsonType": "object",
                    "description": "Per-participant battle metrics"
                }
            }
        }
    }


def create_battle_metrics_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for battle metrics."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "model_id": {"bsonType": "string"},
                "latency_ms": {"bsonType": "double"},
                "ttft_ms": {"bsonType": "double"},
                "cost_usd": {"bsonType": "double"},
                "input_tokens": {"bsonType": "int"},
                "output_tokens": {"bsonType": "int"},
                "consistency_runs": {"bsonType": "int"},
                "response_variance": {"bsonType": "double"},
                "constraints_satisfied": {"bsonType": "int"},
                "total_constraints": {"bsonType": "int"},
                "hallucination_count": {"bsonType": "int"},
                "verifiable_claims": {"bsonType": "int"},
                "context_length": {"bsonType": "int"},
                "quality_at_length": {"bsonType": "double"},
            }
        }
    }


def create_domain_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for domains collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "name"],
            "properties": {
                "_id": {"bsonType": "string"},
                "name": {"bsonType": "string"},
                "description": {"bsonType": "string"}
            }
        }
    }


def create_correlation_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for correlations collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "domain_a", "domain_b"],
            "properties": {
                "_id": {"bsonType": "string"},
                "domain_a": {"bsonType": "string"},
                "domain_b": {"bsonType": "string"},
                "prior_correlation": {
                    "bsonType": "double",
                    "minimum": -1.0,
                    "maximum": 1.0
                },
                "data_correlation": {
                    "bsonType": "double",
                    "minimum": -1.0,
                    "maximum": 1.0
                },
                "blended_correlation": {
                    "bsonType": "double",
                    "minimum": -1.0,
                    "maximum": 1.0
                }
            }
        }
    }


# =============================================================================
# User Session Document Schema
# =============================================================================

USER_SESSION_SCHEMA: Dict[str, Any] = {
    "_id": str,                    # session_<uuid>
    "session_id": str,             # Same as _id
    "user_id": str,                # Optional user identifier
    "domain": str,                 # Domain evaluated
    "use_case": str,               # Use case category (e.g., "patient_education")
    "models_tested": List[str],    # Model IDs tested
    "n_battles": int,              # Number of battles conducted
    
    # Pre-testing predictions
    "predictions": {
        # "<model_id>": {
        #     "model_id": str,
        #     "mu_pred": float,
        #     "sigma_pred": float,
        #     "timestamp": str
        # }
    },
    
    # Post-testing deltas
    "deltas": {
        # "<model_id>": {
        #     "model_id": str,
        #     "mu_pred": float,
        #     "mu_post": float,
        #     "sigma_pred": float,
        #     "sigma_post": float,
        #     "delta": float,       # μ_post - μ_pred
        #     "z_score": float      # Normalized delta
        # }
    },
    
    # Conservation check (must be ~0)
    "delta_sum": float,
    
    # Accuracy metrics
    "prediction_accuracy": float,  # Pairwise ordering accuracy
    "kendall_tau": float,          # Rank correlation coefficient
    
    # Timestamps
    "timestamp_start": str,
    "timestamp_end": str
}


# =============================================================================
# Judge Calibration Document Schema
# =============================================================================

JUDGE_CALIBRATION_SCHEMA: Dict[str, Any] = {
    "_id": str,                    # judge_id or judge_id:domain
    "judge_id": str,
    "domain": str,                 # Optional domain for domain-specific calibration
    
    # Calibration data
    "agreed": int,                 # Number of user-aligned judgments
    "total": int,                  # Total overlapping judgments
    "calibration": float,          # agreed / total
    
    # Timestamps
    "created_at": str,
    "updated_at": str
}


# =============================================================================
# Parameter Document Schema
# =============================================================================

PARAMETER_SCHEMA: Dict[str, Any] = {
    "_id": str,                    # param_type or param_type:domain
    "param_type": str,             # Parameter type identifier
    "domain": str,                 # Optional domain for overrides
    "params": Dict[str, Any],      # Actual parameter values
    "updated_at": str,             # ISO timestamp
}


def create_parameter_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for parameters collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "param_type", "params"],
            "properties": {
                "_id": {"bsonType": "string"},
                "param_type": {
                    "bsonType": "string",
                    "enum": [
                        "trueskill", "strata", "match", "temperature",
                        "dimension_weights", "transfer", "feedback",
                        "penalty", "consistency", "long_context"
                    ]
                },
                "domain": {"bsonType": ["string", "null"]},
                "params": {"bsonType": "object"},
                "updated_at": {"bsonType": "string"}
            }
        }
    }


def create_user_session_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for user sessions collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "domain", "use_case"],
            "properties": {
                "_id": {"bsonType": "string"},
                "session_id": {"bsonType": "string"},
                "user_id": {"bsonType": ["string", "null"]},
                "domain": {"bsonType": "string"},
                "use_case": {"bsonType": "string"},
                "models_tested": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"}
                },
                "n_battles": {"bsonType": "int"},
                "delta_sum": {"bsonType": "double"},
                "prediction_accuracy": {
                    "bsonType": "double",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "kendall_tau": {
                    "bsonType": "double",
                    "minimum": -1.0,
                    "maximum": 1.0
                }
            }
        }
    }


# =============================================================================
# Phase 11: Base Model Document Schema
# =============================================================================

BASE_MODEL_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # e.g., "meta-llama/llama-3.1-70b"
    "name": str,                           # Human-readable name
    "family": str,                         # Model family
    "version": str,                        # Version within family
    "organization": str,                   # Creating organization
    
    # Architecture
    "architecture": {
        "type": str,                       # "decoder-only", "encoder-decoder", "moe"
        "parameters": int,                 # Raw parameter count
        "parameters_display": str,         # e.g., "70B"
        "hidden_size": int,
        "num_layers": int,
        "num_attention_heads": int,
        "num_kv_heads": int,
        "vocab_size": int,
        "max_position_embeddings": int,
        "native_precision": str,
        "native_size_gb": float,
    },
    
    # Capabilities
    "capabilities": {
        "modalities": List[str],
        "languages": List[str],
        "supports_function_calling": bool,
        "supports_json_mode": bool,
        "supports_vision": bool,
        "supports_audio": bool,
        "supports_system_prompt": bool,
        "supports_streaming": bool,
        "trained_for": List[str],
    },
    
    # Context window
    "context": {
        "max_context_length": int,
        "recommended_context": int,
        "max_output_tokens": int,
        "quality_at_context": Dict[int, float],
    },
    
    # License
    "license": {
        "type": str,
        "commercial_use": bool,
        "requires_agreement": bool,
        "restrictions": List[str],
        "url": str,
    },
    
    # Quantizations available
    "quantizations": {
        # "<quant_name>": {
        #     "bits": float,
        #     "vram_gb": float,
        #     "quality_retention": float,
        #     "supported_frameworks": List[str],
        # }
    },
    
    # Serving requirements by framework
    "serving_requirements": Dict[str, Any],
    
    # Sources
    "sources": Dict[str, str],
    
    # External benchmarks (denormalized for fast access)
    "benchmarks": {
        # "<benchmark_name>": {
        #     "score": float,
        #     "variant": str,
        #     "percentile": int,
        #     "updated_at": str,
        # }
    },
    
    "benchmark_aggregates": {
        "open_llm_average": float,
        "knowledge_average": float,
        "reasoning_average": float,
        "coding_average": float,
        "math_average": float,
        "chat_average": float,
        "overall_rank": int,
        "total_models_ranked": int,
        "last_computed": str,
    },
    
    "released_at": str,
    "created_at": str,
    "updated_at": str,
}


# =============================================================================
# Phase 11: Model Deployment Document Schema
# =============================================================================

MODEL_DEPLOYMENT_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # e.g., "groq/llama-3.1-70b-versatile"
    "base_model_id": str,                  # Link to base model
    
    "provider": {
        "id": str,
        "name": str,
        "type": str,                       # "cloud", "self-hosted", "edge"
        "api_base": str,
        "api_compatible": str,
        "website": str,
        "docs": str,
    },
    
    "deployment": {
        "model_id": str,                   # What to send in API calls
        "display_name": str,
        "quantization": str,
        "serving_framework": str,
        "max_context_length": int,
        "max_output_tokens": int,
        "notes": str,
    },
    
    "pricing": {
        "input_cost_per_million": float,
        "output_cost_per_million": float,
        "currency": str,
        "batch_pricing": Dict[str, Any],   # Optional
        "free_tier": Dict[str, Any],       # Optional
        "source_id": str,
        "source_url": str,
        "last_verified": str,
        "price_hash": str,
    },
    
    "performance": {
        "avg_latency_ms": float,
        "p50_latency_ms": float,
        "p95_latency_ms": float,
        "p99_latency_ms": float,
        "avg_ttft_ms": float,
        "p50_ttft_ms": float,
        "p95_ttft_ms": float,
        "tokens_per_second": float,
        "uptime_30d": float,
        "error_rate_30d": float,
        "measured_at": str,
        "measurement_source": str,
        "measurement_count": int,
    },
    
    "rate_limits": {
        "requests_per_minute": int,
        "tokens_per_minute": int,
        "tokens_per_day": int,
        "concurrent_requests": int,
        "tiers": Dict[str, Any],
    },
    
    "availability": {
        "status": str,                     # "active", "deprecated", "beta", "limited"
        "regions": List[str],
        "requires_waitlist": bool,
        "requires_enterprise": bool,
        "deprecation_date": str,
        "deprecation_replacement": str,
    },
    
    # TrueSkill ratings for THIS deployment
    "trueskill": {
        "raw": {"mu": float, "sigma": float},
        "cost_adjusted": {"mu": float, "sigma": float},
    },
    "trueskill_by_domain": Dict[str, Any],
    "multi_trueskill": Dict[str, Any],
    "multi_trueskill_by_domain": Dict[str, Any],
    
    "performance_stats": {
        "total_matches_played": int,
        "avg_rank_raw": float,
        "avg_rank_cost": float,
        "last_match": str,
    },
    
    "created_at": str,
    "updated_at": str,
}


# =============================================================================
# Phase 11: Self-Hosted Deployment Document Schema
# =============================================================================

SELF_HOSTED_DEPLOYMENT_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # e.g., "user123/llama-3.1-70b-vllm-a100"
    "base_model_id": str,
    
    "owner": {
        "user_id": str,
        "organization": str,
        "email": str,
        "is_public": bool,
    },
    
    "provider": {
        "id": str,                         # Always "self-hosted"
        "name": str,
        "type": str,
        "api_base": str,
        "api_compatible": str,
    },
    
    "hardware": {
        "gpu_type": str,
        "gpu_count": int,
        "gpu_memory_gb": float,
        "cpu_cores": int,
        "ram_gb": float,
        "cloud_provider": str,             # "aws", "gcp", "azure", "on-prem"
        "instance_type": str,
        "region": str,
        "datacenter": str,
        "rack": str,
    },
    
    "software": {
        "serving_framework": str,          # "vllm", "tgi", "llama.cpp", etc.
        "framework_version": str,
        "quantization": str,
        "quantization_source": str,
        "framework_settings": Dict[str, Any],
        "tensor_parallel_size": int,
        "max_model_len": int,
        "gpu_memory_utilization": float,
    },
    
    "costs": {
        "input_cost_per_million": float,
        "output_cost_per_million": float,
        "hourly_compute_cost": float,
        "calculation": {
            "method": str,                 # "user_provided", "estimated"
            "assumed_utilization": float,
            "assumed_batch_size": int,
            "assumed_throughput_tps": float,
        },
        "monthly_fixed_costs": {
            "storage": float,
            "network": float,
            "monitoring": float,
            "other": float,
        },
        "notes": str,
    },
    
    "performance": {
        "expected_latency_ms": float,
        "expected_ttft_ms": float,
        "expected_tokens_per_second": float,
        "measured_latency_ms": float,
        "measured_ttft_ms": float,
        "measured_throughput": float,
        "measurement_count": int,
        "last_measured": str,
    },
    
    "availability": {
        "status": str,
        "schedule": str,
        "health_check_url": str,
        "last_health_check": str,
        "is_healthy": bool,
        "timezone": str,
        "availability_hours": Dict[str, Any],
    },
    
    # TrueSkill ratings
    "trueskill": {
        "raw": {"mu": float, "sigma": float},
        "cost_adjusted": {"mu": float, "sigma": float},
    },
    "trueskill_by_domain": Dict[str, Any],
    "multi_trueskill": Dict[str, Any],
    "multi_trueskill_by_domain": Dict[str, Any],
    
    "performance_stats": {
        "total_matches_played": int,
        "avg_rank_raw": float,
        "avg_rank_cost": float,
        "last_match": str,
    },
    
    "created_at": str,
    "updated_at": str,
}


# =============================================================================
# MongoDB Validators for Phase 11
# =============================================================================

def create_base_model_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for base_models collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "name"],
            "properties": {
                "_id": {"bsonType": "string"},
                "name": {"bsonType": "string"},
                "family": {"bsonType": "string"},
                "version": {"bsonType": "string"},
                "organization": {"bsonType": "string"},
                "architecture": {"bsonType": "object"},
                "capabilities": {"bsonType": "object"},
                "context": {"bsonType": "object"},
                "license": {"bsonType": "object"},
                "quantizations": {"bsonType": "object"},
                "benchmarks": {"bsonType": "object"},
            }
        }
    }


def create_model_deployment_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for model_deployments collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "base_model_id", "provider"],
            "properties": {
                "_id": {"bsonType": "string"},
                "base_model_id": {"bsonType": "string"},
                "provider": {
                    "bsonType": "object",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {"bsonType": "string"},
                        "name": {"bsonType": "string"},
                        "type": {"bsonType": "string"},
                    }
                },
                "pricing": {
                    "bsonType": "object",
                    "properties": {
                        "input_cost_per_million": {"bsonType": "double"},
                        "output_cost_per_million": {"bsonType": "double"},
                    }
                },
                "availability": {
                    "bsonType": "object",
                    "properties": {
                        "status": {"bsonType": "string"},
                    }
                },
                "trueskill": {"bsonType": "object"},
            }
        }
    }


def create_self_hosted_deployment_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for self_hosted_deployments collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "base_model_id", "owner"],
            "properties": {
                "_id": {"bsonType": "string"},
                "base_model_id": {"bsonType": "string"},
                "owner": {
                    "bsonType": "object",
                    "required": ["user_id"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "is_public": {"bsonType": "bool"},
                    }
                },
                "hardware": {"bsonType": "object"},
                "software": {"bsonType": "object"},
                "costs": {"bsonType": "object"},
                "trueskill": {"bsonType": "object"},
            }
        }
    }


# =============================================================================
# Phase 17-24: Ground Truth Evaluation Schemas
# =============================================================================

GROUND_TRUTH_SAMPLE_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # sample_<uuid>
    "domain": str,                         # Domain name (asr, tts, needle_in_haystack, etc.)
    "category": str,                       # "multimodal" or "long_context"
    
    # Sample metadata
    "base_path": str,                      # Path to sample data directory
    "difficulty": str,                     # "easy", "medium", "hard"
    "language": str,                       # Language code
    
    # Input specification
    "inputs": {
        # Domain-specific input fields
        # ASR: {"audio": "path/to/audio.wav", "duration_seconds": float}
        # TTS: {"text": str, "voice_ref": "path/to/ref.wav"}
        # Needle: {"haystack": "path/to/text.txt"}
        # VQA: {"image": "path/to/image.jpg", "question": str}
    },
    
    # Ground truth
    "ground_truth": {
        # Domain-specific ground truth
        # ASR: {"transcript_file": "path/to/transcript.txt"}
        # TTS: {"expected_text": str}
        # Needle: {"answer": str, "answer_variants": List[str]}
        # VQA: {"answer": str, "answer_type": str}
        # Code: {"expected_code": str, "test_cases": List[Dict]}
    },
    
    # Domain-specific metadata
    "metadata": {
        # Needle: {"context_length": int, "needle_position": float}
        # ASR: {"audio_quality": str, "speaker_id": str}
        # TTS: {"style": str, "emotion": str}
    },
    
    # Usage tracking
    "usage_count": int,
    "last_used": str,                      # ISO timestamp
    
    # Timestamps
    "created_at": str,
    "updated_at": str,
}


GROUND_TRUTH_MATCH_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # gt_match_<uuid>
    "domain": str,                         # Ground truth domain
    "timestamp": str,                      # ISO timestamp
    
    "sample_id": str,                      # Reference to sample
    "sample_metadata": Dict,               # Denormalized sample metadata
    
    "participants": List[str],             # Deployment IDs
    
    # Responses from all participants
    "responses": {
        # "<deployment_id>": {
        #     "response_content": Any,  # str for text, encoded for audio
        #     "latency_ms": float,
        #     "ttft_ms": float,
        #     "input_tokens": int,
        #     "output_tokens": int,
        #     "cost_usd": float,
        #     "error": str,
        # }
    },
    
    # Scores per participant
    "scores": {
        # "<deployment_id>": {
        #     "metrics": {"wer": float, "cer": float, ...},
        #     "overall": float,
        #     "semantic_match": float,
        # }
    },
    
    # Overall scores (denormalized for quick access)
    "overall_scores": {
        # "<deployment_id>": float
    },
    
    # Rankings based on scores
    "rankings": {
        # "<deployment_id>": int  # 1 = best
    },
    
    # TrueSkill updates
    "trueskill_updates": {
        # "<deployment_id>": {
        #     "before": {"mu": float, "sigma": float},
        #     "after": {"mu": float, "sigma": float}
        # }
    },
    
    "evaluation_mode": str,                # "metrics_only", "hybrid", "llm_judge"
    "metrics_used": List[str],             # List of metrics computed
    
    "status": str,                         # "completed", "failed", "partial"
    "error_message": str,
    "duration_ms": float,
}


GROUND_TRUTH_RATING_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # deployment_id:domain
    "deployment_id": str,
    "domain": str,
    
    # TrueSkill rating for this GT domain
    "trueskill": {
        "mu": float,
        "sigma": float,
    },
    
    # Aggregate statistics
    "statistics": {
        "total_evaluations": int,
        "avg_score": float,
        "best_score": float,
        "worst_score": float,
        "score_std": float,
    },
    
    # Per-metric averages
    "metric_averages": {
        # "<metric_name>": float
    },
    
    # Leaderboard position (cached)
    "leaderboard_rank": int,
    "leaderboard_total": int,
    
    "created_at": str,
    "updated_at": str,
}


GROUND_TRUTH_LEADERBOARD_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # domain name
    "domain": str,
    
    "entries": [
        # {
        #     "deployment_id": str,
        #     "rank": int,
        #     "trueskill_mu": float,
        #     "trueskill_sigma": float,
        #     "primary_metric_avg": float,
        #     "total_evaluations": int,
        #     "last_evaluation": str,
        # }
    ],
    
    "primary_metric": str,
    "metric_direction": str,               # "asc" or "desc"
    
    "last_updated": str,
    "total_entries": int,
}


GROUND_TRUTH_COVERAGE_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # domain name
    "domain": str,
    
    "total_samples": int,
    "used_samples": int,
    "coverage_percentage": float,
    
    # Stratification coverage
    "stratification_coverage": {
        # "<dimension>": {
        #     "<value>": {"total": int, "used": int, "coverage": float}
        # }
    },
    
    # Samples needing more evaluation
    "underrepresented_strata": List[Dict],
    
    "last_computed": str,
}


def create_ground_truth_sample_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for ground_truth_samples collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "domain", "category"],
            "properties": {
                "_id": {"bsonType": "string"},
                "domain": {"bsonType": "string"},
                "category": {
                    "bsonType": "string",
                    "enum": ["multimodal", "long_context"]
                },
                "difficulty": {
                    "bsonType": "string",
                    "enum": ["easy", "medium", "hard"]
                },
                "inputs": {"bsonType": "object"},
                "ground_truth": {"bsonType": "object"},
                "usage_count": {"bsonType": "int"},
            }
        }
    }


def create_ground_truth_match_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for ground_truth_matches collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "domain", "timestamp", "sample_id"],
            "properties": {
                "_id": {"bsonType": "string"},
                "domain": {"bsonType": "string"},
                "timestamp": {"bsonType": "string"},
                "sample_id": {"bsonType": "string"},
                "participants": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"}
                },
                "status": {
                    "bsonType": "string",
                    "enum": ["completed", "failed", "partial"]
                },
                "evaluation_mode": {
                    "bsonType": "string",
                    "enum": ["metrics_only", "hybrid", "llm_judge"]
                },
            }
        }
    }


def create_ground_truth_rating_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for ground_truth_ratings collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "deployment_id", "domain"],
            "properties": {
                "_id": {"bsonType": "string"},
                "deployment_id": {"bsonType": "string"},
                "domain": {"bsonType": "string"},
                "trueskill": {
                    "bsonType": "object",
                    "properties": {
                        "mu": {"bsonType": "double"},
                        "sigma": {"bsonType": "double"},
                    }
                },
                "statistics": {"bsonType": "object"},
                "metric_averages": {"bsonType": "object"},
            }
        }
    }


# =============================================================================
# User Model Document Schema (Private Testing)
# =============================================================================

USER_MODEL_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # um_<uuid>
    "user_id": str,                        # Owner user ID
    "name": str,                           # Model name
    "description": str,                    # Optional description
    "model_type": str,                     # LLM, ASR, TTS, VLM, etc.
    "version": str,                        # Model version
    "base_model_id": str,                  # Link to base model (optional)
    
    # API Configuration
    "api_config": {
        "endpoint": str,                   # API endpoint URL
        "model_id": str,                   # Model ID for API calls
        "api_format": str,                 # openai, anthropic, google, custom
        "api_key_hash": str,               # Hashed API key (not actual key)
        "has_api_key": bool,
        "headers": Dict,
        "extra_params": Dict,
    },
    
    # Type-specific configuration
    "type_config": {
        "language": str,                   # For ASR/TTS
        "sample_rate": int,                # For ASR/TTS
        "image_detail": str,               # For VLM
        "image_size": str,                 # For ImageGen
        "steps": int,                      # For ImageGen/VideoGen
        "guidance_scale": float,           # For ImageGen/VideoGen
        "embedding_dimension": int,        # For Embedding
        "normalize": bool,                 # For Embedding
        "max_tokens": int,                 # For LLM/VLM
        "temperature": float,              # For LLM/VLM
    },
    
    # Cost configuration
    "costs": {
        "input_cost_per_million": float,
        "output_cost_per_million": float,
        "currency": str,
        "is_estimate": bool,
        "notes": str,
    },
    
    # Visibility
    "is_public": bool,
    "is_active": bool,
    
    # TrueSkill ratings (for LLM/VLM subjective evaluation)
    "trueskill": {
        # 10-dimensional TrueSkill
        # "raw_quality": {"mu": float, "sigma": float},
        # ... etc
    },
    "trueskill_by_domain": {
        # Domain-specific TrueSkill
    },
    
    # Ground truth metrics (for ASR, TTS, etc.)
    "ground_truth_metrics": {
        # e.g., "wer": 0.05, "cer": 0.02, "mos": 4.2
    },
    "ground_truth_by_domain": {
        # Domain-specific ground truth metrics
    },
    
    # Tracking
    "total_evaluations": int,
    "domains_evaluated": List[str],
    "last_evaluated_at": str,
    "created_at": str,
    "updated_at": str,
}


USER_MODEL_EVALUATION_SCHEMA: Dict[str, Any] = {
    "_id": str,                            # eval_<uuid>
    "model_id": str,                       # User model ID
    "user_id": str,                        # Owner user ID
    "domain": str,                         # Evaluation domain
    "evaluation_type": str,                # subjective, ground_truth
    "status": str,                         # queued, running, completed, failed
    "progress": float,                     # 0.0 to 1.0
    "current_step": str,                   # Current step description
    
    # Results
    "trueskill_before": Dict,              # TrueSkill before evaluation
    "trueskill_after": Dict,               # TrueSkill after evaluation
    "metrics": Dict,                       # Evaluation metrics
    "competitors": List[str],              # Models competed against
    "matches_played": int,                 # Number of matches
    
    # Metadata
    "error": str,                          # Error message if failed
    "started_at": str,
    "completed_at": str,
    "timestamp": str,
}


def create_user_model_validator() -> Dict[str, Any]:
    """Create MongoDB JSON Schema validator for user_models collection."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "user_id", "name", "model_type"],
            "properties": {
                "_id": {"bsonType": "string"},
                "user_id": {"bsonType": "string"},
                "name": {"bsonType": "string"},
                "model_type": {
                    "bsonType": "string",
                    "enum": ["LLM", "ASR", "TTS", "VLM", "V2V", "STT", 
                            "ImageGen", "VideoGen", "Embedding", "Reranker"]
                },
                "version": {"bsonType": "string"},
                "is_public": {"bsonType": "bool"},
                "is_active": {"bsonType": "bool"},
                "api_config": {"bsonType": "object"},
                "type_config": {"bsonType": "object"},
                "costs": {"bsonType": "object"},
                "trueskill": {"bsonType": "object"},
                "ground_truth_metrics": {"bsonType": "object"},
            }
        }
    }


