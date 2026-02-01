"""
GraphQL query resolvers for LangScope.
"""

import strawberry
from typing import Optional, List
from langscope.graphql.types import (
    BaseModelType,
    ModelDeploymentType,
    SelfHostedDeploymentType,
    DomainType,
    LeaderboardEntryType,
    ModelDeploymentConnection,
    ModelDeploymentEdge,
    PageInfo,
    _base_model_to_type,
    _deployment_to_type,
    # Ground Truth types
    GroundTruthDomainType,
    GroundTruthSampleType,
    GroundTruthMatchType,
    GroundTruthLeaderboardEntryType,
    GroundTruthScoreType,
    GroundTruthMetricType,
    NeedleHeatmapType,
    NeedleHeatmapCellType,
    GroundTruthCoverageType,
    # Prompt Management types
    ClassificationResultType,
    PromptProcessingResultType,
    PromptMetricsType,
    # Cache Management types
    CacheStatsType,
    CacheCategoryStatsType,
    CacheTotalsType,
    CacheConnectionsType,
    SessionType,
    RateLimitStatusType,
)


@strawberry.type
class Query:
    """Root GraphQL query type."""
    
    # === Base Models ===
    
    @strawberry.field
    async def base_model(self, info, id: str) -> Optional[BaseModelType]:
        """Get a base model by ID."""
        db = info.context.get("db")
        if db:
            data = await db.get_base_model(id)
            if data:
                return _base_model_to_type(data)
        return None
    
    @strawberry.field
    async def base_models(
        self,
        info,
        organization: Optional[str] = None,
        family: Optional[str] = None,
        limit: int = 50
    ) -> List[BaseModelType]:
        """List base models with optional filtering."""
        db = info.context.get("db")
        if not db:
            return []
        
        filters = {}
        if organization:
            filters["organization"] = organization
        if family:
            filters["family"] = family
        
        models = await db.get_all_base_models(filters=filters, limit=limit)
        return [_base_model_to_type(m) for m in models]
    
    @strawberry.field
    async def search_base_models(
        self,
        info,
        query: str,
        limit: int = 20
    ) -> List[BaseModelType]:
        """Search base models by name."""
        db = info.context.get("db")
        if not db:
            return []
        
        # Simple text search
        models = await db.search_base_models(query, limit=limit)
        return [_base_model_to_type(m) for m in models]
    
    # === Deployments ===
    
    @strawberry.field
    async def deployment(self, info, id: str) -> Optional[ModelDeploymentType]:
        """Get a deployment by ID."""
        db = info.context.get("db")
        if db:
            data = await db.get_model_deployment(id)
            if data:
                return _deployment_to_type(data)
        return None
    
    @strawberry.field
    async def deployments(
        self,
        info,
        base_model_id: Optional[str] = None,
        provider: Optional[str] = None,
        available_only: bool = True,
        first: int = 50,
        after: Optional[str] = None
    ) -> ModelDeploymentConnection:
        """List deployments with pagination."""
        db = info.context.get("db")
        if not db:
            return ModelDeploymentConnection(
                edges=[],
                page_info=PageInfo(
                    has_next_page=False,
                    has_previous_page=False,
                    start_cursor=None,
                    end_cursor=None,
                ),
                total_count=0,
            )
        
        filters = {}
        if base_model_id:
            filters["base_model_id"] = base_model_id
        if provider:
            filters["provider.name"] = provider
        if available_only:
            filters["available"] = True
        
        # Get total count
        total = await db.count_model_deployments(filters)
        
        # Get page
        skip = 0
        if after:
            # Decode cursor (base64 encoded offset)
            import base64
            try:
                skip = int(base64.b64decode(after).decode())
            except (ValueError, Exception):
                skip = 0
        
        deployments = await db.get_all_model_deployments(
            filters=filters,
            skip=skip,
            limit=first + 1  # Get one extra to check if there's more
        )
        
        has_next = len(deployments) > first
        if has_next:
            deployments = deployments[:first]
        
        edges = []
        for i, d in enumerate(deployments):
            import base64
            cursor = base64.b64encode(str(skip + i).encode()).decode()
            edges.append(ModelDeploymentEdge(
                cursor=cursor,
                node=_deployment_to_type(d),
            ))
        
        return ModelDeploymentConnection(
            edges=edges,
            page_info=PageInfo(
                has_next_page=has_next,
                has_previous_page=skip > 0,
                start_cursor=edges[0].cursor if edges else None,
                end_cursor=edges[-1].cursor if edges else None,
            ),
            total_count=total,
        )
    
    @strawberry.field
    async def best_deployment(
        self,
        info,
        base_model_id: str,
        optimize_for: str = "cost"  # "cost", "latency", "quality"
    ) -> Optional[ModelDeploymentType]:
        """Get the best deployment for a base model based on optimization criteria."""
        db = info.context.get("db")
        if not db:
            return None
        
        deployment = await db.get_best_deployment(base_model_id, optimize_for)
        if deployment:
            return _deployment_to_type(deployment)
        return None
    
    # === Leaderboard ===
    
    @strawberry.field
    async def leaderboard(
        self,
        info,
        domain: Optional[str] = None,
        dimension: str = "raw_quality",
        limit: int = 20
    ) -> List[LeaderboardEntryType]:
        """Get the leaderboard for a domain and dimension."""
        db = info.context.get("db")
        if not db:
            return []
        
        entries = await db.get_leaderboard(
            domain=domain or "",
            dimension=dimension,
            limit=limit
        )
        
        results = []
        for rank, entry in enumerate(entries, 1):
            rating = entry.get("ratings", {}).get(dimension, {})
            results.append(LeaderboardEntryType(
                rank=rank,
                deployment_id=str(entry.get("_id", "")),
                name=entry.get("name", ""),
                provider=entry.get("provider", {}).get("name", ""),
                rating_mu=rating.get("mu", 25.0),
                rating_sigma=rating.get("sigma", 8.333),
                conservative_rating=rating.get("mu", 25.0) - 3 * rating.get("sigma", 8.333),
                matches_played=entry.get("matches_played", 0),
            ))
        
        return results
    
    # === Domains ===
    
    @strawberry.field
    async def domains(self, info) -> List[DomainType]:
        """List all evaluation domains."""
        db = info.context.get("db")
        if not db:
            return []
        
        domains = await db.get_all_domains()
        return [
            DomainType(
                id=str(d.get("_id", "")),
                name=d.get("name", ""),
                description=d.get("description", ""),
                case_prompt=d.get("case_prompt", ""),
                evaluation_criteria=d.get("evaluation_criteria", []),
                created_at=d.get("created_at"),
            )
            for d in domains
        ]
    
    @strawberry.field
    async def domain(self, info, name: str) -> Optional[DomainType]:
        """Get a domain by name."""
        db = info.context.get("db")
        if not db:
            return None
        
        d = await db.get_domain(name)
        if d:
            return DomainType(
                id=str(d.get("_id", "")),
                name=d.get("name", ""),
                description=d.get("description", ""),
                case_prompt=d.get("case_prompt", ""),
                evaluation_criteria=d.get("evaluation_criteria", []),
                created_at=d.get("created_at"),
            )
        return None
    
    # === Statistics ===
    
    @strawberry.field
    async def stats(self, info) -> "SystemStatsType":
        """Get system statistics."""
        db = info.context.get("db")
        if not db:
            return SystemStatsType(
                total_base_models=0,
                total_deployments=0,
                total_matches=0,
                total_domains=0,
            )
        
        return SystemStatsType(
            total_base_models=await db.count_base_models(),
            total_deployments=await db.count_model_deployments(),
            total_matches=await db.count_matches(),
            total_domains=await db.count_domains(),
        )


@strawberry.type
class SystemStatsType:
    """System-wide statistics."""
    total_base_models: int
    total_deployments: int
    total_matches: int
    total_domains: int
    total_ground_truth_samples: int = 0
    total_ground_truth_matches: int = 0


# =============================================================================
# Ground Truth Queries
# =============================================================================

@strawberry.type
class GroundTruthQuery:
    """Ground truth query resolvers."""
    
    @strawberry.field
    async def ground_truth_domains(self, info) -> List[GroundTruthDomainType]:
        """List all ground truth domains."""
        from langscope.ground_truth.metrics import MetricRegistry
        
        db = info.context.get("db")
        registry = MetricRegistry()
        
        domains = []
        for domain_name in registry.DOMAIN_METRICS.keys():
            sample_count = 0
            if db and db.connected:
                sample_count = db.get_ground_truth_sample_count(domain_name)
            
            # Determine category
            if domain_name in ("asr", "tts", "visual_qa", "document_extraction", 
                              "image_captioning", "ocr"):
                category = "multimodal"
            else:
                category = "long_context"
            
            # Determine evaluation mode
            eval_mode = "metrics_only"
            if domain_name in ("tts", "visual_qa", "long_document_qa", "long_summarization"):
                eval_mode = "hybrid"
            
            domains.append(GroundTruthDomainType(
                name=domain_name,
                category=category,
                primary_metric=registry.PRIMARY_METRIC.get(domain_name, ""),
                metrics=registry.DOMAIN_METRICS.get(domain_name, []),
                evaluation_mode=eval_mode,
                sample_count=sample_count,
            ))
        
        return domains
    
    @strawberry.field
    async def ground_truth_samples(
        self,
        info,
        domain: str,
        difficulty: Optional[str] = None,
        limit: int = 50
    ) -> List[GroundTruthSampleType]:
        """List ground truth samples for a domain."""
        db = info.context.get("db")
        if not db or not db.connected:
            return []
        
        samples = db.get_ground_truth_samples(
            domain=domain,
            difficulty=difficulty,
            limit=limit
        )
        
        return [
            GroundTruthSampleType(
                id=str(s.get("_id", "")),
                domain=s.get("domain", ""),
                category=s.get("category", ""),
                difficulty=s.get("difficulty", "medium"),
                language=s.get("language", "en"),
                usage_count=s.get("usage_count", 0),
            )
            for s in samples
        ]
    
    @strawberry.field
    async def ground_truth_match(
        self,
        info,
        match_id: str
    ) -> Optional[GroundTruthMatchType]:
        """Get a specific ground truth match result."""
        db = info.context.get("db")
        if not db or not db.connected:
            return None
        
        match = db.get_ground_truth_match(match_id)
        if not match:
            return None
        
        from langscope.ground_truth.metrics import MetricRegistry
        registry = MetricRegistry()
        domain = match.get("domain", "")
        primary_metric = registry.PRIMARY_METRIC.get(domain, "")
        
        # Convert scores
        scores = []
        for model_id, score_data in match.get("scores", {}).items():
            metrics = [
                GroundTruthMetricType(
                    name=m_name,
                    value=m_value,
                    is_primary=(m_name == primary_metric)
                )
                for m_name, m_value in score_data.get("metrics", {}).items()
                if isinstance(m_value, (int, float))
            ]
            
            scores.append(GroundTruthScoreType(
                model_id=model_id,
                sample_id=match.get("sample_id", ""),
                overall=score_data.get("overall", 0.0),
                metrics=metrics,
                semantic_match=score_data.get("semantic_match"),
                evaluation_mode=score_data.get("evaluation_mode", "metrics_only"),
            ))
        
        # Convert rankings to ordered list
        rankings_dict = match.get("rankings", {})
        rankings = sorted(rankings_dict.keys(), key=lambda m: rankings_dict[m])
        
        from datetime import datetime
        timestamp = match.get("timestamp", "")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.utcnow()
        
        return GroundTruthMatchType(
            match_id=str(match.get("_id", "")),
            domain=domain,
            timestamp=timestamp,
            sample_id=match.get("sample_id", ""),
            participants=match.get("participants", []),
            rankings=rankings,
            scores=scores,
            status=match.get("status", "completed"),
            duration_ms=match.get("duration_ms", 0.0),
        )
    
    @strawberry.field
    async def ground_truth_leaderboard(
        self,
        info,
        domain: str,
        limit: int = 50
    ) -> List[GroundTruthLeaderboardEntryType]:
        """Get ground truth leaderboard for a domain."""
        db = info.context.get("db")
        if not db or not db.connected:
            return []
        
        from langscope.ground_truth.metrics import MetricRegistry
        registry = MetricRegistry()
        primary_metric = registry.PRIMARY_METRIC.get(domain, "accuracy")
        
        ratings = db.get_ground_truth_leaderboard(domain, limit)
        
        entries = []
        for i, r in enumerate(ratings):
            entries.append(GroundTruthLeaderboardEntryType(
                rank=i + 1,
                deployment_id=r.get("deployment_id", ""),
                trueskill_mu=r.get("trueskill", {}).get("mu", 25.0),
                trueskill_sigma=r.get("trueskill", {}).get("sigma", 8.33),
                primary_metric_avg=r.get("metric_averages", {}).get(primary_metric, 0.0),
                total_evaluations=r.get("statistics", {}).get("total_evaluations", 0),
            ))
        
        return entries
    
    @strawberry.field
    async def needle_heatmap(
        self,
        info,
        model_id: str
    ) -> Optional[NeedleHeatmapType]:
        """Get needle in haystack accuracy heatmap for a model."""
        db = info.context.get("db")
        if not db or not db.connected:
            return None
        
        from langscope.ground_truth.analytics import compute_accuracy_heatmap
        
        heatmap = compute_accuracy_heatmap(db, model_id, "needle_in_haystack")
        
        cells = []
        total_accuracy = 0.0
        count = 0
        
        for ctx_len_str, positions in heatmap.items():
            try:
                ctx_len = int(ctx_len_str)
            except ValueError:
                continue
            
            for pos_str, accuracy in positions.items():
                try:
                    pos = float(pos_str)
                except ValueError:
                    continue
                
                cells.append(NeedleHeatmapCellType(
                    context_length=ctx_len,
                    needle_position=pos,
                    accuracy=accuracy,
                    sample_count=1,  # Could aggregate this from DB
                ))
                total_accuracy += accuracy
                count += 1
        
        return NeedleHeatmapType(
            model_id=model_id,
            cells=cells,
            overall_accuracy=total_accuracy / count if count > 0 else 0.0,
        )
    
    @strawberry.field
    async def ground_truth_coverage(
        self,
        info,
        domain: str
    ) -> Optional[GroundTruthCoverageType]:
        """Get sample coverage statistics for a domain."""
        db = info.context.get("db")
        if not db or not db.connected:
            return None
        
        coverage = db.compute_ground_truth_coverage(domain)
        
        return GroundTruthCoverageType(
            domain=coverage.get("domain", domain),
            total_samples=coverage.get("total_samples", 0),
            used_samples=coverage.get("used_samples", 0),
            coverage_percentage=coverage.get("coverage_percentage", 0.0),
        )


# =============================================================================
# Prompt Management Queries
# =============================================================================

@strawberry.type
class PromptQuery:
    """Prompt management query resolvers."""
    
    @strawberry.field
    async def classify_prompt(
        self,
        info,
        prompt: str
    ) -> ClassificationResultType:
        """Classify a prompt into domain hierarchy."""
        from langscope.api.dependencies import get_prompt_manager
        
        pm = await get_prompt_manager()
        result = pm.classify_prompt(prompt)
        
        return ClassificationResultType(
            category=result.category,
            base_domain=result.base_domain,
            variant=result.variant,
            confidence=result.confidence,
            is_ground_truth=result.is_ground_truth,
            full_domain_name=result.full_domain_name,
            template_name=result.template_name,
        )
    
    @strawberry.field
    async def process_prompt(
        self,
        info,
        prompt: str,
        model: Optional[str] = None,
        domain: Optional[str] = None,
        skip_cache: bool = False
    ) -> PromptProcessingResultType:
        """Process a prompt with classification and cache lookup."""
        from langscope.api.dependencies import get_prompt_manager
        
        pm = await get_prompt_manager()
        result = await pm.process_prompt(
            prompt=prompt,
            model=model,
            user_domain=domain,
            skip_cache=skip_cache,
        )
        
        return PromptProcessingResultType(
            prompt=result.prompt,
            domain=ClassificationResultType(
                category=result.domain.category,
                base_domain=result.domain.base_domain,
                variant=result.domain.variant,
                confidence=result.domain.confidence,
                is_ground_truth=result.domain.is_ground_truth,
                full_domain_name=result.domain.full_domain_name,
                template_name=result.domain.template_name,
            ),
            cache_hit=result.cache_hit,
            cache_layer=result.cache_layer,
            cache_similarity=result.cache_similarity,
            evaluation_type=result.evaluation_type,
            processing_time_ms=result.processing_time_ms,
        )
    
    @strawberry.field
    async def prompt_metrics(self, info) -> PromptMetricsType:
        """Get prompt processing metrics."""
        from langscope.api.dependencies import get_prompt_manager
        
        pm = await get_prompt_manager()
        metrics = pm.get_metrics()
        
        return PromptMetricsType(**metrics)
    
    @strawberry.field
    async def domain_categories(self, info) -> List[str]:
        """List all domain categories."""
        from langscope.prompt.constants import CATEGORIES
        return list(CATEGORIES.keys())
    
    @strawberry.field
    async def supported_languages(self, info) -> List[str]:
        """List supported languages for detection."""
        from langscope.prompt.constants import LANGUAGE_PATTERNS
        return list(LANGUAGE_PATTERNS.keys())


# =============================================================================
# Cache Management Queries
# =============================================================================

@strawberry.type
class CacheQuery:
    """Cache management query resolvers."""
    
    @strawberry.field
    async def cache_stats(self, info) -> CacheStatsType:
        """Get comprehensive cache statistics."""
        from langscope.api.dependencies import get_cache_manager
        
        cache = await get_cache_manager()
        stats = cache.get_stats()
        
        categories = [
            CacheCategoryStatsType(
                category=cat_name,
                total_hits=cat_stats.get("total_hits", 0),
                misses=cat_stats.get("misses", 0),
                errors=cat_stats.get("errors", 0),
                writes=cat_stats.get("writes", 0),
                hit_rate=cat_stats.get("hit_rate", 0.0),
                local_entries=cat_stats.get("local_entries", 0),
            )
            for cat_name, cat_stats in stats.get("by_category", {}).items()
        ]
        
        totals = stats.get("totals", {})
        connections = stats.get("connections", {})
        
        return CacheStatsType(
            by_category=categories,
            totals=CacheTotalsType(
                hits=totals.get("hits", 0),
                misses=totals.get("misses", 0),
                errors=totals.get("errors", 0),
                writes=totals.get("writes", 0),
                hit_rate=totals.get("hit_rate", 0.0),
            ),
            connections=CacheConnectionsType(
                redis=connections.get("redis", False),
                mongodb=connections.get("mongodb", False),
                qdrant=connections.get("qdrant", False),
            ),
        )
    
    @strawberry.field
    async def session(self, info, session_id: str) -> Optional[SessionType]:
        """Get session by ID."""
        from langscope.api.dependencies import get_cache_manager
        from langscope.cache.session import SessionManager
        
        cache = await get_cache_manager()
        session_mgr = SessionManager(cache)
        
        session = await session_mgr.get_session(session_id)
        if not session:
            return None
        
        return SessionType(
            session_id=session.session_id,
            session_type=session.session_type,
            user_id=session.user_id,
            status=session.status,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            expires_at=session.expires_at.isoformat(),
            domain=session.domain,
        )
    
    @strawberry.field
    async def rate_limit_status(
        self,
        info,
        identifier: str,
        endpoint: str = "default"
    ) -> RateLimitStatusType:
        """Get rate limit status for an identifier."""
        from langscope.api.dependencies import get_cache_manager
        from langscope.cache.rate_limit import SlidingWindowRateLimiter
        
        cache = await get_cache_manager()
        limiter = SlidingWindowRateLimiter(cache._redis)
        
        usage = await limiter.get_usage(identifier, endpoint)
        
        return RateLimitStatusType(
            current_count=usage.get("current_count", 0),
            limit=usage.get("limit", 100),
            window=usage.get("window", 60),
            remaining=usage.get("remaining", 100),
        )


# =============================================================================
# Faceted Transfer Learning Queries (Model Rank API)
# =============================================================================

# Import transfer learning types
from langscope.graphql.types import (
    ModelRatingType,
    ModelAllRatingsType,
    DimensionRatingType,
    DomainFacetsType,
    SimilarDomainType,
    FacetContributionType,
    TransferDetailsType,
    FacetedLeaderboardType,
    FacetedLeaderboardEntryType,
    DomainSimilarityExplanationType,
    FacetSimilaritiesType,
    FacetSimilarityType,
    DomainIndexStatsType,
)


@strawberry.type
class TransferLearningQuery:
    """Faceted Transfer Learning query resolvers (Model Rank API)."""
    
    @strawberry.field
    async def model_rating(
        self,
        info,
        model_id: str,
        domain: str,
        dimension: str = "raw_quality",
        explain: bool = False
    ) -> Optional[ModelRatingType]:
        """
        Get model rating in a domain.
        
        Uses direct rating if available, otherwise transfers from similar domains.
        """
        from langscope.api.dependencies import get_model_by_id, get_faceted_transfer
        
        model = get_model_by_id(model_id)
        if not model:
            return None
        
        faceted_transfer = get_faceted_transfer()
        result = faceted_transfer.get_rating_or_transfer(model, domain, dimension)
        
        transfer_details = None
        if explain and result.source == "transfer":
            facet_contributions = []
            for facet, data in result.facet_contributions.items():
                facet_contributions.append(FacetContributionType(
                    facet=facet,
                    source_value=data.get("source_value", ""),
                    target_value=data.get("target_value", ""),
                    similarity=data.get("similarity", 0.0),
                    weight=data.get("weight", 0.0),
                    contribution=data.get("contribution", 0.0),
                ))
            
            transfer_details = TransferDetailsType(
                source_domains=result.source_domains,
                source_weights=[result.source_weights.get(d, 0.0) for d in result.source_domains],
                correlation_used=result.correlation_used,
                facet_contributions=facet_contributions,
            )
        
        from datetime import datetime
        return ModelRatingType(
            model_id=model_id,
            domain=domain,
            dimension=dimension,
            mu=result.target_mu,
            sigma=result.target_sigma,
            conservative_estimate=result.target_mu - 3 * result.target_sigma,
            source=result.source,
            confidence=result.confidence,
            match_count=model.performance.total_matches_played if result.source == "direct" else None,
            transfer_details=transfer_details,
            last_updated=datetime.utcnow().isoformat() + "Z",
        )
    
    @strawberry.field
    async def model_ratings(
        self,
        info,
        model_id: str,
        domain: str
    ) -> Optional[ModelAllRatingsType]:
        """Get all dimension ratings for a model in a domain."""
        from langscope.api.dependencies import get_model_by_id, get_faceted_transfer
        
        model = get_model_by_id(model_id)
        if not model:
            return None
        
        dimensions = [
            "raw_quality", "cost_adjusted", "latency", "ttft",
            "consistency", "token_efficiency", "instruction_following",
            "hallucination_resistance", "long_context", "combined"
        ]
        
        faceted_transfer = get_faceted_transfer()
        ratings = []
        
        for dim in dimensions:
            result = faceted_transfer.get_rating_or_transfer(model, domain, dim)
            ratings.append(DimensionRatingType(
                dimension=dim,
                mu=result.target_mu,
                sigma=result.target_sigma,
                conservative_estimate=result.target_mu - 3 * result.target_sigma,
                source=result.source,
                confidence=result.confidence,
            ))
        
        return ModelAllRatingsType(
            model_id=model_id,
            domain=domain,
            ratings=ratings,
        )
    
    @strawberry.field
    async def similar_domains(
        self,
        info,
        domain: str,
        limit: int = 10,
        min_correlation: float = 0.25
    ) -> List[SimilarDomainType]:
        """Get similar domains for transfer learning transparency."""
        from langscope.api.dependencies import get_domain_index
        
        domain_index = get_domain_index()
        similar = domain_index.get_similar_domains(
            domain, k=limit, min_correlation=min_correlation
        )
        
        result = []
        for name, corr in similar:
            breakdown = domain_index.get_facet_breakdown(name, domain)
            facet_contributions = []
            for facet, data in breakdown.items():
                facet_contributions.append(FacetContributionType(
                    facet=facet,
                    source_value=data.get("source_value", ""),
                    target_value=data.get("target_value", ""),
                    similarity=data.get("similarity", 0.0),
                    weight=data.get("weight", 0.0),
                    contribution=data.get("contribution", 0.0),
                ))
            
            result.append(SimilarDomainType(
                name=name,
                correlation=corr,
                facet_breakdown=facet_contributions,
            ))
        
        return result
    
    @strawberry.field
    async def domain_facets(self, info, domain: str) -> DomainFacetsType:
        """Get facets for a domain."""
        from langscope.api.dependencies import get_domain_index
        
        domain_index = get_domain_index()
        descriptor = domain_index.get_or_create_descriptor(domain)
        facets = descriptor.get_all_facets()
        
        return DomainFacetsType(
            name=domain,
            language=facets.get("language", "english"),
            field=facets.get("field", "general"),
            modality=facets.get("modality", "text"),
            task=facets.get("task", "general"),
            specialty=facets.get("specialty", "general"),
        )
    
    @strawberry.field
    async def explain_correlation(
        self,
        info,
        source: str,
        target: str
    ) -> DomainSimilarityExplanationType:
        """Explain the correlation between two domains."""
        from langscope.api.dependencies import get_domain_index
        
        domain_index = get_domain_index()
        correlation = domain_index.get_correlation(source, target)
        breakdown = domain_index.get_facet_breakdown(source, target)
        
        facet_contributions = []
        for facet, data in breakdown.items():
            facet_contributions.append(FacetContributionType(
                facet=facet,
                source_value=data.get("source_value", ""),
                target_value=data.get("target_value", ""),
                similarity=data.get("similarity", 0.0),
                weight=data.get("weight", 0.0),
                contribution=data.get("contribution", 0.0),
            ))
        
        return DomainSimilarityExplanationType(
            source=source,
            target=target,
            correlation=correlation,
            facet_contributions=facet_contributions,
        )
    
    @strawberry.field
    async def faceted_leaderboard(
        self,
        info,
        domain: str,
        dimension: str = "raw_quality",
        include_transferred: bool = True,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> FacetedLeaderboardType:
        """Get leaderboard with transfer-included entries."""
        from langscope.api.dependencies import get_models, get_faceted_transfer
        from datetime import datetime
        
        models = get_models()
        faceted_transfer = get_faceted_transfer()
        
        entries = []
        direct_count = 0
        transferred_count = 0
        
        for model in models:
            result = faceted_transfer.get_rating_or_transfer(model, domain, dimension)
            
            if result.confidence < min_confidence:
                continue
            
            if not include_transferred and result.source == "transfer":
                continue
            
            if result.source == "direct":
                direct_count += 1
            else:
                transferred_count += 1
            
            transfer_note = None
            if result.source == "transfer" and result.source_domains:
                transfer_note = f"From {', '.join(result.source_domains[:3])}"
            
            entries.append({
                "model_id": model.model_id,
                "mu": result.target_mu,
                "sigma": result.target_sigma,
                "conservative_estimate": result.target_mu - 3 * result.target_sigma,
                "source": result.source,
                "confidence": result.confidence,
                "transfer_note": transfer_note,
            })
        
        # Sort by conservative estimate
        entries.sort(key=lambda x: -x["conservative_estimate"])
        
        ranked_entries = []
        for i, entry in enumerate(entries[:limit], 1):
            ranked_entries.append(FacetedLeaderboardEntryType(
                rank=i,
                model_id=entry["model_id"],
                deployment_id=None,
                mu=entry["mu"],
                sigma=entry["sigma"],
                conservative_estimate=entry["conservative_estimate"],
                source=entry["source"],
                confidence=entry["confidence"],
                transfer_note=entry["transfer_note"],
            ))
        
        return FacetedLeaderboardType(
            domain=domain,
            dimension=dimension,
            evaluation_type="subjective",
            entries=ranked_entries,
            total_models=len(entries),
            direct_count=direct_count,
            transferred_count=transferred_count,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
    
    @strawberry.field
    async def domain_index_stats(self, info) -> DomainIndexStatsType:
        """Get statistics about the domain index."""
        from langscope.api.dependencies import get_domain_index
        
        domain_index = get_domain_index()
        
        return DomainIndexStatsType(
            total_domains=len(domain_index.descriptors),
            domains_with_facets=sum(
                1 for d in domain_index.descriptors.values() if d.facets
            ),
            precomputed_similarities=len(domain_index._top_k_cache),
            last_refresh=domain_index.last_refresh.isoformat() + "Z" if domain_index.last_refresh else None,
        )

