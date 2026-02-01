"""
GraphQL mutation resolvers for LangScope.
"""

import strawberry
from typing import Optional, List
from datetime import datetime


# === Input Types ===

@strawberry.input
class ProviderInput:
    """Input for provider information."""
    name: str
    region: str = "us"
    endpoint: str = ""


@strawberry.input
class PricingInput:
    """Input for pricing information."""
    input_cost_per_million: float
    output_cost_per_million: float
    currency: str = "USD"


@strawberry.input
class CreateDeploymentInput:
    """Input for creating a new deployment."""
    base_model_id: str
    name: str
    provider: ProviderInput
    pricing: PricingInput


@strawberry.input
class UpdateDeploymentInput:
    """Input for updating a deployment."""
    name: Optional[str] = None
    pricing: Optional[PricingInput] = None
    available: Optional[bool] = None


@strawberry.input
class RegisterSelfHostedInput:
    """Input for registering a self-hosted deployment."""
    base_model_id: str
    name: str
    hardware_config: str  # JSON string for flexibility
    software_config: str  # JSON string for flexibility


@strawberry.input
class SubmitFeedbackInput:
    """Input for submitting user feedback on a match."""
    match_id: str
    deployment_id: str
    feedback_type: str  # "agree", "disagree", "report"
    comment: Optional[str] = None


# === Result Types ===

@strawberry.type
class MutationResult:
    """Generic mutation result."""
    success: bool
    message: str
    id: Optional[str] = None


@strawberry.type
class DeploymentMutationResult:
    """Result of a deployment mutation."""
    success: bool
    message: str
    deployment_id: Optional[str] = None


@strawberry.type
class MatchMutationResult:
    """Result of a match-related mutation."""
    success: bool
    message: str
    match_id: Optional[str] = None


# === Mutations ===

@strawberry.type
class Mutation:
    """Root GraphQL mutation type."""
    
    # === Deployment Management ===
    
    @strawberry.mutation
    async def create_deployment(
        self,
        info,
        input: CreateDeploymentInput
    ) -> DeploymentMutationResult:
        """Create a new model deployment."""
        db = info.context.get("db")
        if not db:
            return DeploymentMutationResult(
                success=False,
                message="Database not available",
            )
        
        # Verify base model exists
        base_model = await db.get_base_model(input.base_model_id)
        if not base_model:
            return DeploymentMutationResult(
                success=False,
                message=f"Base model {input.base_model_id} not found",
            )
        
        # Create deployment document
        now = datetime.utcnow()
        deployment = {
            "base_model_id": input.base_model_id,
            "name": input.name,
            "provider": {
                "name": input.provider.name,
                "region": input.provider.region,
                "endpoint": input.provider.endpoint,
            },
            "pricing": {
                "input_cost_per_million": input.pricing.input_cost_per_million,
                "output_cost_per_million": input.pricing.output_cost_per_million,
                "currency": input.pricing.currency,
                "effective_date": now.isoformat() + "Z",
            },
            "performance": {
                "median_latency_ms": 0,
                "p95_latency_ms": 0,
                "median_ttft_ms": 0,
                "p95_ttft_ms": 0,
                "tokens_per_second": 0,
                "uptime_pct": 100.0,
                "sample_size": 0,
            },
            "available": True,
            "created_at": now,
            "updated_at": now,
        }
        
        deployment_id = await db.save_model_deployment(deployment)
        
        return DeploymentMutationResult(
            success=True,
            message="Deployment created successfully",
            deployment_id=deployment_id,
        )
    
    @strawberry.mutation
    async def update_deployment(
        self,
        info,
        id: str,
        input: UpdateDeploymentInput
    ) -> DeploymentMutationResult:
        """Update an existing deployment."""
        db = info.context.get("db")
        if not db:
            return DeploymentMutationResult(
                success=False,
                message="Database not available",
            )
        
        # Get existing deployment
        deployment = await db.get_model_deployment(id)
        if not deployment:
            return DeploymentMutationResult(
                success=False,
                message=f"Deployment {id} not found",
            )
        
        # Build update
        update = {"updated_at": datetime.utcnow()}
        
        if input.name is not None:
            update["name"] = input.name
        
        if input.pricing is not None:
            update["pricing"] = {
                "input_cost_per_million": input.pricing.input_cost_per_million,
                "output_cost_per_million": input.pricing.output_cost_per_million,
                "currency": input.pricing.currency,
                "effective_date": datetime.utcnow().isoformat() + "Z",
            }
        
        if input.available is not None:
            update["available"] = input.available
        
        await db.update_model_deployment(id, update)
        
        return DeploymentMutationResult(
            success=True,
            message="Deployment updated successfully",
            deployment_id=id,
        )
    
    @strawberry.mutation
    async def delete_deployment(
        self,
        info,
        id: str
    ) -> MutationResult:
        """Delete a deployment."""
        db = info.context.get("db")
        if not db:
            return MutationResult(
                success=False,
                message="Database not available",
            )
        
        deleted = await db.delete_model_deployment(id)
        
        return MutationResult(
            success=deleted,
            message="Deployment deleted" if deleted else "Deployment not found",
            id=id if deleted else None,
        )
    
    # === Self-Hosted Management ===
    
    @strawberry.mutation
    async def register_self_hosted(
        self,
        info,
        input: RegisterSelfHostedInput
    ) -> DeploymentMutationResult:
        """Register a self-hosted deployment."""
        import json
        
        db = info.context.get("db")
        if not db:
            return DeploymentMutationResult(
                success=False,
                message="Database not available",
            )
        
        # Parse JSON configs
        try:
            hardware = json.loads(input.hardware_config)
            software = json.loads(input.software_config)
        except json.JSONDecodeError as e:
            return DeploymentMutationResult(
                success=False,
                message=f"Invalid JSON config: {e}",
            )
        
        # Get user from context
        user = info.context.get("user", {})
        owner_id = user.get("id", "anonymous")
        
        now = datetime.utcnow()
        deployment = {
            "base_model_id": input.base_model_id,
            "owner_id": owner_id,
            "name": input.name,
            "hardware_config": hardware,
            "software_config": software,
            "available": True,
            "created_at": now,
            "updated_at": now,
        }
        
        deployment_id = await db.save_self_hosted_deployment(deployment)
        
        return DeploymentMutationResult(
            success=True,
            message="Self-hosted deployment registered",
            deployment_id=deployment_id,
        )
    
    # === Feedback ===
    
    @strawberry.mutation
    async def submit_feedback(
        self,
        info,
        input: SubmitFeedbackInput
    ) -> MutationResult:
        """Submit user feedback on a match result."""
        db = info.context.get("db")
        if not db:
            return MutationResult(
                success=False,
                message="Database not available",
            )
        
        # Get user from context
        user = info.context.get("user", {})
        user_id = user.get("id", "anonymous")
        
        feedback = {
            "match_id": input.match_id,
            "deployment_id": input.deployment_id,
            "user_id": user_id,
            "feedback_type": input.feedback_type,
            "comment": input.comment,
            "created_at": datetime.utcnow(),
        }
        
        feedback_id = await db.save_feedback(feedback)
        
        return MutationResult(
            success=True,
            message="Feedback submitted",
            id=feedback_id,
        )
    
    # === Match Operations ===
    
    @strawberry.mutation
    async def trigger_match(
        self,
        info,
        domain: str,
        deployment_ids: Optional[List[str]] = None
    ) -> MatchMutationResult:
        """Trigger a new match evaluation."""
        db = info.context.get("db")
        match_service = info.context.get("match_service")
        
        if not db or not match_service:
            return MatchMutationResult(
                success=False,
                message="Required services not available",
            )
        
        try:
            match_id = await match_service.create_match(
                domain=domain,
                deployment_ids=deployment_ids,
            )
            
            return MatchMutationResult(
                success=True,
                message="Match created",
                match_id=match_id,
            )
        except Exception as e:
            return MatchMutationResult(
                success=False,
                message=str(e),
            )
    
    # === Ground Truth Operations ===
    
    @strawberry.mutation
    async def trigger_ground_truth_match(
        self,
        info,
        domain: str,
        deployment_ids: List[str],
        sample_id: Optional[str] = None,
        filters: Optional[str] = None  # JSON string for filters
    ) -> MatchMutationResult:
        """
        Trigger a ground truth evaluation match.
        
        Uses objective metrics (WER, BLEU, etc.) instead of LLM judges.
        Supports domains: asr, tts, visual_qa, needle_in_haystack, code_completion, etc.
        
        Args:
            domain: Ground truth domain name
            deployment_ids: List of deployment IDs to evaluate (min 2)
            sample_id: Optional specific sample to use
            filters: Optional JSON filters for sample selection
        """
        import json
        
        db = info.context.get("db")
        llm_caller = info.context.get("llm_caller")
        
        if not db:
            return MatchMutationResult(
                success=False,
                message="Database not available",
            )
        
        if len(deployment_ids) < 2:
            return MatchMutationResult(
                success=False,
                message="At least 2 deployment IDs required",
            )
        
        try:
            # Parse filters if provided
            filter_dict = {}
            if filters:
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    pass
            
            # Import workflow
            from langscope.ground_truth.workflow import GroundTruthMatchWorkflow
            from langscope.core.model import LLMModel
            
            # Get models
            models = []
            for dep_id in deployment_ids:
                deployment = db.get_deployment(dep_id)
                if deployment:
                    model = LLMModel.from_dict(deployment)
                    models.append(model)
            
            if len(models) < 2:
                return MatchMutationResult(
                    success=False,
                    message="Could not find at least 2 valid deployments",
                )
            
            # Create and run workflow
            workflow = GroundTruthMatchWorkflow(
                domain=domain,
                models=models,
                db=db,
                llm_caller=llm_caller
            )
            
            result = await workflow.run_single_match(
                model_ids=deployment_ids,
                sample_id=sample_id,
                filters=filter_dict
            )
            
            if result and result.status == "completed":
                return MatchMutationResult(
                    success=True,
                    message="Ground truth match completed",
                    match_id=result.match_id,
                )
            else:
                return MatchMutationResult(
                    success=False,
                    message=result.error_message if result else "Match failed",
                )
                
        except Exception as e:
            return MatchMutationResult(
                success=False,
                message=str(e),
            )
    
    @strawberry.mutation
    async def trigger_ground_truth_tournament(
        self,
        info,
        domain: str,
        deployment_ids: List[str],
        n_rounds: int = 10
    ) -> MutationResult:
        """
        Trigger a multi-round ground truth tournament.
        
        Runs multiple matches with random samples to build robust rankings.
        
        Args:
            domain: Ground truth domain name
            deployment_ids: List of deployment IDs to evaluate
            n_rounds: Number of rounds to run
        """
        db = info.context.get("db")
        llm_caller = info.context.get("llm_caller")
        
        if not db:
            return MutationResult(
                success=False,
                message="Database not available",
            )
        
        if len(deployment_ids) < 2:
            return MutationResult(
                success=False,
                message="At least 2 deployment IDs required",
            )
        
        if n_rounds < 1 or n_rounds > 100:
            return MutationResult(
                success=False,
                message="n_rounds must be between 1 and 100",
            )
        
        try:
            from langscope.ground_truth.workflow import run_ground_truth_tournament
            from langscope.core.model import LLMModel
            
            # Get models
            models = []
            for dep_id in deployment_ids:
                deployment = db.get_deployment(dep_id)
                if deployment:
                    model = LLMModel.from_dict(deployment)
                    models.append(model)
            
            if len(models) < 2:
                return MutationResult(
                    success=False,
                    message="Could not find at least 2 valid deployments",
                )
            
            # Run tournament
            results = await run_ground_truth_tournament(
                domain=domain,
                models=models,
                n_rounds=n_rounds,
                db=db,
                llm_caller=llm_caller
            )
            
            completed = sum(1 for r in results if r.status == "completed")
            
            return MutationResult(
                success=True,
                message=f"Tournament completed: {completed}/{n_rounds} rounds successful",
                id=results[0].match_id if results else None,
            )
            
        except Exception as e:
            return MutationResult(
                success=False,
                message=str(e),
            )

