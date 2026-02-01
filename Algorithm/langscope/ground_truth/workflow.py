"""
Ground Truth Match Workflow.

Complete workflow for running ground truth evaluation matches:
1. Select stratified sample
2. Collect responses from models
3. Evaluate responses against ground truth
4. Compute rankings
5. Update TrueSkill ratings
6. Persist results
"""

import asyncio
import uuid
import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime

from langscope.ground_truth.manager import GroundTruthManager, GroundTruthSample
from langscope.ground_truth.judge import (
    EvaluationMode,
    GroundTruthScore,
    GroundTruthJudge,
)
from langscope.ground_truth.metrics import MetricRegistry
from langscope.core.rating import TrueSkillRating
from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater, create_updater_from_manager

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthMatchResult:
    """Result of a ground truth evaluation match."""
    match_id: str
    match_type: str = "ground_truth"
    domain: str = ""
    category: str = ""
    timestamp: str = ""
    
    # Sample reference
    sample_id: str = ""
    sample_version: str = ""
    ground_truth_hash: str = ""
    sample_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Responses
    responses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scores
    scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Rankings
    rankings: Dict[str, int] = field(default_factory=dict)
    
    # TrueSkill updates
    trueskill_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    evaluation_mode: str = "metrics_only"
    metrics_used: List[str] = field(default_factory=list)
    participant_count: int = 0
    info_bits: float = 0.0
    status: str = "completed"
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "_id": self.match_id,
            "match_type": self.match_type,
            "domain": self.domain,
            "category": self.category,
            "timestamp": self.timestamp,
            "sample_id": self.sample_id,
            "sample_version": self.sample_version,
            "ground_truth_hash": self.ground_truth_hash,
            "sample_metadata": self.sample_metadata,
            "responses": self.responses,
            "scores": self.scores,
            "rankings": self.rankings,
            "trueskill_updates": self.trueskill_updates,
            "evaluation_mode": self.evaluation_mode,
            "metrics_used": self.metrics_used,
            "participant_count": self.participant_count,
            "info_bits": self.info_bits,
            "status": self.status,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


class GroundTruthMatchWorkflow:
    """
    Workflow for ground truth evaluation matches.
    
    Flow:
    1. Select stratified sample from dataset
    2. Collect responses from all participating models
    3. Evaluate responses with ground truth judge
    4. Compute rankings from scores
    5. Update TrueSkill ratings
    6. Persist match result to database
    """
    
    def __init__(
        self,
        domain: str,
        models: List['LLMModel'] = None,
        db: 'MongoDB' = None,
        llm_caller: Any = None,
        evaluation_mode: EvaluationMode = EvaluationMode.METRICS_ONLY,
        config: Dict = None
    ):
        """
        Initialize workflow.
        
        Args:
            domain: Domain for evaluation
            models: List of models to evaluate (optional)
            db: Database instance
            llm_caller: LLM calling interface
            evaluation_mode: Evaluation mode
            config: Additional configuration
        """
        self.domain = domain
        self.models = models or []
        self.db = db
        self.llm_caller = llm_caller
        self.evaluation_mode = evaluation_mode
        self.config = config or {}
        
        # Initialize components
        self.manager = GroundTruthManager(db=db)
        self.judge = GroundTruthJudge(
            domain=domain,
            llm_caller=llm_caller,
            config=config
        )
        self.trueskill_updater = create_updater_from_manager(domain)
        self.metric_registry = MetricRegistry()
    
    async def run_single_match(
        self,
        model_ids: List[str] = None,
        sample_id: str = None,
        filters: Dict[str, Any] = None
    ) -> Optional[GroundTruthMatchResult]:
        """
        Run a single ground truth evaluation match.
        
        Args:
            model_ids: Specific model IDs to evaluate (optional)
            sample_id: Specific sample to use (optional)
            filters: Filters for sample selection
        
        Returns:
            GroundTruthMatchResult or None
        """
        start_time = time.perf_counter()
        match_id = f"gt_match_{uuid.uuid4().hex[:16]}"
        
        try:
            # 1. Select sample
            sample = await self._select_sample(sample_id, filters)
            if not sample:
                logger.warning("Could not select sample")
                return None
            
            # Load ground truth
            ground_truth = sample.load_ground_truth()
            if ground_truth is None:
                logger.warning(f"Could not load ground truth for {sample.sample_id}")
                return None
            
            # 2. Get models to evaluate
            models = self._get_models(model_ids)
            if len(models) < 2:
                logger.warning("Need at least 2 models")
                return None
            
            # 3. Collect responses
            responses = await self._collect_responses(sample, models)
            
            # 4. Evaluate responses
            scores = await self._evaluate_responses(responses, ground_truth, sample)
            
            # 5. Compute rankings
            rankings = self.judge.get_ranking_from_scores(scores)
            
            # 6. Update TrueSkill
            trueskill_updates = self._update_trueskill(models, rankings)
            
            # Build result
            end_time = time.perf_counter()
            
            result = GroundTruthMatchResult(
                match_id=match_id,
                domain=self.domain,
                category=sample.category,
                sample_id=sample.sample_id,
                sample_version=sample.version,
                ground_truth_hash=sample.ground_truth_hash,
                sample_metadata={
                    "difficulty": sample.difficulty,
                    "language": sample.language,
                    "context_length": sample.context_length,
                    "needle_position": sample.needle_position,
                },
                responses=[
                    {
                        "model_id": r["model_id"],
                        "response_text": r.get("response_text", ""),
                        "latency_ms": r.get("latency_ms", 0),
                        "cost_usd": r.get("cost_usd", 0),
                        "error": r.get("error"),
                    }
                    for r in responses.values()
                ],
                scores={mid: s.to_dict() for mid, s in scores.items()},
                rankings=rankings,
                trueskill_updates=trueskill_updates,
                evaluation_mode=self.evaluation_mode.value,
                metrics_used=self.metric_registry.get_metrics_for_domain(self.domain),
                participant_count=len(models),
                info_bits=math.log2(math.factorial(len(models))) if len(models) > 1 else 0,
                duration_ms=(end_time - start_time) * 1000,
            )
            
            # 7. Persist
            self._persist_match(result, sample)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GT match: {e}")
            return GroundTruthMatchResult(
                match_id=match_id,
                domain=self.domain,
                status="failed",
                error_message=str(e),
            )
    
    async def _select_sample(
        self,
        sample_id: str = None,
        filters: Dict[str, Any] = None
    ) -> Optional[GroundTruthSample]:
        """Select a sample for evaluation."""
        if sample_id:
            return self.manager.load_sample(sample_id)
        return self.manager.get_random_sample(self.domain, filters)
    
    def _get_models(self, model_ids: List[str] = None) -> List['LLMModel']:
        """Get models to evaluate."""
        if model_ids:
            return [m for m in self.models if m.model_id in model_ids]
        return self.models
    
    async def _collect_responses(
        self,
        sample: GroundTruthSample,
        models: List['LLMModel']
    ) -> Dict[str, Dict[str, Any]]:
        """Collect responses from all models."""
        responses = {}
        
        # Build prompt based on domain
        prompt = self._build_prompt(sample)
        
        for model in models:
            response = await self._generate_response(model, prompt, sample)
            responses[model.model_id] = response
        
        return responses
    
    def _build_prompt(self, sample: GroundTruthSample) -> str:
        """Build prompt from sample."""
        if sample.domain == "needle_in_haystack":
            haystack_path = sample.get_input_path("text")
            if haystack_path and os.path.exists(haystack_path):
                with open(haystack_path, "r") as f:
                    haystack = f.read()
                question = sample.ground_truth.get("question", "")
                return f"{haystack}\n\nQuestion: {question}"
        
        elif sample.domain == "asr":
            return "[Audio transcription task - audio file provided]"
        
        elif sample.domain == "visual_qa":
            question = sample.ground_truth.get("question", "")
            return f"[Image provided]\n\nQuestion: {question}"
        
        elif sample.domain in ("long_document_qa", "code_completion"):
            doc_path = sample.get_input_path("document") or sample.get_input_path("context")
            if doc_path and os.path.exists(doc_path):
                with open(doc_path, "r") as f:
                    document = f.read()
                question = sample.ground_truth.get("question", sample.ground_truth.get("prompt", ""))
                return f"{document}\n\n{question}"
        
        # Default: return inputs as string
        return str(sample.inputs)
    
    async def _generate_response(
        self,
        model: 'LLMModel',
        prompt: str,
        sample: GroundTruthSample
    ) -> Dict[str, Any]:
        """
        Generate response from a model.
        
        Uses the llm_caller (LiteLLM) for actual LLM API calls.
        Supports multimodal inputs for ASR, visual QA, etc.
        """
        start_time = time.perf_counter()
        response_text = ""
        cost_usd = 0.0
        ttft_ms = 0.0
        input_tokens = 0
        output_tokens = 0
        
        try:
            if self.llm_caller:
                # Build messages based on domain
                messages = self._build_messages(prompt, sample)
                
                # Build API call parameters
                call_params = {
                    "model": model.model_id,
                    "messages": messages,
                    "temperature": self.config.get("temperature", 0),
                    "max_tokens": self.config.get("max_tokens", 4096),
                }
                
                # Add model-specific parameters if needed
                if sample.domain == "asr":
                    # For ASR, use speech-to-text endpoint if available
                    audio_path = sample.get_input_path("audio")
                    if audio_path and os.path.exists(audio_path):
                        call_params["audio"] = audio_path
                
                # Make the API call
                ttft_start = None
                
                # Use async completion if available
                if hasattr(self.llm_caller, 'acompletion'):
                    result = await self.llm_caller.acompletion(**call_params)
                else:
                    # Fallback to sync completion
                    result = self.llm_caller.completion(**call_params)
                
                # Extract response
                if hasattr(result, 'choices') and result.choices:
                    response_text = result.choices[0].message.content or ""
                    
                    # Extract usage if available
                    if hasattr(result, 'usage') and result.usage:
                        input_tokens = getattr(result.usage, 'prompt_tokens', 0)
                        output_tokens = getattr(result.usage, 'completion_tokens', 0)
                        
                        # Calculate cost if pricing available
                        cost_usd = self._calculate_cost(
                            model.model_id, input_tokens, output_tokens
                        )
            else:
                # No LLM caller configured - use placeholder for testing
                logger.warning(f"No LLM caller configured, using placeholder response")
                await asyncio.sleep(0.1)  # Simulate latency
                response_text = f"[Placeholder response from {model.name}]"
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            return {
                "model_id": model.model_id,
                "response_text": response_text,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "cost_usd": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            
        except Exception as e:
            logger.error(f"Error generating response from {model.name}: {e}")
            end_time = time.perf_counter()
            return {
                "model_id": model.model_id,
                "response_text": "",
                "latency_ms": (end_time - start_time) * 1000,
                "error": str(e),
            }
    
    def _build_messages(
        self,
        prompt: str,
        sample: GroundTruthSample
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM API call, including multimodal content."""
        messages = []
        
        # System message if configured
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build user message content
        content = []
        
        # Handle multimodal inputs
        if sample.domain == "visual_qa":
            image_path = sample.get_input_path("image")
            if image_path and os.path.exists(image_path):
                from langscope.ground_truth.utils.image import encode_image_base64
                try:
                    img_b64 = encode_image_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })
                except Exception as e:
                    logger.warning(f"Could not encode image: {e}")
        
        elif sample.domain in ("document_extraction", "ocr"):
            doc_path = sample.get_input_path("document") or sample.get_input_path("image")
            if doc_path and os.path.exists(doc_path):
                from langscope.ground_truth.utils.image import encode_image_base64
                try:
                    doc_b64 = encode_image_base64(doc_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{doc_b64}"
                        }
                    })
                except Exception as e:
                    logger.warning(f"Could not encode document: {e}")
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        # If no multimodal content, use simple string format
        if len(content) == 1 and content[0]["type"] == "text":
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": content})
        
        return messages
    
    def _calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for API call based on token usage."""
        # Try to get pricing from config or database
        pricing = self.config.get("pricing", {}).get(model_id, {})
        
        if not pricing:
            # Default pricing fallback (conservative estimate)
            pricing = {
                "input_cost_per_million": 1.0,
                "output_cost_per_million": 2.0,
            }
        
        input_cost = (input_tokens / 1_000_000) * pricing.get("input_cost_per_million", 1.0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output_cost_per_million", 2.0)
        
        return input_cost + output_cost
    
    async def _evaluate_responses(
        self,
        responses: Dict[str, Dict[str, Any]],
        ground_truth: Any,
        sample: GroundTruthSample
    ) -> Dict[str, GroundTruthScore]:
        """Evaluate all responses."""
        response_texts = {
            model_id: r.get("response_text", "")
            for model_id, r in responses.items()
        }
        
        sample_dict = sample.to_dict()
        
        return await self.judge.evaluate_batch(
            response_texts, ground_truth, sample_dict
        )
    
    def _update_trueskill(
        self,
        models: List['LLMModel'],
        rankings: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:
        """Update TrueSkill ratings."""
        updates = {}
        
        # Get current ratings
        current_ratings = []
        for model in models:
            ts = model.get_domain_trueskill(self.domain)
            current_ratings.append(TrueSkillRating(mu=ts.raw.mu, sigma=ts.raw.sigma))
        
        # Get ranking order
        rank_list = [rankings.get(m.model_id, len(models)) for m in models]
        
        # Update
        new_ratings = self.trueskill_updater.update_from_ranking(
            current_ratings, rank_list
        )
        
        # Apply and record updates
        for i, model in enumerate(models):
            before = {
                "mu": current_ratings[i].mu,
                "sigma": current_ratings[i].sigma
            }
            after = {
                "mu": new_ratings[i].mu,
                "sigma": new_ratings[i].sigma
            }
            
            updates[model.model_id] = {
                "before": before,
                "after": after,
            }
            
            # Apply to model
            model.set_domain_trueskill(
                self.domain,
                raw_mu=new_ratings[i].mu,
                raw_sigma=new_ratings[i].sigma,
                cost_mu=new_ratings[i].mu,  # For GT, raw == cost
                cost_sigma=new_ratings[i].sigma,
            )
        
        return updates
    
    def _persist_match(
        self,
        result: GroundTruthMatchResult,
        sample: GroundTruthSample
    ):
        """Persist match result to database."""
        if not self.db or not self.db.connected:
            return
        
        # Save match
        self.db.db["ground_truth_matches"].insert_one(result.to_dict())
        
        # Update sample usage
        self.manager.register_sample_usage(sample.sample_id, result.match_id)
        
        # Save model updates
        for model in self.models:
            if model.model_id in result.rankings:
                self.db.save_model(model.to_dict())


# Need to import os at the top
import os


async def run_ground_truth_tournament(
    domain: str,
    models: List['LLMModel'],
    n_rounds: int = 10,
    db: 'MongoDB' = None,
    llm_caller: Any = None
) -> List[GroundTruthMatchResult]:
    """
    Run a tournament of ground truth matches.
    
    Args:
        domain: Domain for evaluation
        models: Models to evaluate
        n_rounds: Number of rounds
        db: Database instance
        llm_caller: LLM caller interface
    
    Returns:
        List of match results
    """
    workflow = GroundTruthMatchWorkflow(
        domain=domain,
        models=models,
        db=db,
        llm_caller=llm_caller
    )
    
    results = []
    for i in range(n_rounds):
        logger.info(f"GT Tournament round {i+1}/{n_rounds}")
        result = await workflow.run_single_match()
        if result:
            results.append(result)
    
    return results

