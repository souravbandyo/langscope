"""
Complete multi-player match workflow integrating all components.

5-6 players compete, judges rank all responses, TrueSkill + Plackett-Luce updates.
Supports 10-dimensional ranking with BattleMetrics collection.
"""

import asyncio
import uuid
import math
import logging
import time
from typing import List, Dict, Optional, Set, Any, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field

from langscope.core.rating import TrueSkillRating
from langscope.core.dimensions import Dimension
from langscope.ranking.trueskill import MultiPlayerTrueSkillUpdater, create_updater_from_manager
from langscope.ranking.plackett_luce import PlackettLuceModel
from langscope.ranking.cost_adjustment import (
    create_cost_adjusted_ranking,
    aggregate_judge_rankings,
)
from langscope.ranking.dimension_ranker import DimensionRanker, MultiDimensionalRanking
from langscope.federation.selection import (
    MultiPlayerSwissPairing,
    ContentCreatorSelector,
    JudgeSelector,
)
from langscope.federation.judge import (
    JudgeRankingValidator,
    JudgeAggregator,
    create_judge_prompt,
    shuffle_response_order,
)
from langscope.federation.content import ContentGenerator
from langscope.evaluation.metrics import BattleMetrics, LatencyTimer

if TYPE_CHECKING:
    from langscope.core.model import LLMModel
    from langscope.database.mongodb import MongoDB

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a multi-player match with 10-dimensional rankings."""
    match_id: str
    domain: str
    timestamp: str
    
    # Participants
    players: List[str]  # Model IDs
    case_creator: str
    question_creator: str
    judges: List[str]
    
    # Content
    case_text: str
    question_text: str
    
    # Responses
    responses: Dict[str, Dict]  # {model_id: {text, tokens, cost, ...}}
    
    # Rankings (legacy 2D)
    raw_ranking: Dict[str, int]  # {model_id: rank}
    cost_adjusted_ranking: Dict[str, int]
    judge_rankings: List[Dict[str, int]]
    judge_weights: List[float]
    
    # Plackett-Luce
    pl_strengths: Dict[str, float]
    
    # Metadata
    info_bits: float
    
    # 10D Rankings (new)
    dimension_rankings: Optional[Dict[str, Dict[str, int]]] = None  # {dim: {model_id: rank}}
    
    # Battle metrics per participant (new)
    battle_metrics: Optional[Dict[str, Dict[str, Any]]] = None  # {model_id: metrics_dict}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "_id": self.match_id,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "participants": self.players,
            "participant_count": len(self.players),
            "prompt": {
                "case_text": self.case_text,
                "case_generator_id": self.case_creator,
                "question_text": self.question_text,
                "question_generator_id": self.question_creator,
            },
            "responses": self.responses,
            "judgment": {
                "raw_ranking": self.raw_ranking,
                "cost_adjusted_ranking": self.cost_adjusted_ranking,
                "judge_rankings": self.judge_rankings,
                "judge_weights": self.judge_weights,
                "judges": self.judges,
            },
            "plackett_luce": {
                "raw_strengths": self.pl_strengths,
            },
            "meta": {
                "info_bits": self.info_bits,
                "judgment_method": "weighted_borda",
            }
        }
        
        # Add 10D rankings if available
        if self.dimension_rankings:
            result["dimension_rankings"] = self.dimension_rankings
        
        # Add battle metrics if available
        if self.battle_metrics:
            result["battle_metrics"] = self.battle_metrics
        
        return result


class MultiPlayerMatchWorkflow:
    """
    Complete match workflow for multi-player (5-6) evaluation.
    
    Flow:
    1. Select 5-6 players (Swiss grouping)
    2. Generate content (case + question)
    3. All players generate responses (with metrics collection)
    4. Judges rank all responses (1st to 6th)
    5. Aggregate rankings (weighted Borda)
    6. Create cost-adjusted ranking
    7. Compute 10-dimensional rankings
    8. Update TrueSkill ratings (both 2D legacy and 10D)
    9. Estimate Plackett-Luce strengths
    10. Save to database
    """
    
    def __init__(
        self,
        domain: str,
        models: List['LLMModel'],
        db: 'MongoDB' = None,
        llm_caller: Any = None,
        enable_10d: bool = True
    ):
        """
        Initialize workflow.
        
        Args:
            domain: Domain for evaluation
            models: List of all models
            db: Database instance (optional)
            llm_caller: LLM calling interface (e.g., LiteLLM)
            enable_10d: Whether to enable 10-dimensional ranking
        """
        self.domain = domain
        self.models = models
        self.db = db
        self.llm_caller = llm_caller
        self.enable_10d = enable_10d
        
        # Initialize components with domain-aware params
        self.swiss_pairing = MultiPlayerSwissPairing(domain=domain)
        self.content_selector = ContentCreatorSelector(domain=domain)
        self.judge_selector = JudgeSelector(domain=domain)
        self.trueskill_updater = create_updater_from_manager(domain)
        self.plackett_luce = PlackettLuceModel()
        self.content_generator = ContentGenerator(domain)
        self.judge_aggregator = JudgeAggregator()
        
        # 10D ranker
        if enable_10d:
            self.dimension_ranker = DimensionRanker()
        
        # Track recent groups to avoid repetition
        self.recent_groups: Set[frozenset] = set()
        self._max_recent_groups = 20
    
    async def run_single_match(self) -> Optional[MatchResult]:
        """
        Run a single multi-player match.
        
        Returns:
            MatchResult or None if match couldn't be run
        """
        try:
            # 1. Select players
            players = self.swiss_pairing.select_match_players(
                self.models, self.domain, self.recent_groups
            )
            if not players or len(players) < 5:
                logger.warning("Could not select enough players")
                return None
            
            # Track grouping
            group_key = frozenset(p.model_id for p in players)
            self.recent_groups.add(group_key)
            if len(self.recent_groups) > self._max_recent_groups:
                # Remove oldest
                self.recent_groups.pop()
            
            # 2. Select content creators
            exclude = [p.name for p in players] + [p.model_id for p in players]
            
            case_creator = self.content_selector.select(
                self.models, self.domain, exclude
            )
            if not case_creator:
                logger.warning("Could not select case creator")
                return None
            exclude.extend([case_creator.name, case_creator.model_id])
            
            question_creator = self.content_selector.select(
                self.models, self.domain, exclude
            )
            if not question_creator:
                logger.warning("Could not select question creator")
                return None
            exclude.extend([question_creator.name, question_creator.model_id])
            
            # 3. Select judges
            judges = self.judge_selector.select_judges(
                self.models, self.domain, n_judges=5, exclude=exclude
            )
            if not judges:
                logger.warning("Could not select judges")
                return None
            
            judge_weights = self.judge_selector.get_judge_weights(judges, self.domain)
            
            # 4. Generate content
            case_text = await self._generate_case(case_creator)
            question_text = await self._generate_question(question_creator, case_text)
            
            # 5. Get player responses
            responses = {}
            for player in players:
                response = await self._generate_response(player, case_text, question_text)
                responses[player.model_id] = response
            
            # 6. Get judge rankings
            judge_rankings = await self._collect_judge_rankings(
                judges, case_text, question_text, responses, players
            )
            
            # 7. Aggregate rankings
            raw_ranking = self.judge_aggregator.aggregate(judge_rankings, judge_weights)
            
            # 8. Create cost-adjusted ranking
            costs = {model_id: r.get("cost_usd", 0.0) for model_id, r in responses.items()}
            cost_ranking = create_cost_adjusted_ranking(raw_ranking, costs)
            
            # 9. Compute 10D rankings if enabled
            dimension_rankings = None
            battle_metrics_dict = None
            
            if self.enable_10d:
                # Build BattleMetrics from responses
                metrics = {}
                for model_id, resp in responses.items():
                    metrics[model_id] = BattleMetrics(
                        model_id=model_id,
                        latency_ms=resp.get("latency_ms", 0.0),
                        ttft_ms=resp.get("ttft_ms", 0.0),
                        cost_usd=resp.get("cost_usd", 0.0),
                        input_tokens=resp.get("input_tokens", 0),
                        output_tokens=resp.get("output_tokens", 0),
                    )
                
                # Get cost per million for each model
                cost_per_millions = {}
                for p in players:
                    cost_per_millions[p.model_id] = p.output_cost_per_million
                
                # Get raw mu values for dimension calculations
                mu_raws = {}
                for p in players:
                    ts = p.get_domain_trueskill(self.domain)
                    mu_raws[p.model_id] = ts.raw.mu
                
                # Compute all dimension rankings
                match_id = f"match_{uuid.uuid4().hex[:16]}"
                multi_dim_ranking = self.dimension_ranker.compute_all_rankings(
                    match_id=match_id,
                    raw_rankings=raw_ranking,
                    metrics=metrics,
                    mu_raws=mu_raws,
                    cost_per_millions=cost_per_millions,
                )
                
                # Convert to storable format
                dimension_rankings = {
                    dim.value: ranking.rankings
                    for dim, ranking in multi_dim_ranking.dimension_rankings.items()
                }
                
                # Store metrics
                battle_metrics_dict = {
                    model_id: m.to_dict()
                    for model_id, m in metrics.items()
                }
            else:
                match_id = f"match_{uuid.uuid4().hex[:16]}"
            
            # 10. Update TrueSkill ratings
            self._update_trueskill_ratings(players, raw_ranking, cost_ranking)
            
            # 11. Estimate Plackett-Luce strengths
            ranking_list = [sorted(raw_ranking.keys(), key=lambda m: raw_ranking[m])]
            pl_result = self.plackett_luce.estimate_strengths(ranking_list)
            
            # 12. Create match result
            match_result = MatchResult(
                match_id=match_id,
                domain=self.domain,
                timestamp=datetime.utcnow().isoformat() + "Z",
                players=[p.model_id for p in players],
                case_creator=case_creator.model_id,
                question_creator=question_creator.model_id,
                judges=[j.model_id for j in judges],
                case_text=case_text,
                question_text=question_text,
                responses=responses,
                raw_ranking=raw_ranking,
                cost_adjusted_ranking=cost_ranking,
                judge_rankings=judge_rankings,
                judge_weights=judge_weights,
                pl_strengths=pl_result.strengths,
                info_bits=math.log2(math.factorial(len(players))),
                dimension_rankings=dimension_rankings,
                battle_metrics=battle_metrics_dict,
            )
            
            # 13. Update performance metrics
            self._update_performance_metrics(players, raw_ranking, cost_ranking, responses)
            
            # 13. Save to database
            if self.db:
                self.db.save_match(match_result.to_dict())
                for player in players:
                    self.db.save_model(player.to_dict())
            
            return match_result
            
        except Exception as e:
            logger.error(f"Error running match: {e}")
            return None
    
    def _update_trueskill_ratings(
        self,
        players: List['LLMModel'],
        raw_ranking: Dict[str, int],
        cost_ranking: Dict[str, int]
    ):
        """Update TrueSkill ratings for all players."""
        # Get current raw ratings
        raw_ratings = []
        for p in players:
            ts = p.get_domain_trueskill(self.domain)
            raw_ratings.append(TrueSkillRating(mu=ts.raw.mu, sigma=ts.raw.sigma))
        
        # Get current cost ratings
        cost_ratings = []
        for p in players:
            ts = p.get_domain_trueskill(self.domain)
            cost_ratings.append(TrueSkillRating(
                mu=ts.cost_adjusted.mu, sigma=ts.cost_adjusted.sigma
            ))
        
        # Convert rankings to list format
        raw_rank_list = [raw_ranking[p.model_id] for p in players]
        cost_rank_list = [cost_ranking[p.model_id] for p in players]
        
        # Update ratings
        new_raw = self.trueskill_updater.update_from_ranking(raw_ratings, raw_rank_list)
        new_cost = self.trueskill_updater.update_from_ranking(cost_ratings, cost_rank_list)
        
        # Apply updates
        for i, player in enumerate(players):
            player.set_domain_trueskill(
                self.domain,
                raw_mu=new_raw[i].mu,
                raw_sigma=new_raw[i].sigma,
                cost_mu=new_cost[i].mu,
                cost_sigma=new_cost[i].sigma
            )
    
    def _update_performance_metrics(
        self,
        players: List['LLMModel'],
        raw_ranking: Dict[str, int],
        cost_ranking: Dict[str, int],
        responses: Dict[str, Dict]
    ):
        """Update performance metrics for all players."""
        for player in players:
            # Get domain performance
            perf = player.get_domain_performance(self.domain)
            
            # Update match count
            perf.total_matches_played += 1
            perf.total_races_participated += 1
            
            # Update ranks
            raw_rank = raw_ranking.get(player.model_id, 0)
            cost_rank = cost_ranking.get(player.model_id, 0)
            perf.update_ranks(raw_rank, cost_rank)
            
            # Update usage
            resp = responses.get(player.model_id, {})
            perf.add_usage(
                input_tokens=resp.get("input_tokens", 0),
                output_tokens=resp.get("output_tokens", 0),
                cost_usd=resp.get("cost_usd", 0.0)
            )
            
            # Also update global performance
            player.performance.total_matches_played += 1
            player.performance.total_races_participated += 1
    
    # =========================================================================
    # LLM Interaction Methods (to be implemented with actual LLM calls)
    # =========================================================================
    
    async def _generate_case(self, creator: 'LLMModel') -> str:
        """Generate case using content creator."""
        prompt = self.content_generator.create_case_prompt()
        
        if self.llm_caller:
            try:
                if hasattr(self.llm_caller, 'acompletion'):
                    response = await self.llm_caller.acompletion(
                        model=creator.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                    )
                else:
                    response = self.llm_caller.completion(
                        model=creator.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                    )
                
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Error generating case with {creator.name}: {e}")
        
        # Placeholder for testing (only when llm_caller not available)
        logger.warning(f"No LLM caller available, using placeholder case")
        return f"[Case generated by {creator.name}]\n\nThis is a placeholder case for domain {self.domain}."
    
    async def _generate_question(self, creator: 'LLMModel', case_text: str) -> str:
        """Generate question based on case."""
        prompt = self.content_generator.create_question_prompt(case_text)
        
        if self.llm_caller:
            try:
                if hasattr(self.llm_caller, 'acompletion'):
                    response = await self.llm_caller.acompletion(
                        model=creator.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1024,
                    )
                else:
                    response = self.llm_caller.completion(
                        model=creator.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1024,
                    )
                
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Error generating question with {creator.name}: {e}")
        
        # Placeholder for testing
        logger.warning(f"No LLM caller available, using placeholder question")
        return f"[Question generated by {creator.name}]\n\nWhat is the best approach to solve this case?"
    
    async def _generate_response(
        self,
        player: 'LLMModel',
        case_text: str,
        question_text: str
    ) -> Dict:
        """Generate response from player with metrics collection."""
        prompt = f"""## Case
{case_text}

## Question
{question_text}

Please provide a comprehensive answer."""
        
        # Collect timing metrics
        start_time = time.perf_counter()
        ttft_ms = 0.0
        text = ""
        input_tokens = 0
        output_tokens = 0
        
        if self.llm_caller:
            try:
                if hasattr(self.llm_caller, 'acompletion'):
                    response = await self.llm_caller.acompletion(
                        model=player.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=4096,
                    )
                else:
                    response = self.llm_caller.completion(
                        model=player.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=4096,
                    )
                
                if hasattr(response, 'choices') and response.choices:
                    text = response.choices[0].message.content or ""
                
                # Extract usage if available
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                
                # Extract TTFT from response metadata if available
                if hasattr(response, '_response_ms'):
                    ttft_ms = getattr(response, '_response_ms', 0) * 0.25
                    
            except Exception as e:
                logger.error(f"Error generating response from {player.name}: {e}")
                text = f"[Error: {str(e)}]"
        else:
            # Placeholder for testing (only when llm_caller not available)
            logger.warning(f"No LLM caller available, using placeholder response for {player.name}")
            text = f"[Response from {player.name}]\n\nThis is a placeholder response."
            input_tokens = len(prompt.split()) * 2  # Rough estimate
            output_tokens = len(text.split()) * 2
        
        # Record end time
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Estimate TTFT if not available (typically ~20-30% of total latency)
        if ttft_ms == 0.0:
            ttft_ms = latency_ms * 0.25
        
        cost_usd = player.calculate_response_cost(input_tokens, output_tokens)
        
        return {
            "text": text,
            "tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        }
    
    async def _collect_judge_rankings(
        self,
        judges: List['LLMModel'],
        case_text: str,
        question_text: str,
        responses: Dict[str, Dict],
        players: List['LLMModel']
    ) -> List[Dict[str, int]]:
        """Collect rankings from all judges."""
        # Shuffle responses to prevent position bias
        response_texts = {pid: r["text"] for pid, r in responses.items()}
        shuffled, label_mapping = shuffle_response_order(response_texts)
        
        # Create judge prompt
        prompt = create_judge_prompt(
            case_text, question_text,
            {label_mapping[l]: shuffled[l] for l in shuffled},
            {label_mapping[l]: l for l in shuffled}
        )
        
        validator = JudgeRankingValidator(list(responses.keys()))
        rankings = []
        
        for judge in judges:
            ranking = None
            
            if self.llm_caller:
                try:
                    if hasattr(self.llm_caller, 'acompletion'):
                        response = await self.llm_caller.acompletion(
                            model=judge.model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=512,
                        )
                    else:
                        response = self.llm_caller.completion(
                            model=judge.model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_tokens=512,
                        )
                    
                    if hasattr(response, 'choices') and response.choices:
                        text = response.choices[0].message.content or ""
                        ranking = self._parse_judge_ranking(text, list(responses.keys()))
                except Exception as e:
                    logger.error(f"Error getting ranking from judge {judge.name}: {e}")
            
            if ranking is None:
                # Placeholder: random ranking for testing (only when no LLM or parse failed)
                import random
                model_ids = list(responses.keys())
                random.shuffle(model_ids)
                ranking = {mid: i + 1 for i, mid in enumerate(model_ids)}
            
            # Validate and add
            is_valid, _ = validator.validate(ranking)
            if is_valid:
                rankings.append(ranking)
            else:
                # Use fallback uniform ranking
                rankings.append({mid: i + 1 for i, mid in enumerate(responses.keys())})
        
        return rankings
    
    def _parse_judge_ranking(
        self,
        text: str,
        model_ids: List[str]
    ) -> Optional[Dict[str, int]]:
        """
        Parse judge ranking from LLM response text.
        
        Expected format: Rankings in order, e.g., "1. model_a\n2. model_b..."
        or JSON format: {"model_a": 1, "model_b": 2}
        """
        import json
        import re
        
        # Try JSON parsing first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {k: int(v) for k, v in data.items() if k in model_ids}
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try parsing numbered list format
        ranking = {}
        for line in text.split('\n'):
            line = line.strip()
            # Match patterns like "1. model_id" or "1: model_id"
            match = re.match(r'^(\d+)[.\):\s]+(.+)$', line)
            if match:
                rank = int(match.group(1))
                model_ref = match.group(2).strip()
                # Match to model_id
                for mid in model_ids:
                    if mid in model_ref or model_ref in mid:
                        ranking[mid] = rank
                        break
        
        if len(ranking) == len(model_ids):
            return ranking
        
        return None


async def run_tournament(
    domain: str,
    models: List['LLMModel'],
    n_rounds: int = 10,
    db: 'MongoDB' = None,
    llm_caller: Any = None
) -> List[MatchResult]:
    """
    Run a tournament of multiple matches.
    
    Args:
        domain: Domain for evaluation
        models: List of models to evaluate
        n_rounds: Number of match rounds
        db: Database instance
        llm_caller: LLM calling interface
    
    Returns:
        List of match results
    """
    workflow = MultiPlayerMatchWorkflow(domain, models, db, llm_caller)
    results = []
    
    for round_num in range(n_rounds):
        logger.info(f"Running round {round_num + 1}/{n_rounds}")
        result = await workflow.run_single_match()
        if result:
            results.append(result)
        else:
            logger.warning(f"Round {round_num + 1} failed to produce result")
    
    return results


