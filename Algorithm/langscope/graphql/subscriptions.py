"""
GraphQL subscription resolvers for LangScope.

Provides real-time updates for:
- Rating changes
- New match results
- Price updates
"""

import strawberry
from typing import AsyncGenerator, Optional
from datetime import datetime
import asyncio


@strawberry.type
class RatingUpdateType:
    """Real-time rating update."""
    deployment_id: str
    domain: str
    dimension: str
    old_mu: float
    new_mu: float
    old_sigma: float
    new_sigma: float
    match_id: str
    timestamp: datetime


@strawberry.type
class MatchCompletedType:
    """Notification when a match completes."""
    match_id: str
    domain: str
    winner_deployment_id: str
    participants: int
    timestamp: datetime


@strawberry.type
class PriceUpdateType:
    """Notification when prices change."""
    deployment_id: str
    provider: str
    old_input_cost: float
    new_input_cost: float
    old_output_cost: float
    new_output_cost: float
    change_pct: float
    timestamp: datetime


@strawberry.type
class GroundTruthMatchCompletedType:
    """Notification when a ground truth match completes."""
    match_id: str
    domain: str
    sample_id: str
    participants: int
    winner_deployment_id: str
    primary_metric: str
    winner_score: float
    timestamp: datetime


@strawberry.type
class GroundTruthLeaderboardUpdateType:
    """Notification when ground truth leaderboard changes."""
    domain: str
    deployment_id: str
    old_rank: int
    new_rank: int
    old_mu: float
    new_mu: float
    primary_metric_avg: float
    timestamp: datetime


@strawberry.type
class Subscription:
    """Root GraphQL subscription type."""
    
    @strawberry.subscription
    async def rating_updates(
        self,
        info,
        deployment_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> AsyncGenerator[RatingUpdateType, None]:
        """
        Subscribe to rating updates.
        
        Optionally filter by deployment_id or domain.
        """
        pubsub = info.context.get("pubsub")
        
        if not pubsub:
            # Fallback: yield nothing
            return
        
        async for update in pubsub.subscribe("rating_updates"):
            # Apply filters
            if deployment_id and update.get("deployment_id") != deployment_id:
                continue
            if domain and update.get("domain") != domain:
                continue
            
            yield RatingUpdateType(
                deployment_id=update["deployment_id"],
                domain=update["domain"],
                dimension=update["dimension"],
                old_mu=update["old_mu"],
                new_mu=update["new_mu"],
                old_sigma=update["old_sigma"],
                new_sigma=update["new_sigma"],
                match_id=update["match_id"],
                timestamp=update.get("timestamp", datetime.utcnow()),
            )
    
    @strawberry.subscription
    async def match_completed(
        self,
        info,
        domain: Optional[str] = None
    ) -> AsyncGenerator[MatchCompletedType, None]:
        """Subscribe to match completion events."""
        pubsub = info.context.get("pubsub")
        
        if not pubsub:
            return
        
        async for event in pubsub.subscribe("match_completed"):
            if domain and event.get("domain") != domain:
                continue
            
            yield MatchCompletedType(
                match_id=event["match_id"],
                domain=event["domain"],
                winner_deployment_id=event["winner_deployment_id"],
                participants=event["participants"],
                timestamp=event.get("timestamp", datetime.utcnow()),
            )
    
    @strawberry.subscription
    async def price_updates(
        self,
        info,
        provider: Optional[str] = None
    ) -> AsyncGenerator[PriceUpdateType, None]:
        """Subscribe to price update events."""
        pubsub = info.context.get("pubsub")
        
        if not pubsub:
            return
        
        async for event in pubsub.subscribe("price_updates"):
            if provider and event.get("provider") != provider:
                continue
            
            yield PriceUpdateType(
                deployment_id=event["deployment_id"],
                provider=event["provider"],
                old_input_cost=event["old_input_cost"],
                new_input_cost=event["new_input_cost"],
                old_output_cost=event["old_output_cost"],
                new_output_cost=event["new_output_cost"],
                change_pct=event["change_pct"],
                timestamp=event.get("timestamp", datetime.utcnow()),
            )
    
    # === Ground Truth Subscriptions ===
    
    @strawberry.subscription
    async def ground_truth_match_completed(
        self,
        info,
        domain: Optional[str] = None
    ) -> AsyncGenerator[GroundTruthMatchCompletedType, None]:
        """
        Subscribe to ground truth match completion events.
        
        Optionally filter by domain.
        """
        pubsub = info.context.get("pubsub")
        
        if not pubsub:
            return
        
        async for event in pubsub.subscribe("ground_truth_match_completed"):
            if domain and event.get("domain") != domain:
                continue
            
            yield GroundTruthMatchCompletedType(
                match_id=event["match_id"],
                domain=event["domain"],
                sample_id=event["sample_id"],
                participants=event["participants"],
                winner_deployment_id=event["winner_deployment_id"],
                primary_metric=event["primary_metric"],
                winner_score=event["winner_score"],
                timestamp=event.get("timestamp", datetime.utcnow()),
            )
    
    @strawberry.subscription
    async def ground_truth_leaderboard_updated(
        self,
        info,
        domain: str
    ) -> AsyncGenerator[GroundTruthLeaderboardUpdateType, None]:
        """
        Subscribe to ground truth leaderboard updates for a specific domain.
        
        Fires when a model's rank changes in the leaderboard.
        """
        pubsub = info.context.get("pubsub")
        
        if not pubsub:
            return
        
        async for event in pubsub.subscribe("ground_truth_leaderboard_updated"):
            if event.get("domain") != domain:
                continue
            
            yield GroundTruthLeaderboardUpdateType(
                domain=event["domain"],
                deployment_id=event["deployment_id"],
                old_rank=event["old_rank"],
                new_rank=event["new_rank"],
                old_mu=event["old_mu"],
                new_mu=event["new_mu"],
                primary_metric_avg=event["primary_metric_avg"],
                timestamp=event.get("timestamp", datetime.utcnow()),
            )


# === PubSub Implementation ===

class InMemoryPubSub:
    """Simple in-memory pub/sub for development."""
    
    def __init__(self):
        self._subscribers: dict = {}
    
    async def publish(self, channel: str, message: dict):
        """Publish a message to a channel."""
        if channel in self._subscribers:
            for queue in self._subscribers[channel]:
                await queue.put(message)
    
    async def subscribe(self, channel: str) -> AsyncGenerator[dict, None]:
        """Subscribe to a channel."""
        queue: asyncio.Queue = asyncio.Queue()
        
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(queue)
        
        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            self._subscribers[channel].remove(queue)
            if not self._subscribers[channel]:
                del self._subscribers[channel]


# Global pubsub instance
pubsub = InMemoryPubSub()


async def publish_rating_update(
    deployment_id: str,
    domain: str,
    dimension: str,
    old_mu: float,
    new_mu: float,
    old_sigma: float,
    new_sigma: float,
    match_id: str
):
    """Publish a rating update event."""
    await pubsub.publish("rating_updates", {
        "deployment_id": deployment_id,
        "domain": domain,
        "dimension": dimension,
        "old_mu": old_mu,
        "new_mu": new_mu,
        "old_sigma": old_sigma,
        "new_sigma": new_sigma,
        "match_id": match_id,
        "timestamp": datetime.utcnow(),
    })


async def publish_match_completed(
    match_id: str,
    domain: str,
    winner_deployment_id: str,
    participants: int
):
    """Publish a match completed event."""
    await pubsub.publish("match_completed", {
        "match_id": match_id,
        "domain": domain,
        "winner_deployment_id": winner_deployment_id,
        "participants": participants,
        "timestamp": datetime.utcnow(),
    })


async def publish_price_update(
    deployment_id: str,
    provider: str,
    old_input_cost: float,
    new_input_cost: float,
    old_output_cost: float,
    new_output_cost: float
):
    """Publish a price update event."""
    # Calculate change percentage
    avg_old = (old_input_cost + old_output_cost) / 2
    avg_new = (new_input_cost + new_output_cost) / 2
    change_pct = ((avg_new - avg_old) / avg_old * 100) if avg_old > 0 else 0
    
    await pubsub.publish("price_updates", {
        "deployment_id": deployment_id,
        "provider": provider,
        "old_input_cost": old_input_cost,
        "new_input_cost": new_input_cost,
        "old_output_cost": old_output_cost,
        "new_output_cost": new_output_cost,
        "change_pct": change_pct,
        "timestamp": datetime.utcnow(),
    })


# === Ground Truth Event Publishers ===

async def publish_ground_truth_match_completed(
    match_id: str,
    domain: str,
    sample_id: str,
    participants: int,
    winner_deployment_id: str,
    primary_metric: str,
    winner_score: float
):
    """Publish a ground truth match completed event."""
    await pubsub.publish("ground_truth_match_completed", {
        "match_id": match_id,
        "domain": domain,
        "sample_id": sample_id,
        "participants": participants,
        "winner_deployment_id": winner_deployment_id,
        "primary_metric": primary_metric,
        "winner_score": winner_score,
        "timestamp": datetime.utcnow(),
    })


async def publish_ground_truth_leaderboard_updated(
    domain: str,
    deployment_id: str,
    old_rank: int,
    new_rank: int,
    old_mu: float,
    new_mu: float,
    primary_metric_avg: float
):
    """Publish a ground truth leaderboard update event."""
    await pubsub.publish("ground_truth_leaderboard_updated", {
        "domain": domain,
        "deployment_id": deployment_id,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_mu": old_mu,
        "new_mu": new_mu,
        "primary_metric_avg": primary_metric_avg,
        "timestamp": datetime.utcnow(),
    })

