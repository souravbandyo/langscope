"""
Billing routes for LangScope API.

Provides endpoints for:
- Subscription management
- Plan information
- Usage tracking
- Invoice history
- Payment methods
"""

import uuid
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from langscope.api.middleware.auth import UserContext, get_current_user
from langscope.api.dependencies import get_db


router = APIRouter(prefix="/billing", tags=["billing"])


# =============================================================================
# Request/Response Models
# =============================================================================

class PlanFeatures(BaseModel):
    """Features included in a plan."""
    evaluations_per_month: int
    team_members: int
    api_access: bool
    all_domains: bool
    custom_domains: bool
    priority_support: bool
    export_reports: bool
    sso_saml: bool
    sla_guarantee: bool


class Plan(BaseModel):
    """Subscription plan details."""
    id: str
    name: str
    price_monthly: int  # cents
    price_yearly: int  # cents
    features: PlanFeatures
    popular: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "id": "pro",
                "name": "Pro",
                "price_monthly": 2900,
                "price_yearly": 29000,
                "features": {
                    "evaluations_per_month": 5000,
                    "team_members": 25,
                    "api_access": True,
                    "all_domains": True,
                    "custom_domains": False,
                    "priority_support": True,
                    "export_reports": True,
                    "sso_saml": False,
                    "sla_guarantee": False
                },
                "popular": True
            }
        }


class PlansResponse(BaseModel):
    """Response model for available plans."""
    plans: List[Plan]


class UsageStats(BaseModel):
    """Current usage statistics."""
    evaluations: int
    evaluations_limit: int
    api_calls: int
    api_calls_limit: int
    team_members: int
    team_members_limit: int
    period_start: str
    period_end: str


class Subscription(BaseModel):
    """Subscription details."""
    id: str
    organization_id: str
    plan: str
    plan_name: str
    status: str  # active, past_due, canceled, trialing
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool = False
    usage: UsageStats

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sub_123",
                "organization_id": "org_123",
                "plan": "pro",
                "plan_name": "Pro",
                "status": "active",
                "current_period_start": "2024-01-01T00:00:00Z",
                "current_period_end": "2024-02-01T00:00:00Z",
                "cancel_at_period_end": False,
                "usage": {
                    "evaluations": 1523,
                    "evaluations_limit": 5000,
                    "api_calls": 45000,
                    "api_calls_limit": 100000,
                    "team_members": 8,
                    "team_members_limit": 25,
                    "period_start": "2024-01-01T00:00:00Z",
                    "period_end": "2024-02-01T00:00:00Z"
                }
            }
        }


class SubscribeRequest(BaseModel):
    """Request model for subscribing to a plan."""
    plan_id: str = Field(..., pattern="^(free|pro|enterprise)$")
    payment_method_id: Optional[str] = None


class ChangePlanRequest(BaseModel):
    """Request model for changing plan."""
    plan_id: str = Field(..., pattern="^(free|pro|enterprise)$")


class PaymentMethod(BaseModel):
    """Payment method details."""
    id: str
    type: str  # card, bank_account
    last_four: str
    brand: Optional[str] = None  # visa, mastercard, etc.
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    is_default: bool = False


class PaymentMethodsResponse(BaseModel):
    """Response model for payment methods."""
    payment_methods: List[PaymentMethod]


class AddPaymentMethodRequest(BaseModel):
    """Request model for adding payment method."""
    # In production, this would be a Stripe token/payment method ID
    token: str


class Invoice(BaseModel):
    """Invoice details."""
    id: str
    organization_id: str
    amount: int  # cents
    currency: str
    status: str  # paid, pending, failed
    description: str
    created_at: str
    pdf_url: Optional[str] = None


class InvoicesResponse(BaseModel):
    """Response model for invoices list."""
    invoices: List[Invoice]
    total: int


# =============================================================================
# Plan Definitions
# =============================================================================

PLANS = {
    "free": Plan(
        id="free",
        name="Free",
        price_monthly=0,
        price_yearly=0,
        features=PlanFeatures(
            evaluations_per_month=100,
            team_members=3,
            api_access=False,
            all_domains=False,
            custom_domains=False,
            priority_support=False,
            export_reports=False,
            sso_saml=False,
            sla_guarantee=False,
        ),
    ),
    "pro": Plan(
        id="pro",
        name="Pro",
        price_monthly=2900,  # $29
        price_yearly=29000,  # $290 (2 months free)
        features=PlanFeatures(
            evaluations_per_month=5000,
            team_members=25,
            api_access=True,
            all_domains=True,
            custom_domains=False,
            priority_support=True,
            export_reports=True,
            sso_saml=False,
            sla_guarantee=False,
        ),
        popular=True,
    ),
    "enterprise": Plan(
        id="enterprise",
        name="Enterprise",
        price_monthly=0,  # Contact for pricing
        price_yearly=0,
        features=PlanFeatures(
            evaluations_per_month=999999,  # Unlimited
            team_members=999999,
            api_access=True,
            all_domains=True,
            custom_domains=True,
            priority_support=True,
            export_reports=True,
            sso_saml=True,
            sla_guarantee=True,
        ),
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_user_organization(db, user_id: str) -> Optional[dict]:
    """Get user's organization."""
    membership = db.db.team_members.find_one({
        "user_id": user_id,
        "status": "active"
    })
    
    if not membership:
        return None
    
    return db.db.organizations.find_one({"_id": membership["organization_id"]})


def get_usage_stats(db, org_id: str, plan_id: str) -> UsageStats:
    """Get current usage stats for an organization."""
    plan = PLANS.get(plan_id, PLANS["free"])
    
    # Get current period
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (period_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    period_end = next_month - timedelta(seconds=1)
    
    # Get usage document
    usage_doc = db.db.usage.find_one({
        "organization_id": org_id,
        "period": period_start.strftime("%Y-%m")
    }) or {}
    
    # Count team members
    member_count = db.db.team_members.count_documents({
        "organization_id": org_id,
        "status": "active"
    })
    
    return UsageStats(
        evaluations=usage_doc.get("evaluations", 0),
        evaluations_limit=plan.features.evaluations_per_month,
        api_calls=usage_doc.get("api_calls", 0),
        api_calls_limit=100000 if plan.features.api_access else 0,
        team_members=member_count,
        team_members_limit=plan.features.team_members,
        period_start=period_start.isoformat() + "Z",
        period_end=period_end.isoformat() + "Z",
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/plans",
    response_model=PlansResponse,
    summary="List available plans",
    description="Get all available subscription plans.",
)
async def list_plans() -> PlansResponse:
    """
    Get all available subscription plans.
    
    Returns plan details including features and pricing.
    """
    return PlansResponse(plans=list(PLANS.values()))


@router.get(
    "/subscription",
    response_model=Optional[Subscription],
    summary="Get current subscription",
    description="Get the current subscription for the user's organization.",
)
async def get_subscription(
    user: UserContext = Depends(get_current_user)
) -> Optional[Subscription]:
    """
    Get the current subscription.
    
    Returns subscription details including plan, status, and usage.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            return None
        
        # Get or create subscription
        sub = db.db.subscriptions.find_one({"organization_id": org["_id"]})
        
        if not sub:
            # Create free subscription
            now = datetime.utcnow()
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (period_start.replace(day=28) + timedelta(days=4)).replace(day=1)
            
            sub_id = f"sub_{uuid.uuid4().hex[:12]}"
            sub = {
                "_id": sub_id,
                "organization_id": org["_id"],
                "plan": "free",
                "status": "active",
                "current_period_start": period_start.isoformat() + "Z",
                "current_period_end": next_month.isoformat() + "Z",
                "cancel_at_period_end": False,
            }
            db.db.subscriptions.insert_one(sub)
        
        plan = PLANS.get(sub["plan"], PLANS["free"])
        usage = get_usage_stats(db, org["_id"], sub["plan"])
        
        return Subscription(
            id=sub["_id"],
            organization_id=org["_id"],
            plan=sub["plan"],
            plan_name=plan.name,
            status=sub["status"],
            current_period_start=sub["current_period_start"],
            current_period_end=sub["current_period_end"],
            cancel_at_period_end=sub.get("cancel_at_period_end", False),
            usage=usage,
        )
        
    except RuntimeError:
        return None


@router.post(
    "/subscribe",
    response_model=Subscription,
    summary="Subscribe to plan",
    description="Subscribe to a new plan.",
)
async def subscribe_to_plan(
    request: SubscribeRequest,
    user: UserContext = Depends(get_current_user)
) -> Subscription:
    """
    Subscribe to a plan.
    
    For paid plans, a payment method is required.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You must create an organization first"
            )
        
        # Check if owner
        membership = db.db.team_members.find_one({
            "user_id": user.user_id,
            "organization_id": org["_id"]
        })
        
        if membership["role"] != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can manage subscriptions"
            )
        
        plan = PLANS.get(request.plan_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid plan"
            )
        
        # For paid plans, require payment method (in production)
        if plan.price_monthly > 0 and not request.payment_method_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment method required for paid plans"
            )
        
        # Create/update subscription
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (period_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        
        sub_id = f"sub_{uuid.uuid4().hex[:12]}"
        
        db.db.subscriptions.update_one(
            {"organization_id": org["_id"]},
            {
                "$set": {
                    "plan": request.plan_id,
                    "status": "active",
                    "current_period_start": period_start.isoformat() + "Z",
                    "current_period_end": next_month.isoformat() + "Z",
                    "cancel_at_period_end": False,
                },
                "$setOnInsert": {
                    "_id": sub_id,
                    "organization_id": org["_id"],
                }
            },
            upsert=True
        )
        
        # Update organization plan
        db.db.organizations.update_one(
            {"_id": org["_id"]},
            {"$set": {"plan": request.plan_id}}
        )
        
        return await get_subscription(user)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.put(
    "/subscription",
    response_model=Subscription,
    summary="Change plan",
    description="Change to a different subscription plan.",
)
async def change_plan(
    request: ChangePlanRequest,
    user: UserContext = Depends(get_current_user)
) -> Subscription:
    """
    Change subscription plan.
    
    Upgrades take effect immediately. Downgrades at end of period.
    """
    # Reuse subscribe logic for now
    return await subscribe_to_plan(
        SubscribeRequest(plan_id=request.plan_id),
        user
    )


@router.delete(
    "/subscription",
    summary="Cancel subscription",
    description="Cancel the current subscription.",
)
async def cancel_subscription(
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Cancel the subscription.
    
    The subscription will remain active until the end of the billing period.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No organization found"
            )
        
        # Check if owner
        membership = db.db.team_members.find_one({
            "user_id": user.user_id,
            "organization_id": org["_id"]
        })
        
        if membership["role"] != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can cancel subscriptions"
            )
        
        # Mark for cancellation at period end
        db.db.subscriptions.update_one(
            {"organization_id": org["_id"]},
            {"$set": {"cancel_at_period_end": True}}
        )
        
        return {"message": "Subscription will be canceled at end of billing period"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.get(
    "/usage",
    response_model=UsageStats,
    summary="Get usage stats",
    description="Get current usage statistics.",
)
async def get_usage(
    user: UserContext = Depends(get_current_user)
) -> UsageStats:
    """
    Get current usage statistics.
    
    Returns usage for the current billing period.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No organization found"
            )
        
        sub = db.db.subscriptions.find_one({"organization_id": org["_id"]})
        plan_id = sub["plan"] if sub else "free"
        
        return get_usage_stats(db, org["_id"], plan_id)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.get(
    "/invoices",
    response_model=InvoicesResponse,
    summary="List invoices",
    description="Get billing history.",
)
async def list_invoices(
    user: UserContext = Depends(get_current_user),
    limit: int = 12
) -> InvoicesResponse:
    """
    List invoices.
    
    Returns billing history for the organization.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            return InvoicesResponse(invoices=[], total=0)
        
        invoice_docs = list(db.db.invoices.find(
            {"organization_id": org["_id"]}
        ).sort("created_at", -1).limit(limit))
        
        invoices = [
            Invoice(
                id=inv["_id"],
                organization_id=org["_id"],
                amount=inv["amount"],
                currency=inv.get("currency", "usd"),
                status=inv["status"],
                description=inv.get("description", f"Subscription - {inv.get('plan', 'Pro')}"),
                created_at=inv["created_at"],
                pdf_url=inv.get("pdf_url"),
            )
            for inv in invoice_docs
        ]
        
        total = db.db.invoices.count_documents({"organization_id": org["_id"]})
        
        return InvoicesResponse(invoices=invoices, total=total)
        
    except RuntimeError:
        return InvoicesResponse(invoices=[], total=0)


@router.get(
    "/invoices/{invoice_id}/pdf",
    summary="Download invoice PDF",
    description="Get PDF download URL for an invoice.",
)
async def get_invoice_pdf(
    invoice_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Get invoice PDF download URL.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No organization found"
            )
        
        invoice = db.db.invoices.find_one({
            "_id": invoice_id,
            "organization_id": org["_id"]
        })
        
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        # In production, return Stripe invoice PDF URL
        return {
            "pdf_url": invoice.get("pdf_url", f"/invoices/{invoice_id}.pdf"),
            "invoice_id": invoice_id
        }
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.get(
    "/payment-methods",
    response_model=PaymentMethodsResponse,
    summary="List payment methods",
    description="Get saved payment methods.",
)
async def list_payment_methods(
    user: UserContext = Depends(get_current_user)
) -> PaymentMethodsResponse:
    """
    List saved payment methods.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            return PaymentMethodsResponse(payment_methods=[])
        
        pm_docs = list(db.db.payment_methods.find({"organization_id": org["_id"]}))
        
        payment_methods = [
            PaymentMethod(
                id=pm["_id"],
                type=pm.get("type", "card"),
                last_four=pm.get("last_four", "****"),
                brand=pm.get("brand"),
                exp_month=pm.get("exp_month"),
                exp_year=pm.get("exp_year"),
                is_default=pm.get("is_default", False),
            )
            for pm in pm_docs
        ]
        
        return PaymentMethodsResponse(payment_methods=payment_methods)
        
    except RuntimeError:
        return PaymentMethodsResponse(payment_methods=[])


@router.post(
    "/payment-method",
    response_model=PaymentMethod,
    summary="Add payment method",
    description="Add a new payment method.",
)
async def add_payment_method(
    request: AddPaymentMethodRequest,
    user: UserContext = Depends(get_current_user)
) -> PaymentMethod:
    """
    Add a new payment method.
    
    In production, this would create a Stripe payment method.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No organization found"
            )
        
        # Check if owner
        membership = db.db.team_members.find_one({
            "user_id": user.user_id,
            "organization_id": org["_id"]
        })
        
        if membership["role"] != "owner":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can manage payment methods"
            )
        
        # In production, create Stripe payment method here
        # For now, create mock payment method
        pm_id = f"pm_{uuid.uuid4().hex[:12]}"
        
        pm_doc = {
            "_id": pm_id,
            "organization_id": org["_id"],
            "type": "card",
            "last_four": "4242",
            "brand": "visa",
            "exp_month": 12,
            "exp_year": 2025,
            "is_default": True,
        }
        
        # Set other cards as non-default
        db.db.payment_methods.update_many(
            {"organization_id": org["_id"]},
            {"$set": {"is_default": False}}
        )
        
        db.db.payment_methods.insert_one(pm_doc)
        
        return PaymentMethod(**{k: v for k, v in pm_doc.items() if k != "_id"}, id=pm_id)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.delete(
    "/payment-method/{payment_method_id}",
    summary="Remove payment method",
    description="Remove a saved payment method.",
)
async def remove_payment_method(
    payment_method_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Remove a payment method.
    """
    try:
        db = get_db()
        
        org = get_user_organization(db, user.user_id)
        if not org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No organization found"
            )
        
        result = db.db.payment_methods.delete_one({
            "_id": payment_method_id,
            "organization_id": org["_id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment method not found"
            )
        
        return {"message": "Payment method removed successfully"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
