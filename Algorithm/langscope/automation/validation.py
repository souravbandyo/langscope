"""
Validation and approval workflow for automated changes.

Provides validation rules and pending change management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class ChangeType(str, Enum):
    """Types of changes that can occur."""
    PRICE_UPDATE = "price_update"
    MODEL_ADDED = "model_added"
    MODEL_REMOVED = "model_removed"
    BENCHMARK_UPDATE = "benchmark_update"
    CAPABILITY_UPDATE = "capability_update"


class ApprovalStatus(str, Enum):
    """Status of a pending change."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


@dataclass
class PendingChange:
    """
    A change awaiting approval.
    
    Used when automated changes exceed thresholds or require manual review.
    """
    id: str
    source_id: str
    change_type: ChangeType
    entity_id: str  # model_id, deployment_id, etc.
    entity_name: str
    
    # What changed
    field_name: str
    old_value: Any
    new_value: Any
    
    # Why it needs approval
    reason: str
    
    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Metadata
    detected_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.utcnow().isoformat() + "Z"
    
    def approve(self, reviewed_by: str, notes: str = ""):
        """Approve the change."""
        self.status = ApprovalStatus.APPROVED
        self.reviewed_at = datetime.utcnow().isoformat() + "Z"
        self.reviewed_by = reviewed_by
        self.notes = notes
    
    def reject(self, reviewed_by: str, notes: str = ""):
        """Reject the change."""
        self.status = ApprovalStatus.REJECTED
        self.reviewed_at = datetime.utcnow().isoformat() + "Z"
        self.reviewed_by = reviewed_by
        self.notes = notes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "_id": self.id,
            "source_id": self.source_id,
            "change_type": self.change_type.value if isinstance(self.change_type, Enum) else self.change_type,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "detected_at": self.detected_at,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PendingChange':
        """Create from dictionary."""
        change_type = data.get("change_type", "price_update")
        if isinstance(change_type, str):
            try:
                change_type = ChangeType(change_type)
            except ValueError:
                change_type = ChangeType.PRICE_UPDATE
        
        status = data.get("status", "pending")
        if isinstance(status, str):
            try:
                status = ApprovalStatus(status)
            except ValueError:
                status = ApprovalStatus.PENDING
        
        return cls(
            id=data.get("_id", ""),
            source_id=data.get("source_id", ""),
            change_type=change_type,
            entity_id=data.get("entity_id", ""),
            entity_name=data.get("entity_name", ""),
            field_name=data.get("field_name", ""),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            reason=data.get("reason", ""),
            status=status,
            detected_at=data.get("detected_at", ""),
            reviewed_at=data.get("reviewed_at", ""),
            reviewed_by=data.get("reviewed_by", ""),
            notes=data.get("notes", ""),
        )


def validate_price_change(
    old_price: float,
    new_price: float,
    max_change_pct: float = 50.0
) -> tuple:
    """
    Validate a price change.
    
    Args:
        old_price: Previous price
        new_price: New price
        max_change_pct: Maximum allowed percentage change
    
    Returns:
        Tuple of (is_valid, change_pct, reason)
    """
    if old_price <= 0:
        # New model or free tier
        return (True, 0.0, "")
    
    change_pct = abs(new_price - old_price) / old_price * 100
    
    if change_pct > max_change_pct:
        return (
            False,
            change_pct,
            f"Price change of {change_pct:.1f}% exceeds threshold of {max_change_pct}%"
        )
    
    return (True, change_pct, "")


def check_requires_approval(
    change_type: ChangeType,
    old_value: Any,
    new_value: Any,
    max_price_change_pct: float = 50.0,
    require_approval_for_new: bool = False,
    require_approval_for_delete: bool = True
) -> tuple:
    """
    Check if a change requires manual approval.
    
    Args:
        change_type: Type of change
        old_value: Previous value
        new_value: New value
        max_price_change_pct: Max price change before requiring approval
        require_approval_for_new: Require approval for new entities
        require_approval_for_delete: Require approval for deletions
    
    Returns:
        Tuple of (requires_approval, reason)
    """
    if change_type == ChangeType.MODEL_ADDED:
        if require_approval_for_new:
            return (True, "New model requires manual approval")
        return (False, "")
    
    if change_type == ChangeType.MODEL_REMOVED:
        if require_approval_for_delete:
            return (True, "Model removal requires manual approval")
        return (False, "")
    
    if change_type == ChangeType.PRICE_UPDATE:
        if old_value is None or old_value <= 0:
            return (False, "")
        
        is_valid, change_pct, reason = validate_price_change(
            old_value, new_value, max_price_change_pct
        )
        if not is_valid:
            return (True, reason)
    
    return (False, "")


def compute_price_change_severity(
    old_price: float,
    new_price: float
) -> str:
    """
    Compute severity level of a price change.
    
    Returns:
        Severity level: "minor", "moderate", "significant", "major"
    """
    if old_price <= 0:
        return "minor"
    
    change_pct = abs(new_price - old_price) / old_price * 100
    
    if change_pct <= 10:
        return "minor"
    elif change_pct <= 25:
        return "moderate"
    elif change_pct <= 50:
        return "significant"
    else:
        return "major"


