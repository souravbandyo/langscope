"""
Organization routes for LangScope API.

Provides endpoints for:
- Organization CRUD operations
- Team member management
- Invitation system
"""

import os
import uuid
import secrets
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel, EmailStr, Field

from langscope.api.middleware.auth import UserContext, get_current_user
from langscope.api.dependencies import get_db


router = APIRouter(prefix="/organizations", tags=["organizations"])


# =============================================================================
# Request/Response Models
# =============================================================================

class OrganizationSettings(BaseModel):
    """Organization settings."""
    default_domain: Optional[str] = None
    allowed_email_domains: List[str] = []
    sso_enabled: bool = False


class Organization(BaseModel):
    """Organization data."""
    id: str
    name: str
    slug: str
    logo_url: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    owner_id: str
    plan: str = "free"
    settings: OrganizationSettings = OrganizationSettings()
    created_at: str
    member_count: int = 1

    class Config:
        json_schema_extra = {
            "example": {
                "id": "org_123",
                "name": "Acme Corp",
                "slug": "acme-corp",
                "logo_url": "https://example.com/logo.png",
                "description": "AI-powered solutions",
                "website": "https://acme.com",
                "owner_id": "user_123",
                "plan": "pro",
                "settings": {
                    "default_domain": "code_generation",
                    "allowed_email_domains": ["acme.com"],
                    "sso_enabled": False
                },
                "created_at": "2024-01-01T00:00:00Z",
                "member_count": 5
            }
        }


class CreateOrganizationRequest(BaseModel):
    """Request model for creating an organization."""
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=200)


class UpdateOrganizationRequest(BaseModel):
    """Request model for updating an organization."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=200)
    settings: Optional[OrganizationSettings] = None


class TeamMember(BaseModel):
    """Team member data."""
    id: str
    user_id: str
    organization_id: str
    email: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str  # owner, admin, member, viewer
    status: str  # active, pending, suspended
    joined_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "member_123",
                "user_id": "user_456",
                "organization_id": "org_123",
                "email": "john@acme.com",
                "display_name": "John Doe",
                "avatar_url": "https://example.com/avatar.jpg",
                "role": "admin",
                "status": "active",
                "joined_at": "2024-01-05T00:00:00Z"
            }
        }


class TeamMembersResponse(BaseModel):
    """Response model for team members list."""
    members: List[TeamMember]
    total: int


class InviteMemberRequest(BaseModel):
    """Request model for inviting a team member."""
    email: EmailStr
    role: str = Field("member", pattern="^(admin|member|viewer)$")


class UpdateMemberRoleRequest(BaseModel):
    """Request model for updating a member's role."""
    role: str = Field(..., pattern="^(admin|member|viewer)$")


class Invitation(BaseModel):
    """Invitation data."""
    id: str
    organization_id: str
    organization_name: str
    email: str
    role: str
    invited_by: str
    invite_code: str
    expires_at: str
    status: str  # pending, accepted, expired, revoked

    class Config:
        json_schema_extra = {
            "example": {
                "id": "invite_123",
                "organization_id": "org_123",
                "organization_name": "Acme Corp",
                "email": "newuser@example.com",
                "role": "member",
                "invited_by": "user_123",
                "invite_code": "ABC123XYZ",
                "expires_at": "2024-01-15T00:00:00Z",
                "status": "pending"
            }
        }


class InvitationsResponse(BaseModel):
    """Response model for invitations list."""
    invitations: List[Invitation]


class JoinOrganizationRequest(BaseModel):
    """Request model for joining an organization."""
    invite_code: str


# =============================================================================
# Helper Functions
# =============================================================================

def generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a name."""
    import re
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return f"{slug}-{secrets.token_hex(3)}"


def check_org_permission(db, user_id: str, org_id: str, required_roles: List[str]) -> TeamMember:
    """Check if user has required permission in organization."""
    member = db.db.team_members.find_one({
        "user_id": user_id,
        "organization_id": org_id,
        "status": "active"
    })
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization"
        )
    
    if member["role"] not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This action requires one of these roles: {', '.join(required_roles)}"
        )
    
    return member


# =============================================================================
# Organization Endpoints
# =============================================================================

@router.post(
    "",
    response_model=Organization,
    status_code=status.HTTP_201_CREATED,
    summary="Create organization",
    description="Create a new organization. The creating user becomes the owner.",
)
async def create_organization(
    request: CreateOrganizationRequest,
    user: UserContext = Depends(get_current_user)
) -> Organization:
    """
    Create a new organization.
    
    The user creating the organization automatically becomes the owner.
    """
    try:
        db = get_db()
        
        # Check if user already has an organization
        existing_membership = db.db.team_members.find_one({
            "user_id": user.user_id,
            "role": "owner"
        })
        
        if existing_membership:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You already own an organization"
            )
        
        # Create organization
        org_id = f"org_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat() + "Z"
        
        org_doc = {
            "_id": org_id,
            "name": request.name,
            "slug": generate_slug(request.name),
            "description": request.description,
            "website": request.website,
            "owner_id": user.user_id,
            "plan": "free",
            "settings": {
                "default_domain": None,
                "allowed_email_domains": [],
                "sso_enabled": False
            },
            "created_at": now,
        }
        
        db.db.organizations.insert_one(org_doc)
        
        # Add creator as owner
        member_id = f"member_{uuid.uuid4().hex[:12]}"
        member_doc = {
            "_id": member_id,
            "user_id": user.user_id,
            "organization_id": org_id,
            "email": user.email,
            "role": "owner",
            "status": "active",
            "invited_by": user.user_id,
            "joined_at": now,
        }
        
        db.db.team_members.insert_one(member_doc)
        
        # Update user's organization reference
        db.db.users.update_one(
            {"_id": user.user_id},
            {
                "$set": {
                    "organization_id": org_id,
                    "role_in_org": "owner",
                    "updated_at": now
                },
                "$setOnInsert": {
                    "email": user.email,
                    "created_at": now,
                }
            },
            upsert=True
        )
        
        return Organization(
            id=org_id,
            name=request.name,
            slug=org_doc["slug"],
            description=request.description,
            website=request.website,
            owner_id=user.user_id,
            plan="free",
            settings=OrganizationSettings(),
            created_at=now,
            member_count=1,
        )
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.get(
    "/me",
    response_model=Optional[Organization],
    summary="Get user's organization",
    description="Get the organization the current user belongs to.",
)
async def get_my_organization(
    user: UserContext = Depends(get_current_user)
) -> Optional[Organization]:
    """
    Get the current user's organization.
    
    Returns null if user doesn't belong to any organization.
    """
    try:
        db = get_db()
        
        # Find user's membership
        membership = db.db.team_members.find_one({
            "user_id": user.user_id,
            "status": "active"
        })
        
        if not membership:
            return None
        
        # Get organization
        org = db.db.organizations.find_one({"_id": membership["organization_id"]})
        
        if not org:
            return None
        
        # Count members
        member_count = db.db.team_members.count_documents({
            "organization_id": org["_id"],
            "status": "active"
        })
        
        return Organization(
            id=org["_id"],
            name=org["name"],
            slug=org["slug"],
            logo_url=org.get("logo_url"),
            description=org.get("description"),
            website=org.get("website"),
            owner_id=org["owner_id"],
            plan=org.get("plan", "free"),
            settings=OrganizationSettings(**org.get("settings", {})),
            created_at=org["created_at"],
            member_count=member_count,
        )
        
    except RuntimeError:
        return None


@router.put(
    "/{org_id}",
    response_model=Organization,
    summary="Update organization",
    description="Update organization details. Requires owner or admin role.",
)
async def update_organization(
    org_id: str,
    request: UpdateOrganizationRequest,
    user: UserContext = Depends(get_current_user)
) -> Organization:
    """
    Update organization details.
    
    Only owners and admins can update organization settings.
    """
    try:
        db = get_db()
        
        # Check permission
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        # Build update
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.website is not None:
            update_data["website"] = request.website
        if request.settings is not None:
            update_data["settings"] = request.settings.model_dump()
        
        if update_data:
            db.db.organizations.update_one(
                {"_id": org_id},
                {"$set": update_data}
            )
        
        # Return updated org
        return await get_organization_by_id(db, org_id)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


async def get_organization_by_id(db, org_id: str) -> Organization:
    """Helper to get organization by ID."""
    org = db.db.organizations.find_one({"_id": org_id})
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    member_count = db.db.team_members.count_documents({
        "organization_id": org_id,
        "status": "active"
    })
    
    return Organization(
        id=org["_id"],
        name=org["name"],
        slug=org["slug"],
        logo_url=org.get("logo_url"),
        description=org.get("description"),
        website=org.get("website"),
        owner_id=org["owner_id"],
        plan=org.get("plan", "free"),
        settings=OrganizationSettings(**org.get("settings", {})),
        created_at=org["created_at"],
        member_count=member_count,
    )


@router.post(
    "/{org_id}/logo",
    summary="Upload organization logo",
    description="Upload a new logo for the organization.",
)
async def upload_org_logo(
    org_id: str,
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user)
) -> dict:
    """Upload organization logo."""
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        # Validate file
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 5MB."
            )
        
        # Save file
        ext = file.filename.split(".")[-1] if file.filename else "png"
        upload_dir = os.path.join(os.getcwd(), "uploads", "logos", org_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logo_url = f"/uploads/logos/{org_id}/{filename}"
        
        # Update organization
        db.db.organizations.update_one(
            {"_id": org_id},
            {"$set": {"logo_url": logo_url}}
        )
        
        return {"logo_url": logo_url, "message": "Logo uploaded successfully"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.delete(
    "/{org_id}",
    summary="Delete organization",
    description="Permanently delete the organization. Requires owner role.",
)
async def delete_organization(
    org_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Delete an organization.
    
    Only the owner can delete an organization.
    """
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner"])
        
        # Delete organization
        db.db.organizations.delete_one({"_id": org_id})
        
        # Delete all memberships
        db.db.team_members.delete_many({"organization_id": org_id})
        
        # Delete all invitations
        db.db.invitations.delete_many({"organization_id": org_id})
        
        # Clear organization reference from users
        db.db.users.update_many(
            {"organization_id": org_id},
            {"$set": {"organization_id": None, "role_in_org": None}}
        )
        
        return {"message": "Organization deleted successfully"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


# =============================================================================
# Team Member Endpoints
# =============================================================================

@router.get(
    "/{org_id}/members",
    response_model=TeamMembersResponse,
    summary="List team members",
    description="Get all members of an organization.",
)
async def list_members(
    org_id: str,
    user: UserContext = Depends(get_current_user)
) -> TeamMembersResponse:
    """
    List all members of an organization.
    
    All members can view the member list.
    """
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin", "member", "viewer"])
        
        # Get all members
        member_docs = list(db.db.team_members.find({"organization_id": org_id}))
        
        # Enrich with user data
        members = []
        for m in member_docs:
            user_doc = db.db.users.find_one({"_id": m["user_id"]}) or {}
            members.append(TeamMember(
                id=m["_id"],
                user_id=m["user_id"],
                organization_id=org_id,
                email=m.get("email", user_doc.get("email", "")),
                display_name=user_doc.get("display_name"),
                avatar_url=user_doc.get("avatar_url"),
                role=m["role"],
                status=m["status"],
                joined_at=m["joined_at"],
            ))
        
        return TeamMembersResponse(members=members, total=len(members))
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.post(
    "/{org_id}/members",
    response_model=Invitation,
    status_code=status.HTTP_201_CREATED,
    summary="Invite member",
    description="Invite a new member to the organization.",
)
async def invite_member(
    org_id: str,
    request: InviteMemberRequest,
    user: UserContext = Depends(get_current_user)
) -> Invitation:
    """
    Invite a new member to the organization.
    
    Only owners and admins can invite new members.
    """
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        # Get organization name
        org = db.db.organizations.find_one({"_id": org_id})
        if not org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Check if user already exists in organization
        existing = db.db.team_members.find_one({
            "organization_id": org_id,
            "email": request.email
        })
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member or has a pending invitation"
            )
        
        # Create invitation
        invite_id = f"invite_{uuid.uuid4().hex[:12]}"
        invite_code = secrets.token_urlsafe(16)
        now = datetime.utcnow()
        expires_at = (now + timedelta(days=7)).isoformat() + "Z"
        
        invite_doc = {
            "_id": invite_id,
            "organization_id": org_id,
            "email": request.email,
            "role": request.role,
            "invited_by": user.user_id,
            "invite_code": invite_code,
            "expires_at": expires_at,
            "status": "pending",
            "created_at": now.isoformat() + "Z",
        }
        
        db.db.invitations.insert_one(invite_doc)
        
        # In production, send email with invite link here
        
        return Invitation(
            id=invite_id,
            organization_id=org_id,
            organization_name=org["name"],
            email=request.email,
            role=request.role,
            invited_by=user.user_id,
            invite_code=invite_code,
            expires_at=expires_at,
            status="pending",
        )
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.put(
    "/{org_id}/members/{member_id}",
    response_model=TeamMember,
    summary="Update member role",
    description="Change a team member's role.",
)
async def update_member_role(
    org_id: str,
    member_id: str,
    request: UpdateMemberRoleRequest,
    user: UserContext = Depends(get_current_user)
) -> TeamMember:
    """
    Update a team member's role.
    
    Only owners can change roles. Owners cannot demote themselves.
    """
    try:
        db = get_db()
        current_member = check_org_permission(db, user.user_id, org_id, ["owner"])
        
        # Get target member
        target_member = db.db.team_members.find_one({"_id": member_id, "organization_id": org_id})
        
        if not target_member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )
        
        # Cannot change owner's role
        if target_member["role"] == "owner":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change owner's role"
            )
        
        # Update role
        db.db.team_members.update_one(
            {"_id": member_id},
            {"$set": {"role": request.role}}
        )
        
        # Update user's role reference
        db.db.users.update_one(
            {"_id": target_member["user_id"]},
            {"$set": {"role_in_org": request.role}}
        )
        
        # Return updated member
        user_doc = db.db.users.find_one({"_id": target_member["user_id"]}) or {}
        return TeamMember(
            id=member_id,
            user_id=target_member["user_id"],
            organization_id=org_id,
            email=target_member.get("email", ""),
            display_name=user_doc.get("display_name"),
            avatar_url=user_doc.get("avatar_url"),
            role=request.role,
            status=target_member["status"],
            joined_at=target_member["joined_at"],
        )
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.delete(
    "/{org_id}/members/{member_id}",
    summary="Remove member",
    description="Remove a member from the organization.",
)
async def remove_member(
    org_id: str,
    member_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Remove a member from the organization.
    
    Owners and admins can remove members. Owners cannot be removed.
    """
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        # Get target member
        target_member = db.db.team_members.find_one({"_id": member_id, "organization_id": org_id})
        
        if not target_member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )
        
        # Cannot remove owner
        if target_member["role"] == "owner":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove organization owner"
            )
        
        # Remove member
        db.db.team_members.delete_one({"_id": member_id})
        
        # Clear organization reference from user
        db.db.users.update_one(
            {"_id": target_member["user_id"]},
            {"$set": {"organization_id": None, "role_in_org": None}}
        )
        
        return {"message": "Member removed successfully"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


# =============================================================================
# Invitation Endpoints
# =============================================================================

@router.get(
    "/{org_id}/invitations",
    response_model=InvitationsResponse,
    summary="List invitations",
    description="Get pending invitations for an organization.",
)
async def list_invitations(
    org_id: str,
    user: UserContext = Depends(get_current_user)
) -> InvitationsResponse:
    """List pending invitations."""
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        org = db.db.organizations.find_one({"_id": org_id})
        org_name = org["name"] if org else "Unknown"
        
        invite_docs = list(db.db.invitations.find({
            "organization_id": org_id,
            "status": "pending"
        }))
        
        invitations = [
            Invitation(
                id=inv["_id"],
                organization_id=org_id,
                organization_name=org_name,
                email=inv["email"],
                role=inv["role"],
                invited_by=inv["invited_by"],
                invite_code=inv["invite_code"],
                expires_at=inv["expires_at"],
                status=inv["status"],
            )
            for inv in invite_docs
        ]
        
        return InvitationsResponse(invitations=invitations)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.delete(
    "/{org_id}/invitations/{invitation_id}",
    summary="Revoke invitation",
    description="Revoke a pending invitation.",
)
async def revoke_invitation(
    org_id: str,
    invitation_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """Revoke a pending invitation."""
    try:
        db = get_db()
        check_org_permission(db, user.user_id, org_id, ["owner", "admin"])
        
        result = db.db.invitations.delete_one({
            "_id": invitation_id,
            "organization_id": org_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invitation not found"
            )
        
        return {"message": "Invitation revoked successfully"}
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.post(
    "/join",
    response_model=Organization,
    summary="Join organization",
    description="Join an organization using an invite code.",
)
async def join_organization(
    request: JoinOrganizationRequest,
    user: UserContext = Depends(get_current_user)
) -> Organization:
    """
    Join an organization using an invite code.
    """
    try:
        db = get_db()
        
        # Find invitation
        invitation = db.db.invitations.find_one({
            "invite_code": request.invite_code,
            "status": "pending"
        })
        
        if not invitation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired invite code"
            )
        
        # Check expiry
        expires_at = datetime.fromisoformat(invitation["expires_at"].replace("Z", "+00:00"))
        if datetime.now(expires_at.tzinfo) > expires_at:
            db.db.invitations.update_one(
                {"_id": invitation["_id"]},
                {"$set": {"status": "expired"}}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite code has expired"
            )
        
        # Check if user already in an organization
        existing = db.db.team_members.find_one({
            "user_id": user.user_id,
            "status": "active"
        })
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You are already a member of an organization"
            )
        
        org_id = invitation["organization_id"]
        now = datetime.utcnow().isoformat() + "Z"
        
        # Create membership
        member_id = f"member_{uuid.uuid4().hex[:12]}"
        member_doc = {
            "_id": member_id,
            "user_id": user.user_id,
            "organization_id": org_id,
            "email": user.email,
            "role": invitation["role"],
            "status": "active",
            "invited_by": invitation["invited_by"],
            "joined_at": now,
        }
        
        db.db.team_members.insert_one(member_doc)
        
        # Update invitation status
        db.db.invitations.update_one(
            {"_id": invitation["_id"]},
            {"$set": {"status": "accepted"}}
        )
        
        # Update user's organization reference
        db.db.users.update_one(
            {"_id": user.user_id},
            {
                "$set": {
                    "organization_id": org_id,
                    "role_in_org": invitation["role"],
                    "updated_at": now
                }
            },
            upsert=True
        )
        
        # Return organization
        return await get_organization_by_id(db, org_id)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
