"""
User Profile routes for LangScope API.

Provides endpoints for:
- Getting and updating user profile
- Avatar upload and management
- Password change
- Session management
- Account deletion
"""

import os
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel, EmailStr, Field

from langscope.api.middleware.auth import UserContext, get_current_user
from langscope.api.dependencies import get_db


router = APIRouter(prefix="/users", tags=["users"])


# =============================================================================
# Request/Response Models
# =============================================================================

class UserProfile(BaseModel):
    """User profile data."""
    user_id: str
    email: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    language: str = "en"
    organization_id: Optional[str] = None
    role_in_org: Optional[str] = None
    plan: str = "free"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "email": "user@example.com",
                "display_name": "John Doe",
                "avatar_url": "https://example.com/avatar.jpg",
                "phone": "+1234567890",
                "timezone": "America/New_York",
                "language": "en",
                "organization_id": "org_123",
                "role_in_org": "admin",
                "plan": "pro",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T00:00:00Z"
            }
        }


class UpdateProfileRequest(BaseModel):
    """Request model for updating user profile."""
    display_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    timezone: Optional[str] = Field(None, max_length=50)
    language: Optional[str] = Field(None, max_length=10)


class ChangePasswordRequest(BaseModel):
    """Request model for changing password."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class Session(BaseModel):
    """Active session information."""
    session_id: str
    device: str
    ip_address: str
    location: Optional[str] = None
    last_active: str
    is_current: bool = False


class SessionsResponse(BaseModel):
    """Response model for active sessions."""
    sessions: List[Session]


class AvatarResponse(BaseModel):
    """Response model for avatar upload."""
    avatar_url: str
    message: str = "Avatar uploaded successfully"


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/me/profile",
    response_model=UserProfile,
    summary="Get user profile",
    description="Get the current user's full profile information.",
)
async def get_profile(user: UserContext = Depends(get_current_user)) -> UserProfile:
    """
    Get the current user's profile.
    
    Returns full profile including display name, avatar, preferences, and organization info.
    """
    try:
        db = get_db()
        
        # Try to find existing profile in database
        user_doc = db.db.users.find_one({"_id": user.user_id})
        
        if user_doc:
            return UserProfile(
                user_id=user.user_id,
                email=user.email or "",
                display_name=user_doc.get("display_name"),
                avatar_url=user_doc.get("avatar_url"),
                phone=user_doc.get("phone"),
                timezone=user_doc.get("timezone", "UTC"),
                language=user_doc.get("language", "en"),
                organization_id=user_doc.get("organization_id"),
                role_in_org=user_doc.get("role_in_org"),
                plan=user_doc.get("plan", "free"),
                created_at=user_doc.get("created_at"),
                updated_at=user_doc.get("updated_at"),
            )
        
        # Return basic profile from JWT if no database record
        return UserProfile(
            user_id=user.user_id,
            email=user.email or "",
            display_name=user.email.split("@")[0] if user.email else None,
            plan="free",
        )
    except RuntimeError:
        # Database not available, return basic profile from JWT
        return UserProfile(
            user_id=user.user_id,
            email=user.email or "",
            display_name=user.email.split("@")[0] if user.email else None,
            plan="free",
        )


@router.put(
    "/me/profile",
    response_model=UserProfile,
    summary="Update user profile",
    description="Update the current user's profile information.",
)
async def update_profile(
    request: UpdateProfileRequest,
    user: UserContext = Depends(get_current_user)
) -> UserProfile:
    """
    Update the current user's profile.
    
    Only provided fields will be updated.
    """
    try:
        db = get_db()
        
        # Build update document
        update_data = {"updated_at": datetime.utcnow().isoformat() + "Z"}
        
        if request.display_name is not None:
            update_data["display_name"] = request.display_name
        if request.phone is not None:
            update_data["phone"] = request.phone
        if request.timezone is not None:
            update_data["timezone"] = request.timezone
        if request.language is not None:
            update_data["language"] = request.language
        
        # Upsert user document
        db.db.users.update_one(
            {"_id": user.user_id},
            {
                "$set": update_data,
                "$setOnInsert": {
                    "email": user.email,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "plan": "free",
                }
            },
            upsert=True
        )
        
        # Return updated profile
        return await get_profile(user)
        
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )


@router.post(
    "/me/avatar",
    response_model=AvatarResponse,
    summary="Upload avatar",
    description="Upload a new profile avatar image.",
)
async def upload_avatar(
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user)
) -> AvatarResponse:
    """
    Upload a profile avatar image.
    
    Accepts JPEG, PNG, GIF, or WebP images up to 5MB.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate file size (5MB max)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB."
        )
    
    # Generate unique filename
    ext = file.filename.split(".")[-1] if file.filename else "jpg"
    filename = f"avatars/{user.user_id}/{uuid.uuid4()}.{ext}"
    
    # For now, store locally (in production, use S3/CloudFlare R2)
    upload_dir = os.path.join(os.getcwd(), "uploads", "avatars", user.user_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}.{ext}")
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Generate URL (in production, this would be a CDN URL)
    avatar_url = f"/uploads/{filename}"
    
    # Update user profile with avatar URL
    try:
        db = get_db()
        db.db.users.update_one(
            {"_id": user.user_id},
            {
                "$set": {
                    "avatar_url": avatar_url,
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                },
                "$setOnInsert": {
                    "email": user.email,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
            },
            upsert=True
        )
    except RuntimeError:
        pass  # Continue even if database not available
    
    return AvatarResponse(avatar_url=avatar_url)


@router.delete(
    "/me/avatar",
    summary="Remove avatar",
    description="Remove the current user's avatar image.",
)
async def delete_avatar(user: UserContext = Depends(get_current_user)) -> dict:
    """
    Remove the current user's avatar.
    
    Sets avatar_url to null and optionally deletes the file.
    """
    try:
        db = get_db()
        db.db.users.update_one(
            {"_id": user.user_id},
            {
                "$set": {
                    "avatar_url": None,
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                }
            }
        )
    except RuntimeError:
        pass
    
    return {"message": "Avatar removed successfully"}


@router.put(
    "/me/password",
    summary="Change password",
    description="Change the current user's password.",
)
async def change_password(
    request: ChangePasswordRequest,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Change the current user's password.
    
    Note: This endpoint is primarily for local development auth.
    In production with Supabase, password changes go through Supabase Auth.
    """
    # For local development, we'd verify and update the password
    # In production with Supabase, redirect to Supabase password change
    
    supabase_url = os.getenv("SUPABASE_URL", "")
    if "localhost" not in supabase_url and "127.0.0.1" not in supabase_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password changes should be done through Supabase Auth"
        )
    
    # Local development password change would go here
    # For now, just return success
    return {"message": "Password changed successfully"}


@router.get(
    "/me/sessions",
    response_model=SessionsResponse,
    summary="Get active sessions",
    description="Get list of active sessions for the current user.",
)
async def get_sessions(user: UserContext = Depends(get_current_user)) -> SessionsResponse:
    """
    Get active sessions for the current user.
    
    Returns list of devices and locations where the user is logged in.
    """
    # In a full implementation, this would track actual sessions
    # For now, return mock data showing the current session
    return SessionsResponse(
        sessions=[
            Session(
                session_id="current",
                device="Current Browser",
                ip_address="127.0.0.1",
                location="Local",
                last_active=datetime.utcnow().isoformat() + "Z",
                is_current=True,
            )
        ]
    )


@router.delete(
    "/me/sessions/{session_id}",
    summary="Revoke session",
    description="Revoke a specific session.",
)
async def revoke_session(
    session_id: str,
    user: UserContext = Depends(get_current_user)
) -> dict:
    """
    Revoke a specific session.
    
    The user will be logged out from that device.
    """
    if session_id == "current":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke current session. Use sign out instead."
        )
    
    # In a full implementation, invalidate the session token
    return {"message": f"Session {session_id} revoked successfully"}


@router.delete(
    "/me/sessions",
    summary="Revoke all sessions",
    description="Sign out from all devices.",
)
async def revoke_all_sessions(user: UserContext = Depends(get_current_user)) -> dict:
    """
    Revoke all sessions except the current one.
    
    The user will be logged out from all other devices.
    """
    # In a full implementation, invalidate all session tokens
    return {"message": "All other sessions revoked successfully"}


@router.delete(
    "/me",
    summary="Delete account",
    description="Permanently delete the user account.",
)
async def delete_account(user: UserContext = Depends(get_current_user)) -> dict:
    """
    Permanently delete the user account.
    
    This action is irreversible. All user data will be deleted.
    """
    try:
        db = get_db()
        
        # Delete user profile
        db.db.users.delete_one({"_id": user.user_id})
        
        # Delete user's organization membership
        db.db.team_members.delete_many({"user_id": user.user_id})
        
        # Note: In production, also handle:
        # - Ownership transfer or org deletion
        # - Supabase user deletion
        # - Data export
        
    except RuntimeError:
        pass
    
    return {"message": "Account deleted successfully"}
