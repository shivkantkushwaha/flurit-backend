"""Workspace Context and Tenant Ownership Authorization Module for Flurit.

Provides server-side tenant resolution, workspace authorization, and membership
verification. Maps Supabase authenticated users ('sub') to isolated workspaces.
"""

from uuid import uuid4

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import AuthenticatedUser, verify_supabase_jwt
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.workspace import Workspace, WorkspaceMember

logger = get_logger("flurit.workspace")


class WorkspaceContext(BaseModel):
    """Verified workspace and authorization context for the current request."""

    workspace_id: str = Field(..., description="Unique workspace tenant identifier")
    workspace_name: str = Field(..., description="Human-readable workspace title")
    user_id: str = Field(..., description="Authenticated user identifier (sub)")
    role: str = Field(
        ..., description="User role in the authorized workspace (owner, admin, member)"
    )


def get_or_create_default_workspace(
    db: Session,
    user_id: str,
    email: str | None = None,
) -> tuple[Workspace, WorkspaceMember]:
    """Retrieve the primary workspace for a user or provision a default personal workspace.

    Args:
        db: Active SQLAlchemy database session.
        user_id: Unique Supabase subject claim (auth.users.id).
        email: Optional user email address used to name default workspace.

    Returns:
        Tuple of (Workspace, WorkspaceMember).
    """
    # 1. Check for existing active workspace membership
    stmt = (
        select(WorkspaceMember, Workspace)
        .join(Workspace, WorkspaceMember.workspace_id == Workspace.workspace_id)
        .where(WorkspaceMember.user_id == user_id)
        .order_by(WorkspaceMember.created_at)
    )
    result = db.execute(stmt).first()
    if result:
        member, ws = result
        return ws, member

    # 2. Provision new default workspace for this user
    ws_id = f"ws_{uuid4().hex[:16]}"
    display_name = (email or user_id).split("@")[0].capitalize()
    ws_name = f"{display_name}'s Workspace"

    workspace = Workspace(
        workspace_id=ws_id,
        workspace_name=ws_name,
        owner_user_id=user_id,
    )
    membership = WorkspaceMember(
        workspace_id=ws_id,
        user_id=user_id,
        role="owner",
    )
    db.add(workspace)
    db.add(membership)
    db.flush()

    logger.info(
        "Auto-provisioned default personal workspace for user",
        extra={"user_id": user_id, "workspace_id": ws_id},
    )
    return workspace, membership


def verify_user_workspace_access(
    db: Session,
    user_id: str,
    workspace_id: str,
) -> WorkspaceMember | None:
    """Verify server-side whether a user has authorized membership in a workspace.

    Args:
        db: Active database session.
        user_id: Supabase user subject identifier.
        workspace_id: Target workspace identifier.

    Returns:
        WorkspaceMember instance if authorized, or None if forbidden.
    """
    stmt = select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    )
    return db.scalars(stmt).first()


def get_workspace_context(
    current_user: AuthenticatedUser = Depends(verify_supabase_jwt),
    db: Session = Depends(get_db),
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-ID"),
) -> WorkspaceContext:
    """FastAPI dependency resolving and enforcing the authorized workspace context.

    If X-Workspace-ID header is provided:
        Verifies that current_user has an active membership record in that workspace.
        Fails closed with HTTP 403 Forbidden if the user is unauthorized.

    If X-Workspace-ID header is omitted:
        Resolves to the user's primary/default workspace (auto-provisioning if first login).

    Args:
        current_user: Verified Supabase authenticated user.
        db: Active SQLAlchemy database session.
        x_workspace_id: Optional client-requested workspace header.

    Returns:
        WorkspaceContext with verified workspace_id, user_id, and role.

    Raises:
        HTTPException(403): If the user does not have permission for the requested workspace.
        HTTPException(404): If requested workspace does not exist.
    """
    user_id = current_user.user_id

    if x_workspace_id:
        clean_ws_id = x_workspace_id.strip()
        membership = verify_user_workspace_access(db, user_id=user_id, workspace_id=clean_ws_id)
        if not membership:
            logger.warning(
                "Cross-workspace access attempt blocked",
                extra={"user_id": user_id, "attempted_workspace_id": clean_ws_id},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: User '{user_id}' is not an authorized member of workspace '{clean_ws_id}'.",
            )

        workspace = db.get(Workspace, clean_ws_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{clean_ws_id}' not found.",
            )

        return WorkspaceContext(
            workspace_id=clean_ws_id,
            workspace_name=workspace.workspace_name,
            user_id=user_id,
            role=membership.role,
        )

    # Resolve or create default workspace
    ws, member = get_or_create_default_workspace(db, user_id=user_id, email=current_user.email)
    return WorkspaceContext(
        workspace_id=ws.workspace_id,
        workspace_name=ws.workspace_name,
        user_id=user_id,
        role=member.role,
    )
