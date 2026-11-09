"""SQLAlchemy models for multi-tenant workspaces and workspace memberships."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.job import ProcessingJob


class Workspace(Base):
    """Isolated tenant organization workspace."""

    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Supabase auth.users.id (sub) of the workspace creator",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    members: Mapped[list["WorkspaceMember"]] = relationship(
        "WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan"
    )
    accounts: Mapped[list["Account"]] = relationship(
        "Account", back_populates="workspace", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["ProcessingJob"]] = relationship(
        "ProcessingJob", back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceMember(Base):
    """User membership and role assignment within a workspace."""

    __tablename__ = "workspace_members"

    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        index=True,
        comment="Supabase auth.users.id (sub) of the member",
    )
    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="owner",
        comment="Access role within workspace (owner, admin, member)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="members")
