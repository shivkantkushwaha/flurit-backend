"""SQLAlchemy model for asynchronous processing jobs scoped to workspaces."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.workspace import Workspace


class ProcessingJob(Base):
    """Asynchronous dataset processing job record scoped to an authorized tenant workspace."""

    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "idempotency_key", name="uq_jobs_workspace_idempotency_key"
        ),
        Index("ix_jobs_workspace_status", "workspace_id", "status"),
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="Unique processing job identifier (e.g. job_12345678)",
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Supabase auth.users.id (sub) of submitter",
    )
    dataset_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        index=True,
        comment="Job state: queued, running, succeeded, failed, cancelled",
    )
    progress_pct: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Estimated progress percentage (0 - 100)",
    )
    stage_message: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Current execution stage description",
    )
    file_keys: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Storage references/paths for uploaded CSV datasets",
    )
    mapping_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment="Serialized concept and column mapping request",
    )
    options_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Serialized processing execution options",
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Client-supplied key to prevent duplicate job execution",
    )
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Sanitized pipeline summary and metrics upon successful execution",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Sanitized human-readable failure detail without stack traces or secrets",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Current retry count on worker failures",
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        comment="Maximum allowed worker retry attempts",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    workspace: Mapped[Optional["Workspace"]] = relationship("Workspace", back_populates="jobs")
