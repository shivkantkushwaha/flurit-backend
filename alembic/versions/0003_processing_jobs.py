"""Add processing_jobs table for asynchronous dataset execution.

Revision ID: 0003_processing_jobs
Revises: 0002_tenant_isolation
Create Date: 2026-10-15 15:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_processing_jobs"
down_revision: str | None = "0002_tenant_isolation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "processing_jobs",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="queued", nullable=False),
        sa.Column("progress_pct", sa.Integer(), server_default="0", nullable=False),
        sa.Column("stage_message", sa.String(length=255), nullable=True),
        sa.Column("file_keys", sa.JSON(), nullable=True),
        sa.Column("mapping_payload", sa.JSON(), nullable=False),
        sa.Column("options_payload", sa.JSON(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_retries", sa.Integer(), server_default="3", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "workspace_id", "idempotency_key", name="uq_jobs_workspace_idempotency_key"
        ),
    )
    op.create_index(op.f("ix_processing_jobs_workspace_id"), "processing_jobs", ["workspace_id"])
    op.create_index(op.f("ix_processing_jobs_user_id"), "processing_jobs", ["user_id"])
    op.create_index(op.f("ix_processing_jobs_status"), "processing_jobs", ["status"])
    op.create_index(op.f("ix_processing_jobs_created_at"), "processing_jobs", ["created_at"])
    op.create_index("ix_jobs_workspace_status", "processing_jobs", ["workspace_id", "status"])


def downgrade() -> None:
    op.drop_table("processing_jobs")
