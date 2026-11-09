"""Multi-tenant workspace data isolation and prediction persistence integrity.

Revision ID: 0002_multi_tenant_workspace_isolation
Revises: 0001_initial_schema
Create Date: 2026-10-15 14:15:00

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_tenant_isolation"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --------------------------------------------------------------------------
    # 1. Create workspaces and workspace_members tables
    # --------------------------------------------------------------------------
    op.create_table(
        "workspaces",
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_name", sa.String(length=255), nullable=False),
        sa.Column("owner_user_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("workspace_id"),
    )
    op.create_index(op.f("ix_workspaces_owner_user_id"), "workspaces", ["owner_user_id"])

    op.create_table(
        "workspace_members",
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=32), server_default="owner", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id", "user_id"),
    )
    op.create_index(op.f("ix_workspace_members_user_id"), "workspace_members", ["user_id"])

    # Provision system legacy and default workspaces for unassigned or existing records
    op.execute(
        "INSERT INTO workspaces (workspace_id, workspace_name, owner_user_id, created_at) "
        "VALUES ('ws_legacy_unassigned', 'Legacy Unassigned System Workspace', 'system_legacy', now()) "
        "ON CONFLICT DO NOTHING;"
    )
    op.execute(
        "INSERT INTO workspaces (workspace_id, workspace_name, owner_user_id, created_at) "
        "VALUES ('ws_default', 'Default Development Workspace', 'system_default', now()) "
        "ON CONFLICT DO NOTHING;"
    )

    # --------------------------------------------------------------------------
    # 2. Drop existing foreign keys pointing to accounts and subscriptions
    # --------------------------------------------------------------------------
    op.drop_constraint("subscriptions_account_id_fkey", "subscriptions", type_="foreignkey")
    op.drop_constraint("feature_usage_subscription_id_fkey", "feature_usage", type_="foreignkey")
    op.drop_constraint("support_tickets_account_id_fkey", "support_tickets", type_="foreignkey")
    op.drop_constraint("churn_events_account_id_fkey", "churn_events", type_="foreignkey")
    op.drop_constraint(
        "customer_snapshots_account_id_fkey", "customer_snapshots", type_="foreignkey"
    )
    op.drop_constraint("customer_features_account_id_fkey", "customer_features", type_="foreignkey")
    op.drop_constraint(
        "customer_predictions_account_id_fkey", "customer_predictions", type_="foreignkey"
    )
    op.drop_constraint(
        "customer_explanations_account_id_fkey", "customer_explanations", type_="foreignkey"
    )
    op.drop_constraint("customer_segments_account_id_fkey", "customer_segments", type_="foreignkey")

    # Drop old non-workspace unique constraint on customer_snapshots
    op.drop_constraint("uq_account_snapshot_date", "customer_snapshots", type_="unique")

    # --------------------------------------------------------------------------
    # 3. Add workspace_id column to all tenant-scoped tables
    # --------------------------------------------------------------------------
    tenant_tables = [
        "accounts",
        "subscriptions",
        "feature_usage",
        "support_tickets",
        "churn_events",
        "customer_snapshots",
        "customer_features",
        "customer_predictions",
        "customer_explanations",
        "customer_segments",
    ]
    for table_name in tenant_tables:
        op.add_column(
            table_name,
            sa.Column(
                "workspace_id",
                sa.String(length=64),
                nullable=False,
                server_default="ws_legacy_unassigned",
            ),
        )

    # Widen string IDs to 64 chars to support flexible multi-tenant formats
    op.alter_column("accounts", "account_id", type_=sa.String(length=64))
    op.alter_column("subscriptions", "subscription_id", type_=sa.String(length=64))
    op.alter_column("subscriptions", "account_id", type_=sa.String(length=64))
    op.alter_column("feature_usage", "subscription_id", type_=sa.String(length=64))
    op.alter_column("feature_usage", "usage_id", type_=sa.String(length=64))
    op.alter_column("support_tickets", "ticket_id", type_=sa.String(length=64))
    op.alter_column("support_tickets", "account_id", type_=sa.String(length=64))
    op.alter_column("churn_events", "churn_event_id", type_=sa.String(length=64))
    op.alter_column("churn_events", "account_id", type_=sa.String(length=64))
    op.alter_column("customer_snapshots", "account_id", type_=sa.String(length=64))
    op.alter_column("customer_features", "account_id", type_=sa.String(length=64))
    op.alter_column("customer_predictions", "account_id", type_=sa.String(length=64))
    op.alter_column("customer_explanations", "account_id", type_=sa.String(length=64))
    op.alter_column("customer_segments", "account_id", type_=sa.String(length=64))

    # --------------------------------------------------------------------------
    # 4. Update Primary Keys to include workspace_id (composite PKs)
    # --------------------------------------------------------------------------
    op.drop_constraint("accounts_pkey", "accounts", type_="primary")
    op.create_primary_key("accounts_pkey", "accounts", ["workspace_id", "account_id"])

    op.drop_constraint("subscriptions_pkey", "subscriptions", type_="primary")
    op.create_primary_key(
        "subscriptions_pkey", "subscriptions", ["workspace_id", "subscription_id"]
    )

    op.drop_constraint("support_tickets_pkey", "support_tickets", type_="primary")
    op.create_primary_key("support_tickets_pkey", "support_tickets", ["workspace_id", "ticket_id"])

    op.drop_constraint("churn_events_pkey", "churn_events", type_="primary")
    op.create_primary_key("churn_events_pkey", "churn_events", ["workspace_id", "churn_event_id"])

    # --------------------------------------------------------------------------
    # 5. Create new workspace-scoped composite Foreign Keys
    # --------------------------------------------------------------------------
    op.create_foreign_key(
        "subscriptions_workspace_account_fkey",
        "subscriptions",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "feature_usage_workspace_subscription_fkey",
        "feature_usage",
        "subscriptions",
        ["workspace_id", "subscription_id"],
        ["workspace_id", "subscription_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "support_tickets_workspace_account_fkey",
        "support_tickets",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "churn_events_workspace_account_fkey",
        "churn_events",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_snapshots_workspace_account_fkey",
        "customer_snapshots",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_features_workspace_account_fkey",
        "customer_features",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_predictions_workspace_account_fkey",
        "customer_predictions",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_explanations_workspace_account_fkey",
        "customer_explanations",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_segments_workspace_account_fkey",
        "customer_segments",
        "accounts",
        ["workspace_id", "account_id"],
        ["workspace_id", "account_id"],
        ondelete="CASCADE",
    )

    # Add FKs directly to workspaces table
    for table_name in tenant_tables:
        op.create_foreign_key(
            f"fk_{table_name}_workspace",
            table_name,
            "workspaces",
            ["workspace_id"],
            ["workspace_id"],
            ondelete="CASCADE",
        )
        op.create_index(f"ix_{table_name}_workspace_id", table_name, ["workspace_id"])

    # --------------------------------------------------------------------------
    # 6. Database Unique Constraints for Prediction and Segment Idempotency
    # --------------------------------------------------------------------------
    op.create_unique_constraint(
        "uq_workspace_account_snapshot_date",
        "customer_snapshots",
        ["workspace_id", "account_id", "snapshot_date"],
    )
    op.create_unique_constraint(
        "uq_prediction_workspace_model_account_snapshot",
        "customer_predictions",
        ["workspace_id", "model_run_id", "account_id", "snapshot_id"],
    )
    op.create_unique_constraint(
        "uq_segment_workspace_model_account",
        "customer_segments",
        ["workspace_id", "model_run_id", "account_id"],
    )


def downgrade() -> None:
    tenant_tables = [
        "accounts",
        "subscriptions",
        "feature_usage",
        "support_tickets",
        "churn_events",
        "customer_snapshots",
        "customer_features",
        "customer_predictions",
        "customer_explanations",
        "customer_segments",
    ]

    # Drop unique constraints
    op.drop_constraint("uq_segment_workspace_model_account", "customer_segments", type_="unique")
    op.drop_constraint(
        "uq_prediction_workspace_model_account_snapshot", "customer_predictions", type_="unique"
    )
    op.drop_constraint("uq_workspace_account_snapshot_date", "customer_snapshots", type_="unique")

    # Drop composite foreign keys
    op.drop_constraint(
        "customer_segments_workspace_account_fkey", "customer_segments", type_="foreignkey"
    )
    op.drop_constraint(
        "customer_explanations_workspace_account_fkey", "customer_explanations", type_="foreignkey"
    )
    op.drop_constraint(
        "customer_predictions_workspace_account_fkey", "customer_predictions", type_="foreignkey"
    )
    op.drop_constraint(
        "customer_features_workspace_account_fkey", "customer_features", type_="foreignkey"
    )
    op.drop_constraint(
        "customer_snapshots_workspace_account_fkey", "customer_snapshots", type_="foreignkey"
    )
    op.drop_constraint("churn_events_workspace_account_fkey", "churn_events", type_="foreignkey")
    op.drop_constraint(
        "support_tickets_workspace_account_fkey", "support_tickets", type_="foreignkey"
    )
    op.drop_constraint(
        "feature_usage_workspace_subscription_fkey", "feature_usage", type_="foreignkey"
    )
    op.drop_constraint("subscriptions_workspace_account_fkey", "subscriptions", type_="foreignkey")

    for table_name in tenant_tables:
        op.drop_constraint(f"fk_{table_name}_workspace", table_name, type_="foreignkey")
        op.drop_index(f"ix_{table_name}_workspace_id", table_name=table_name)

    # Restore old primary keys
    op.drop_constraint("churn_events_pkey", "churn_events", type_="primary")
    op.create_primary_key("churn_events_pkey", "churn_events", ["churn_event_id"])

    op.drop_constraint("support_tickets_pkey", "support_tickets", type_="primary")
    op.create_primary_key("support_tickets_pkey", "support_tickets", ["ticket_id"])

    op.drop_constraint("subscriptions_pkey", "subscriptions", type_="primary")
    op.create_primary_key("subscriptions_pkey", "subscriptions", ["subscription_id"])

    op.drop_constraint("accounts_pkey", "accounts", type_="primary")
    op.create_primary_key("accounts_pkey", "accounts", ["account_id"])

    for table_name in tenant_tables:
        op.drop_column(table_name, "workspace_id")

    # Restore old foreign keys
    op.create_foreign_key(
        "subscriptions_account_id_fkey",
        "subscriptions",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "feature_usage_subscription_id_fkey",
        "feature_usage",
        "subscriptions",
        ["subscription_id"],
        ["subscription_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "support_tickets_account_id_fkey",
        "support_tickets",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "churn_events_account_id_fkey",
        "churn_events",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_snapshots_account_id_fkey",
        "customer_snapshots",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_features_account_id_fkey",
        "customer_features",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_predictions_account_id_fkey",
        "customer_predictions",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_explanations_account_id_fkey",
        "customer_explanations",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "customer_segments_account_id_fkey",
        "customer_segments",
        "accounts",
        ["account_id"],
        ["account_id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_account_snapshot_date", "customer_snapshots", ["account_id", "snapshot_date"]
    )

    op.drop_table("workspace_members")
    op.drop_table("workspaces")
