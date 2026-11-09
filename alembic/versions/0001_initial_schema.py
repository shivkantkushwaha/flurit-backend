"""Initial schema: RavenStack source entities and derived ML analytics tables.

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-10-13 22:50:00

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. accounts
    op.create_table(
        "accounts",
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("account_name", sa.String(length=255), nullable=False),
        sa.Column("industry", sa.String(length=64), nullable=False),
        sa.Column("country", sa.String(length=8), nullable=False),
        sa.Column("signup_date", sa.Date(), nullable=False),
        sa.Column("referral_source", sa.String(length=64), nullable=False),
        sa.Column("plan_tier", sa.String(length=64), nullable=False),
        sa.Column("seats", sa.Integer(), nullable=False),
        sa.Column("is_trial", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("churn_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.PrimaryKeyConstraint("account_id"),
    )
    op.create_index(op.f("ix_accounts_signup_date"), "accounts", ["signup_date"])

    # 2. subscriptions
    op.create_table(
        "subscriptions",
        sa.Column("subscription_id", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("plan_tier", sa.String(length=64), nullable=False),
        sa.Column("seats", sa.Integer(), nullable=False),
        sa.Column("mrr_amount", sa.Float(), nullable=False),
        sa.Column("arr_amount", sa.Float(), nullable=False),
        sa.Column("is_trial", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("upgrade_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("downgrade_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("churn_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("billing_frequency", sa.String(length=32), nullable=False),
        sa.Column("auto_renew_flag", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("subscription_id"),
    )
    op.create_index(op.f("ix_subscriptions_account_id"), "subscriptions", ["account_id"])
    op.create_index(op.f("ix_subscriptions_start_date"), "subscriptions", ["start_date"])
    op.create_index(op.f("ix_subscriptions_end_date"), "subscriptions", ["end_date"])

    # 3. feature_usage (surrogate PK id, non-unique indexed usage_id to prevent collision)
    op.create_table(
        "feature_usage",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("usage_id", sa.String(length=32), nullable=False),
        sa.Column("subscription_id", sa.String(length=32), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("usage_duration_secs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_beta_feature", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(
            ["subscription_id"], ["subscriptions.subscription_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_feature_usage_usage_id"), "feature_usage", ["usage_id"])
    op.create_index(op.f("ix_feature_usage_subscription_id"), "feature_usage", ["subscription_id"])
    op.create_index(op.f("ix_feature_usage_usage_date"), "feature_usage", ["usage_date"])
    op.create_index(op.f("ix_feature_usage_feature_name"), "feature_usage", ["feature_name"])

    # 4. support_tickets
    op.create_table(
        "support_tickets",
        sa.Column("ticket_id", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("submitted_at", sa.DateTime(), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("resolution_time_hours", sa.Float(), nullable=True),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("first_response_time_minutes", sa.Integer(), nullable=True),
        sa.Column("satisfaction_score", sa.Float(), nullable=True),
        sa.Column("escalation_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("ticket_id"),
    )
    op.create_index(op.f("ix_support_tickets_account_id"), "support_tickets", ["account_id"])
    op.create_index(op.f("ix_support_tickets_submitted_at"), "support_tickets", ["submitted_at"])
    op.create_index(op.f("ix_support_tickets_priority"), "support_tickets", ["priority"])

    # 5. churn_events
    op.create_table(
        "churn_events",
        sa.Column("churn_event_id", sa.String(length=32), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("churn_date", sa.Date(), nullable=False),
        sa.Column("reason_code", sa.String(length=64), nullable=False),
        sa.Column("refund_amount_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "preceding_upgrade_flag", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "preceding_downgrade_flag", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("is_reactivation", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("churn_event_id"),
    )
    op.create_index(op.f("ix_churn_events_account_id"), "churn_events", ["account_id"])
    op.create_index(op.f("ix_churn_events_churn_date"), "churn_events", ["churn_date"])
    op.create_index(op.f("ix_churn_events_reason_code"), "churn_events", ["reason_code"])

    # 6. customer_snapshots
    op.create_table(
        "customer_snapshots",
        sa.Column("snapshot_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("target_churn_60d", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("account_id", "snapshot_date", name="uq_account_snapshot_date"),
    )
    op.create_index(op.f("ix_customer_snapshots_account_id"), "customer_snapshots", ["account_id"])
    op.create_index(
        op.f("ix_customer_snapshots_snapshot_date"), "customer_snapshots", ["snapshot_date"]
    )

    # 7. customer_features
    op.create_table(
        "customer_features",
        sa.Column("feature_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("current_mrr", sa.Float(), nullable=True),
        sa.Column("current_arr", sa.Float(), nullable=True),
        sa.Column("tenure_days", sa.Integer(), nullable=True),
        sa.Column("usage_count_30d", sa.Integer(), nullable=True),
        sa.Column("usage_duration_30d", sa.Integer(), nullable=True),
        sa.Column("usage_change_pct", sa.Float(), nullable=True),
        sa.Column("ticket_count_30d", sa.Integer(), nullable=True),
        sa.Column("avg_satisfaction", sa.Float(), nullable=True),
        sa.Column("has_unrated_tickets", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("feature_payload", sa.JSON(), nullable=True),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["customer_snapshots.snapshot_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("feature_id"),
        sa.UniqueConstraint("snapshot_id"),
    )
    op.create_index(op.f("ix_customer_features_snapshot_id"), "customer_features", ["snapshot_id"])
    op.create_index(op.f("ix_customer_features_account_id"), "customer_features", ["account_id"])
    op.create_index(
        op.f("ix_customer_features_snapshot_date"), "customer_features", ["snapshot_date"]
    )

    # 8. model_runs
    op.create_table(
        "model_runs",
        sa.Column("model_run_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("decision_threshold", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("hyperparameters", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(length=255), nullable=True),
        sa.Column("trained_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.PrimaryKeyConstraint("model_run_id"),
    )

    # 9. customer_predictions
    op.create_table(
        "customer_predictions",
        sa.Column("prediction_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.BigInteger(), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=True),
        sa.Column("churn_probability", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("predicted_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.model_run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["customer_snapshots.snapshot_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("prediction_id"),
    )
    op.create_index(
        op.f("ix_customer_predictions_model_run_id"), "customer_predictions", ["model_run_id"]
    )
    op.create_index(
        op.f("ix_customer_predictions_account_id"), "customer_predictions", ["account_id"]
    )
    op.create_index(
        op.f("ix_customer_predictions_snapshot_id"), "customer_predictions", ["snapshot_id"]
    )

    # 10. customer_explanations
    op.create_table(
        "customer_explanations",
        sa.Column("explanation_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("top_risk_factors", sa.JSON(), nullable=False),
        sa.Column("shap_values", sa.JSON(), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["customer_predictions.prediction_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("explanation_id"),
        sa.UniqueConstraint("prediction_id"),
    )
    op.create_index(
        op.f("ix_customer_explanations_prediction_id"), "customer_explanations", ["prediction_id"]
    )
    op.create_index(
        op.f("ix_customer_explanations_account_id"), "customer_explanations", ["account_id"]
    )

    # 11. customer_segments
    op.create_table(
        "customer_segments",
        sa.Column("segment_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_run_id", sa.BigInteger(), nullable=True),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("cluster_name", sa.String(length=100), nullable=False),
        sa.Column("risk_tier", sa.String(length=32), nullable=True),
        sa.Column("value_tier", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.model_run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.account_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("segment_id"),
    )
    op.create_index(op.f("ix_customer_segments_account_id"), "customer_segments", ["account_id"])
    op.create_index(
        op.f("ix_customer_segments_model_run_id"), "customer_segments", ["model_run_id"]
    )


def downgrade() -> None:
    op.drop_table("customer_segments")
    op.drop_table("customer_explanations")
    op.drop_table("customer_predictions")
    op.drop_table("model_runs")
    op.drop_table("customer_features")
    op.drop_table("customer_snapshots")
    op.drop_table("churn_events")
    op.drop_table("support_tickets")
    op.drop_table("feature_usage")
    op.drop_table("subscriptions")
    op.drop_table("accounts")
