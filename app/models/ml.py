"""SQLAlchemy models for derived machine learning and analytics entities.

Encapsulates point-in-time snapshots, feature matrices, model runs,
predictions, SHAP explanations, and customer segments.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.account import Account


class CustomerSnapshot(Base):
    """Point-in-time customer snapshot for leakage-safe target labeling scoped to a workspace."""

    __tablename__ = "customer_snapshots"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "workspace_id", "account_id", "snapshot_date", name="uq_workspace_account_snapshot_date"
        ),
    )

    snapshot_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        default="ws_default",
        index=True,
    )
    account_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    target_churn_60d: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="1 if churned in (T, T+60d], 0 if active throughout, NULL if right-censored.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")
    features: Mapped[Optional["CustomerFeature"]] = relationship(
        "CustomerFeature", back_populates="snapshot", uselist=False, cascade="all, delete-orphan"
    )
    predictions: Mapped[list["CustomerPrediction"]] = relationship(
        "CustomerPrediction", back_populates="snapshot", cascade="all, delete-orphan"
    )


class CustomerFeature(Base):
    """Engineered behavioral and subscription feature vectors evaluated strictly at snapshot date."""

    __tablename__ = "customer_features"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
    )

    feature_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        default="ws_default",
        index=True,
    )
    snapshot_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("customer_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    account_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Structured features
    current_mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_arr: Mapped[float | None] = mapped_column(Float, nullable=True)
    tenure_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_count_30d: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_duration_30d: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ticket_count_30d: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_unrated_tickets: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # Complete serialized dictionary for flexible feature sets
    feature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    snapshot: Mapped["CustomerSnapshot"] = relationship(
        "CustomerSnapshot", back_populates="features"
    )


class ModelRun(Base):
    """Trained ML model run registry and evaluation metadata."""

    __tablename__ = "model_runs"

    model_run_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Evaluation metrics: PR-AUC, ROC-AUC, F1, Brier score, Recall.",
    )
    feature_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")

    # Relationships
    predictions: Mapped[list["CustomerPrediction"]] = relationship(
        "CustomerPrediction", back_populates="model_run", cascade="all, delete-orphan"
    )
    segments: Mapped[list["CustomerSegment"]] = relationship(
        "CustomerSegment", back_populates="model_run", cascade="all, delete-orphan"
    )


class CustomerPrediction(Base):
    """Model inference outputs and risk scores per customer snapshot scoped to a tenant workspace."""

    __tablename__ = "customer_predictions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "workspace_id",
            "model_run_id",
            "account_id",
            "snapshot_id",
            name="uq_prediction_workspace_model_account_snapshot",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        default="ws_default",
        index=True,
    )
    model_run_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("model_runs.model_run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    account_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    snapshot_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("customer_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    model_run: Mapped["ModelRun"] = relationship("ModelRun", back_populates="predictions")
    account: Mapped["Account"] = relationship("Account", back_populates="predictions")
    snapshot: Mapped[Optional["CustomerSnapshot"]] = relationship(
        "CustomerSnapshot", back_populates="predictions"
    )
    explanation: Mapped[Optional["CustomerExplanation"]] = relationship(
        "CustomerExplanation",
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


class CustomerExplanation(Base):
    """SHAP feature attribution explanations linked to customer predictions scoped to a workspace."""

    __tablename__ = "customer_explanations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
    )

    explanation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        default="ws_default",
        index=True,
    )
    prediction_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("customer_predictions.prediction_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    account_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    top_risk_factors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        comment="Ordered list of top SHAP risk factors with feature names, values, and attributions.",
    )
    shap_values: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    prediction: Mapped["CustomerPrediction"] = relationship(
        "CustomerPrediction", back_populates="explanation"
    )


class CustomerSegment(Base):
    """Unsupervised K-Means behavioral and value cluster assignments scoped to a tenant workspace."""

    __tablename__ = "customer_segments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "workspace_id",
            "model_run_id",
            "account_id",
            name="uq_segment_workspace_model_account",
        ),
    )

    segment_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        default="ws_default",
        index=True,
    )
    model_run_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("model_runs.model_run_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    account_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_name: Mapped[str] = mapped_column(String(100), nullable=False)
    risk_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    value_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    model_run: Mapped[Optional["ModelRun"]] = relationship("ModelRun", back_populates="segments")
