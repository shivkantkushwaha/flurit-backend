"""SQLAlchemy model for feature_usage table.

CRITICAL DATA QUALITY ARCHITECTURE:
The source dataset contains 21 duplicate usage_id strings across 25,000 rows.
To prevent primary key collisions and preserve all usage records without data loss,
this model defines a surrogate primary key (id) and retains usage_id as an indexed,
non-unique natural source identifier.
"""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Date, ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.subscription import Subscription


class FeatureUsage(Base):
    """Customer product feature usage telemetry events scoped to a tenant workspace."""

    __tablename__ = "feature_usage"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "subscription_id"],
            ["subscriptions.workspace_id", "subscriptions.subscription_id"],
            ondelete="CASCADE",
        ),
    )

    # Surrogate primary key to accommodate duplicate source usage_ids without collision
    id: Mapped[int] = mapped_column(
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
    # Natural source identifier (indexed, non-unique)
    usage_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Source identifier from CSV; contains non-unique values in synthetic data.",
    )
    subscription_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    usage_duration_secs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_beta_feature: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    subscription: Mapped["Subscription"] = relationship(
        "Subscription", back_populates="feature_usage"
    )
