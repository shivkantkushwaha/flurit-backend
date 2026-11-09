"""SQLAlchemy model for subscriptions table."""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, Float, ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.usage import FeatureUsage


class Subscription(Base):
    """Customer subscription contracts scoped to a tenant workspace."""

    __tablename__ = "subscriptions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "account_id"],
            ["accounts.workspace_id", "accounts.account_id"],
            ondelete="CASCADE",
        ),
    )

    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        primary_key=True,
        default="ws_default",
        index=True,
    )
    subscription_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    plan_tier: Mapped[str] = mapped_column(String(64), nullable=False)
    seats: Mapped[int] = mapped_column(Integer, nullable=False)
    mrr_amount: Mapped[float] = mapped_column(Float, nullable=False)
    arr_amount: Mapped[float] = mapped_column(Float, nullable=False)
    is_trial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    upgrade_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    downgrade_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    churn_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    billing_frequency: Mapped[str] = mapped_column(String(32), nullable=False)
    auto_renew_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    account: Mapped["Account"] = relationship("Account", back_populates="subscriptions")
    feature_usage: Mapped[list["FeatureUsage"]] = relationship(
        "FeatureUsage", back_populates="subscription", cascade="all, delete-orphan"
    )
