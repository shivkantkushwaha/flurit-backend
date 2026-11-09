"""SQLAlchemy model for accounts table."""

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.churn import ChurnEvent
    from app.models.ml import CustomerPrediction, CustomerSnapshot
    from app.models.subscription import Subscription
    from app.models.ticket import SupportTicket
    from app.models.workspace import Workspace


class Account(Base):
    """Customer account profiles scoped to a tenant workspace."""

    __tablename__ = "accounts"

    workspace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        primary_key=True,
        default="ws_default",
        index=True,
    )
    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[str] = mapped_column(String(64), nullable=False)
    country: Mapped[str] = mapped_column(String(8), nullable=False)
    signup_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    referral_source: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(64), nullable=False)
    seats: Mapped[int] = mapped_column(Integer, nullable=False)
    is_trial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    churn_flag: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Retrospective flag from source dataset; not to be used as predictive feature.",
    )

    # Relationships
    subscriptions: Mapped[list["Subscription"]] = relationship(
        "Subscription", back_populates="account", cascade="all, delete-orphan"
    )
    support_tickets: Mapped[list["SupportTicket"]] = relationship(
        "SupportTicket", back_populates="account", cascade="all, delete-orphan"
    )
    churn_events: Mapped[list["ChurnEvent"]] = relationship(
        "ChurnEvent", back_populates="account", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list["CustomerSnapshot"]] = relationship(
        "CustomerSnapshot", back_populates="account", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["CustomerPrediction"]] = relationship(
        "CustomerPrediction", back_populates="account", cascade="all, delete-orphan"
    )
    workspace: Mapped[Optional["Workspace"]] = relationship("Workspace", back_populates="accounts")
