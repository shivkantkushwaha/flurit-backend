"""SQLAlchemy model for churn_events table."""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, Float, ForeignKey, ForeignKeyConstraint, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.account import Account


class ChurnEvent(Base):
    """Historical churn occurrences and exit feedback scoped to a tenant workspace."""

    __tablename__ = "churn_events"
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
    churn_event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    churn_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    refund_amount_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    preceding_upgrade_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preceding_downgrade_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_reactivation: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if customer previously churned and subsequently reactivated.",
    )
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    account: Mapped["Account"] = relationship("Account", back_populates="churn_events")
