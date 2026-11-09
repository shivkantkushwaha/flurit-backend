"""SQLAlchemy model for support_tickets table."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.account import Account


class SupportTicket(Base):
    """Customer support interactions and resolutions scoped to a tenant workspace."""

    __tablename__ = "support_tickets"
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
    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    priority: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    first_response_time_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    satisfaction_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Customer CSAT rating (1.0 to 5.0). Null in 41.2% of source tickets.",
    )
    escalation_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    account: Mapped["Account"] = relationship("Account", back_populates="support_tickets")
