"""Flurit Database Models Package.

Exports all SQLAlchemy models for source RavenStack entities
and derived machine learning / analytics entities.
"""

from app.db.base import Base
from app.models.account import Account
from app.models.churn import ChurnEvent
from app.models.job import ProcessingJob
from app.models.ml import (
    CustomerExplanation,
    CustomerFeature,
    CustomerPrediction,
    CustomerSegment,
    CustomerSnapshot,
    ModelRun,
)
from app.models.subscription import Subscription
from app.models.ticket import SupportTicket
from app.models.usage import FeatureUsage
from app.models.workspace import Workspace, WorkspaceMember

__all__ = [
    "Base",
    "Workspace",
    "WorkspaceMember",
    "ProcessingJob",
    "Account",
    "Subscription",
    "FeatureUsage",
    "SupportTicket",
    "ChurnEvent",
    "CustomerSnapshot",
    "CustomerFeature",
    "ModelRun",
    "CustomerPrediction",
    "CustomerExplanation",
    "CustomerSegment",
]
