"""Flurit Snapshot & Target Construction Package.

Provides point-in-time customer snapshot generation and leakage-safe
future churn target labeling across chronological observation dates.
"""

from app.ml.snapshots.engine import SnapshotEngine, SnapshotRecord
from app.ml.snapshots.targets import TargetConstructor

__all__ = ["SnapshotEngine", "SnapshotRecord", "TargetConstructor"]
