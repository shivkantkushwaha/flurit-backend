"""Customer Segmentation Package for Flurit ML.

Provides unsupervised K-Means clustering across point-in-time behavioral and value
features, candidate cluster evaluation (K=2..6), empirical cluster profiling,
and Customer Value x Model Risk prioritization analysis.
"""

from app.ml.segmentation.clustering import (
    ClusterEvaluationResult,
    ClusterProfile,
    CustomerSegmenter,
    PrioritizationQuadrant,
)

__all__ = [
    "CustomerSegmenter",
    "ClusterEvaluationResult",
    "ClusterProfile",
    "PrioritizationQuadrant",
]
