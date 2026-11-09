"""Model Explainability Package for Flurit ML.

Provides local and global feature attribution using SHAP LinearExplainer,
aggregated to canonical point-in-time features with plain-language summaries
and explicit observational-association disclaimers.
"""

from app.ml.explainability.explainer import (
    GlobalFeatureImportance,
    ModelExplainer,
    PredictionExplanation,
)

__all__ = [
    "ModelExplainer",
    "PredictionExplanation",
    "GlobalFeatureImportance",
]
