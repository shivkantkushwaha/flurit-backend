"""Flurit Feature Engineering Package.

Provides point-in-time, leakage-safe customer feature extraction across
account, subscription, usage telemetry, support tickets, and churn history.
"""

from app.ml.features.account_features import AccountFeatureExtractor
from app.ml.features.churn_features import ChurnFeatureExtractor
from app.ml.features.pipeline import FeaturePipeline
from app.ml.features.schema import (
    FEATURE_DEFINITIONS,
    FEATURE_NAME_TO_DEF,
    FEATURE_NAMES,
    FeatureDefinition,
    FeatureGroup,
)
from app.ml.features.subscription_features import SubscriptionFeatureExtractor
from app.ml.features.support_features import SupportFeatureExtractor
from app.ml.features.usage_features import UsageFeatureExtractor

__all__ = [
    "FEATURE_DEFINITIONS",
    "FEATURE_NAME_TO_DEF",
    "FEATURE_NAMES",
    "AccountFeatureExtractor",
    "ChurnFeatureExtractor",
    "FeatureDefinition",
    "FeatureGroup",
    "FeaturePipeline",
    "SubscriptionFeatureExtractor",
    "SupportFeatureExtractor",
    "UsageFeatureExtractor",
]
