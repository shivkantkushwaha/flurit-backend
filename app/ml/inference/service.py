"""Flurit Reusable Model Inference Service.

Orchestrates read-only model predictions, SHAP explainability factor breakdowns,
customer segmentation assignments, and Value x Risk portfolio prioritization.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.logging import get_logger
from app.ml.explainability.explainer import ASSOCIATION_DISCLAIMER, ModelExplainer
from app.ml.features.schema import FEATURE_NAMES
from app.ml.inference.schemas import (
    CanonicalFeaturesPayload,
    CustomerRiskRequest,
    CustomerRiskResponse,
    FactorAttributionResponse,
    PrioritizationQuadrantDetail,
    PrioritizationQuadrantsRequest,
    PrioritizationQuadrantsResponse,
    ScorePredictionRequest,
    ScorePredictionResponse,
)
from app.ml.segmentation.clustering import CustomerSegmenter
from app.ml.training.trainer import ModelTrainer

logger = get_logger("flurit.ml.inference")

RiskTier = Literal["High Risk", "Moderate Risk", "Low Risk"]

SCORE_INTERPRETATION_NOTICE: str = (
    "Model Score Interpretation: Flurit model churn scores represent uncalibrated risk rankings "
    "learned by balanced Logistic Regression. Because Phase 6.1 demonstrated negative Brier skill "
    "scores (-0.2554) due to class_weight='balanced' probability shifts, the score is an operational "
    "prioritization index in [0.0, 1.0], NOT an actuarially calibrated probability of customer churn."
)


class ModelInferenceService:
    """Thread-safe, read-only inference engine serving predictions and explanations."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] | None = None
        self.explainer: ModelExplainer | None = None
        self.segmenter: CustomerSegmenter | None = None
        self.is_loaded: bool = False
        self._decision_threshold: float = 0.50
        self._model_version: str = "1.0.0"
        self._model_name: str = "flurit_churn_predictor"

    def load_artifacts(
        self,
        models_dir: Path | str = Path("artifacts/models/churn_prediction_v1"),
        explainability_dir: Path | str = Path("artifacts/explainability"),
        segmentation_dir: Path | str = Path("artifacts/segmentation"),
    ) -> None:
        """Load and validate all frozen ML artifacts into memory.

        Args:
            models_dir: Directory containing model.joblib and metadata.json.
            explainability_dir: Directory containing explainer.joblib.
            segmentation_dir: Directory containing segmentation_model.joblib and cluster_profiles.json.

        Raises:
            FileNotFoundError: If an artifact file is missing.
            ValueError: If an artifact is corrupt or incompatible with the feature schema.
        """
        models_path = Path(models_dir)
        explainability_path = Path(explainability_dir)
        segmentation_path = Path(segmentation_dir)

        logger.info(
            "Loading Flurit ML artifacts for inference...",
            extra={
                "models_dir": str(models_path.resolve()),
                "explainability_dir": str(explainability_path.resolve()),
                "segmentation_dir": str(segmentation_path.resolve()),
            },
        )

        # 1. Load Champion Model Pipeline & Metadata
        if not (models_path / "model.joblib").exists():
            raise FileNotFoundError(f"Missing model artifact: {models_path / 'model.joblib'}")
        if not (models_path / "metadata.json").exists():
            raise FileNotFoundError(f"Missing metadata artifact: {models_path / 'metadata.json'}")

        self.pipeline, self.metadata = ModelTrainer.load_artifact(models_path)

        # Validate feature schema compatibility
        meta_features = self.metadata.get("feature_names", [])
        if meta_features != FEATURE_NAMES:
            raise ValueError(
                f"Model artifact feature list mismatch. Expected {len(FEATURE_NAMES)} canonical "
                f"features, but metadata specifies {len(meta_features)} features."
            )

        self._decision_threshold = float(self.metadata.get("decision_threshold", 0.50))
        self._model_version = str(self.metadata.get("version", "1.0.0"))
        self._model_name = str(self.metadata.get("model_name", "flurit_churn_predictor"))

        # 2. Load Explainer Artifact
        if not (explainability_path / "explainer.joblib").exists():
            raise FileNotFoundError(
                f"Missing explainer artifact: {explainability_path / 'explainer.joblib'}"
            )

        self.explainer = ModelExplainer.load_artifact(explainability_path, self.pipeline)

        # 3. Load Segmentation Artifact
        if not (segmentation_path / "segmentation_model.joblib").exists():
            raise FileNotFoundError(
                f"Missing segmentation artifact: {segmentation_path / 'segmentation_model.joblib'}"
            )

        self.segmenter = CustomerSegmenter.load_artifact(segmentation_path)
        if not self.segmenter.is_fitted:
            raise ValueError("Loaded CustomerSegmenter is not fitted.")

        self.is_loaded = True
        logger.info(
            "Flurit ML inference service successfully initialized with all artifacts.",
            extra={
                "model_version": self._model_version,
                "decision_threshold": self._decision_threshold,
                "n_clusters": self.segmenter.n_clusters,
            },
        )

    def _ensure_loaded(self) -> None:
        """Internal guard ensuring artifacts are ready before scoring."""
        if (
            not self.is_loaded
            or self.pipeline is None
            or self.explainer is None
            or self.segmenter is None
        ):
            raise RuntimeError(
                "ModelInferenceService is not initialized. Call load_artifacts() before inference."
            )

    def _payload_to_dataframe(
        self,
        features: CanonicalFeaturesPayload | dict[str, Any],
    ) -> pd.DataFrame:
        """Convert input payload to DataFrame strictly ordered by canonical FEATURE_NAMES."""
        if isinstance(features, CanonicalFeaturesPayload):
            feat_dict = features.to_canonical_dict()
        else:
            # Validate presence of all canonical features
            missing = set(FEATURE_NAMES) - set(features.keys())
            if missing:
                raise ValueError(f"Missing required canonical features: {sorted(missing)}")
            feat_dict = {col: features[col] for col in FEATURE_NAMES}

        return pd.DataFrame([feat_dict])[FEATURE_NAMES]

    def score(
        self,
        request: ScorePredictionRequest,
    ) -> ScorePredictionResponse:
        """Score a single canonical feature vector.

        Args:
            request: Validated ScorePredictionRequest.

        Returns:
            ScorePredictionResponse with model score, risk label, and risk tier.
        """
        self._ensure_loaded()
        assert self.pipeline is not None

        df = self._payload_to_dataframe(request.features)
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        label = int(prob >= self._decision_threshold)

        risk_tier: RiskTier
        if prob >= 0.50:
            risk_tier = "High Risk"
        elif prob >= 0.35:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

        return ScorePredictionResponse(
            account_id=request.account_id,
            snapshot_date=request.snapshot_date,
            model_name=self._model_name,
            model_version=self._model_version,
            model_score=round(prob, 4),
            predicted_risk_label=label,
            decision_threshold=self._decision_threshold,
            risk_tier=risk_tier,
            score_interpretation=SCORE_INTERPRETATION_NOTICE,
        )

    def explain_and_segment(
        self,
        account_id: str,
        request: CustomerRiskRequest,
    ) -> CustomerRiskResponse:
        """Generate combined score, SHAP explanation, and customer segment assignment.

        Args:
            account_id: Customer account identifier.
            request: Validated CustomerRiskRequest payload.

        Returns:
            CustomerRiskResponse containing score, local factor attributions, and cluster info.
        """
        self._ensure_loaded()
        assert self.pipeline is not None
        assert self.explainer is not None
        assert self.segmenter is not None

        df = self._payload_to_dataframe(request.features)
        snapshot_date = request.snapshot_date or "latest"

        # 1. Scoring
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        label = int(prob >= self._decision_threshold)
        risk_tier: RiskTier
        if prob >= 0.50:
            risk_tier = "High Risk"
        elif prob >= 0.35:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

        # 2. Local SHAP Explanation
        feat_dict = df.iloc[0].to_dict()
        exp = self.explainer.explain_instance(
            account_id=account_id,
            snapshot_date=snapshot_date,
            feature_row=feat_dict,
            top_k=5,
        )

        top_increasing = [
            FactorAttributionResponse(
                feature_name=f["feature_name"],
                feature_value=f["feature_value"],
                attribution=round(f["attribution"], 4),
                direction="increases_risk",
                feature_description=f["feature_description"],
            )
            for f in exp.top_risk_increasing_factors
        ]

        top_decreasing = [
            FactorAttributionResponse(
                feature_name=f["feature_name"],
                feature_value=f["feature_value"],
                attribution=round(f["attribution"], 4),
                direction="decreases_risk",
                feature_description=f["feature_description"],
            )
            for f in exp.top_risk_decreasing_factors
        ]

        # 3. Customer Segment Assignment
        cluster_id = int(self.segmenter.predict_segments(df)[0])
        matching_profile = next(
            (p for p in self.segmenter.cluster_profiles if p.cluster_id == cluster_id),
            None,
        )
        if matching_profile is not None:
            segment_name = matching_profile.cluster_name
            segment_summary = matching_profile.empirical_summary
        else:
            segment_name = f"Cluster {cluster_id}"
            segment_summary = f"Assigned to cluster segment {cluster_id}."

        return CustomerRiskResponse(
            account_id=account_id,
            snapshot_date=snapshot_date,
            model_version=self._model_version,
            model_score=round(prob, 4),
            predicted_risk_label=label,
            decision_threshold=self._decision_threshold,
            risk_tier=risk_tier,
            segment_id=cluster_id,
            segment_name=segment_name,
            segment_summary=segment_summary,
            top_risk_increasing_factors=top_increasing,
            top_risk_decreasing_factors=top_decreasing,
            plain_language_summary=exp.plain_language_summary,
            association_disclaimer=ASSOCIATION_DISCLAIMER,
            score_interpretation=SCORE_INTERPRETATION_NOTICE,
        )

    def prioritize_quadrants(
        self,
        request: PrioritizationQuadrantsRequest,
    ) -> PrioritizationQuadrantsResponse:
        """Compute portfolio-level Value x Risk quadrants and revenue at risk from scored records.

        Args:
            request: Validated PrioritizationQuadrantsRequest.

        Returns:
            PrioritizationQuadrantsResponse with aggregated portfolio tiers and estimates.
        """
        self._ensure_loaded()
        assert self.segmenter is not None

        # Build DataFrame of customer records
        records_data = [
            {
                "account_id": r.account_id,
                "sub_current_mrr": r.sub_current_mrr,
                "model_score": r.model_score,
            }
            for r in request.records
        ]
        records_df = pd.DataFrame(records_data)

        # Perform quadrant analysis using segmenter's logic
        analysis = self.segmenter.analyze_value_risk_quadrants(
            features_df=records_df,
            model_scores=records_df["model_score"].values,
            mrr_high_threshold=request.mrr_high_threshold,
            mrr_mid_threshold=request.mrr_mid_threshold,
            risk_high_threshold=request.score_high_threshold,
            risk_mod_threshold=request.score_mid_threshold,
        )

        quadrants = [
            PrioritizationQuadrantDetail(
                quadrant_name=q["quadrant_name"],
                priority_level=q["priority_level"],
                account_count=q["account_count"],
                account_percentage=round(q["account_percentage"], 2),
                total_mrr=round(q["total_mrr"], 2),
                estimated_revenue_at_risk_mrr=round(q["estimated_revenue_at_risk_mrr"], 2),
                action_recommendation=q["action_recommendation"],
            )
            for q in analysis["quadrants"]
        ]

        return PrioritizationQuadrantsResponse(
            total_accounts=analysis["total_accounts"],
            total_mrr=round(analysis["total_mrr"], 2),
            total_estimated_revenue_at_risk_mrr=round(
                analysis["total_estimated_revenue_at_risk_mrr"], 2
            ),
            quadrants=quadrants,
            risk_metric_interpretation=analysis["risk_metric_interpretation"],
        )


# Global Service Singleton
_inference_service: ModelInferenceService | None = None


def get_inference_service() -> ModelInferenceService:
    """Retrieve or initialize the global ModelInferenceService instance."""
    global _inference_service
    if _inference_service is None:
        service = ModelInferenceService()
        # Default artifact locations
        service.load_artifacts(
            models_dir=Path("artifacts/models/churn_prediction_v1"),
            explainability_dir=Path("artifacts/explainability"),
            segmentation_dir=Path("artifacts/segmentation"),
        )
        _inference_service = service
    return _inference_service
