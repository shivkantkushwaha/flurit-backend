"""Model Explainability Engine using SHAP LinearExplainer.

Computes exact additive Shapley feature attributions for Flurit's Logistic Regression
pipeline, mapping one-hot and scaled features back to canonical point-in-time features,
generating natural-language risk summaries, and strictly distinguishing model associations
from causal claims.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES

ASSOCIATION_DISCLAIMER: str = (
    "Statistical Association Note: Feature attributions reflect correlational relationships "
    "learned by the Logistic Regression model from historical observations. They quantify how "
    "much each feature shifts the predicted log-odds relative to the background baseline, and "
    "do NOT imply a direct causal relationship or guaranteed real-world outcome."
)


@dataclass(frozen=True)
class FactorAttribution:
    """Attribution breakdown for a single feature factor."""

    feature_name: str
    feature_value: Any
    attribution: float
    direction: str  # 'increases_risk' or 'decreases_risk'
    feature_description: str


@dataclass(frozen=True)
class PredictionExplanation:
    """Full explainability payload for an individual customer snapshot prediction."""

    account_id: str
    snapshot_date: str
    model_score: float  # Uncalibrated churn risk score in [0.0, 1.0]
    decision_threshold: float
    predicted_risk_label: int  # 1 if score >= threshold, else 0
    base_value_log_odds: float  # Expected log-odds across background dataset
    predicted_log_odds: float  # Total model decision function value
    top_risk_increasing_factors: list[dict[str, Any]]
    top_risk_decreasing_factors: list[dict[str, Any]]
    feature_attributions: dict[str, float]  # All 45 canonical feature attributions
    plain_language_summary: str
    association_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GlobalFeatureImportance:
    """Global feature attribution summary across the dataset."""

    feature_name: str
    rank: int
    mean_abs_attribution: float
    mean_directional_attribution: float
    feature_group: str
    feature_description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert global importance to serializable dictionary."""
        return asdict(self)


class ModelExplainer:
    """Explains predictions of Flurit's trained Logistic Regression pipeline using SHAP."""

    def __init__(
        self,
        pipeline: Pipeline,
        background_features_df: pd.DataFrame,
        decision_threshold: float = 0.5,
    ) -> None:
        """Initialize the ModelExplainer with a fitted pipeline and background data.

        Args:
            pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'classifier' steps.
            background_features_df: Feature DataFrame (e.g. training set) used as reference background.
            decision_threshold: Decision threshold used for binary risk labeling.
        """
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier: LogisticRegression = pipeline.named_steps["classifier"]
        self.decision_threshold = decision_threshold

        if not isinstance(self.classifier, LogisticRegression):
            raise TypeError(
                f"ModelExplainer requires LogisticRegression classifier, got {type(self.classifier).__name__}."
            )

        # Build feature descriptions lookup
        self._feature_meta = {f.name: f for f in FEATURE_DEFINITIONS}

        # Transform background dataset
        X_bg = background_features_df[FEATURE_NAMES]
        self.X_bg_trans = self.preprocessor.transform(X_bg)

        # Initialize SHAP LinearExplainer with explicit independent masker
        masker = shap.maskers.Independent(self.X_bg_trans, max_samples=len(self.X_bg_trans))
        self.explainer = shap.LinearExplainer(self.classifier, masker)
        self.transformed_names = list(self.preprocessor.get_feature_names_out())

        # Map transformed feature indices to original 45 canonical features
        self._transformed_to_canonical_map = self._build_feature_mapping()

    def _build_feature_mapping(self) -> list[str]:
        """Maps each column in the transformed matrix to its canonical feature name."""
        mapping: list[str] = []
        for feat_out in self.transformed_names:
            # Strip transformer prefixes 'cat__' or 'num__'
            clean = feat_out
            if clean.startswith("cat__"):
                clean = clean[len("cat__") :]
            elif clean.startswith("num__"):
                clean = clean[len("num__") :]

            # Find matching canonical feature name
            matched = False
            for canonical in FEATURE_NAMES:
                if clean == canonical or clean.startswith(canonical + "_"):
                    mapping.append(canonical)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Could not map transformed feature '{feat_out}' to canonical feature."
                )
        return mapping

    def _aggregate_shap_to_canonical(
        self,
        sample_shap_values: np.ndarray,
    ) -> dict[str, float]:
        """Aggregate transformed feature Shapley values to the 45 canonical features."""
        aggregated: dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)
        for val, canonical_name in zip(
            sample_shap_values, self._transformed_to_canonical_map, strict=True
        ):
            aggregated[canonical_name] += float(val)
        return aggregated

    def explain_instance(
        self,
        account_id: str,
        snapshot_date: str,
        feature_row: pd.Series | dict[str, Any],
        top_k: int = 5,
    ) -> PredictionExplanation:
        """Generate local SHAP explanation for a single customer snapshot.

        Args:
            account_id: Account identifier.
            snapshot_date: Snapshot observation date string.
            feature_row: Single row of 45 features (Series or dict).
            top_k: Number of top factors to extract for risk increase/decrease.

        Returns:
            PredictionExplanation object containing local factors, summary, and disclaimer.
        """
        row = dict(feature_row)
        row["account_id"] = account_id
        row["snapshot_date"] = snapshot_date
        return self.explain_batch(pd.DataFrame([row]), top_k=top_k)[0]

    def _format_explanation(
        self, account_id: str, snapshot_date: str, raw_vals: dict[str, Any],
        prob: float, decision_val: float, raw_shap: np.ndarray, base_val: float, top_k: int,
    ) -> PredictionExplanation:
        canonical_attributions = self._aggregate_shap_to_canonical(raw_shap)

        increasing: list[FactorAttribution] = []
        decreasing: list[FactorAttribution] = []

        for f_name, attr in canonical_attributions.items():
            f_val = raw_vals.get(f_name)
            desc = (
                self._feature_meta[f_name].description if f_name in self._feature_meta else f_name
            )
            if attr > 1e-5:
                increasing.append(
                    FactorAttribution(
                        feature_name=f_name,
                        feature_value=f_val,
                        attribution=attr,
                        direction="increases_risk",
                        feature_description=desc,
                    )
                )
            elif attr < -1e-5:
                decreasing.append(
                    FactorAttribution(
                        feature_name=f_name,
                        feature_value=f_val,
                        attribution=attr,
                        direction="decreases_risk",
                        feature_description=desc,
                    )
                )

        # Sort factors
        increasing.sort(key=lambda x: x.attribution, reverse=True)
        decreasing.sort(key=lambda x: x.attribution)  # Most negative first

        top_increasing = [asdict(x) for x in increasing[:top_k]]
        top_decreasing = [asdict(x) for x in decreasing[:top_k]]

        summary = self._generate_plain_language_summary(
            model_score=prob,
            top_increasing=top_increasing,
            top_decreasing=top_decreasing,
        )

        return PredictionExplanation(
            account_id=str(account_id),
            snapshot_date=str(snapshot_date),
            model_score=prob,
            decision_threshold=self.decision_threshold,
            predicted_risk_label=int(prob >= self.decision_threshold),
            base_value_log_odds=base_val,
            predicted_log_odds=decision_val,
            top_risk_increasing_factors=top_increasing,
            top_risk_decreasing_factors=top_decreasing,
            feature_attributions=canonical_attributions,
            plain_language_summary=summary,
            association_disclaimer=ASSOCIATION_DISCLAIMER,
        )

    def explain_batch(
        self,
        features_df: pd.DataFrame,
        top_k: int = 5,
    ) -> list[PredictionExplanation]:
        """Explain predictions for a batch of customer snapshots."""
        if features_df.empty:
            return []
        transformed = self.preprocessor.transform(features_df[FEATURE_NAMES])
        probabilities = self.classifier.predict_proba(transformed)[:, 1]
        decisions = self.classifier.decision_function(transformed)
        attributions = self.explainer(transformed)
        return [
            self._format_explanation(
                str(row.get("account_id", "unknown")), str(row.get("snapshot_date", "unknown")),
                row, float(probabilities[i]), float(decisions[i]), attributions.values[i],
                float(attributions.base_values[i]), top_k,
            )
            for i, row in enumerate(features_df.to_dict("records"))
        ]

    def compute_global_importance(
        self,
        features_df: pd.DataFrame,
    ) -> list[GlobalFeatureImportance]:
        """Compute global feature importance by averaging absolute SHAP values across a cohort."""
        X_df = features_df[FEATURE_NAMES]
        X_trans = self.preprocessor.transform(X_df)
        shap_obj = self.explainer(X_trans)
        raw_shaps = shap_obj.values  # Shape: (N, 61)

        # Aggregate for each sample
        N = len(features_df)
        canonical_matrix = np.zeros((N, len(FEATURE_NAMES)))

        for i in range(N):
            agg_dict = self._aggregate_shap_to_canonical(raw_shaps[i])
            for j, f in enumerate(FEATURE_NAMES):
                canonical_matrix[i, j] = agg_dict[f]

        mean_abs = np.mean(np.abs(canonical_matrix), axis=0)
        mean_dir = np.mean(canonical_matrix, axis=0)

        # Rank features by mean absolute attribution
        sorted_indices = np.argsort(mean_abs)[::-1]

        results: list[GlobalFeatureImportance] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            f_name = FEATURE_NAMES[idx]
            meta = self._feature_meta.get(f_name)
            group_name = meta.group.value if meta else "unknown"
            desc = meta.description if meta else f_name

            results.append(
                GlobalFeatureImportance(
                    feature_name=f_name,
                    rank=rank,
                    mean_abs_attribution=float(mean_abs[idx]),
                    mean_directional_attribution=float(mean_dir[idx]),
                    feature_group=group_name,
                    feature_description=desc,
                )
            )

        return results

    @staticmethod
    def _generate_plain_language_summary(
        model_score: float,
        top_increasing: list[dict[str, Any]],
        top_decreasing: list[dict[str, Any]],
    ) -> str:
        """Synthesize a human-interpretable risk factor summary."""
        risk_str = f"Estimated model churn score is {model_score:.2f}."

        inc_parts = []
        for factor in top_increasing[:3]:
            f_name = factor["feature_name"].replace("_", " ")
            f_val = factor["feature_value"]
            inc_parts.append(f"{f_name} ({f_val})")

        dec_parts = []
        for factor in top_decreasing[:2]:
            f_name = factor["feature_name"].replace("_", " ")
            f_val = factor["feature_value"]
            dec_parts.append(f"{f_name} ({f_val})")

        inc_text = ""
        if inc_parts:
            inc_text = f" Main factors associated with higher score: {', '.join(inc_parts)}."

        dec_text = ""
        if dec_parts:
            dec_text = f" Mitigating factors associated with lower score: {', '.join(dec_parts)}."

        return f"{risk_str}{inc_text}{dec_text}"

    def save_artifact(self, output_dir: Path) -> Path:
        """Persist fitted explainer bundle to disk.

        Args:
            output_dir: Directory where explainer.joblib will be saved.

        Returns:
            Path to the saved artifact file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "explainer.joblib"
        joblib.dump(
            {
                "explainer": self.explainer,
                "transformed_names": self.transformed_names,
                "transformed_to_canonical_map": self._transformed_to_canonical_map,
                "decision_threshold": self.decision_threshold,
            },
            artifact_path,
        )
        return artifact_path

    @classmethod
    def load_artifact(
        cls,
        artifact_dir: Path,
        pipeline: Pipeline,
    ) -> "ModelExplainer":
        """Load persisted explainer bundle from disk.

        Args:
            artifact_dir: Directory containing explainer.joblib.
            pipeline: Loaded model Pipeline matching the explainer.

        Returns:
            Reconstituted ModelExplainer instance ready for inference.
        """
        artifact_path = artifact_dir / "explainer.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing explainer artifact: {artifact_path}")

        bundle = joblib.load(artifact_path)
        instance = cls.__new__(cls)
        instance.pipeline = pipeline
        instance.preprocessor = pipeline.named_steps["preprocessor"]
        instance.classifier = pipeline.named_steps["classifier"]
        instance.decision_threshold = float(bundle.get("decision_threshold", 0.5))
        instance._feature_meta = {f.name: f for f in FEATURE_DEFINITIONS}
        instance.explainer = bundle["explainer"]
        instance.transformed_names = bundle["transformed_names"]
        instance._transformed_to_canonical_map = bundle["transformed_to_canonical_map"]
        return instance
