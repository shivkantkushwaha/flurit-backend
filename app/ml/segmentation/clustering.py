"""Unsupervised Customer Segmentation and Value x Risk Analysis.

Implements K-Means clustering across point-in-time behavioral and commercial features,
evaluates candidate cluster counts (K=2 to K=6) using silhouette analysis and inertia,
generates empirical cluster profiles, and maps customer value x model risk quadrants.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES

PROHIBITED_SEGMENTATION_FIELDS: tuple[str, ...] = (
    "target_churn_60d",
    "churn_flag",
    "is_churned",
    "churn_date",
    "reason_code",
    "feedback_text",
)


@dataclass(frozen=True)
class ClusterEvaluationResult:
    """Metrics and properties for a candidate cluster count K."""

    k: int
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    inertia: float
    cluster_sizes: list[int]
    cluster_percentages: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ClusterProfile:
    """Empirical profile of a customer segment based on observed feature centroids."""

    cluster_id: int
    cluster_name: str
    sample_count: int
    percentage: float
    key_feature_means: dict[str, float]
    dominant_categories: dict[str, str]
    empirical_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PrioritizationQuadrant:
    """Summary of a Customer Value x Model Score Risk quadrant."""

    quadrant_name: str
    priority_level: int
    account_count: int
    account_percentage: float
    total_mrr: float
    estimated_revenue_at_risk_mrr: float
    action_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert quadrant to serializable dictionary."""
        return asdict(self)


class CustomerSegmenter:
    """Unsupervised customer segmentation engine with Value x Risk quadrant mapping."""

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ) -> None:
        """Initialize CustomerSegmenter.

        Args:
            n_clusters: Chosen number of clusters (default: 3).
            random_state: Random state for deterministic K-Means initialization.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.categorical_features = [f.name for f in FEATURE_DEFINITIONS if f.dtype == "category"]
        self.numerical_features = [f.name for f in FEATURE_DEFINITIONS if f.dtype != "category"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
                (
                    "num",
                    StandardScaler(),
                    self.numerical_features,
                ),
            ]
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.is_fitted = False
        self.cluster_profiles: list[ClusterProfile] = []
        self.evaluation_results: list[ClusterEvaluationResult] = []

    def _verify_no_leakage(self, df: pd.DataFrame) -> None:
        """Verify that target labels and future outcome fields are completely absent."""
        for field in PROHIBITED_SEGMENTATION_FIELDS:
            if field in df.columns:
                raise ValueError(
                    f"Segmentation Leakage Error: Prohibited field '{field}' detected in clustering input. "
                    "Clustering must be strictly unsupervised and blind to churn outcomes."
                )

    def evaluate_k_range(
        self,
        features_df: pd.DataFrame,
        k_range: range = range(2, 7),
    ) -> list[ClusterEvaluationResult]:
        """Evaluate clustering performance across candidate cluster counts K=2 to K=6.

        Args:
            features_df: Development feature matrix (target labels excluded).
            k_range: Range of candidate cluster counts to evaluate.

        Returns:
            List of ClusterEvaluationResult objects containing metrics for each K.
        """
        self._verify_no_leakage(features_df)
        X = features_df[FEATURE_NAMES]
        X_trans = self.preprocessor.fit_transform(X)

        results: list[ClusterEvaluationResult] = []
        n_total = len(X)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_trans)

            sil = float(silhouette_score(X_trans, labels))
            db = float(davies_bouldin_score(X_trans, labels))
            ch = float(calinski_harabasz_score(X_trans, labels))
            inertia = float(km.inertia_)

            counts = [int(c) for c in np.bincount(labels, minlength=k)]
            pcts = [float(c / n_total * 100.0) for c in counts]

            res = ClusterEvaluationResult(
                k=k,
                silhouette_score=sil,
                davies_bouldin_score=db,
                calinski_harabasz_score=ch,
                inertia=inertia,
                cluster_sizes=counts,
                cluster_percentages=pcts,
            )
            results.append(res)

        self.evaluation_results = results
        return results

    def fit(self, features_df: pd.DataFrame) -> "CustomerSegmenter":
        """Fit segmentation preprocessor and K-Means exclusively on development features.

        Args:
            features_df: Development features (e.g. training partition).

        Returns:
            Self (fitted segmenter).
        """
        self._verify_no_leakage(features_df)
        X = features_df[FEATURE_NAMES]
        X_trans = self.preprocessor.fit_transform(X)
        self.kmeans.fit(X_trans)
        self.is_fitted = True

        # Generate empirical cluster profiles
        labels = self.kmeans.labels_
        df_with_clusters = features_df.copy()
        df_with_clusters["cluster_id"] = labels

        self.cluster_profiles = self._build_cluster_profiles(df_with_clusters)
        return self

    def _build_cluster_profiles(self, df_with_clusters: pd.DataFrame) -> list[ClusterProfile]:
        """Build empirical profiles from observed cluster centroids and distributions."""
        profiles: list[ClusterProfile] = []
        n_total = len(df_with_clusters)

        key_metrics = [
            "sub_current_mrr",
            "sub_current_arr",
            "sub_total_active_seats",
            "sub_tenure_days",
            "usage_count_30d",
            "usage_active_days_30d",
            "usage_duration_secs_30d",
            "usage_error_rate_30d",
            "tickets_count_90d",
            "tickets_avg_satisfaction_90d",
            "churn_prior_count",
        ]

        for cid in range(self.n_clusters):
            c_df = df_with_clusters[df_with_clusters["cluster_id"] == cid]
            c_size = len(c_df)
            c_pct = float(c_size / n_total * 100.0)

            # Feature means
            means = {col: float(c_df[col].mean()) for col in key_metrics if col in c_df.columns}

            # Dominant categorical values
            dom_cats = {}
            for col in self.categorical_features:
                if col in c_df.columns and not c_df[col].dropna().empty:
                    dom_cats[col] = str(c_df[col].mode().iloc[0])

            # Generate empirical descriptive name based on observed MRR and usage
            mrr_val = means.get("sub_current_mrr", 0.0)
            usage_val = means.get("usage_count_30d", 0.0)
            tenure_val = means.get("sub_tenure_days", 0.0)

            if mrr_val >= 8000:
                name = f"Cluster {cid}: High-Value Enterprise Accounts"
            elif usage_val >= 250:
                name = f"Cluster {cid}: Highly Active Engagement Accounts"
            else:
                name = f"Cluster {cid}: Standard Established Accounts"

            summary = (
                f"Cluster {cid} (N={c_size:,}, {c_pct:.1f}%): "
                f"Avg MRR=${mrr_val:,.2f}, Avg Seats={means.get('sub_total_active_seats', 0):.1f}, "
                f"Avg 30d Usage={usage_val:.1f} events ({means.get('usage_active_days_30d', 0):.1f} active days), "
                f"Commercial Tenure={tenure_val:.1f} days, Avg 90d Tickets={means.get('tickets_count_90d', 0):.1f}."
            )

            profiles.append(
                ClusterProfile(
                    cluster_id=cid,
                    cluster_name=name,
                    sample_count=c_size,
                    percentage=c_pct,
                    key_feature_means=means,
                    dominant_categories=dom_cats,
                    empirical_summary=summary,
                )
            )

        return profiles

    def predict_segments(
        self,
        features_df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict cluster assignments for new customer snapshots.

        Args:
            features_df: Feature DataFrame.

        Returns:
            1D array of assigned cluster integer IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("CustomerSegmenter must be fitted before predict_segments.")
        self._verify_no_leakage(features_df)
        X = features_df[FEATURE_NAMES]
        X_trans = self.preprocessor.transform(X)
        return np.asarray(self.kmeans.predict(X_trans), dtype=int)

    def analyze_value_risk_quadrants(
        self,
        features_df: pd.DataFrame,
        model_scores: np.ndarray | list[float],
        mrr_high_threshold: float = 8000.0,
        mrr_mid_threshold: float = 4000.0,
        risk_high_threshold: float = 0.50,
        risk_mod_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """Perform Customer Value x Model Score Risk analysis across snapshots.

        Args:
            features_df: Feature DataFrame with 'sub_current_mrr'.
            model_scores: Model churn prediction risk scores in [0.0, 1.0].
            mrr_high_threshold: Cutoff for High Value tier ($).
            mrr_mid_threshold: Cutoff for Mid Value tier ($).
            risk_high_threshold: Cutoff for High Risk score (default 0.50).
            risk_mod_threshold: Cutoff for Moderate Risk score (default 0.35).

        Returns:
            Dictionary containing quadrant summaries, revenue at risk, and accounts breakdown.
        """
        data = features_df.copy()
        scores = np.asarray(model_scores, dtype=float)
        if len(data) != len(scores):
            raise ValueError(f"Length mismatch: {len(data)} rows vs {len(scores)} scores.")

        data["model_score"] = scores
        # Revenue-weighted risk metric (explicitly labeled as an estimate)
        mrr = data["sub_current_mrr"].fillna(0.0).values
        data["estimated_revenue_at_risk_mrr"] = mrr * scores

        # Assign Value Tiers
        def get_value_tier(val: float) -> str:
            if val >= mrr_high_threshold:
                return "High Value"
            if val >= mrr_mid_threshold:
                return "Mid Value"
            return "Core Value"

        # Assign Risk Tiers
        def get_risk_tier(score: float) -> str:
            if score >= risk_high_threshold:
                return "High Risk"
            if score >= risk_mod_threshold:
                return "Moderate Risk"
            return "Low Risk"

        data["value_tier"] = [get_value_tier(v) for v in mrr]
        data["risk_tier"] = [get_risk_tier(s) for s in scores]

        # Prioritization quadrants
        def get_priority(row: pd.Series) -> tuple[str, int, str]:
            v = row["value_tier"]
            r = row["risk_tier"]
            if v == "High Value" and r == "High Risk":
                return (
                    "Priority 1: High Value - High Risk",
                    1,
                    "Immediate executive outreach, contract renegotiation, and dedicated technical review.",
                )
            if v == "Mid Value" and r == "High Risk":
                return (
                    "Priority 2: Mid Value - High Risk",
                    2,
                    "Proactive customer success contact, satisfaction review, and concession modeling.",
                )
            if v == "High Value" and r == "Moderate Risk":
                return (
                    "Priority 3: High Value - Moderate Risk",
                    3,
                    "Account health monitoring and executive sponsor check-in.",
                )
            if v == "Core Value" and r == "High Risk":
                return (
                    "Priority 4: Core Value - High Risk",
                    4,
                    "Automated retention workflows, product education surveys, and self-service re-engagement.",
                )
            return (
                "Priority 5: Standard Monitoring",
                5,
                "Ongoing lifecycle engagement, standard health check-ins, and roadmap updates.",
            )

        priorities = [get_priority(row) for _, row in data.iterrows()]
        data["priority_quadrant"] = [p[0] for p in priorities]
        data["priority_level"] = [p[1] for p in priorities]
        data["action_recommendation"] = [p[2] for p in priorities]

        # Aggregate quadrant statistics
        total_mrr = float(data["sub_current_mrr"].sum())
        total_rev_at_risk = float(data["estimated_revenue_at_risk_mrr"].sum())
        total_accounts = len(data)

        quadrants_summary: list[PrioritizationQuadrant] = []
        for q_name in sorted(
            data["priority_quadrant"].unique(),
            key=lambda x: data[data["priority_quadrant"] == x]["priority_level"].iloc[0],
        ):
            q_df = data[data["priority_quadrant"] == q_name]
            q_mrr = float(q_df["sub_current_mrr"].sum())
            q_at_risk = float(q_df["estimated_revenue_at_risk_mrr"].sum())
            q_count = len(q_df)
            q_pct = float(q_count / total_accounts * 100.0)
            rec = q_df["action_recommendation"].iloc[0]
            plevel = int(q_df["priority_level"].iloc[0])

            quadrants_summary.append(
                PrioritizationQuadrant(
                    quadrant_name=q_name,
                    priority_level=plevel,
                    account_count=q_count,
                    account_percentage=q_pct,
                    total_mrr=q_mrr,
                    estimated_revenue_at_risk_mrr=q_at_risk,
                    action_recommendation=rec,
                )
            )

        return {
            "total_accounts": total_accounts,
            "total_mrr": total_mrr,
            "total_estimated_revenue_at_risk_mrr": total_rev_at_risk,
            "risk_metric_interpretation": (
                "Estimated Revenue at Risk is calculated as MRR * Model Churn Score. "
                "Because Phase 6.1 demonstrated negative Brier skill scores for raw model probabilities, "
                "this metric serves strictly as an operational prioritization score, NOT an actuarially "
                "calibrated financial guarantee or literal probability of monetary loss."
            ),
            "quadrants": [q.to_dict() for q in quadrants_summary],
            "value_tiers_breakdown": data.groupby("value_tier")["sub_current_mrr"]
            .agg(["count", "sum"])
            .to_dict(),
            "risk_tiers_breakdown": data.groupby("risk_tier")["model_score"]
            .agg(["count", "mean"])
            .to_dict(),
        }

    def save_artifact(self, output_dir: Path) -> Path:
        """Persist fitted segmentation models, cluster evaluations, and profiles to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist model pipeline
        model_file = output_dir / "segmentation_model.joblib"
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "kmeans": self.kmeans,
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
            },
            model_file,
        )

        # 2. Persist cluster evaluations (K=2..6)
        eval_file = output_dir / "cluster_evaluation_metrics.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump([res.to_dict() for res in self.evaluation_results], f, indent=2)

        # 3. Persist cluster profiles
        profiles_file = output_dir / "cluster_profiles.json"
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in self.cluster_profiles], f, indent=2)

        return output_dir

    @classmethod
    def load_artifact(cls, artifact_dir: Path) -> "CustomerSegmenter":
        """Load fitted segmentation models and metadata from disk."""
        model_file = artifact_dir / "segmentation_model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing segmentation artifact: {model_file}")

        bundle = joblib.load(model_file)
        instance = cls(
            n_clusters=bundle["n_clusters"],
            random_state=bundle["random_state"],
        )
        instance.preprocessor = bundle["preprocessor"]
        instance.kmeans = bundle["kmeans"]
        instance.is_fitted = True

        profiles_file = artifact_dir / "cluster_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, encoding="utf-8") as f:
                data = json.load(f)
                instance.cluster_profiles = [ClusterProfile(**item) for item in data]

        eval_file = artifact_dir / "cluster_evaluation_metrics.json"
        if eval_file.exists():
            with open(eval_file, encoding="utf-8") as f:
                data = json.load(f)
                instance.evaluation_results = [ClusterEvaluationResult(**item) for item in data]

        return instance
