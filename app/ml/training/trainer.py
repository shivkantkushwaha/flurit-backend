"""Model Training, Validation Selection, and Artifact Persistence Orchestrator.

Manages training across candidate models, evaluates on validation split,
selects champion model by validation PR-AUC, assesses on held-out test split once,
and serializes reproducible model artifacts and metadata.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from app.ml.evaluation.metrics import ModelMetrics, evaluate_binary_classifier
from app.ml.features.schema import FEATURE_NAMES
from app.ml.training.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_model_pipeline,
)
from app.ml.training.splits import ChronologicalSplitter, PartitionSummary

ModelType = Literal[
    "logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting"
]
ALL_CANDIDATE_MODELS: list[ModelType] = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
]


class ModelTrainer:
    """Trains and compares churn classifiers with strict chronological discipline."""

    def __init__(
        self,
        random_state: int = 42,
        decision_threshold: float = 0.5,
    ) -> None:
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.splitter = ChronologicalSplitter()

    def run_training_experiment(
        self,
        features_df: pd.DataFrame,
        candidate_models: list[ModelType] | None = None,
        target_col: str = "target_churn_60d",
    ) -> dict[str, Any]:
        """Execute full training, validation comparison, and held-out test evaluation.

        Args:
            features_df: Full feature matrix with 'snapshot_date', features, and target.
            candidate_models: List of model architectures to evaluate (default: all 4).
            target_col: Target column name.

        Returns:
            Dictionary containing trained pipelines, validation metrics, selected champion,
            held-out test metrics, and partition summaries.
        """
        models_to_train = candidate_models or ALL_CANDIDATE_MODELS

        # 1. Chronological splitting
        train_df, val_df, test_df = self.splitter.split(features_df)
        partitions = self.splitter.summarize_partitions(
            train_df, val_df, test_df, target_col=target_col
        )

        X_train = train_df[FEATURE_NAMES]
        y_train = train_df[target_col].astype(int).values

        X_val = val_df[FEATURE_NAMES]
        y_val = val_df[target_col].astype(int).values

        X_test = test_df[FEATURE_NAMES]
        y_test = test_df[target_col].astype(int).values

        # 2. Train candidates and evaluate on Validation set
        fitted_pipelines: dict[str, Pipeline] = {}
        val_metrics: dict[str, ModelMetrics] = {}
        train_metrics: dict[str, ModelMetrics] = {}

        for m_type in models_to_train:
            pipe = create_model_pipeline(model_type=m_type, random_state=self.random_state)
            pipe.fit(X_train, y_train)
            fitted_pipelines[m_type] = pipe

            # Training metrics (to detect severe overfitting)
            y_train_prob = pipe.predict_proba(X_train)[:, 1]
            train_metrics[m_type] = evaluate_binary_classifier(
                y_train, y_train_prob, threshold=self.decision_threshold
            )

            # Validation metrics
            y_val_prob = pipe.predict_proba(X_val)[:, 1]
            val_metrics[m_type] = evaluate_binary_classifier(
                y_val, y_val_prob, threshold=self.decision_threshold
            )

        # 3. Model selection based exclusively on Validation PR-AUC (Average Precision)
        # Primary: PR-AUC (Average Precision)
        # Tie-breaker: ROC-AUC
        selected_model_type = max(
            val_metrics.keys(),
            key=lambda m: (val_metrics[m].pr_auc, val_metrics[m].roc_auc),
        )
        champion_pipeline = fitted_pipelines[selected_model_type]

        # 4. Evaluate champion model ONCE on held-out Test set
        y_test_prob = champion_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_binary_classifier(
            y_test, y_test_prob, threshold=self.decision_threshold
        )

        discarded_cohorts = self.splitter.get_discarded_cohorts(features_df)

        return {
            "partitions": partitions,
            "discarded_cohorts": discarded_cohorts,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "selected_model_type": selected_model_type,
            "champion_pipeline": champion_pipeline,
            "test_metrics": test_metrics,
            "fitted_pipelines": fitted_pipelines,
            "X_train_shape": X_train.shape,
            "X_val_shape": X_val.shape,
            "X_test_shape": X_test.shape,
        }

    @staticmethod
    def save_artifact(
        experiment_result: dict[str, Any],
        artifact_dir: Path,
        version: str = "1.0.0",
        overwrite: bool = False,
    ) -> Path:
        """Persist champion model pipeline and comprehensive metadata to disk.

        Args:
            experiment_result: Result dict from run_training_experiment.
            artifact_dir: Target directory path.
            version: Model semantic version string.
            overwrite: If False, raises FileExistsError if directory already exists.

        Returns:
            Path to saved artifact directory.
        """
        if artifact_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Artifact directory '{artifact_dir}' already exists. Pass overwrite=True to replace."
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.joblib"
        metadata_path = artifact_dir / "metadata.json"

        # Serialize fitted pipeline
        champion_pipe: Pipeline = experiment_result["champion_pipeline"]
        joblib.dump(champion_pipe, model_path)

        # Prepare metadata
        partitions: dict[str, PartitionSummary] = experiment_result["partitions"]
        val_metrics: dict[str, ModelMetrics] = experiment_result["validation_metrics"]
        test_metrics: ModelMetrics = experiment_result["test_metrics"]
        selected_type = experiment_result["selected_model_type"]

        classifier_step = champion_pipe.named_steps["classifier"]
        hyperparameters = classifier_step.get_params()

        # Sanitize hyperparameters for JSON serialization
        sanitized_params = {}
        for k, v in hyperparameters.items():
            if isinstance(v, (int, float, str, bool, list, type(None))):
                sanitized_params[k] = v
            else:
                sanitized_params[k] = str(v)

        discarded_info = experiment_result.get("discarded_cohorts", {})

        metadata = {
            "model_name": "flurit_churn_predictor",
            "version": version,
            "selected_model_type": selected_type,
            "trained_at": datetime.now(UTC).isoformat(),
            "purged_evaluation_audit": "Phase 6.1 Purged Temporal Partitions",
            "prior_evaluation_disclosure": (
                "An unpurged test partition (2024-09-01 to 2024-11-01) was previously inspected "
                "in Phase 6 prior to implementing this target-window purging audit."
            ),
            "decision_threshold": experiment_result["test_metrics"].decision_threshold,
            "decision_threshold_tuned": False,
            "threshold_notes": (
                "Decision threshold 0.50 is the un-tuned default classification cutoff. "
                "No post-hoc threshold optimization was performed."
            ),
            "split_dates": {
                "train_start": str(partitions["train"].start_date),
                "train_end": str(partitions["train"].end_date),
                "val_start": str(partitions["val"].start_date),
                "val_end": str(partitions["val"].end_date),
                "test_start": str(partitions["test"].start_date),
                "test_end": str(partitions["test"].end_date),
            },
            "discarded_boundary_cohorts": discarded_info,
            "partition_sizes": {name: p.to_dict() for name, p in partitions.items()},
            "calibration_interpretation": (
                "Brier scores must be evaluated against the constant-prevalence baseline P*(1-P). "
                "Due to class_weight='balanced' upward probability shift, raw models exhibit "
                "negative Brier skill scores. Calibration was not fitted post-hoc to prevent test contamination."
            ),
            "feature_names": FEATURE_NAMES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "hyperparameters": sanitized_params,
            "validation_comparison": {m: val_metrics[m].to_dict() for m in val_metrics},
            "test_evaluation": test_metrics.to_dict(),
            "environment": {
                "python_version": sys.version,
                "sklearn_version": sklearn.__version__,
                "joblib_version": joblib.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return artifact_dir

    @staticmethod
    def load_artifact(artifact_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
        """Load persisted model pipeline and metadata from disk.

        Args:
            artifact_dir: Path to artifact directory.

        Returns:
            Tuple of (fitted_pipeline, metadata_dict).
        """
        model_path = artifact_dir / "model.joblib"
        metadata_path = artifact_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")

        pipeline: Pipeline = joblib.load(model_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        return pipeline, metadata
