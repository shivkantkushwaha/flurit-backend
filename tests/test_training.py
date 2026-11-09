"""Unit tests for ML model training, chronological splitting, and artifact lifecycle."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES
from app.ml.training.pipeline import (
    create_model_pipeline,
)
from app.ml.training.splits import ChronologicalSplitter
from app.ml.training.trainer import ModelTrainer, ModelType


def _generate_synthetic_features(n_samples: int = 120) -> pd.DataFrame:
    """Helper to generate valid synthetic feature matrix for unit tests."""
    np.random.seed(42)

    # 4 distinct snapshot dates across time
    dates = [date(2023, 6, 1), date(2023, 12, 1), date(2024, 7, 1), date(2024, 10, 1)]
    assigned_dates = [dates[i % len(dates)] for i in range(n_samples)]

    data: dict[str, list] = {
        "account_id": [f"A{i}" for i in range(n_samples)],
        "snapshot_date": assigned_dates,
    }

    for f_def in FEATURE_DEFINITIONS:
        if f_def.dtype == "category":
            if f_def.name == "account_industry":
                data[f_def.name] = np.random.choice(
                    ["DevTools", "FinTech", "Health"], size=n_samples
                ).tolist()
            elif f_def.name == "account_country":
                data[f_def.name] = np.random.choice(["US", "UK", "DE"], size=n_samples).tolist()
            elif f_def.name == "account_referral_source":
                data[f_def.name] = np.random.choice(
                    ["organic", "partner", "paid"], size=n_samples
                ).tolist()
            elif f_def.name == "sub_billing_frequency":
                data[f_def.name] = np.random.choice(["monthly", "annual"], size=n_samples).tolist()
            else:
                data[f_def.name] = ["val"] * n_samples
        elif f_def.dtype == "float64":
            data[f_def.name] = np.random.uniform(0.0, 100.0, size=n_samples).tolist()
        else:
            data[f_def.name] = np.random.randint(0, 50, size=n_samples).tolist()

    data["target_churn_60d"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]).tolist()
    return pd.DataFrame(data)


class TestChronologicalSplitter(unittest.TestCase):
    """Tests for chronological train/val/test splitting invariants with target-window purging."""

    def setUp(self) -> None:
        self.df = _generate_synthetic_features(n_samples=120)
        self.splitter = ChronologicalSplitter(
            train_end_date=date(2023, 12, 31),
            val_start_date=date(2024, 7, 1),
            val_end_date=date(2024, 8, 1),
            test_start_date=date(2024, 10, 1),
        )

    def test_strict_temporal_isolation_no_date_overlap(self) -> None:
        """Verify train, val, and test splits share zero snapshot dates."""
        train_df, val_df, test_df = self.splitter.split(self.df)

        train_dates = set(train_df["snapshot_date"].unique())
        val_dates = set(val_df["snapshot_date"].unique())
        test_dates = set(test_df["snapshot_date"].unique())

        self.assertEqual(len(train_dates & val_dates), 0)
        self.assertEqual(len(val_dates & test_dates), 0)
        self.assertEqual(len(train_dates & test_dates), 0)

        # Max train date < Min val date < Min test date
        self.assertLess(max(train_dates), min(val_dates))
        self.assertLess(max(val_dates), min(test_dates))

    def test_row_count_conservation(self) -> None:
        """Verify sum of partitioned rows plus purged rows matches original DataFrame length."""
        train_df, val_df, test_df = self.splitter.split(self.df)
        discarded = self.splitter.get_discarded_cohorts(self.df)
        total = len(train_df) + len(val_df) + len(test_df) + discarded["total_purged_samples"]
        self.assertEqual(total, len(self.df))

    def test_target_window_purging_enforcement(self) -> None:
        """Verify splitter rejects partitions where min val date overlaps train 60d label window."""
        # Train ends 2024-05-01, but val starts 2024-06-01 (overlaps 60d window ending 2024-06-30)
        bad_splitter = ChronologicalSplitter(
            train_end_date=date(2024, 5, 1),
            val_start_date=date(2024, 6, 1),
            val_end_date=date(2024, 8, 1),
            test_start_date=date(2024, 10, 1),
            prediction_window_days=60,
        )
        test_df = pd.DataFrame(
            {
                "snapshot_date": [date(2024, 5, 1), date(2024, 6, 1), date(2024, 10, 1)],
                "account_id": ["A1", "A2", "A3"],
                "target_churn_60d": [0, 1, 0],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            bad_splitter.split(test_df)
        self.assertIn("Purge violation", str(ctx.exception))

    def test_purged_boundary_cohorts_identified(self) -> None:
        """Verify discarded cohorts are accurately identified and extracted."""
        # Dates with intermediate buffer cohorts: 2024-06-01 and 2024-09-01
        test_df = pd.DataFrame(
            {
                "snapshot_date": [
                    date(2024, 5, 1),
                    date(2024, 6, 1),  # discarded train-val buffer
                    date(2024, 7, 1),
                    date(2024, 8, 1),
                    date(2024, 9, 1),  # discarded val-test buffer
                    date(2024, 10, 1),
                ],
                "account_id": [f"A{i}" for i in range(6)],
                "target_churn_60d": [0, 1, 0, 1, 0, 1],
            }
        )
        splitter = ChronologicalSplitter(
            train_end_date=date(2024, 5, 1),
            val_start_date=date(2024, 7, 1),
            val_end_date=date(2024, 8, 1),
            test_start_date=date(2024, 10, 1),
            prediction_window_days=60,
        )
        discarded = splitter.get_discarded_cohorts(test_df)
        self.assertEqual(discarded["purged_train_val_dates"], ["2024-06-01"])
        self.assertEqual(discarded["purged_val_test_dates"], ["2024-09-01"])
        self.assertEqual(discarded["total_purged_samples"], 2)

    def test_invalid_boundary_dates_raise_error(self) -> None:
        """Verify splitter rejects train_end_date >= val_start_date."""
        with self.assertRaises(ValueError):
            ChronologicalSplitter(
                train_end_date=date(2024, 8, 1),
                val_start_date=date(2024, 5, 1),
            )


class TestPreprocessingLeakage(unittest.TestCase):
    """Tests verifying preprocessing isolation and handling of unseen categories."""

    def test_unseen_categories_in_validation_are_handled_gracefully(self) -> None:
        """Verify OneHotEncoder ignores new categories in val/test without raising errors."""
        train_df = _generate_synthetic_features(n_samples=40)
        val_df = _generate_synthetic_features(n_samples=20)

        # Introduce completely unseen category in val set
        val_df["account_industry"] = "QuantumComputing"

        pipe = create_model_pipeline("logistic_regression", random_state=42)
        pipe.fit(train_df[FEATURE_NAMES], train_df["target_churn_60d"].values)

        # Predicting on val_df must succeed without error
        probs = pipe.predict_proba(val_df[FEATURE_NAMES])
        self.assertEqual(probs.shape, (20, 2))


class TestDeterministicTraining(unittest.TestCase):
    """Tests verifying 100% deterministic training reproducibility."""

    def test_identical_random_seed_yields_identical_predictions(self) -> None:
        """Verify two pipelines trained with same random_state output identical probabilities."""
        df = _generate_synthetic_features(n_samples=80)
        X = df[FEATURE_NAMES]
        y = df["target_churn_60d"].values

        pipe1 = create_model_pipeline("random_forest", random_state=42)
        pipe1.fit(X, y)
        probs1 = pipe1.predict_proba(X)

        pipe2 = create_model_pipeline("random_forest", random_state=42)
        pipe2.fit(X, y)
        probs2 = pipe2.predict_proba(X)

        np.testing.assert_array_almost_equal(probs1, probs2, decimal=6)


class TestArtifactPersistenceAndLoading(unittest.TestCase):
    """Tests for artifact serialization, metadata, and loading."""

    def test_save_and_load_roundtrip(self) -> None:
        """Verify pipeline and metadata can be saved, loaded, and generate identical predictions."""
        df = _generate_synthetic_features(n_samples=120)

        trainer = ModelTrainer(random_state=42)
        # Use custom boundaries matching synthetic dates
        trainer.splitter = ChronologicalSplitter(
            train_end_date=date(2023, 12, 31),
            val_start_date=date(2024, 7, 1),
            val_end_date=date(2024, 8, 1),
            test_start_date=date(2024, 10, 1),
        )

        result = trainer.run_training_experiment(
            df, candidate_models=["logistic_regression", "random_forest"]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "test_model_v1"
            trainer.save_artifact(result, artifact_dir=artifact_dir, version="1.0.0")

            # Check files exist
            self.assertTrue((artifact_dir / "model.joblib").exists())
            self.assertTrue((artifact_dir / "metadata.json").exists())

            # Load artifact
            loaded_pipe, metadata = trainer.load_artifact(artifact_dir)

            self.assertEqual(metadata["model_name"], "flurit_churn_predictor")
            self.assertEqual(metadata["version"], "1.0.0")
            self.assertIn("validation_comparison", metadata)
            self.assertIn("test_evaluation", metadata)
            self.assertIn("discarded_boundary_cohorts", metadata)
            self.assertFalse(metadata["decision_threshold_tuned"])

            # Test predictions parity
            X_test = df[FEATURE_NAMES].head(10)
            orig_preds = result["champion_pipeline"].predict_proba(X_test)
            loaded_preds = loaded_pipe.predict_proba(X_test)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds, decimal=6)


class TestPredictionOutputValidity(unittest.TestCase):
    """Tests verifying output probability bounds and classification shape."""

    def test_predict_proba_bounds(self) -> None:
        """Verify output probabilities are strictly in [0.0, 1.0] and sum to 1.0."""
        df = _generate_synthetic_features(n_samples=60)
        X = df[FEATURE_NAMES]
        y = df["target_churn_60d"].values

        m_types: list[ModelType] = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "hist_gradient_boosting",
        ]
        for m_type in m_types:
            pipe = create_model_pipeline(m_type, random_state=42)
            pipe.fit(X, y)
            probs = pipe.predict_proba(X)

            self.assertEqual(probs.shape, (len(df), 2))
            self.assertTrue(np.all(probs >= 0.0))
            self.assertTrue(np.all(probs <= 1.0))
            np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(len(df)), decimal=5)

    def test_brier_baseline_and_skill_score_calculation(self) -> None:
        """Verify Brier baseline P*(1-P) and Brier skill score computation."""
        from app.ml.evaluation.metrics import evaluate_binary_classifier

        # 20% prevalence (2 positive, 8 negative)
        y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # Predict constant prevalence for all samples
        y_prob_dummy = [0.2] * 10
        metrics = evaluate_binary_classifier(y_true, y_prob_dummy, threshold=0.5)

        expected_baseline = 0.2 * 0.8  # 0.16
        self.assertAlmostEqual(metrics.brier_score_baseline, expected_baseline, places=4)
        self.assertAlmostEqual(metrics.brier_score, expected_baseline, places=4)
        self.assertAlmostEqual(metrics.brier_skill_score, 0.0, places=4)
        self.assertFalse(metrics.threshold_tuned)


if __name__ == "__main__":
    unittest.main()
