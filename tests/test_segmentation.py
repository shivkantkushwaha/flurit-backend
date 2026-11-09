"""Unit tests for CustomerSegmenter and Value x Risk clustering analysis."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES
from app.ml.segmentation.clustering import CustomerSegmenter


def _generate_synthetic_segmentation_features(n_samples: int = 60) -> pd.DataFrame:
    """Helper to generate valid synthetic feature matrix for segmentation unit tests."""
    np.random.seed(42)
    data: dict[str, list] = {
        "account_id": [f"A{i}" for i in range(n_samples)],
        "snapshot_date": ["2024-05-01"] * n_samples,
    }

    for f_def in FEATURE_DEFINITIONS:
        if f_def.dtype == "category":
            if f_def.name == "account_industry":
                data[f_def.name] = np.random.choice(
                    ["FinTech", "DevTools", "E-commerce"], size=n_samples
                ).tolist()
            elif f_def.name == "account_country":
                data[f_def.name] = np.random.choice(["US", "UK", "DE"], size=n_samples).tolist()
            elif f_def.name == "account_referral_source":
                data[f_def.name] = np.random.choice(
                    ["organic", "paid_ad", "referral"], size=n_samples
                ).tolist()
            elif f_def.name == "sub_billing_frequency":
                data[f_def.name] = np.random.choice(["monthly", "annual"], size=n_samples).tolist()
            else:
                data[f_def.name] = ["standard"] * n_samples
        else:
            if f_def.name == "sub_current_mrr":
                data[f_def.name] = np.random.uniform(500.0, 15000.0, size=n_samples).tolist()
            else:
                data[f_def.name] = np.random.randint(5, 500, size=n_samples).tolist()

    return pd.DataFrame(data)


class TestCustomerSegmenter(unittest.TestCase):
    """Unit test suite for K-Means CustomerSegmenter."""

    def setUp(self) -> None:
        self.df = _generate_synthetic_segmentation_features(n_samples=80)
        self.segmenter = CustomerSegmenter(n_clusters=3, random_state=42)

    def test_prohibited_target_labels_rejection(self) -> None:
        """Verify segmenter strictly rejects DataFrames containing churn labels or future target fields."""
        leaky_df = self.df.copy()
        leaky_df["target_churn_60d"] = 1

        with self.assertRaises(ValueError) as ctx:
            self.segmenter.fit(leaky_df)
        self.assertIn("target_churn_60d", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            self.segmenter.evaluate_k_range(leaky_df)
        self.assertIn("target_churn_60d", str(ctx2.exception))

    def test_k_range_evaluation(self) -> None:
        """Verify evaluation runs across K=2 to K=6 and produces valid silhouette and inertia values."""
        results = self.segmenter.evaluate_k_range(self.df, k_range=range(2, 7))

        self.assertEqual(len(results), 5)
        for i, res in enumerate(results, start=2):
            self.assertEqual(res.k, i)
            self.assertGreater(res.inertia, 0.0)
            self.assertTrue(-1.0 <= res.silhouette_score <= 1.0)
            self.assertEqual(sum(res.cluster_sizes), len(self.df))

    def test_deterministic_clustering_and_profiling(self) -> None:
        """Verify two fits with identical random_state produce identical cluster assignments."""
        seg1 = CustomerSegmenter(n_clusters=3, random_state=42).fit(self.df)
        seg2 = CustomerSegmenter(n_clusters=3, random_state=42).fit(self.df)

        preds1 = seg1.predict_segments(self.df)
        preds2 = seg2.predict_segments(self.df)
        np.testing.assert_array_equal(preds1, preds2)

        # Profiles must be generated for all 3 clusters
        self.assertEqual(len(seg1.cluster_profiles), 3)
        for prof in seg1.cluster_profiles:
            self.assertIn("Cluster", prof.cluster_name)
            self.assertGreater(prof.sample_count, 0)
            self.assertIn("sub_current_mrr", prof.key_feature_means)

    def test_feature_order_consistency(self) -> None:
        """Verify clustering is invariant to permutation of feature column order."""
        seg = CustomerSegmenter(n_clusters=3, random_state=42).fit(self.df)
        orig_preds = seg.predict_segments(self.df)

        # Shuffle feature columns
        shuffled_cols = list(FEATURE_NAMES)
        np.random.seed(99)
        np.random.shuffle(shuffled_cols)
        shuffled_df = self.df[shuffled_cols].copy()

        shuffled_preds = seg.predict_segments(shuffled_df)
        np.testing.assert_array_equal(orig_preds, shuffled_preds)

    def test_artifact_save_and_load_roundtrip(self) -> None:
        """Verify segmenter can be persisted, loaded, and produce identical predictions."""
        self.segmenter.evaluate_k_range(self.df, k_range=range(2, 5))
        self.segmenter.fit(self.df)
        orig_preds = self.segmenter.predict_segments(self.df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "segmentation_v1"
            self.segmenter.save_artifact(out_dir)

            self.assertTrue((out_dir / "segmentation_model.joblib").exists())
            self.assertTrue((out_dir / "cluster_profiles.json").exists())
            self.assertTrue((out_dir / "cluster_evaluation_metrics.json").exists())

            loaded = CustomerSegmenter.load_artifact(out_dir)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(len(loaded.cluster_profiles), 3)

            loaded_preds = loaded.predict_segments(self.df)
            np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_value_risk_quadrant_analysis(self) -> None:
        """Verify Customer Value x Risk analysis correctly computes revenue at risk and priority levels."""
        # Synthetic model scores between 0.1 and 0.9
        scores = np.linspace(0.1, 0.9, len(self.df))
        analysis = self.segmenter.analyze_value_risk_quadrants(
            features_df=self.df,
            model_scores=scores,
        )

        self.assertEqual(analysis["total_accounts"], len(self.df))
        self.assertGreater(analysis["total_mrr"], 0.0)
        self.assertGreater(analysis["total_estimated_revenue_at_risk_mrr"], 0.0)
        self.assertIn("risk_metric_interpretation", analysis)
        self.assertIsInstance(analysis["quadrants"], list)

        # Revenue at risk must be less than or equal to total MRR (since scores <= 1.0)
        self.assertLessEqual(analysis["total_estimated_revenue_at_risk_mrr"], analysis["total_mrr"])


if __name__ == "__main__":
    unittest.main()
