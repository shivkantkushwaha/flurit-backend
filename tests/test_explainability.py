"""Unit tests for ModelExplainer and SHAP LinearExplainer integration."""

import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.ml.explainability.explainer import ASSOCIATION_DISCLAIMER, ModelExplainer
from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES
from app.ml.training.pipeline import create_model_pipeline


def _generate_synthetic_features(n_samples: int = 40) -> pd.DataFrame:
    """Helper to generate valid synthetic feature matrix for explainability unit tests."""
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
            data[f_def.name] = np.random.randint(10, 100, size=n_samples).tolist()

    data["target_churn_60d"] = np.random.choice([0, 1], size=n_samples).tolist()
    return pd.DataFrame(data)


class TestModelExplainer(unittest.TestCase):
    """Unit test suite for SHAP ModelExplainer mathematical validity and outputs."""

    def setUp(self) -> None:
        self.df = _generate_synthetic_features(n_samples=50)
        self.pipeline = create_model_pipeline("logistic_regression", random_state=42)
        X = self.df[FEATURE_NAMES]
        y = self.df["target_churn_60d"].values
        self.pipeline.fit(X, y)

        self.explainer = ModelExplainer(
            pipeline=self.pipeline,
            background_features_df=self.df,
            decision_threshold=0.5,
        )

    def test_additive_shapley_efficiency_identity(self) -> None:
        """Verify sum of 45 canonical SHAP attributions + base value equals model decision function (log-odds)."""
        sample_row = self.df.iloc[0]
        explanation = self.explainer.explain_instance(
            account_id="A0",
            snapshot_date="2024-05-01",
            feature_row=sample_row,
        )

        # Sum of canonical attributions
        sum_canonical = sum(explanation.feature_attributions.values())
        total_attributed_log_odds = sum_canonical + explanation.base_value_log_odds

        # Compare directly with model decision function
        self.assertAlmostEqual(
            total_attributed_log_odds,
            explanation.predicted_log_odds,
            places=5,
            msg="Additive efficiency violated: sum of SHAP values + base value must equal decision function.",
        )

    def test_deterministic_explanations(self) -> None:
        """Verify identical inputs produce strictly identical attributions."""
        sample_row = self.df.iloc[5]
        exp1 = self.explainer.explain_instance("A5", "2024-05-01", sample_row)
        exp2 = self.explainer.explain_instance("A5", "2024-05-01", sample_row)

        self.assertEqual(exp1.model_score, exp2.model_score)
        self.assertEqual(exp1.predicted_log_odds, exp2.predicted_log_odds)
        for f in FEATURE_NAMES:
            self.assertAlmostEqual(
                exp1.feature_attributions[f],
                exp2.feature_attributions[f],
                places=7,
            )

    def test_local_explanation_structure_and_disclaimer(self) -> None:
        """Verify local explanation contains risk factors, plain summary, and association disclaimer."""
        sample_row = self.df.iloc[2]
        exp = self.explainer.explain_instance("A2", "2024-05-01", sample_row)

        self.assertIn("Estimated model churn score is", exp.plain_language_summary)
        self.assertEqual(exp.association_disclaimer, ASSOCIATION_DISCLAIMER)
        self.assertIsInstance(exp.top_risk_increasing_factors, list)
        self.assertIsInstance(exp.top_risk_decreasing_factors, list)

        # Factors increasing risk must have positive attribution
        for f in exp.top_risk_increasing_factors:
            self.assertGreater(f["attribution"], 0.0)
            self.assertEqual(f["direction"], "increases_risk")

        # Factors decreasing risk must have negative attribution
        for f in exp.top_risk_decreasing_factors:
            self.assertLess(f["attribution"], 0.0)
            self.assertEqual(f["direction"], "decreases_risk")

    def test_global_feature_importance_completeness(self) -> None:
        """Verify global importance ranks all 45 canonical features."""
        global_imp = self.explainer.compute_global_importance(self.df)

        self.assertEqual(len(global_imp), len(FEATURE_NAMES))
        # Ranks must be 1 to 45
        ranks = [item.rank for item in global_imp]
        self.assertEqual(ranks, list(range(1, len(FEATURE_NAMES) + 1)))

        # Monotonically non-increasing mean absolute attribution
        mean_abs_vals = [item.mean_abs_attribution for item in global_imp]
        for i in range(len(mean_abs_vals) - 1):
            self.assertGreaterEqual(mean_abs_vals[i], mean_abs_vals[i + 1])

    def test_rejection_of_incompatible_classifier(self) -> None:
        """Verify ModelExplainer rejects non-LogisticRegression classifier with TypeError."""
        tree_pipe = Pipeline(
            [
                ("preprocessor", self.explainer.preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        )
        with self.assertRaises(TypeError):
            ModelExplainer(
                pipeline=tree_pipe,
                background_features_df=self.df,
            )


if __name__ == "__main__":
    unittest.main()
