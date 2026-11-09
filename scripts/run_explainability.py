"""CLI Script to generate SHAP feature attributions and global/local explanations."""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from app.ml.explainability.explainer import ModelExplainer
from app.ml.features.pipeline import FeaturePipeline
from app.ml.snapshots.engine import SnapshotEngine
from app.ml.training.splits import ChronologicalSplitter


def run_explainability_pipeline(
    data_dir: Path,
    model_artifact_dir: Path,
    output_dir: Path,
) -> None:
    """Execute SHAP explainability generation for global and local insights."""
    print("=" * 85)
    print("FLURIT ML MODEL EXPLAINABILITY PIPELINE (SHAP LinearExplainer)")
    print(f"Data Directory:           {data_dir.resolve()}")
    print(f"Model Artifact Directory: {model_artifact_dir.resolve()}")
    print(f"Output Directory:         {output_dir.resolve()}")
    print("=" * 85)

    # 1. Load trained champion pipeline
    print("\n[1/4] Loading trained champion model pipeline...")
    pipe_path = model_artifact_dir / "model.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Missing champion pipeline at {pipe_path}")
    pipeline = joblib.load(pipe_path)
    print("  * Successfully loaded pipeline with steps:", list(pipeline.named_steps.keys()))

    # 2. Extract feature matrix
    print("\n[2/4] Generating customer snapshots and point-in-time feature matrix...")
    t0 = time.time()
    snapshot_engine = SnapshotEngine.from_csv_directory(
        data_dir=data_dir,
        prediction_window_days=60,
        eligibility_policy="subscription_active",
    )
    snaps_df = snapshot_engine.generate_monthly_snapshots()
    feat_pipeline = FeaturePipeline.from_csv_directory(data_dir=data_dir)
    features_df = feat_pipeline.build_feature_matrix(snaps_df, include_metadata=True)
    print(f"  * Generated {len(features_df):,} feature rows in {time.time() - t0:.2f}s")

    # Split: use training partition as reference background dataset
    splitter = ChronologicalSplitter()
    train_df, val_df, test_df = splitter.split(features_df)
    print(f"  * Background training cohort size: {len(train_df):,} rows")

    # 3. Initialize ModelExplainer
    print("\n[3/4] Initializing SHAP LinearExplainer and computing attributions...")
    t1 = time.time()
    explainer = ModelExplainer(
        pipeline=pipeline,
        background_features_df=train_df,
        decision_threshold=0.5,
    )

    # Compute global feature importance across test cohort (unseen evaluation data)
    print("  * Computing global feature importance on test partition...")
    global_importance = explainer.compute_global_importance(test_df)
    print(f"  * Global feature attributions computed in {time.time() - t1:.2f}s")

    print("\nTop 15 Global Churn Drivers (Test Partition):")
    print("-" * 85)
    print(
        f"{'Rank':<5} | {'Feature Name':<35} | {'Mean |SHAP|':<12} | {'Direction':<12} | {'Group':<15}"
    )
    print("-" * 85)
    for item in global_importance[:15]:
        direction = "Positive (+)" if item.mean_directional_attribution > 0 else "Negative (-)"
        print(
            f"{item.rank:<5} | {item.feature_name:<35} | {item.mean_abs_attribution:<12.4f} | {direction:<12} | {item.feature_group:<15}"
        )
    print("-" * 85)

    # 4. Generate sample local explanations
    print("\n[4/4] Generating representative local explanations across risk tiers...")
    # Select high risk, moderate risk, and low risk test accounts
    probs = pipeline.predict_proba(
        test_df[explainer.transformed_names if False else explainer._feature_meta.keys()]
    )[:, 1]
    test_eval_df = test_df.copy()
    test_eval_df["model_score"] = probs

    high_risk_sample = test_eval_df[test_eval_df["model_score"] >= 0.50].head(2)
    mod_risk_sample = test_eval_df[
        (test_eval_df["model_score"] >= 0.35) & (test_eval_df["model_score"] < 0.50)
    ].head(2)
    low_risk_sample = test_eval_df[test_eval_df["model_score"] < 0.20].head(2)

    sample_cohort = pd.concat([high_risk_sample, mod_risk_sample, low_risk_sample])
    local_explanations = explainer.explain_batch(sample_cohort, top_k=5)

    print(f"  * Generated {len(local_explanations)} detailed local explanations.")
    for i, exp in enumerate(local_explanations[:3], 1):
        print(f"\n  [Sample {i}] Account: {exp.account_id} | Date: {exp.snapshot_date}")
        print(
            f"    - Model Churn Score: {exp.model_score:.4f} (Decision: {'High Risk (1)' if exp.predicted_risk_label == 1 else 'Low Risk (0)'})"
        )
        print(f"    - Summary: {exp.plain_language_summary}")
        if exp.top_risk_increasing_factors:
            print("    - Top Factors Raising Risk:")
            for f in exp.top_risk_increasing_factors[:2]:
                print(
                    f"        * {f['feature_name']}: val={f['feature_value']}, SHAP=+{f['attribution']:.4f}"
                )
        if exp.top_risk_decreasing_factors:
            print("    - Top Factors Lowering Risk:")
            for f in exp.top_risk_decreasing_factors[:2]:
                print(
                    f"        * {f['feature_name']}: val={f['feature_value']}, SHAP={f['attribution']:.4f}"
                )

    # 5. Persist artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    global_file = output_dir / "global_feature_importance.json"
    local_file = output_dir / "sample_local_explanations.json"
    explainer_file = explainer.save_artifact(output_dir)

    with open(global_file, "w", encoding="utf-8") as f_out:
        json.dump([item.to_dict() for item in global_importance], f_out, indent=2)

    with open(local_file, "w", encoding="utf-8") as f_out:
        json.dump([exp.to_dict() for exp in local_explanations], f_out, indent=2)

    print("\n" + "=" * 85)
    print("EXPLAINABILITY ARTIFACTS PERSISTED SUCCESSFULLY:")
    print(f"  * Pipeline Explainer: {explainer_file.resolve()}")
    print(f"  * Global Importance:  {global_file.resolve()}")
    print(f"  * Local Explanations:  {local_file.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Flurit ML Explanations.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ravenstack"))
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        default=Path("artifacts/models/churn_prediction_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/explainability"),
    )
    args = parser.parse_args()

    run_explainability_pipeline(
        data_dir=args.data_dir,
        model_artifact_dir=args.model_artifact_dir,
        output_dir=args.output_dir,
    )
