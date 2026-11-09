"""CLI Script to run customer segmentation evaluation, profiling, and Value x Risk analysis."""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from app.ml.features.pipeline import FeaturePipeline
from app.ml.segmentation.clustering import CustomerSegmenter
from app.ml.snapshots.engine import SnapshotEngine
from app.ml.training.splits import ChronologicalSplitter


def run_segmentation_pipeline(
    data_dir: Path,
    model_artifact_dir: Path,
    output_dir: Path,
    n_clusters: int = 3,
) -> None:
    """Execute customer segmentation evaluation (K=2..6) and Value x Risk mapping."""
    print("=" * 85)
    print("FLURIT CUSTOMER SEGMENTATION & VALUE x RISK ANALYSIS PIPELINE")
    print(f"Data Directory:           {data_dir.resolve()}")
    print(f"Model Artifact Directory: {model_artifact_dir.resolve()}")
    print(f"Output Directory:         {output_dir.resolve()}")
    print(f"Selected Cluster Count K: {n_clusters}")
    print("=" * 85)

    # 1. Load feature matrix and split
    print("\n[1/4] Generating customer snapshots and point-in-time feature matrix...")
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

    # Split: fit clustering exclusively on training development partition
    splitter = ChronologicalSplitter()
    train_df, val_df, test_df = splitter.split(features_df)
    print(f"  * Training development cohort size: {len(train_df):,} rows")

    # Strictly exclude target labels to enforce unsupervised invariant
    prohibited = ["target_churn_60d", "churn_flag", "is_churned"]
    train_features = train_df.drop(columns=[c for c in prohibited if c in train_df.columns])
    test_features = test_df.drop(columns=[c for c in prohibited if c in test_df.columns])

    # 2. Evaluate K-Range from K=2 to K=6
    print("\n[2/4] Evaluating candidate cluster counts from K=2 to K=6 on development features...")
    segmenter = CustomerSegmenter(n_clusters=n_clusters, random_state=42)
    eval_results = segmenter.evaluate_k_range(train_features, k_range=range(2, 7))

    print("\nCandidate Cluster Evaluation Matrix (Development Partition):")
    print("-" * 95)
    print(
        f"{'K':<4} | {'Silhouette':<12} | {'Davies-Bouldin':<15} | {'Calinski-Harabasz':<18} | {'Inertia':<12} | {'Cluster Sizes'}"
    )
    print("-" * 95)
    for res in eval_results:
        sizes_str = str(res.cluster_sizes)
        print(
            f"{res.k:<4} | {res.silhouette_score:<12.4f} | {res.davies_bouldin_score:<15.4f} | {res.calinski_harabasz_score:<18.1f} | {res.inertia:<12.1f} | {sizes_str}"
        )
    print("-" * 95)

    # 3. Fit Champion K-Means model on training features
    print(f"\n[3/4] Fitting champion K={n_clusters} segmentation model on development set...")
    segmenter.fit(train_features)
    print("  * Empirical Cluster Profiles:")
    for prof in segmenter.cluster_profiles:
        print(f"    - {prof.cluster_name} (N={prof.sample_count:,}, {prof.percentage:.1f}%)")
        print(f"        {prof.empirical_summary}")

    # 4. Customer Value x Model Score Risk Quadrants
    print("\n[4/4] Conducting Customer Value x Model Score Risk Analysis...")
    # Load champion Logistic Regression model for risk scores
    pipe_path = model_artifact_dir / "model.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Missing champion pipeline at {pipe_path}")
    churn_pipeline = joblib.load(pipe_path)

    # Evaluate across test partition (or full dataset)
    from app.ml.features.schema import FEATURE_NAMES

    test_scores = churn_pipeline.predict_proba(test_features[FEATURE_NAMES])[:, 1]
    risk_analysis = segmenter.analyze_value_risk_quadrants(
        features_df=test_features,
        model_scores=test_scores,
    )

    print("\nCustomer Value x Risk Prioritization Quadrants (Held-Out Test Cohort):")
    print("-" * 95)
    print(
        f"{'Quadrant Name':<35} | {'Priority':<8} | {'Accounts':<10} | {'Total MRR ($)':<15} | {'Est. Rev at Risk ($)'}"
    )
    print("-" * 95)
    for q in risk_analysis["quadrants"]:
        print(
            f"{q['quadrant_name']:<35} | {q['priority_level']:<8} | {q['account_count']:<10} | ${q['total_mrr']:<14,.2f} | ${q['estimated_revenue_at_risk_mrr']:<,.2f}"
        )
    print("-" * 95)
    print(f"Total Test Portfolio MRR:            ${risk_analysis['total_mrr']:,.2f}")
    print(
        f"Total Estimated Revenue at Risk:     ${risk_analysis['total_estimated_revenue_at_risk_mrr']:,.2f}"
    )
    print(f"  Note: {risk_analysis['risk_metric_interpretation']}")

    # 5. Persist artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter.save_artifact(output_dir)

    matrix_file = output_dir / "risk_value_matrix.json"
    with open(matrix_file, "w", encoding="utf-8") as f:
        json.dump(risk_analysis, f, indent=2)

    print("\n" + "=" * 85)
    print("SEGMENTATION ARTIFACTS PERSISTED SUCCESSFULLY:")
    print(f"  * Pipeline Artifact: {output_dir / 'segmentation_model.joblib'}")
    print(f"  * Evaluation Table:  {output_dir / 'cluster_evaluation_metrics.json'}")
    print(f"  * Cluster Profiles:  {output_dir / 'cluster_profiles.json'}")
    print(f"  * Value-Risk Matrix: {matrix_file.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flurit Customer Segmentation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ravenstack"))
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        default=Path("artifacts/models/churn_prediction_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/segmentation"),
    )
    parser.add_argument("--n-clusters", type=int, default=3)
    args = parser.parse_args()

    run_segmentation_pipeline(
        data_dir=args.data_dir,
        model_artifact_dir=args.model_artifact_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
    )
