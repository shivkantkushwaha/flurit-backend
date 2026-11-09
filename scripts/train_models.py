"""CLI Script to train, compare, select, and evaluate churn prediction models."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.features.pipeline import FeaturePipeline
from app.ml.snapshots.engine import SnapshotEngine
from app.ml.training.trainer import ModelTrainer


def run_training_pipeline(
    data_dir: Path,
    artifact_dir: Path,
    version: str = "1.0.0",
    overwrite: bool = True,
) -> None:
    """Execute complete end-to-end model training, validation selection, and testing."""
    print("=" * 85)
    print("FLURIT CHURN PREDICTION ML TRAINING & EVALUATION PIPELINE")
    print(f"Data Directory:     {data_dir.resolve()}")
    print(f"Artifact Directory: {artifact_dir.resolve()}")
    print(f"Model Version:      {version}")
    print("=" * 85)

    # 1. Generate approved snapshots
    t0 = time.time()
    print("\n[1/4] Generating approved 'subscription_active' customer snapshots...")
    snapshot_engine = SnapshotEngine.from_csv_directory(
        data_dir=data_dir,
        prediction_window_days=60,
        eligibility_policy="subscription_active",
    )
    snapshots_df = snapshot_engine.generate_monthly_snapshots()
    print(f"  * Generated {len(snapshots_df):,} customer snapshots in {time.time() - t0:.2f}s")

    # 2. Extract feature matrix
    t1 = time.time()
    print("\n[2/4] Extracting point-in-time feature matrix (45 features across 5 domains)...")
    feature_pipeline = FeaturePipeline.from_csv_directory(data_dir=data_dir)
    features_df = feature_pipeline.build_feature_matrix(snapshots_df, include_metadata=True)
    feature_pipeline.verify_no_leakage(features_df)
    print(f"  * Feature matrix complete: shape={features_df.shape} in {time.time() - t1:.2f}s")

    # 3. Training & Validation
    t2 = time.time()
    print("\n[3/4] Running chronological training experiment across candidate models...")
    trainer = ModelTrainer(random_state=42, decision_threshold=0.5)
    result = trainer.run_training_experiment(features_df)
    t_train = time.time() - t2
    print(f"  * Training and evaluation complete in {t_train:.2f}s")

    # Report Partitions
    partitions = result["partitions"]
    discarded = result.get("discarded_cohorts", {})
    print("\nPurged Chronological Partitions Breakdown (60-day target buffer):")
    print("-" * 85)
    print(
        f"{'Partition':<10} | {'Dates Range':<25} {'Cohorts':<8} {'Samples':<8} {'Positives':<10} {'Prevalence':<10}"
    )
    print("-" * 85)
    for p_name in ["train", "val", "test"]:
        p = partitions[p_name]
        d_range = f"{p.start_date} to {p.end_date}"
        print(
            f"{p_name:<10} | {d_range:<25} {p.dates_count:<8} {p.sample_count:<8} {p.positive_count:<10} {p.prevalence * 100:6.2f}%"
        )
    print("-" * 85)
    if discarded:
        print("Discarded Boundary Cohorts (Purged to eliminate 60-day target window overlap):")
        print(
            f"  * Train-Val Buffer: dates={discarded.get('purged_train_val_dates')} (N={discarded.get('purged_train_val_samples')} samples)"
        )
        print(
            f"  * Val-Test Buffer:   dates={discarded.get('purged_val_test_dates')} (N={discarded.get('purged_val_test_samples')} samples)"
        )
        print(f"  * Total Purged:      {discarded.get('total_purged_samples')} samples discarded")
        print("-" * 85)

    # Report Validation Comparison
    val_metrics = result["validation_metrics"]
    val_prev = partitions["val"].prevalence
    val_brier_base = val_prev * (1.0 - val_prev)
    print(
        f"\nCandidate Models Validation Comparison (Baseline Prevalence = {val_prev * 100:.2f}%, Brier Baseline = {val_brier_base:.4f}):"
    )
    print("-" * 95)
    print(
        f"{'Model Architecture':<24} | {'PR-AUC':<8} {'ROC-AUC':<8} {'Brier':<8} {'BSS':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}"
    )
    print("-" * 95)
    for m_name, m in val_metrics.items():
        print(
            f"{m_name:<24} | {m.pr_auc:<8.4f} {m.roc_auc:<8.4f} {m.brier_score:<8.4f} {m.brier_skill_score:<8.4f} {m.precision:<10.4f} {m.recall:<8.4f} {m.f1:<8.4f}"
        )
    print("-" * 95)

    selected_model = result["selected_model_type"]
    print(f"\nChampion Model Selected: '{selected_model}' (Highest Validation PR-AUC)")

    # Report Held-Out Test Evaluation
    test_metrics = result["test_metrics"]
    test_prev = partitions["test"].prevalence
    test_brier_base = test_prev * (1.0 - test_prev)
    print(f"\nHeld-Out Test Evaluation for Champion Model ('{selected_model}'):")
    print(
        f"  (Test Partition: {partitions['test'].start_date} to {partitions['test'].end_date} | Baseline Prevalence: {test_prev * 100:.2f}%)"
    )
    print(
        "  [SCIENTIFIC DISCLOSURE: An unpurged test partition was previously inspected in Phase 6 before this audit]"
    )
    print("-" * 85)
    print(
        f"  * PR-AUC (Average Precision):  {test_metrics.pr_auc:.4f}  (vs prevalence baseline {test_prev:.4f})"
    )
    print(f"  * ROC-AUC:                     {test_metrics.roc_auc:.4f}")
    print(
        f"  * Brier Score:                 {test_metrics.brier_score:.4f}  (vs constant-prev baseline {test_brier_base:.4f}, BSS: {test_metrics.brier_skill_score:.4f})"
    )
    print(
        f"  * Decision Cutoff:             {test_metrics.decision_threshold:.2f} (un-tuned default threshold; not optimized)"
    )
    print(f"  * Precision @ 0.50:            {test_metrics.precision:.4f}")
    print(f"  * Recall @ 0.50:               {test_metrics.recall:.4f}")
    print(f"  * F1 Score @ 0.50:             {test_metrics.f1:.4f}")
    print("  * Confusion Matrix (@ 0.50 threshold):")
    print(f"      - True Negatives (TN):     {test_metrics.true_negatives:,}")
    print(f"      - False Positives (FP):    {test_metrics.false_positives:,}")
    print(f"      - False Negatives (FN):    {test_metrics.false_negatives:,}")
    print(f"      - True Positives (TP):     {test_metrics.true_positives:,}")

    # 4. Save Artifacts
    print("\n[4/4] Persisting model artifacts and metadata to disk...")
    saved_dir = trainer.save_artifact(
        experiment_result=result,
        artifact_dir=artifact_dir,
        version=version,
        overwrite=overwrite,
    )
    print(f"  * Artifact directory: {saved_dir.resolve()}")
    print("  * Files created: 'model.joblib', 'metadata.json'")

    # Verify loading reproducibility
    loaded_pipe, loaded_meta = trainer.load_artifact(saved_dir)
    assert loaded_meta["selected_model_type"] == selected_model
    print("  * Artifact loading verified: SUCCESS!")

    print("\n" + "=" * 85)
    print("FLURIT ML TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flurit Churn Prediction Pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ravenstack"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/models/churn_prediction_v1")
    )
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        version=args.version,
        overwrite=args.overwrite,
    )
