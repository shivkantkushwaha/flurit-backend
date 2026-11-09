"""CLI Script to generate and audit point-in-time feature matrices."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.features.pipeline import FeaturePipeline
from app.ml.features.schema import FEATURE_DEFINITIONS, FEATURE_NAMES
from app.ml.snapshots.engine import SnapshotEngine


def run_feature_generation(
    data_dir: Path,
    output_path: Path | None = None,
) -> None:
    """Run full point-in-time feature engineering pipeline."""
    print("=" * 80)
    print("FLURIT POINT-IN-TIME FEATURE ENGINEERING PIPELINE")
    print(f"Data Directory: {data_dir.resolve()}")
    print("=" * 80)

    # 1. Generate approved subscription_active snapshots
    t0 = time.time()
    print("\n[1/3] Generating approved 'subscription_active' customer snapshots...")
    snapshot_engine = SnapshotEngine.from_csv_directory(
        data_dir=data_dir,
        prediction_window_days=60,
        eligibility_policy="subscription_active",
    )
    snapshots_df = snapshot_engine.generate_monthly_snapshots()
    t_snap = time.time() - t0
    print(f"  * Generated {len(snapshots_df):,} customer snapshots in {t_snap:.2f}s")
    print(f"  * Target balance: {snapshots_df['target_churn_60d'].value_counts().to_dict()}")

    # 2. Extract feature matrix
    t1 = time.time()
    print("\n[2/3] Extracting point-in-time feature vectors across all 5 domains...")
    pipeline = FeaturePipeline.from_csv_directory(data_dir=data_dir)
    features_df = pipeline.build_feature_matrix(snapshots_df, include_metadata=True)
    t_feat = time.time() - t1
    print(
        f"  * Feature extraction complete in {t_feat:.2f}s ({len(features_df) / max(t_feat, 0.001):.1f} rows/s)"
    )

    # 3. Validation & Auditing
    print("\n[3/3] Validating feature matrix and anti-leakage invariants...")
    pipeline.verify_no_leakage(features_df)
    print("  * Anti-leakage checks: 100% PASSED (prohibited fields strictly absent)")

    dup_keys = features_df.duplicated(subset=["account_id", "snapshot_date"]).sum()
    print(f"  * Duplicate snapshot keys (account_id, snapshot_date): {dup_keys}")
    assert dup_keys == 0, "Duplicate composite snapshot keys detected!"

    print("\nFeature Matrix Summary:")
    print("-" * 80)
    print(f"  * Total Rows:            {len(features_df):,}")
    print(
        f"  * Total Columns:         {len(features_df.columns):,} (2 metadata + {len(FEATURE_NAMES)} features + 1 target)"
    )
    print(f"  * Feature Count:         {len(FEATURE_NAMES)}")

    # Null value inspection
    null_counts = features_df[FEATURE_NAMES].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        print("  * Missing Values:        0 nulls across all 45 features (100% complete)")
    else:
        print("  * Features with Nulls:")
        for col, cnt in cols_with_nulls.items():
            print(f"      - {col}: {cnt} nulls ({cnt / len(features_df) * 100:.1f}%)")

    # Domain feature count
    by_group: dict[str, int] = {}
    for f in FEATURE_DEFINITIONS:
        by_group[f.group.value] = by_group.get(f.group.value, 0) + 1
    print("\nFeatures by Domain Group:")
    for grp, count in sorted(by_group.items()):
        print(f"  * {grp:<15}: {count} features")

    # Optional output serialization
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_path, index=False)
        print(f"\nSaved feature dataset to: {output_path.resolve()}")

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Flurit customer feature matrix.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ravenstack"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_feature_generation(
        data_dir=args.data_dir,
        output_path=args.output,
    )
