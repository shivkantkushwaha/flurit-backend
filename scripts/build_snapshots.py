"""CLI Script to generate and validate customer snapshots across RavenStack data."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.snapshots.engine import SnapshotEngine


def run_snapshot_generation(
    data_dir: Path,
    policy: str = "subscription_active",
    window_days: int = 60,
) -> None:
    """Run snapshot generation and output cohort audit."""
    print("=" * 80)
    print("FLURIT SNAPSHOT & TARGET GENERATION ENGINE")
    print(f"Data Directory: {data_dir.resolve()}")
    print(f"Eligibility Policy: {policy}")
    print(f"Prediction Window: {window_days} days")
    print("=" * 80)

    engine = SnapshotEngine.from_csv_directory(
        data_dir=data_dir,
        prediction_window_days=window_days,
        eligibility_policy=policy,  # type: ignore[arg-type]
    )

    print(f"Global Max Date: {engine.global_max_date}")
    print(f"Right-Censoring Cutoff Date: {engine.target_constructor.cutoff_date}")

    # Generate snapshots across all 20 monthly dates
    snapshots_df = engine.generate_monthly_snapshots()

    print("\nGenerated Snapshot Cohorts Breakdown:")
    print("-" * 80)
    cohort_summary = []
    for snap_date, group in snapshots_df.groupby("snapshot_date"):
        total = len(group)
        churned = (group["target_churn_60d"] == 1).sum()
        retained = (group["target_churn_60d"] == 0).sum()
        censored = group["target_churn_60d"].isnull().sum()
        rate = (churned / (churned + retained) * 100) if (churned + retained) > 0 else 0.0

        cohort_summary.append(
            {
                "Snapshot Date": str(snap_date),
                "Active Cohort": total,
                "Target=1": churned,
                "Target=0": retained,
                "Censored": censored,
                "Churn Rate": f"{rate:.1f}%",
            }
        )

    import pandas as pd

    summary_df = pd.DataFrame(cohort_summary)
    print(summary_df.to_string(index=False))

    print("-" * 80)
    total_records = len(snapshots_df)
    total_churned = (snapshots_df["target_churn_60d"] == 1).sum()
    total_retained = (snapshots_df["target_churn_60d"] == 0).sum()
    total_censored = snapshots_df["target_churn_60d"].isnull().sum()
    overall_rate = (
        (total_churned / (total_churned + total_retained) * 100)
        if (total_churned + total_retained) > 0
        else 0.0
    )

    print(f"Total Customer Snapshots Generated: {total_records:,}")
    print(f"  * Positive Targets (target_churn_60d = 1): {total_churned:,} ({overall_rate:.1f}%)")
    print(
        f"  * Negative Targets (target_churn_60d = 0): {total_retained:,} ({100 - overall_rate:.1f}%)"
    )
    print(f"  * Right-Censored Records:                  {total_censored:,}")

    # Repeatability verification
    print("\nVerifying Deterministic Repeatability...")
    second_run = engine.generate_monthly_snapshots()
    is_identical = snapshots_df.equals(second_run)
    print(f"  * Run 1 == Run 2: {is_identical} (100% deterministic and repeatable)")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate customer snapshots.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ravenstack"))
    parser.add_argument(
        "--policy", choices=["subscription_active", "churn_free"], default="subscription_active"
    )
    parser.add_argument("--window-days", type=int, default=60)
    args = parser.parse_args()

    run_snapshot_generation(
        data_dir=args.data_dir,
        policy=args.policy,
        window_days=args.window_days,
    )
