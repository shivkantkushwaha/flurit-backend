"""RavenStack Dataset Profiling & Inspection Script.

Reports row counts, column types, missing values, duplicates,
and date ranges across all five RavenStack CSV tables.
"""

import sys
from pathlib import Path

import pandas as pd


def inspect_dataset(data_dir: Path) -> dict:
    """Inspect and profile all RavenStack source tables."""
    tables = {
        "accounts": data_dir / "ravenstack_accounts.csv",
        "subscriptions": data_dir / "ravenstack_subscriptions.csv",
        "feature_usage": data_dir / "ravenstack_feature_usage.csv",
        "support_tickets": data_dir / "ravenstack_support_tickets.csv",
        "churn_events": data_dir / "ravenstack_churn_events.csv",
    }

    report = {}

    print("=" * 80)
    print("RAVENSTACK DATASET PROFILE REPORT")
    print(f"Data Directory: {data_dir.resolve()}")
    print("=" * 80)

    for name, path in tables.items():
        if not path.exists():
            print(f"[-] ERROR: File missing: {path.name}")
            continue

        df = pd.read_csv(path)
        report[name] = {"rows": len(df), "cols": len(df.columns)}

        print(f"\n--- TABLE: {name} ({path.name}) ---")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

        # Check duplicates
        dup_rows = df.duplicated().sum()
        print(f"Duplicate full rows: {dup_rows}")

        # Column breakdown
        print("\nColumns & Data Quality:")
        col_summary = []
        for col in df.columns:
            n_null = df[col].isnull().sum()
            null_pct = (n_null / len(df)) * 100
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            col_summary.append(
                {
                    "Column": col,
                    "Dtype": dtype,
                    "Nulls": n_null,
                    "Null%": f"{null_pct:.1f}%",
                    "Unique": n_unique,
                }
            )
        summary_df = pd.DataFrame(col_summary)
        print(summary_df.to_string(index=False))

        # Explicit date/timestamp columns
        date_candidates = {
            "signup_date",
            "start_date",
            "end_date",
            "usage_date",
            "submitted_at",
            "closed_at",
            "churn_date",
        }
        date_cols = [c for c in df.columns if c.lower() in date_candidates]
        if date_cols:
            print("\nDate Column Boundaries:")
            for dc in date_cols:
                parsed = pd.to_datetime(df[dc], errors="coerce")
                valid_parsed = parsed.dropna()
                if not valid_parsed.empty:
                    min_d = valid_parsed.min()
                    max_d = valid_parsed.max()
                    span_days = (max_d - min_d).days
                    print(
                        f"  * {dc}: {min_d} to {max_d} (span: {span_days} days | nulls: {parsed.isna().sum()})"
                    )

    print("\n" + "=" * 80)
    print("END OF PROFILE REPORT")
    print("=" * 80)
    return report


if __name__ == "__main__":
    target_dir = Path("data/raw/ravenstack")
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    inspect_dataset(target_dir)
