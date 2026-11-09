"""Chronological Train/Validation/Test Partitioning Engine.

Enforces strict time-aware splitting by snapshot date:
- All customer rows from the same snapshot date reside exclusively in one partition.
- Train dates < Validation dates < Test dates (zero temporal overlap or forward leakage).
- Random row splitting is strictly prohibited.
"""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PartitionSummary:
    """Summary statistics for a single temporal partition."""

    name: str
    start_date: date
    end_date: date
    dates_count: int
    sample_count: int
    positive_count: int
    negative_count: int
    prevalence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert partition summary to dictionary."""
        d = asdict(self)
        d["start_date"] = str(self.start_date)
        d["end_date"] = str(self.end_date)
        return d


class ChronologicalSplitter:
    """Partitions time-aware snapshot DataFrames into train, val, and test splits with target-window purging.

    Enforces that:
    1. Max train date + prediction_window_days < Min val date (no train label overlap with val observation).
    2. Max val date + prediction_window_days < Min test date (no val label overlap with test observation).
    3. Boundary cohorts falling inside the purge buffers are discarded.
    """

    def __init__(
        self,
        train_end_date: date = date(2024, 5, 1),
        val_start_date: date = date(2024, 7, 1),
        val_end_date: date = date(2024, 8, 1),
        test_start_date: date = date(2024, 10, 1),
        prediction_window_days: int = 60,
    ) -> None:
        if train_end_date >= val_start_date:
            raise ValueError(
                f"train_end_date ({train_end_date}) must be strictly before val_start_date ({val_start_date})"
            )
        if val_start_date > val_end_date:
            raise ValueError(
                f"val_start_date ({val_start_date}) must be before or equal to val_end_date ({val_end_date})"
            )
        if val_end_date >= test_start_date:
            raise ValueError(
                f"val_end_date ({val_end_date}) must be strictly before test_start_date ({test_start_date})"
            )

        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.prediction_window_days = prediction_window_days

    def verify_target_window_purging(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Verify that no training label window overlaps validation observation dates,
        and no validation label window overlaps test observation dates."""
        from datetime import timedelta

        if not train_df.empty and not val_df.empty:
            max_train_date = max(train_df["snapshot_date"])
            min_val_date = min(val_df["snapshot_date"])
            train_label_end = max_train_date + timedelta(days=self.prediction_window_days)
            if min_val_date <= train_label_end:
                raise ValueError(
                    f"Purge violation: Min validation date ({min_val_date}) overlaps "
                    f"training {self.prediction_window_days}d target window (ends {train_label_end})."
                )

        if not val_df.empty and not test_df.empty:
            max_val_date = max(val_df["snapshot_date"])
            min_test_date = min(test_df["snapshot_date"])
            val_label_end = max_val_date + timedelta(days=self.prediction_window_days)
            if min_test_date <= val_label_end:
                raise ValueError(
                    f"Purge violation: Min test date ({min_test_date}) overlaps "
                    f"validation {self.prediction_window_days}d target window (ends {val_label_end})."
                )

    def get_discarded_cohorts(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return information about boundary cohorts discarded to purge target-window leakage."""
        data = df.copy()
        if not isinstance(data["snapshot_date"].iloc[0], date):
            data["snapshot_date"] = pd.to_datetime(data["snapshot_date"]).dt.date

        all_dates = sorted(data["snapshot_date"].unique())
        purged_train_val = [d for d in all_dates if self.train_end_date < d < self.val_start_date]
        purged_val_test = [d for d in all_dates if self.val_end_date < d < self.test_start_date]

        purged_tv_df = data[data["snapshot_date"].isin(purged_train_val)]
        purged_vt_df = data[data["snapshot_date"].isin(purged_val_test)]

        return {
            "purged_train_val_dates": [str(d) for d in purged_train_val],
            "purged_val_test_dates": [str(d) for d in purged_val_test],
            "purged_train_val_samples": len(purged_tv_df),
            "purged_val_test_samples": len(purged_vt_df),
            "total_purged_samples": len(purged_tv_df) + len(purged_vt_df),
        }

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train, val, and test partitions by snapshot_date with purging.

        Args:
            df: Feature DataFrame containing 'snapshot_date' column.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        data = df.copy()
        if "snapshot_date" not in data.columns:
            raise KeyError("DataFrame must contain 'snapshot_date' column.")

        # Ensure snapshot_date is comparable date object
        if not isinstance(data["snapshot_date"].iloc[0], date):
            data["snapshot_date"] = pd.to_datetime(data["snapshot_date"]).dt.date

        train_mask = data["snapshot_date"] <= self.train_end_date
        val_mask = (data["snapshot_date"] >= self.val_start_date) & (
            data["snapshot_date"] <= self.val_end_date
        )
        test_mask = data["snapshot_date"] >= self.test_start_date

        train_df = data[train_mask].copy()
        val_df = data[val_mask].copy()
        test_df = data[test_mask].copy()

        # Invariant checks: zero date overlap
        train_dates = set(train_df["snapshot_date"].unique())
        val_dates = set(val_df["snapshot_date"].unique())
        test_dates = set(test_df["snapshot_date"].unique())

        assert len(train_dates & val_dates) == 0, "Leakage: Train and Val share snapshot dates!"
        assert len(val_dates & test_dates) == 0, "Leakage: Val and Test share snapshot dates!"
        assert len(train_dates & test_dates) == 0, "Leakage: Train and Test share snapshot dates!"

        # Invariant check: verify target-window purging
        self.verify_target_window_purging(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def summarize_partitions(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "target_churn_60d",
    ) -> dict[str, PartitionSummary]:
        """Compute demographic and class prevalence summaries across partitions."""
        summaries: dict[str, PartitionSummary] = {}
        partitions = [("train", train_df), ("val", val_df), ("test", test_df)]

        for name, p_df in partitions:
            dates = sorted(p_df["snapshot_date"].unique())
            n_samples = len(p_df)
            n_pos = int((p_df[target_col] == 1).sum()) if target_col in p_df.columns else 0
            n_neg = int((p_df[target_col] == 0).sum()) if target_col in p_df.columns else 0
            prev = float(n_pos / n_samples) if n_samples > 0 else 0.0

            summaries[name] = PartitionSummary(
                name=name,
                start_date=dates[0] if dates else date.min,
                end_date=dates[-1] if dates else date.min,
                dates_count=len(dates),
                sample_count=n_samples,
                positive_count=n_pos,
                negative_count=n_neg,
                prevalence=prev,
            )

        return summaries
