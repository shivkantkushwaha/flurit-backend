"""Leakage-Safe Churn Target Construction.

Defines target labeling logic for forward prediction windows:
Target = 1 if at least one qualifying churn event occurs in (T_obs, T_obs + W]
Target = 0 if customer remains active throughout (T_obs, T_obs + W] and T_obs <= T_cutoff
Target = NULL (None) if T_obs > T_cutoff (right-censored where future is unobserved)
"""

from datetime import date, timedelta

import pandas as pd


class TargetConstructor:
    """Constructs future churn prediction targets with strict anti-leakage safeguards."""

    def __init__(
        self,
        prediction_window_days: int = 60,
        global_max_date: date | None = None,
    ) -> None:
        self.window_days = prediction_window_days
        self.window_delta = timedelta(days=prediction_window_days)
        self.global_max_date = global_max_date

    @property
    def cutoff_date(self) -> date | None:
        """Right-censoring cutoff date (max observable date - prediction window)."""
        if self.global_max_date is None:
            return None
        return self.global_max_date - self.window_delta

    def is_right_censored(self, snapshot_date: date) -> bool:
        """Check if a snapshot date falls beyond the right-censoring cutoff."""
        if self.cutoff_date is None:
            return False
        return snapshot_date > self.cutoff_date

    def compute_target(
        self,
        account_id: str,
        snapshot_date: date,
        churn_events_df: pd.DataFrame,
    ) -> int | None:
        """Compute binary churn target for a single account snapshot.

        Args:
            account_id: Customer account identifier.
            snapshot_date: Observation snapshot date (T_obs).
            churn_events_df: Historical churn events DataFrame with 'account_id' and 'churn_date'.

        Returns:
            1 if churned in (T_obs, T_obs + W],
            0 if fully observed non-churner,
            None if right-censored.
        """
        # If beyond cutoff date, the future is unobserved (right-censored)
        if self.is_right_censored(snapshot_date):
            return None

        window_end = snapshot_date + self.window_delta

        # Find churn events for this account strictly in (snapshot_date, window_end]
        acc_churns = churn_events_df[churn_events_df["account_id"] == account_id]
        if acc_churns.empty:
            return 0

        # Filter strictly within the forward window
        in_window = acc_churns[
            (acc_churns["parsed_churn_date"].dt.date > snapshot_date)
            & (acc_churns["parsed_churn_date"].dt.date <= window_end)
        ]

        return 1 if len(in_window) > 0 else 0
