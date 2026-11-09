"""Subscription point-in-time feature extraction.

Computes active subscription volume, MRR/ARR, contract terms, tenure,
and historical plan movement strictly at snapshot date.
"""

from datetime import date
from typing import Any

import pandas as pd


class SubscriptionFeatureExtractor:
    """Extracts subscription and contract features without lookahead leakage."""

    def __init__(self, subscriptions_df: pd.DataFrame) -> None:
        self.subscriptions = subscriptions_df.copy()
        self.subscriptions["parsed_start"] = pd.to_datetime(self.subscriptions["start_date"])
        self.subscriptions["parsed_end"] = pd.to_datetime(self.subscriptions["end_date"])

        # Group by account_id for fast retrieval
        self._grouped = dict(iter(self.subscriptions.groupby("account_id")))

    def extract_features(self, account_id: str, snapshot_date: date) -> dict[str, Any]:
        """Extract subscription features for an account at snapshot date."""
        account_subs = self._grouped.get(account_id)
        if account_subs is None or account_subs.empty:
            return self._empty_features()

        snap_ts = pd.to_datetime(snapshot_date)

        # Filter strictly to subscriptions started on or before snapshot date
        started_subs = account_subs[account_subs["parsed_start"] <= snap_ts]
        if started_subs.empty:
            return self._empty_features()

        # Active subscriptions at snapshot date:
        # start_date <= snapshot_date AND (end_date IS NULL OR end_date > snapshot_date)
        active_subs = started_subs[
            started_subs["parsed_end"].isnull() | (started_subs["parsed_end"] > snap_ts)
        ]

        active_count = len(active_subs)
        current_mrr = float(active_subs["mrr_amount"].sum()) if active_count > 0 else 0.0
        current_arr = float(active_subs["arr_amount"].sum()) if active_count > 0 else 0.0
        active_seats = int(active_subs["seats"].sum()) if active_count > 0 else 0

        # Commercial tenure from earliest subscription started
        earliest_start = started_subs["parsed_start"].min().date()
        tenure_days = max(0, (snapshot_date - earliest_start).days)

        # Billing frequency
        if active_count == 0:
            freq = "none"
        else:
            unique_freqs = active_subs["billing_frequency"].dropna().unique()
            if len(unique_freqs) == 1:
                freq = str(unique_freqs[0])
            elif len(unique_freqs) > 1:
                freq = "mixed"
            else:
                freq = "none"

        has_auto_renew = (
            1 if active_count > 0 and bool((active_subs["auto_renew_flag"] == True).any()) else 0  # noqa: E712
        )
        has_active_trial = (
            1 if active_count > 0 and bool((active_subs["is_trial"] == True).any()) else 0  # noqa: E712
        )

        # Historical upgrade/downgrade counts across all started contracts to date
        upgrade_count = int((started_subs["upgrade_flag"] == True).sum())  # noqa: E712
        downgrade_count = int((started_subs["downgrade_flag"] == True).sum())  # noqa: E712

        return {
            "sub_active_count": active_count,
            "sub_current_mrr": current_mrr,
            "sub_current_arr": current_arr,
            "sub_total_active_seats": active_seats,
            "sub_tenure_days": tenure_days,
            "sub_billing_frequency": freq,
            "sub_has_auto_renew": has_auto_renew,
            "sub_has_active_trial": has_active_trial,
            "sub_historical_upgrade_count": upgrade_count,
            "sub_historical_downgrade_count": downgrade_count,
        }

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "sub_active_count": 0,
            "sub_current_mrr": 0.0,
            "sub_current_arr": 0.0,
            "sub_total_active_seats": 0,
            "sub_tenure_days": 0,
            "sub_billing_frequency": "none",
            "sub_has_auto_renew": 0,
            "sub_has_active_trial": 0,
            "sub_historical_upgrade_count": 0,
            "sub_historical_downgrade_count": 0,
        }
