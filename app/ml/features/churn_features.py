"""Historical churn point-in-time feature extraction.

Computes prior churn event frequencies, recency, and refund totals
strictly using events logged on or before the snapshot date.
"""

from datetime import date
from typing import Any

import pandas as pd


class ChurnFeatureExtractor:
    """Extracts historical churn records without future lookahead leakage."""

    def __init__(self, churn_events_df: pd.DataFrame) -> None:
        churns = churn_events_df.copy()
        churns["parsed_churn"] = pd.to_datetime(churns["churn_date"])

        # Group by account_id for fast retrieval
        self._grouped = dict(iter(churns.groupby("account_id")))

    def extract_features(self, account_id: str, snapshot_date: date) -> dict[str, Any]:
        """Extract prior churn features for an account at snapshot date."""
        acc_churns = self._grouped.get(account_id)
        if acc_churns is None or acc_churns.empty:
            return self._empty_features()

        snap_ts = pd.to_datetime(snapshot_date)

        # Filter strictly to churn events occurring on or before snapshot date
        hist = acc_churns[acc_churns["parsed_churn"] <= snap_ts]
        if hist.empty:
            return self._empty_features()

        prior_count = len(hist)
        has_prior = 1 if prior_count > 0 else 0
        latest_date = hist["parsed_churn"].max().date()
        days_since_last = float(max(0, (snapshot_date - latest_date).days))
        refund_total = float(hist["refund_amount_usd"].sum())

        return {
            "churn_prior_count": prior_count,
            "churn_has_prior": has_prior,
            "churn_days_since_last": days_since_last,
            "churn_prior_refund_total": refund_total,
        }

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "churn_prior_count": 0,
            "churn_has_prior": 0,
            "churn_days_since_last": -1.0,
            "churn_prior_refund_total": 0.0,
        }
