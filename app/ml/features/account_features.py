"""Account-level point-in-time feature extraction.

Computes customer profile and account age attributes strictly at snapshot date.
"""

from datetime import date
from typing import Any

import pandas as pd


class AccountFeatureExtractor:
    """Extracts account profile and age features without lookahead leakage."""

    def __init__(self, accounts_df: pd.DataFrame) -> None:
        self.accounts = accounts_df.copy()
        self.accounts["parsed_signup"] = pd.to_datetime(self.accounts["signup_date"]).dt.date
        # Index by account_id for fast O(1) lookups
        self._indexed_accounts = self.accounts.set_index("account_id")

    def extract_features(self, account_id: str, snapshot_date: date) -> dict[str, Any]:
        """Extract account features for a single customer at snapshot date."""
        if account_id not in self._indexed_accounts.index:
            return {
                "account_age_days": 0,
                "account_industry": "unknown",
                "account_country": "unknown",
                "account_referral_source": "unknown",
                "account_initial_seats": 1,
                "account_is_trial": 0,
            }

        row = self._indexed_accounts.loc[account_id]
        signup = row["parsed_signup"]
        age_days = max(0, (snapshot_date - signup).days)

        return {
            "account_age_days": age_days,
            "account_industry": str(row["industry"]),
            "account_country": str(row["country"]),
            "account_referral_source": str(row["referral_source"]),
            "account_initial_seats": int(row["seats"]),
            "account_is_trial": 1 if bool(row["is_trial"]) else 0,
        }
