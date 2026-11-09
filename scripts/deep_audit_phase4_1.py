"""Deep audit for Phase 4.1:
- Accounts with churn events and active subscriptions
- Accounts without active subscriptions and no churn
- Subscription start date vs signup date lag
- Boundary condition analysis (events exactly on T_obs or T_obs + 60d)
- Repeat churners during the prediction window
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd


def deep_audit():
    data_dir = Path("data/raw/ravenstack")
    accounts = pd.read_csv(data_dir / "ravenstack_accounts.csv")
    subscriptions = pd.read_csv(data_dir / "ravenstack_subscriptions.csv")
    churn_events = pd.read_csv(data_dir / "ravenstack_churn_events.csv")

    accounts["parsed_signup"] = pd.to_datetime(accounts["signup_date"]).dt.date
    subscriptions["parsed_start"] = pd.to_datetime(subscriptions["start_date"]).dt.date
    subscriptions["parsed_end"] = pd.to_datetime(subscriptions["end_date"]).dt.date
    churn_events["parsed_churn"] = pd.to_datetime(churn_events["churn_date"]).dt.date

    # 1. Investigate signup vs first subscription start lag
    first_sub = subscriptions.groupby("account_id")["parsed_start"].min().reset_index()
    acc_sub = accounts.merge(first_sub, on="account_id", how="left")
    acc_sub["lag_days"] = acc_sub.apply(
        lambda r: (r["parsed_start"] - r["parsed_signup"]).days, axis=1
    )

    print("=" * 80)
    print("1. ACCOUNT SIGNUP VS FIRST SUBSCRIPTION START LAG")
    print("=" * 80)
    print(
        f"Accounts with first_sub == signup_date: {(acc_sub['lag_days'] == 0).sum()} ({(acc_sub['lag_days'] == 0).mean() * 100:.1f}%)"
    )
    print(
        f"Accounts with first_sub > signup_date:  {(acc_sub['lag_days'] > 0).sum()} ({(acc_sub['lag_days'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"Max lag: {acc_sub['lag_days'].max()} days | Median lag for lagged: {acc_sub[acc_sub['lag_days'] > 0]['lag_days'].median()} days"
    )

    # 2. Investigate churn events vs subscription end dates
    print("\n" + "=" * 80)
    print("2. CHURN EVENTS VS SUBSCRIPTIONS RELATIONSHIP")
    print("=" * 80)
    # For each churn event, what was the status of the account's subscriptions on churn_date?
    churn_analysis = []
    for _, crow in churn_events.iterrows():
        acc = crow["account_id"]
        c_date = crow["parsed_churn"]

        # Subscriptions for this account
        subs = subscriptions[subscriptions["account_id"] == acc]

        # Subs active on churn_date: start <= c_date and (end is null or end >= c_date)
        active_on_churn = subs[
            (subs["parsed_start"] <= c_date)
            & (subs["parsed_end"].isnull() | (subs["parsed_end"] >= c_date))
        ]

        # Subs ending exactly on churn_date
        ending_on_churn = subs[subs["parsed_end"] == c_date]

        # Subs starting after churn_date
        starting_after = subs[subs["parsed_start"] > c_date]

        churn_analysis.append(
            {
                "churn_id": crow["churn_event_id"],
                "account_id": acc,
                "churn_date": c_date,
                "is_reactivation": crow["is_reactivation"],
                "total_subs": len(subs),
                "active_on_churn": len(active_on_churn),
                "ending_on_churn": len(ending_on_churn),
                "subs_starting_after": len(starting_after),
            }
        )
    cadf = pd.DataFrame(churn_analysis)
    print(f"Total Churn Events: {len(cadf)}")
    print(
        f"Churn events where account had active subscriptions ON churn_date: {(cadf['active_on_churn'] > 0).sum()} ({(cadf['active_on_churn'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"Churn events where a subscription ended EXACTLY on churn_date:     {(cadf['ending_on_churn'] > 0).sum()}"
    )
    print(
        f"Churn events where account had NEW subscription starting AFTER churn_date: {(cadf['subs_starting_after'] > 0).sum()}"
    )

    # 3. Look at accounts that churned: did their subscriptions actually end?
    print("\n" + "=" * 80)
    print("3. DO SUBSCRIPTIONS END WHEN AN ACCOUNT CHURNS?")
    print("=" * 80)
    # Check how many subscriptions have end_date not null in the whole dataset
    total_subs = len(subscriptions)
    ended_subs = subscriptions["parsed_end"].notnull().sum()
    null_end_subs = subscriptions["parsed_end"].isnull().sum()
    print(f"Total subscriptions: {total_subs}")
    print(
        f"Ended subscriptions (end_date not null): {ended_subs} ({ended_subs / total_subs * 100:.1f}%)"
    )
    print(
        f"Open-ended subscriptions (end_date is null): {null_end_subs} ({null_end_subs / total_subs * 100:.1f}%)"
    )

    # Check subscriptions for accounts that have churn_flag == True in accounts table
    acc_churn_true = set(accounts[accounts["churn_flag"]]["account_id"])
    subs_churn_accs = subscriptions[subscriptions["account_id"].isin(acc_churn_true)]
    print("\nFor 110 accounts with accounts.churn_flag == True:")
    print(f"  Total subscriptions: {len(subs_churn_accs)}")
    print(
        f"  Subscriptions with end_date IS NULL: {subs_churn_accs['parsed_end'].isnull().sum()} ({subs_churn_accs['parsed_end'].isnull().mean() * 100:.1f}%)"
    )

    # 4. Multi-subscription behavior per account
    sub_per_acc = subscriptions.groupby("account_id").size()
    print(
        f"\nSubscriptions per account: min={sub_per_acc.min()}, median={sub_per_acc.median()}, max={sub_per_acc.max()}"
    )

    # 5. Boundary testing: events exactly on T_obs vs T_obs + 60d
    print("\n" + "=" * 80)
    print("4. BOUNDARY ANALYSIS ON PREDICTION WINDOWS")
    print("=" * 80)
    dates = [d.date() for d in pd.date_range(start="2023-04-01", end="2024-11-01", freq="MS")]
    exact_on_tobs = 0
    exact_on_window_end = 0
    for snap in dates:
        window_end = snap + timedelta(days=60)
        on_snap = len(churn_events[churn_events["parsed_churn"] == snap])
        on_end = len(churn_events[churn_events["parsed_churn"] == window_end])
        exact_on_tobs += on_snap
        exact_on_window_end += on_end
        if on_snap > 0 or on_end > 0:
            print(
                f"  Date {snap}: {on_snap} churns exactly on T_obs, {on_end} churns exactly on T_obs + 60d ({window_end})"
            )
    print(f"Total churn events falling EXACTLY on any T_obs: {exact_on_tobs}")
    print(f"Total churn events falling EXACTLY on any T_obs + 60d: {exact_on_window_end}")

    # 6. Repeat churn events within the same 60-day prediction window
    print("\n" + "=" * 80)
    print("5. REPEAT CHURN EVENTS WITHIN A SINGLE 60-DAY WINDOW")
    print("=" * 80)
    repeat_in_window = 0
    for snap in dates:
        window_end = snap + timedelta(days=60)
        in_win = churn_events[
            (churn_events["parsed_churn"] > snap) & (churn_events["parsed_churn"] <= window_end)
        ]
        counts = in_win["account_id"].value_counts()
        mult = counts[counts > 1]
        if len(mult) > 0:
            print(
                f"  Snapshot {snap}: {len(mult)} accounts have >1 churn event in window: {dict(mult)}"
            )
            repeat_in_window += len(mult)
    print(
        f"Total snapshot-account instances with >1 churn event in the 60-day window: {repeat_in_window}"
    )


if __name__ == "__main__":
    deep_audit()
