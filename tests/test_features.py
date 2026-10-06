"""Point-in-time correctness: a feature may only use transactions strictly before the one it describes."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import FEATURES, NO_PREVIOUS_MINUTES, build_features


@pytest.fixture(scope="module")
def features(tiny_transactions: pd.DataFrame) -> pd.DataFrame:
    return build_features(tiny_transactions)


def _cutoff(frame: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(frame["ts"].iloc[len(frame) // 2])


def test_deleting_future_rows_does_not_change_past_features(
    tiny_transactions: pd.DataFrame, features: pd.DataFrame
) -> None:
    cutoff = _cutoff(tiny_transactions)
    past_only = build_features(tiny_transactions[tiny_transactions["ts"] < cutoff])
    full_past = features[features["ts"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_only[FEATURES], full_past[FEATURES])


def test_rewriting_future_rows_does_not_change_past_features(
    tiny_transactions: pd.DataFrame, features: pd.DataFrame
) -> None:
    cutoff = _cutoff(tiny_transactions)
    future = tiny_transactions["ts"] >= cutoff
    rewritten = tiny_transactions.copy()
    rewritten.loc[future, "amount"] *= 100
    rewritten.loc[future, "device_id"] = "attacker"
    rewritten.loc[future, "merchant_country"] = "US"
    again = build_features(rewritten)
    past = features["ts"] < cutoff
    pd.testing.assert_frame_equal(again.loc[past, FEATURES], features.loc[past, FEATURES])
    # And the rewrite did reach the future rows, so the comparison above is not vacuous.
    assert not again.loc[~past, "amount_log"].equals(features.loc[~past, "amount_log"])


def test_row_order_of_the_input_does_not_matter(tiny_transactions: pd.DataFrame, features: pd.DataFrame) -> None:
    shuffled = tiny_transactions.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(shuffled), features)


def test_a_transaction_does_not_see_itself() -> None:
    """A customer's very first transaction has an empty history, whatever comes after it."""
    frame = _frame([("c1", "m1", "d1", 0, 20.0), ("c1", "m1", "d1", 60, 20.0)])
    first = build_features(frame).iloc[0]
    assert first["txn_count_24h"] == 0
    assert first["merchant_is_new"] == 1
    assert first["minutes_since_last"] == NO_PREVIOUS_MINUTES


def test_velocity_and_novelty_on_a_hand_built_history() -> None:
    hour = 3_600
    frame = _frame(
        [
            ("c1", "m1", "d1", 0, 10.0),
            ("c1", "m1", "d1", 23 * hour, 30.0),
            ("c1", "m2", "d1", 23 * hour + 1_800, 50.0),
            ("c1", "m1", "d2", 25 * hour, 70.0),
            ("c2", "m9", "d9", 25 * hour, 5.0),  # another customer never counts towards c1
        ]
    )
    out = build_features(frame).set_index("transaction_id")

    third = out.loc[3]
    assert third["txn_count_1h"] == 1  # the one 30 minutes earlier
    assert third["txn_count_24h"] == 2  # both earlier ones, the first 23.5 hours back
    assert third["minutes_since_last"] == 30
    assert third["merchant_is_new"] == 1

    fourth = out.loc[4]
    assert fourth["txn_count_1h"] == 0
    assert fourth["txn_count_24h"] == 2  # the first transaction is now more than 24 hours old
    assert fourth["amount_sum_24h_log"] == pytest.approx(np.log1p(80.0))
    assert fourth["merchant_is_new"] == 0
    assert fourth["device_is_new"] == 1


def _frame(rows: list[tuple[str, str, str, int, float]]) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "transaction_id": np.arange(1, len(rows) + 1),
            "ts": [start + pd.Timedelta(seconds=r[3]) for r in rows],
            "customer_id": [r[0] for r in rows],
            "customer_country": "NL",
            "merchant_id": [r[1] for r in rows],
            "merchant_category": "grocery",
            "merchant_country": "NL",
            "amount": [r[4] for r in rows],
            "device_id": [r[2] for r in rows],
            "is_fraud": 0,
            "fraud_pattern": "",
        }
    )
