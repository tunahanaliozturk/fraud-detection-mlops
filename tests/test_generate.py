import pandas as pd

from fraud_detection.generate import PATTERNS, GeneratorConfig, generate

SMALL = GeneratorConfig(customers=200, days=30, seed=11)


def test_the_same_seed_gives_identical_rows() -> None:
    pd.testing.assert_frame_equal(generate(SMALL), generate(SMALL))


def test_a_different_seed_gives_different_rows() -> None:
    other = GeneratorConfig(customers=200, days=30, seed=12)
    assert not generate(SMALL).equals(generate(other))


def test_the_default_dataset_has_a_realistic_fraud_rate() -> None:
    rate = generate(GeneratorConfig())["is_fraud"].mean()
    assert 0.002 <= rate <= 0.01


def test_every_pattern_appears_and_only_fraud_carries_one(tiny_transactions: pd.DataFrame) -> None:
    fraud = tiny_transactions[tiny_transactions["is_fraud"] == 1]
    assert set(fraud["fraud_pattern"]) == set(PATTERNS)
    assert (tiny_transactions.loc[tiny_transactions["is_fraud"] == 0, "fraud_pattern"] == "").all()


def test_transaction_ids_follow_arrival_order(tiny_transactions: pd.DataFrame) -> None:
    assert tiny_transactions["transaction_id"].is_unique
    assert tiny_transactions["ts"].is_monotonic_increasing
    assert tiny_transactions["transaction_id"].is_monotonic_increasing
