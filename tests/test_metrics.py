import numpy as np
import pytest

from fraud_detection.metrics import (
    CostModel,
    cost_at,
    cost_minimising_threshold,
    recall_at_precision,
    reliability,
    threshold_for_precision,
)

COST = CostModel(review_cost=5.0, friction_cost=10.0)


def test_cost_minimising_threshold_on_a_hand_worked_example() -> None:
    # Flag nothing: 130. Cut at 0.9: 130 - 100 + 5 = 35. At 0.8: + 15 = 50. At 0.7: 50 - 30 + 5 = 25.
    # At 0.6: 40. At 0.1: 55. So 0.7, at a cost of 25.
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    y = np.array([1, 0, 1, 0, 0])
    amount = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
    threshold = cost_minimising_threshold(y, score, amount, COST)
    assert threshold == 0.7
    assert cost_at(y, score, amount, threshold, COST).total == 25.0


def test_flagging_nothing_wins_when_reviews_cost_more_than_the_fraud() -> None:
    score = np.array([0.9, 0.5, 0.1])
    y = np.array([1, 0, 0])
    amount = np.array([3.0, 50.0, 50.0])
    assert cost_minimising_threshold(y, score, amount, COST) == float("inf")


def test_tied_scores_are_flagged_together() -> None:
    # Cutting between the two 0.8s is not possible, so the only cut-offs are 0.8 and 0.2.
    score = np.array([0.8, 0.8, 0.2])
    y = np.array([1, 0, 0])
    amount = np.array([100.0, 1.0, 1.0])
    assert cost_minimising_threshold(y, score, amount, COST) == 0.8


def test_the_one_pass_search_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    score = np.round(np.clip(rng.normal(0.2 + 0.4 * y, 0.2), 0, 1), 3)  # rounded, so there are ties
    amount = rng.lognormal(3, 1, size=2_000)
    found = cost_minimising_threshold(y, score, amount, COST)
    brute = min(np.unique(score), key=lambda t: cost_at(y, score, amount, float(t), COST).total)
    assert found == pytest.approx(float(brute))


def test_recall_at_precision() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 1, 0, 1])
    assert recall_at_precision(y, score, 1.0) == pytest.approx(2 / 3)
    assert recall_at_precision(y, score, 0.75) == 1.0


def test_threshold_for_precision_is_the_lowest_that_reaches_it() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 1, 0, 1])
    assert threshold_for_precision(y, score, 1.0) == 0.8
    assert threshold_for_precision(y, score, 0.75) == 0.6


def test_reliability_buckets_count_every_row_once() -> None:
    score = np.array([0.0005, 0.002, 0.02, 0.3, 0.9, 1.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    buckets = reliability(y, score)
    assert sum(b.count for b in buckets) == len(score)
    top = buckets[-1]
    assert (top.low, top.high, top.count, top.fraud_rate) == (0.75, 1.0, 2, 1.0)
