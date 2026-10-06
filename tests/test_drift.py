import math

import numpy as np
import pytest

from fraud_detection.drift import psi, status


def test_psi_on_a_hand_worked_example() -> None:
    reference = np.array([0] * 50 + [1] * 50)
    live = np.array([0] * 25 + [1] * 75)
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    assert psi(reference, live) == pytest.approx(expected)  # 0.2747


def test_the_same_distribution_is_stable() -> None:
    rng = np.random.default_rng(0)
    value = psi(rng.normal(0, 1, 50_000), rng.normal(0, 1, 50_000))
    assert value < 0.01
    assert status(value) == "stable"


def test_a_one_standard_deviation_shift_raises_an_alert() -> None:
    rng = np.random.default_rng(0)
    assert status(psi(rng.normal(0, 1, 50_000), rng.normal(1, 1, 50_000))) == "alert"


def test_a_flag_that_changes_rate_is_not_hidden_by_binning() -> None:
    # Quantile edges for a 0/1 feature collapse to one bin; one bin per value keeps the shift visible.
    reference = np.array([0] * 95 + [1] * 5)
    live = np.array([0] * 70 + [1] * 30)
    assert status(psi(reference, live)) == "alert"


def test_status_bands() -> None:
    assert [status(v) for v in (0.05, 0.1, 0.2, 0.25)] == ["stable", "watch", "watch", "alert"]
