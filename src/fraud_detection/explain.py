"""Reason codes: why one transaction got its score, in words a customer or a model reviewer can read.

SHAP values come from shap.TreeExplainer, in log-odds. Each is one feature's push on this transaction's score away
from the average; they add up to the model's raw output. The largest pushes, either way, become the reasons.
For a LightGBM model TreeExplainer hands the work to LightGBM's own TreeSHAP (`pred_contrib=True`), see ADR 0005.
"""

from dataclasses import dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np
import shap  # type: ignore[import-untyped]  # shap ships no type information

from fraud_detection.features import FEATURES

Direction = Literal["increases_risk", "decreases_risk"]

# Feature -> (wording when it raised the score, wording when it lowered it).
WORDING: dict[str, tuple[str, str]] = {
    "amount_log": ("The amount is high", "The amount is low"),
    "hour": ("The time of day is unusual for card spending", "The time of day is ordinary"),
    "is_night": ("The purchase was made at night", "The purchase was made during the day"),
    "is_foreign": ("The merchant is abroad", "The merchant is in the cardholder's home country"),
    "country_is_new": (
        "The cardholder has not used the card in this country before",
        "The card has been used in this country before",
    ),
    "merchant_is_new": (
        "The cardholder has not paid this merchant before",
        "The cardholder has paid this merchant before",
    ),
    "device_is_new": ("The purchase came from a device not seen before", "The purchase came from a known device"),
    "txn_count_1h": ("Many transactions in the last hour", "Few transactions in the last hour"),
    "txn_count_24h": ("Many transactions in the last 24 hours", "Few transactions in the last 24 hours"),
    "amount_sum_24h_log": ("High total spend in the last 24 hours", "Low total spend in the last 24 hours"),
    "minutes_since_last": (
        "Unusual time gap since the previous transaction",
        "Ordinary time gap since the previous transaction",
    ),
    "amount_zscore": (
        "The amount is far from this cardholder's usual spending",
        "The amount is in line with this cardholder's usual spending",
    ),
}


@dataclass(frozen=True)
class Reason:
    feature: str
    direction: Direction
    contribution: float
    text: str


class Explainer:
    def __init__(self, booster: lgb.Booster) -> None:
        self._explainer = shap.TreeExplainer(booster)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """SHAP values in log-odds, one row per transaction and one column per feature."""
        # Called rather than .shap_values(): same numbers, without a warning about a return type we do not use.
        return np.asarray(self._explainer(x).values, dtype=float)


def reason_codes(contributions: np.ndarray, top: int = 3) -> list[Reason]:
    """The `top` features with the largest absolute contribution, largest first."""
    order = np.argsort(-np.abs(contributions), kind="stable")[:top]
    reasons = []
    for i in order:
        value = float(contributions[i])
        name = FEATURES[i]
        raised = value > 0
        reasons.append(
            Reason(
                feature=name,
                direction="increases_risk" if raised else "decreases_risk",
                contribution=round(value, 4),
                text=WORDING[name][0 if raised else 1],
            )
        )
    return reasons


def global_importance(contributions: np.ndarray) -> list[tuple[str, float]]:
    """Mean absolute SHAP value per feature, largest first."""
    mean_abs = np.abs(contributions).mean(axis=0)
    return sorted(zip(FEATURES, mean_abs.tolist(), strict=True), key=lambda pair: -pair[1])
