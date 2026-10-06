from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.bundle import load_bundle
from fraud_detection.explain import WORDING, Explainer, reason_codes
from fraud_detection.features import FEATURES, build_features
from fraud_detection.training import matrix


def test_reasons_are_the_largest_contributions_with_matching_direction_and_wording() -> None:
    contributions = np.zeros(len(FEATURES))
    contributions[FEATURES.index("amount_zscore")] = 2.0
    contributions[FEATURES.index("device_is_new")] = -3.0
    contributions[FEATURES.index("hour")] = 0.5
    contributions[FEATURES.index("is_night")] = 0.1

    reasons = reason_codes(contributions)

    assert [r.feature for r in reasons] == ["device_is_new", "amount_zscore", "hour"]
    assert [r.direction for r in reasons] == ["decreases_risk", "increases_risk", "increases_risk"]
    assert reasons[0].text == WORDING["device_is_new"][1]
    assert reasons[1].text == WORDING["amount_zscore"][0]


def test_every_feature_has_wording() -> None:
    assert set(WORDING) == set(FEATURES)


def test_shap_values_add_up_to_the_model_output(tiny_bundle: Path, tiny_transactions: pd.DataFrame) -> None:
    booster = load_bundle(tiny_bundle).booster
    x = matrix(build_features(tiny_transactions).tail(200))
    contributions = Explainer(booster).contributions(x)
    raw = np.asarray(booster.predict(x, raw_score=True))
    expected_value = np.asarray(booster.predict(x[:1], pred_contrib=True))[0, -1]
    assert contributions.sum(axis=1) + expected_value == pytest.approx(raw, abs=1e-6)
