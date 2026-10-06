from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fraud_detection.bundle import load_bundle
from fraud_detection.service import Scorer, create_app
from tests.conftest import AppClient

VALID: dict[str, Any] = {
    "transaction_id": "t-1",
    "timestamp": "2026-03-01T03:15:00+01:00",
    "customer_id": "c00001",
    "customer_country": "NL",
    "merchant_id": "m00042",
    "merchant_country": "US",
    "amount": 950.0,
    "device_id": "new-phone",
}


@pytest.fixture
def scorer(tiny_bundle: Path, tiny_transactions: pd.DataFrame) -> Scorer:
    return Scorer(load_bundle(tiny_bundle), tiny_transactions)


@pytest.fixture
def client(scorer: Scorer) -> AppClient:
    return AppClient(create_app(scorer))


def test_a_valid_request_gets_a_score_a_decision_and_reasons(client: AppClient) -> None:
    response = client.post("/score", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"transaction_id", "score", "decision", "reasons", "model_version"}
    assert body["transaction_id"] == "t-1"
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["model_version"] == "test"
    assert len(body["reasons"]) == 3
    assert {"feature", "direction", "contribution", "text"} == set(body["reasons"][0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount", -5),
        ("amount", 0),
        ("amount", "lots"),
        ("customer_country", "Netherlands"),
        ("merchant_country", "us"),
        ("timestamp", "2026-03-01T03:15:00"),  # no offset, so the local hour is unknown
        ("customer_id", "c1; DROP TABLE"),
        ("device_id", ""),
    ],
)
def test_invalid_input_is_refused_with_422(client: AppClient, field: str, value: object) -> None:
    assert client.post("/score", json={**VALID, field: value}).status_code == 422


def test_missing_and_unknown_fields_are_refused_with_422(client: AppClient) -> None:
    missing = {k: v for k, v in VALID.items() if k != "amount"}
    assert client.post("/score", json=missing).status_code == 422
    assert client.post("/score", json={**VALID, "is_fraud": 0}).status_code == 422


def test_reason_directions_match_the_sign_of_the_shap_values(client: AppClient) -> None:
    for i, amount in enumerate([4.0, 60.0, 2_500.0]):
        body = client.post("/score", json={**VALID, "transaction_id": f"s-{i}", "amount": amount}).json()
        assert body["reasons"]
        magnitudes = [abs(r["contribution"]) for r in body["reasons"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for reason in body["reasons"]:
            positive = reason["contribution"] > 0
            assert reason["direction"] == ("increases_risk" if positive else "decreases_risk")


def test_a_retried_request_gets_the_same_answer_and_counts_once(
    tiny_bundle: Path, tiny_transactions: pd.DataFrame
) -> None:
    once = AppClient(create_app(Scorer(load_bundle(tiny_bundle), tiny_transactions)))
    twice = AppClient(create_app(Scorer(load_bundle(tiny_bundle), tiny_transactions)))
    request = {**VALID, "transaction_id": "retry"}

    assert once.post("/score", json=request).json() == twice.post("/score", json=request).json()
    assert twice.post("/score", json=request).json() == once.post("/score", json=request).json()

    # If the retry had been added to the history again, velocity features for the next transaction would differ.
    later = {**VALID, "transaction_id": "next", "timestamp": "2026-03-01T03:16:00+01:00"}
    assert once.post("/score", json=later).json() == twice.post("/score", json=later).json()


def test_velocity_builds_up_across_requests(scorer: Scorer) -> None:
    client = AppClient(create_app(scorer))
    burst = [
        {**VALID, "transaction_id": f"b-{i}", "amount": 1.5, "timestamp": f"2026-03-01T03:{20 + i:02d}:00+01:00"}
        for i in range(8)
    ]
    scores = [client.post("/score", json=t).json()["score"] for t in burst]
    assert scores[-1] > scores[0]


def test_health_reports_the_model_version(client: AppClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": "test"}
