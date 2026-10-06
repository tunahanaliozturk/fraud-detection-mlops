"""The scoring service: POST /score returns a score, a decision band and the reasons behind it."""

import threading
from collections import OrderedDict
from datetime import UTC
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from fraud_detection.bundle import Bundle, load_bundle
from fraud_detection.explain import Explainer, reason_codes
from fraud_detection.features import FeatureStore, Transaction, arrival_order

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
REMEMBERED_RESPONSES = 100_000


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: Identifier
    timestamp: AwareDatetime = Field(description="Wall-clock time in the cardholder's local offset")
    customer_id: Identifier
    customer_country: CountryCode
    merchant_id: Identifier
    merchant_country: CountryCode
    amount: float = Field(gt=0, le=1_000_000, allow_inf_nan=False, description="Euros")
    device_id: Identifier


class ReasonOut(BaseModel):
    feature: str
    direction: Literal["increases_risk", "decreases_risk"]
    contribution: float = Field(description="SHAP value in log-odds")
    text: str


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float = Field(ge=0, le=1)
    decision: Literal["approve", "review", "decline"]
    reasons: list[ReasonOut]
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


class Scorer:
    """Holds the model, its explainer and the customer histories. Built once at start-up."""

    def __init__(self, bundle: Bundle, history: pd.DataFrame | None = None) -> None:
        self.version = bundle.version
        self._booster = bundle.booster
        self._policy = bundle.policy
        self._explainer = Explainer(bundle.booster)
        self._store = FeatureStore()
        if history is not None:
            self._store.load(arrival_order(history))
        # A retried request must get the same answer and must not count twice towards velocity features.
        self._responses: OrderedDict[str, ScoreResponse] = OrderedDict()
        # ponytail: one lock serialises scoring (a few ms each); per-customer locks if one process must go faster.
        self._lock = threading.Lock()

    def score(self, request: ScoreRequest) -> ScoreResponse:
        with self._lock:
            if (previous := self._responses.get(request.transaction_id)) is not None:
                return previous
            txn = _transaction(request)
            x = np.array([self._store.features(txn)])
            self._store.add(txn)
            score = float(self._booster.predict(x)[0])
            reasons = reason_codes(self._explainer.contributions(x)[0])
            response = ScoreResponse(
                transaction_id=request.transaction_id,
                score=score,
                decision=self._policy.decide(score),
                reasons=[ReasonOut(**vars(r)) for r in reasons],
                model_version=self.version,
            )
            self._responses[request.transaction_id] = response
            if len(self._responses) > REMEMBERED_RESPONSES:
                self._responses.popitem(last=False)
            return response


def _transaction(request: ScoreRequest) -> Transaction:
    # Features read the local wall clock, so the offset is dropped rather than converted to UTC.
    local = request.timestamp.replace(tzinfo=UTC)
    return Transaction(
        ts=int(local.timestamp()),
        customer_id=request.customer_id,
        customer_country=request.customer_country,
        merchant_id=request.merchant_id,
        merchant_country=request.merchant_country,
        amount=request.amount,
        device_id=request.device_id,
    )


def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="Card fraud scoring", version="0.1.0")

    # Plain `def` handlers: scoring is CPU work, so FastAPI runs it on its worker threads, not the event loop.
    @app.post("/score")
    def score(request: ScoreRequest) -> ScoreResponse:
        return scorer.score(request)

    @app.get("/health")
    def health() -> Health:
        return Health(status="ok", model_version=scorer.version)

    return app


def load_scorer(model_dir: Path, history: Path | None) -> Scorer:
    frame = pd.read_parquet(history) if history is not None else None
    return Scorer(load_bundle(model_dir), frame)
