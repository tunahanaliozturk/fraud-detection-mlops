import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI

from fraud_detection.bundle import MODEL_FILE, POLICY_FILE, VERSION_FILE
from fraud_detection.features import build_features
from fraud_detection.generate import GeneratorConfig, generate
from fraud_detection.metrics import CostModel
from fraud_detection.training import choose_policy, labels, lightgbm_scores, time_split, train_lightgbm

# Small enough to train in a couple of seconds, large enough to have fraud in every window.
TINY = GeneratorConfig(customers=300, days=60, seed=3)


@pytest.fixture(scope="session")
def tiny_transactions() -> pd.DataFrame:
    return generate(TINY)


@pytest.fixture(scope="session")
def tiny_parquet(tiny_transactions: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "transactions.parquet"
    tiny_transactions.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def tiny_bundle(tiny_transactions: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model directory as `fraud export` writes it, trained on tiny data without MLflow."""
    split = time_split(build_features(tiny_transactions))
    booster = train_lightgbm(split.train, split.valid)
    policy = choose_policy(
        labels(split.valid), lightgbm_scores(booster, split.valid), split.valid["amount"].to_numpy(), CostModel()
    )
    directory = tmp_path_factory.mktemp("bundle")
    booster.save_model(str(directory / MODEL_FILE))
    (directory / POLICY_FILE).write_text(json.dumps(policy.as_dict()), encoding="utf-8")
    (directory / VERSION_FILE).write_text(json.dumps({"version": "test"}), encoding="utf-8")
    return directory


class AppClient:
    """Calls the ASGI app in-process through httpx, without starlette's TestClient (which now wants httpx2)."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def post(self, path: str, json: object) -> httpx.Response:
        return asyncio.run(self._request("POST", path, json))

    def get(self, path: str) -> httpx.Response:
        return asyncio.run(self._request("GET", path, None))

    async def _request(self, method: str, path: str, json: object) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, json=json)
