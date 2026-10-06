"""MLflow tracking and model registry on a local SQLite file. No server to run.

Each training run logs its parameters, metrics, the LightGBM model (native text format, not a pickle), the decision
policy and a model card, and registers a new model version. The `champion` alias is the only thing the service
deploys, and it moves only when a candidate beats the current champion on expected cost (ADR 0006).
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from fraud_detection.bundle import CARD_FILE, MODEL_FILE, POLICY_FILE, VERSION_FILE
from fraud_detection.training import DecisionPolicy

MODEL_NAME = "card-fraud"
CHAMPION = "champion"
EXPERIMENT = "card-fraud"

# uv environments have no pip, and MLflow warns on every logged model that it cannot record pip's version. The model's
# own requirement (lightgbm, pinned) is passed explicitly, so the warning says nothing about this model.
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Candidate:
    booster: lgb.Booster
    policy: DecisionPolicy
    params: dict[str, Any]
    metrics: dict[str, float]
    model_card: str
    example: np.ndarray  # a few feature rows, recorded as the model's input signature


class Registry:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self._client = mlflow.MlflowClient()

    def log(self, candidate: Candidate) -> ModelVersion:
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params(candidate.params)
            mlflow.log_metrics(candidate.metrics)
            info = mlflow.lightgbm.log_model(
                candidate.booster,
                name="model",
                signature=infer_signature(candidate.example, candidate.booster.predict(candidate.example)),
                pip_requirements=[f"lightgbm=={lgb.__version__}"],
            )
            mlflow.log_dict(candidate.policy.as_dict(), POLICY_FILE)
            mlflow.log_text(candidate.model_card, CARD_FILE)
        return mlflow.register_model(info.model_uri, MODEL_NAME)

    def champion(self) -> ModelVersion | None:
        try:
            return self._client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
        except MlflowException:
            return None

    def load(self, version: ModelVersion) -> tuple[lgb.Booster, DecisionPolicy]:
        booster = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{version.version}")
        policy = DecisionPolicy(**mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{POLICY_FILE}"))
        return booster, policy

    def promote(self, version: ModelVersion, reason: str) -> None:
        self._client.set_registered_model_alias(MODEL_NAME, CHAMPION, version.version)
        self._client.set_model_version_tag(MODEL_NAME, version.version, "promotion", reason)

    def export_champion(self, out_dir: Path) -> str:
        """Copy the champion's model file, policy and model card to a plain directory the service can read."""
        version = self.champion()
        if version is None:
            raise LookupError(f"no model carries the '{CHAMPION}' alias yet; run `fraud train` first")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version.version}"))
        shutil.copyfile(model_dir / MODEL_FILE, out_dir / MODEL_FILE)
        for name in (POLICY_FILE, CARD_FILE):
            mlflow.artifacts.download_artifacts(f"runs:/{version.run_id}/{name}", dst_path=str(out_dir))
        (out_dir / VERSION_FILE).write_text(
            json.dumps({"model": MODEL_NAME, "version": version.version, "run_id": version.run_id}, indent=2),
            encoding="utf-8",
        )
        return str(version.version)


def should_promote(candidate_cost: float, champion_cost: float | None) -> bool:
    """Strictly cheaper, so retraining an identical model does not churn the alias."""
    return champion_cost is None or candidate_cost < champion_cost
