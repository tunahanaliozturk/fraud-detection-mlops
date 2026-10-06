"""The directory a deployed model lives in: the files `fraud export` writes and the service reads.

Keeping the service off MLflow at run time means it starts in seconds, and the image holds no tracking client.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

from fraud_detection.training import DecisionPolicy

MODEL_FILE = "model.lgb"  # LightGBM's own text format: loading it cannot run code, unlike a pickle
POLICY_FILE = "policy.json"
CARD_FILE = "model_card.md"
VERSION_FILE = "version.json"


@dataclass(frozen=True)
class Bundle:
    booster: lgb.Booster
    policy: DecisionPolicy
    version: str


def load_bundle(directory: Path) -> Bundle:
    booster = lgb.Booster(model_file=str(directory / MODEL_FILE))
    policy = DecisionPolicy(**json.loads((directory / POLICY_FILE).read_text(encoding="utf-8")))
    version = json.loads((directory / VERSION_FILE).read_text(encoding="utf-8"))["version"]
    return Bundle(booster, policy, str(version))
