"""Time-based split, the two models, and the decision policy chosen on the validation window."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.features import FEATURES
from fraud_detection.metrics import CostModel, cost_minimising_threshold, threshold_for_precision

Decision = Literal["approve", "review", "decline"]

LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}
MAX_ROUNDS = 1_000
EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    valid_start: pd.Timestamp
    test_start: pd.Timestamp


def time_split(frame: pd.DataFrame, warmup_days: int = 14, train: float = 0.6, valid: float = 0.2) -> Split:
    """Split by time, never at random: the model is always judged on days after the ones it learnt from.

    The first `warmup_days` only build customer history. Their features are computed from almost nothing (every
    merchant is new on day one), so training on them would teach the model a cold start that live traffic never has.
    """
    start = frame["ts"].min() + pd.Timedelta(days=warmup_days)
    span = frame["ts"].max() - start
    valid_start = start + span * train
    test_start = start + span * (train + valid)
    ts = frame["ts"]
    return Split(
        train=frame[(ts >= start) & (ts < valid_start)],
        valid=frame[(ts >= valid_start) & (ts < test_start)],
        test=frame[ts >= test_start],
        valid_start=valid_start,
        test_start=test_start,
    )


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].to_numpy(dtype=float)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["is_fraud"].to_numpy(dtype=int)


def train_lightgbm(train: pd.DataFrame, valid: pd.DataFrame, pos_weight: float = 1.0, seed: int = 42) -> lgb.Booster:
    """Gradient-boosted trees, stopped when validation PR-AUC stops improving.

    `pos_weight` stays at 1 by default: the cost-based threshold does the work of trading recall for precision, and
    unweighted training keeps the scores usable as probabilities (ADR 0004).
    """
    params = {**LIGHTGBM_PARAMS, "seed": seed, "scale_pos_weight": pos_weight, "metric": "average_precision"}
    train_set = lgb.Dataset(matrix(train), labels(train), feature_name=FEATURES, free_raw_data=False)
    valid_set = lgb.Dataset(matrix(valid), labels(valid), reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )


def train_baseline(train: pd.DataFrame) -> Pipeline:
    """Logistic regression on standardised features: the model a reviewer would ask us to beat."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000))
    model.fit(matrix(train), labels(train))
    return model


def baseline_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix(frame))[:, 1], dtype=float)


def lightgbm_scores(booster: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(booster.predict(matrix(frame)), dtype=float)


@dataclass(frozen=True)
class DecisionPolicy:
    """Score bands. Review costs an analyst's time; decline is kept for scores precise enough to act on unseen."""

    review_threshold: float
    decline_threshold: float
    review_cost: float
    friction_cost: float

    def decide(self, score: float) -> Decision:
        if score >= self.decline_threshold:
            return "decline"
        return "review" if score >= self.review_threshold else "approve"

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def cost_model(self) -> CostModel:
        return CostModel(self.review_cost, self.friction_cost)


def choose_policy(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, cost: CostModel, decline_precision: float = 0.9
) -> DecisionPolicy:
    review = cost_minimising_threshold(y, score, amount, cost)
    decline = max(threshold_for_precision(y, score, decline_precision), review)
    return DecisionPolicy(review, decline, cost.review_cost, cost.friction_cost)
