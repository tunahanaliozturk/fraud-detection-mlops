"""What `fraud train`, `fraud evaluate` and `fraud drift` do, end to end."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.drift import DriftRow, drift_report, psi, simulate_drift, status
from fraud_detection.evaluation import calibration, report
from fraud_detection.explain import Explainer
from fraud_detection.features import FEATURES, build_features
from fraud_detection.metrics import CostModel, cost_at, pr_auc, recall_at_precision
from fraud_detection.registry import Candidate, Registry, should_promote
from fraud_detection.training import (
    LIGHTGBM_PARAMS,
    Split,
    baseline_scores,
    choose_policy,
    labels,
    lightgbm_scores,
    matrix,
    time_split,
    train_baseline,
    train_lightgbm,
)


@dataclass(frozen=True)
class TrainingResult:
    version: str
    promoted: bool
    candidate_cost: float
    champion_cost: float | None
    report: str


def features_and_split(data: Path) -> Split:
    return time_split(build_features(pd.read_parquet(data)))


def train(
    data: Path, tracking_uri: str, cost: CostModel, pos_weight: float = 1.0, decline_precision: float = 0.9
) -> TrainingResult:
    split = features_and_split(data)
    booster = train_lightgbm(split.train, split.valid, pos_weight)
    baseline = train_baseline(split.train)

    y_valid, amount_valid = labels(split.valid), split.valid["amount"].to_numpy()
    valid_scores = lightgbm_scores(booster, split.valid)
    policy = choose_policy(y_valid, valid_scores, amount_valid, cost, decline_precision)
    baseline_policy = choose_policy(
        y_valid, baseline_scores(baseline, split.valid), amount_valid, cost, decline_precision
    )

    y, amount = labels(split.test), split.test["amount"].to_numpy()
    scores, base = lightgbm_scores(booster, split.test), baseline_scores(baseline, split.test)
    candidate_cost = cost_at(y, scores, amount, policy.review_threshold, cost).total

    # The champion is judged on today's test window and today's costs, with the threshold it shipped with.
    registry = Registry(tracking_uri)
    champion = registry.champion()
    champion_cost = None
    if champion is not None:
        champion_booster, champion_policy = registry.load(champion)
        champion_scores = lightgbm_scores(champion_booster, split.test)
        champion_cost = cost_at(y, champion_scores, amount, champion_policy.review_threshold, cost).total

    text = report(split, scores, base, policy, baseline_policy, Explainer(booster))
    version = registry.log(
        Candidate(
            booster=booster,
            policy=policy,
            params={
                **LIGHTGBM_PARAMS,
                "pos_weight": pos_weight,
                "review_cost": cost.review_cost,
                "friction_cost": cost.friction_cost,
                "decline_precision": decline_precision,
                "valid_start": str(split.valid_start),
                "test_start": str(split.test_start),
                "train_rows": len(split.train),
                "data": data.name,
            },
            metrics={
                "best_iteration": float(booster.best_iteration),
                "valid_pr_auc": pr_auc(y_valid, valid_scores),
                "valid_cost": cost_at(y_valid, valid_scores, amount_valid, policy.review_threshold, cost).total,
                "test_pr_auc": pr_auc(y, scores),
                "test_recall_at_p90": recall_at_precision(y, scores, 0.9),
                "test_cost": candidate_cost,
                "test_cost_flag_nothing": float(amount[y == 1].sum()),
                "baseline_test_pr_auc": pr_auc(y, base),
                "baseline_test_cost": cost_at(y, base, amount, baseline_policy.review_threshold, cost).total,
                "review_threshold": policy.review_threshold,
                "decline_threshold": policy.decline_threshold,
            },
            model_card="# Evaluation\n\n" + text,
            example=matrix(split.test.head(5)),
        )
    )
    promoted = should_promote(candidate_cost, champion_cost)
    if promoted:
        reason = "first model" if champion_cost is None else f"test cost {candidate_cost:.0f} < {champion_cost:.0f}"
        registry.promote(version, reason)
    return TrainingResult(str(version.version), promoted, candidate_cost, champion_cost, text)


def evaluate(data: Path, tracking_uri: str) -> str:
    registry = Registry(tracking_uri)
    champion = registry.champion()
    if champion is None:
        raise LookupError("no champion yet; run `fraud train` first")
    booster, policy = registry.load(champion)
    split = features_and_split(data)
    # The baseline is retrained rather than stored: it takes under a second and is deterministic.
    baseline = train_baseline(split.train)
    baseline_policy = choose_policy(
        labels(split.valid),
        baseline_scores(baseline, split.valid),
        split.valid["amount"].to_numpy(),
        policy.cost_model,
    )
    scores, base = lightgbm_scores(booster, split.test), baseline_scores(baseline, split.test)
    header = f"# Champion: version {champion.version}\n\n"
    return header + report(split, scores, base, policy, baseline_policy, Explainer(booster))


@dataclass(frozen=True)
class DriftResult:
    rows: list[DriftRow]
    text: str

    @property
    def alerting(self) -> list[str]:
        return [row.name for row in self.rows if row.status == "alert"]


def drift(data: Path, tracking_uri: str, scenario: str) -> DriftResult:
    """Compare the training window with the live (test) window, optionally after a simulated shift."""
    registry = Registry(tracking_uri)
    champion = registry.champion()
    if champion is None:
        raise LookupError("no champion yet; run `fraud train` first")
    booster, _ = registry.load(champion)

    transactions = pd.read_parquet(data)
    if scenario == "shifted":
        transactions = simulate_drift(transactions, time_split(transactions).test_start)
    split = time_split(build_features(transactions))

    reference, live = split.train, split.test
    rows = drift_report(reference, live, FEATURES)
    score_psi = psi(lightgbm_scores(booster, reference), lightgbm_scores(booster, live))
    rows.append(DriftRow("model score", score_psi, status(score_psi)))

    live_scores, y = lightgbm_scores(booster, live), labels(live)
    table = ["| input | PSI | status |", "|---|---|---|"]
    table += [f"| {row.name} | {row.psi:.3f} | {row.status} |" for row in rows]
    text = "\n".join(
        [
            f"Reference: training window, {len(reference):,} transactions. "
            f"Live: test window, {len(live):,} transactions. Scenario: {scenario}.",
            "",
            "\n".join(table),
            "",
            "PSI under 0.10 is stable, 0.10 to 0.25 is worth watching, 0.25 and above is an alert.",
            "",
            f"Calibration on the live window: mean score {np.mean(live_scores):.4f}, "
            f"observed fraud rate {np.mean(y):.4f}.",
            "",
            calibration(y, live_scores),
        ]
    )
    return DriftResult(rows, text + "\n")
