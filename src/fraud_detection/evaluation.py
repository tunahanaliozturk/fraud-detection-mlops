"""The evaluation report on the test window, as Markdown. The same text becomes the model card artifact in MLflow."""

import numpy as np
import pandas as pd

from fraud_detection.explain import Explainer, global_importance
from fraud_detection.metrics import cost_at, pr_auc, recall_at_precision, reliability
from fraud_detection.training import DecisionPolicy, Split, labels, matrix

CURVE_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
AMOUNT_BANDS = [0.0, 10.0, 100.0, 500.0, float("inf")]
SHAP_SAMPLE = 10_000


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _eur(value: float) -> str:
    return f"{value:,.0f}"


def headline(
    y: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    amount: np.ndarray,
    policy: DecisionPolicy,
    baseline_policy: DecisionPolicy,
) -> str:
    cost = policy.cost_model
    nothing = cost_at(y, model, amount, float("inf"), cost).total
    rows = []
    for name, score, chosen in (("LightGBM", model, policy), ("Logistic regression", baseline, baseline_policy)):
        total = cost_at(y, score, amount, chosen.review_threshold, cost).total
        rows.append(
            [
                name,
                f"{pr_auc(y, score):.3f}",
                f"{recall_at_precision(y, score, 0.9):.3f}",
                f"{recall_at_precision(y, score, 0.5):.3f}",
                f"{chosen.review_threshold:.4f}",
                _eur(total),
                f"{1 - total / nothing:.1%}",
            ]
        )
    header = [
        "model",
        "PR-AUC",
        "recall @ 90% precision",
        "recall @ 50% precision",
        "threshold (from validation)",
        "cost, EUR",
        "saving vs flag nothing",
    ]
    return _table(header, rows) + f"\n\nFlagging nothing costs EUR {_eur(nothing)}: every fraud goes through."


def cost_curve(y: np.ndarray, score: np.ndarray, amount: np.ndarray, policy: DecisionPolicy) -> str:
    rows = []
    for threshold in sorted({*CURVE_THRESHOLDS, policy.review_threshold}):
        c = cost_at(y, score, amount, threshold, policy.cost_model)
        mark = " (chosen)" if threshold == policy.review_threshold else ""
        rows.append(
            [
                f"{threshold:.4f}{mark}",
                str(c.flagged),
                str(c.caught),
                str(c.missed),
                _eur(c.missed_amount),
                str(c.false_alarms),
                _eur(c.total),
            ]
        )
    header = ["threshold", "flagged", "fraud caught", "fraud missed", "missed EUR", "false alarms", "total cost EUR"]
    return _table(header, rows)


def bands(test: pd.DataFrame, score: np.ndarray, policy: DecisionPolicy) -> str:
    decisions = np.array([policy.decide(s) for s in score])
    rows = []
    for band in ("approve", "review", "decline"):
        mask = decisions == band
        frauds = int(test["is_fraud"].to_numpy()[mask].sum())
        rate = f"{frauds / mask.sum():.2%}" if mask.any() else "n/a"
        rows.append([band, str(int(mask.sum())), str(frauds), rate])
    return _table(["decision", "transactions", "fraud", "fraud rate"], rows)


def by_pattern(test: pd.DataFrame, score: np.ndarray, policy: DecisionPolicy) -> str:
    rows = []
    fraud = test.assign(score=score)[test["is_fraud"] == 1]
    for pattern, group in fraud.groupby("fraud_pattern"):
        flagged = group["score"] >= policy.review_threshold
        rows.append([str(pattern), str(len(group)), f"{flagged.mean():.1%}"])
    return _table(["fraud pattern", "transactions", "recall at the review threshold"], rows)


def segments(test: pd.DataFrame, score: np.ndarray, policy: DecisionPolicy) -> str:
    keyed = test.assign(
        score=score,
        amount_band=pd.cut(
            test["amount"], AMOUNT_BANDS, right=False, labels=["< 10", "10 to 100", "100 to 500", ">= 500"]
        ),
        merchant_location=np.where(test["is_foreign"] == 1, "abroad", "home country"),
    )
    rows = []
    for column in ("amount_band", "merchant_location", "customer_country"):
        for value, group in keyed.groupby(column, observed=True):
            y = labels(group)
            s = group["score"].to_numpy()
            flagged = s >= policy.review_threshold
            fraud = y == 1
            rows.append(
                [
                    f"{column} = {value}",
                    str(len(group)),
                    str(int(fraud.sum())),
                    f"{pr_auc(y, s):.3f}" if 0 < fraud.sum() < len(y) else "n/a",
                    f"{flagged[fraud].mean():.1%}" if fraud.any() else "n/a",
                    f"{flagged[~fraud].mean():.2%}",
                ]
            )
    header = ["segment", "transactions", "fraud", "PR-AUC", "recall", "false positive rate"]
    return _table(header, rows)


def calibration(y: np.ndarray, score: np.ndarray) -> str:
    rows = [
        [f"{b.low:.4f} to {b.high:.4f}", str(b.count), f"{b.mean_score:.4f}", f"{b.fraud_rate:.4f}"]
        for b in reliability(y, score)
    ]
    return _table(["score range", "transactions", "mean score", "observed fraud rate"], rows)


def importance(explainer: Explainer, test: pd.DataFrame, seed: int = 0) -> str:
    sample = test.sample(n=min(SHAP_SAMPLE, len(test)), random_state=seed)
    ranked = global_importance(explainer.contributions(matrix(sample)))
    return _table(["feature", "mean abs SHAP (log-odds)"], [[name, f"{value:.3f}"] for name, value in ranked])


def report(
    split: Split,
    model: np.ndarray,
    baseline: np.ndarray,
    policy: DecisionPolicy,
    baseline_policy: DecisionPolicy,
    explainer: Explainer,
) -> str:
    test = split.test
    y, amount = labels(test), test["amount"].to_numpy()
    parts = [
        f"Test window: {split.test_start:%Y-%m-%d %H:%M} to {test['ts'].max():%Y-%m-%d}, {len(test):,} transactions, "
        f"{int(y.sum())} fraud ({y.mean():.2%}). Thresholds were chosen on the validation window "
        f"({split.valid_start:%Y-%m-%d} onwards) and applied unchanged here. Costs: review EUR "
        f"{policy.review_cost:g}, false alarm EUR {policy.cost_model.false_alarm_cost:g}, missed fraud its amount.",
        "## Model against baseline\n\n" + headline(y, model, baseline, amount, policy, baseline_policy),
        "## Cost curve (LightGBM)\n\n" + cost_curve(y, model, amount, policy),
        f"## Decision bands\n\nReview from {policy.review_threshold:.4f}, decline from "
        f"{policy.decline_threshold:.4f}.\n\n" + bands(test, model, policy),
        "## Recall by fraud pattern\n\n" + by_pattern(test, model, policy),
        "## Segments\n\n" + segments(test, model, policy),
        "## Calibration\n\n" + calibration(y, model),
        "## Global importance\n\n" + importance(explainer, test),
    ]
    return "\n\n".join(parts) + "\n"
