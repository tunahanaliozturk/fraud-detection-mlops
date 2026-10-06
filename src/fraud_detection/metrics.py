"""Metrics that mean something when fraud is a fraction of a percent of transactions.

Accuracy is not here on purpose: flagging nothing scores above 99.5% on this data.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

FloatArray = np.ndarray


@dataclass(frozen=True)
class CostModel:
    """What each outcome costs, in euros. A missed fraud costs its amount; caught fraud still costs a review."""

    review_cost: float = 5.0  # analyst time to look at one flagged transaction
    friction_cost: float = 10.0  # a genuine customer whose card was stopped: support call, abandoned purchase, churn

    @property
    def false_alarm_cost(self) -> float:
        return self.review_cost + self.friction_cost


@dataclass(frozen=True)
class CostBreakdown:
    threshold: float
    total: float
    flagged: int
    caught: int
    missed: int
    missed_amount: float
    false_alarms: int


def pr_auc(y: FloatArray, score: FloatArray) -> float:
    return float(average_precision_score(y, score))


def recall_at_precision(y: FloatArray, score: FloatArray, precision: float) -> float:
    """The best recall available at any threshold whose precision is at least `precision`."""
    precisions, recalls, _ = precision_recall_curve(y, score)
    reachable = recalls[precisions >= precision]
    return float(reachable.max()) if len(reachable) else 0.0


def cost_at(y: FloatArray, score: FloatArray, amount: FloatArray, threshold: float, cost: CostModel) -> CostBreakdown:
    flagged = score >= threshold
    fraud = y == 1
    caught = flagged & fraud
    missed = ~flagged & fraud
    false_alarms = flagged & ~fraud
    missed_amount = float(amount[missed].sum())
    total = missed_amount + caught.sum() * cost.review_cost + false_alarms.sum() * cost.false_alarm_cost
    return CostBreakdown(
        threshold=threshold,
        total=float(total),
        flagged=int(flagged.sum()),
        caught=int(caught.sum()),
        missed=int(missed.sum()),
        missed_amount=missed_amount,
        false_alarms=int(false_alarms.sum()),
    )


def cost_minimising_threshold(y: FloatArray, score: FloatArray, amount: FloatArray, cost: CostModel) -> float:
    """The threshold with the lowest total cost, trying every distinct score as a cut-off.

    Sorting by score once makes this one pass: flagging the top k rows costs the fraud amount not yet caught, plus
    a review for every fraud caught and a false alarm for every genuine row flagged. Returns infinity when flagging
    nothing is cheapest.
    """
    order = np.argsort(-score, kind="stable")
    s, fraud, amt = score[order], y[order] == 1, amount[order]
    caught_amount = np.cumsum(np.where(fraud, amt, 0.0))
    caught = np.cumsum(fraud)
    false_alarms = np.cumsum(~fraud)
    totals = (amt[fraud].sum() - caught_amount) + caught * cost.review_cost + false_alarms * cost.false_alarm_cost
    # Rows with equal scores are flagged together, so only the last row of each run of ties is a real cut-off.
    is_cut = np.append(s[:-1] != s[1:], True)
    candidates = np.flatnonzero(is_cut)
    best = candidates[np.argmin(totals[candidates])]
    flag_nothing = float(amt[fraud].sum())
    return float(s[best]) if totals[best] < flag_nothing else float("inf")


def threshold_for_precision(y: FloatArray, score: FloatArray, precision: float) -> float:
    """The lowest threshold whose flagged rows reach `precision`. Infinity when no threshold does."""
    precisions, _, thresholds = precision_recall_curve(y, score)
    reaching = np.flatnonzero(precisions[:-1] >= precision)
    return float(thresholds[reaching[0]]) if len(reaching) else float("inf")


@dataclass(frozen=True)
class ReliabilityBucket:
    low: float
    high: float
    count: int
    mean_score: float
    fraud_rate: float


CALIBRATION_EDGES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def reliability(
    y: FloatArray, score: FloatArray, edges: tuple[float, ...] = CALIBRATION_EDGES
) -> list[ReliabilityBucket]:
    """Predicted versus observed fraud rate per score bucket. Empty buckets are left out.

    The edges are spaced tighter near zero because that is where almost every transaction scores.
    """
    bucket = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    result = []
    for i in range(len(edges) - 1):
        mask = bucket == i
        if mask.any():
            result.append(
                ReliabilityBucket(
                    low=edges[i],
                    high=edges[i + 1],
                    count=int(mask.sum()),
                    mean_score=float(score[mask].mean()),
                    fraud_rate=float(y[mask].mean()),
                )
            )
    return result
