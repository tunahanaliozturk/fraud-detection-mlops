"""Population stability index (PSI) between a training window and a live window."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

WATCH = 0.10
ALERT = 0.25
_EMPTY_BIN = 1e-4  # stands in for a proportion of zero, which the log in PSI cannot take
MOVED_ABROAD = 0.33  # share of live purchases the drift scenario moves to a foreign merchant

Status = Literal["stable", "watch", "alert"]


def bin_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inner bin edges taken from the reference distribution.

    A feature with few distinct values (a flag, a small count) gets one bin per value, because quantile edges would
    collapse into one bin and hide any shift. Anything else gets reference deciles.
    """
    values = np.unique(reference)
    if len(values) <= bins:
        return np.asarray((values[:-1] + values[1:]) / 2, dtype=float)
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return np.clip(counts / max(len(values), 1), _EMPTY_BIN, None)


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    edges = bin_edges(reference, bins)
    expected, actual = proportions(reference, edges), proportions(live, edges)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(value: float) -> Status:
    if value >= ALERT:
        return "alert"
    return "watch" if value >= WATCH else "stable"


@dataclass(frozen=True)
class DriftRow:
    name: str
    psi: float
    status: Status


def drift_report(reference: pd.DataFrame, live: pd.DataFrame, columns: list[str]) -> list[DriftRow]:
    rows = []
    for column in columns:
        value = psi(reference[column].to_numpy(dtype=float), live[column].to_numpy(dtype=float))
        rows.append(DriftRow(column, value, status(value)))
    return rows


def simulate_drift(transactions: pd.DataFrame, since: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """A copy where everything from `since` looks like a holiday season after a price rise.

    Amounts go up by half and a third of purchases move abroad. Fraud labels are left alone; the point is to see the
    monitor fire, not to retrain.
    """
    rng = np.random.default_rng(seed)
    shifted = transactions.copy()
    live = shifted["ts"] >= since
    shifted.loc[live, "amount"] = (shifted.loc[live, "amount"] * 1.5).round(2)
    abroad = live & (rng.random(len(shifted)) < MOVED_ABROAD)
    shifted.loc[abroad, "merchant_country"] = rng.choice(["FR", "ES", "IT", "US"], size=int(abroad.sum()))
    return shifted
