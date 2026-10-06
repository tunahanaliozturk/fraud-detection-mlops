"""Point-in-time features, computed by the same code for training and for live scoring.

A customer's history only ever grows: a transaction is turned into features from what the history holds *before*
it, and only then added. Training replays the whole file in arrival order through that loop; the service runs one
transaction through it per request. One code path means no training/serving skew, and the order of the two calls
means a feature cannot see the transaction it describes or anything after it (tests/test_features.py checks both).
"""

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOUR = 3_600
DAY = 86_400
NO_PREVIOUS_MINUTES = 30 * 24 * 60.0  # used for a customer's first transaction
MIN_HISTORY_FOR_ZSCORE = 5
NIGHT_ENDS = 6  # midnight to 05:59 counts as night

FEATURES = [
    "amount_log",
    "hour",
    "is_night",
    "is_foreign",
    "country_is_new",
    "merchant_is_new",
    "device_is_new",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h_log",
    "minutes_since_last",
    "amount_zscore",
]

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass(frozen=True, slots=True)
class Transaction:
    ts: int  # seconds since the epoch, in the cardholder's local time
    customer_id: str
    customer_country: str
    merchant_id: str
    merchant_country: str
    amount: float
    device_id: str


@dataclass(slots=True)
class CustomerHistory:
    recent: deque[tuple[int, float]] = field(default_factory=deque)  # (ts, amount) for the last 24 hours
    count: int = 0
    mean_log_amount: float = 0.0
    m2_log_amount: float = 0.0  # Welford's running sum of squared deviations
    last_ts: int | None = None
    merchants: set[str] = field(default_factory=set)
    devices: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)

    def features(self, txn: Transaction) -> list[float]:
        last_day = [(ts, amount) for ts, amount in self.recent if ts > txn.ts - DAY]
        last_hour = [amount for ts, amount in last_day if ts > txn.ts - HOUR]
        log_amount = math.log1p(txn.amount)
        hour = (txn.ts % DAY) // HOUR
        return [
            log_amount,
            float(hour),
            float(hour < NIGHT_ENDS),
            float(txn.merchant_country != txn.customer_country),
            float(txn.merchant_country != txn.customer_country and txn.merchant_country not in self.countries),
            float(txn.merchant_id not in self.merchants),
            float(txn.device_id not in self.devices),
            float(len(last_hour)),
            float(len(last_day)),
            math.log1p(sum(amount for _, amount in last_day)),
            NO_PREVIOUS_MINUTES if self.last_ts is None else min((txn.ts - self.last_ts) / 60, NO_PREVIOUS_MINUTES),
            self._zscore(log_amount),
        ]

    def add(self, txn: Transaction) -> None:
        self.recent.append((txn.ts, txn.amount))
        while self.recent and self.recent[0][0] <= txn.ts - DAY:
            self.recent.popleft()
        log_amount = math.log1p(txn.amount)
        self.count += 1
        delta = log_amount - self.mean_log_amount
        self.mean_log_amount += delta / self.count
        self.m2_log_amount += delta * (log_amount - self.mean_log_amount)
        self.last_ts = txn.ts if self.last_ts is None else max(self.last_ts, txn.ts)
        self.merchants.add(txn.merchant_id)
        self.devices.add(txn.device_id)
        self.countries.add(txn.merchant_country)

    def _zscore(self, log_amount: float) -> float:
        """How unusual the amount is for this customer, on a log scale because spending is heavy-tailed."""
        if self.count < MIN_HISTORY_FOR_ZSCORE:
            return 0.0
        std = math.sqrt(self.m2_log_amount / (self.count - 1))
        return (log_amount - self.mean_log_amount) / max(std, 0.1)


class FeatureStore:
    """Per-customer histories held in memory. A stand-in for an online feature store."""

    def __init__(self) -> None:
        self._customers: dict[str, CustomerHistory] = {}

    def features(self, txn: Transaction) -> list[float]:
        history = self._customers.get(txn.customer_id)
        return (history or CustomerHistory()).features(txn)

    def add(self, txn: Transaction) -> None:
        self._customers.setdefault(txn.customer_id, CustomerHistory()).add(txn)

    def load(self, transactions: pd.DataFrame) -> None:
        """Build histories from an arrival-ordered frame without computing features (service start-up)."""
        for txn in iter_transactions(transactions):
            self.add(txn)

    def replay(self, transactions: pd.DataFrame) -> list[list[float]]:
        """Features for every row of an arrival-ordered frame, each computed before the row joins the history."""
        rows = []
        for txn in iter_transactions(transactions):
            rows.append(self.features(txn))
            self.add(txn)
        return rows


def arrival_order(transactions: pd.DataFrame) -> pd.DataFrame:
    return transactions.sort_values(["ts", "transaction_id"], kind="stable").reset_index(drop=True)


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def iter_transactions(frame: pd.DataFrame) -> Iterator[Transaction]:
    columns = zip(
        epoch_seconds(frame["ts"]).tolist(),
        frame["customer_id"].tolist(),
        frame["customer_country"].tolist(),
        frame["merchant_id"].tolist(),
        frame["merchant_country"].tolist(),
        frame["amount"].tolist(),
        frame["device_id"].tolist(),
        strict=True,
    )
    for ts, customer, customer_country, merchant, merchant_country, amount, device in columns:
        yield Transaction(ts, customer, customer_country, merchant, merchant_country, amount, device)


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """The transactions in arrival order, with one column per feature appended."""
    ordered = arrival_order(transactions)
    values = FeatureStore().replay(ordered)
    features = pd.DataFrame(values, columns=FEATURES, index=ordered.index)
    return pd.concat([ordered, features], axis=1)
