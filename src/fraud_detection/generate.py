"""Synthetic card transactions with fraud patterns a human analyst could explain.

Everything comes from one seeded generator, so the same arguments always give the same rows. Legitimate behaviour
is deliberately messy (trips abroad, new phones, the odd large purchase at night) so that no single rule separates
fraud from normal spending. The four fraud patterns are described in docs/data.md.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86_400
HOUR = 3_600

# Category and the multiplier applied to a customer's typical amount when they shop there.
CATEGORIES: dict[str, float] = {
    "grocery": 1.0,
    "fuel": 1.6,
    "restaurant": 1.3,
    "pharmacy": 0.8,
    "clothing": 2.2,
    "online_retail": 1.8,
    "electronics": 4.0,
    "travel": 6.0,
    "digital_goods": 0.4,
    "jewellery": 5.0,
}
CARD_TESTING_CATEGORIES = ["digital_goods"]
HIGH_VALUE_CATEGORIES = ["electronics", "jewellery", "travel"]
ONLINE_CATEGORIES = ["online_retail", "electronics", "digital_goods"]

# Country codes only label "home" versus "abroad": no feature carries the code itself. Fraud abroad picks a foreign
# merchant at random, so fraud rates by country (0.9% to 5.4% of foreign purchases) follow merchant counts and trips,
# not anything about the country.
HOME_COUNTRIES = ["NL", "BE", "DE"]
HOME_WEIGHTS = [0.8, 0.1, 0.1]
COUNTRIES = ["NL", "BE", "DE", "FR", "ES", "IT", "PL", "GB", "US", "TR"]

PATTERNS = ["card_testing", "foreign_high_value", "new_merchant_night", "account_takeover"]

COLUMNS = [
    "transaction_id",
    "ts",
    "customer_id",
    "customer_country",
    "merchant_id",
    "merchant_category",
    "merchant_country",
    "amount",
    "device_id",
    "is_fraud",
    "fraud_pattern",
]


@dataclass(frozen=True)
class GeneratorConfig:
    customers: int = 4_000
    merchants: int = 1_500
    days: int = 120
    start: str = "2026-01-01"
    seed: int = 42
    victim_share: float = 0.20  # share of customers who suffer one fraud incident in the period


@dataclass(frozen=True)
class Merchants:
    ids: np.ndarray
    categories: np.ndarray
    countries: np.ndarray


@dataclass(frozen=True)
class Customer:
    id: str
    country: str
    typical_amount: float
    spread: float
    favourites: np.ndarray  # indexes into Merchants


class _Rows:
    """Column buffers, filled one batch of transactions at a time."""

    def __init__(self) -> None:
        self.columns: dict[str, list[np.ndarray]] = {name: [] for name in COLUMNS if name != "transaction_id"}

    def add(
        self,
        customer: Customer,
        merchants: Merchants,
        ts: np.ndarray,
        merchant_idx: np.ndarray,
        amount: np.ndarray,
        device: str | np.ndarray,
        pattern: str = "",
    ) -> None:
        n = len(ts)
        self.columns["ts"].append(ts.astype(np.int64))
        self.columns["customer_id"].append(np.full(n, customer.id, dtype=object))
        self.columns["customer_country"].append(np.full(n, customer.country, dtype=object))
        self.columns["merchant_id"].append(merchants.ids[merchant_idx])
        self.columns["merchant_category"].append(merchants.categories[merchant_idx])
        self.columns["merchant_country"].append(merchants.countries[merchant_idx])
        self.columns["amount"].append(np.maximum(np.round(amount, 2), 0.5))
        self.columns["device_id"].append(np.full(n, device, dtype=object) if isinstance(device, str) else device)
        self.columns["is_fraud"].append(np.full(n, 1 if pattern else 0, dtype=np.int8))
        self.columns["fraud_pattern"].append(np.full(n, pattern, dtype=object))

    def frame(self, start: pd.Timestamp) -> pd.DataFrame:
        data = {name: np.concatenate(parts) for name, parts in self.columns.items()}
        frame = pd.DataFrame(data)
        frame["ts"] = start + pd.to_timedelta(frame["ts"], unit="s")
        # Ids follow arrival order, so (ts, transaction_id) is a total order that matches what a live system sees.
        frame = frame.sort_values(["ts", "customer_id", "merchant_id", "amount"], kind="stable").reset_index(drop=True)
        frame.insert(0, "transaction_id", np.arange(1, len(frame) + 1, dtype=np.int64))
        return frame[COLUMNS]


def generate(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    merchants = _merchants(config, rng)
    rows = _Rows()
    victims = set(rng.choice(config.customers, size=round(config.customers * config.victim_share), replace=False))
    for number in range(config.customers):
        customer = _customer(number, merchants, rng)
        _legitimate(customer, merchants, config.days, rng, rows)
        if number in victims:
            _fraud(customer, merchants, config.days, rng, rows, incident=number)
    return rows.frame(pd.Timestamp(config.start, tz="UTC"))


def _merchants(config: GeneratorConfig, rng: np.random.Generator) -> Merchants:
    names = list(CATEGORIES)
    # Everyday categories are more common than jewellers and travel agents.
    weights = np.array([6, 3, 5, 2, 3, 4, 2, 1, 2, 1], dtype=float)
    categories = rng.choice(names, size=config.merchants, p=weights / weights.sum())
    country_weights = np.array([45, 10, 10, 6, 6, 6, 5, 5, 5, 2], dtype=float)
    countries = rng.choice(COUNTRIES, size=config.merchants, p=country_weights / country_weights.sum())
    ids = np.array([f"m{i:05d}" for i in range(config.merchants)], dtype=object)
    return Merchants(ids, categories.astype(object), countries.astype(object))


def _customer(number: int, merchants: Merchants, rng: np.random.Generator) -> Customer:
    country = str(rng.choice(HOME_COUNTRIES, p=HOME_WEIGHTS))
    local = np.flatnonzero(merchants.countries == country)
    favourites = rng.choice(local, size=int(rng.integers(8, 25)), replace=False)
    return Customer(
        id=f"c{number:05d}",
        country=country,
        typical_amount=float(np.exp(rng.normal(np.log(30), 0.5))),
        spread=float(rng.uniform(0.5, 0.9)),
        favourites=favourites,
    )


def _amounts(customer: Customer, merchants: Merchants, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    multiplier = np.array([CATEGORIES[c] for c in merchants.categories[idx]])
    return customer.typical_amount * multiplier * rng.lognormal(0.0, customer.spread, size=len(idx))


def _legitimate(customer: Customer, merchants: Merchants, days: int, rng: np.random.Generator, rows: _Rows) -> None:
    n = int(rng.poisson(rng.gamma(2.0, 0.75) * days))
    if n == 0:
        return
    # Most spending happens during the customer's own waking hours; the tail of the normal reaches into the night.
    peak = rng.uniform(11, 17)
    hours = np.mod(rng.normal(peak, 3.5, size=n), 24)
    ts = rng.integers(0, days, size=n) * DAY + (hours * HOUR).astype(np.int64)

    idx = rng.choice(customer.favourites, size=n)
    elsewhere = rng.random(n) < 0.12
    idx[elsewhere] = rng.integers(0, len(merchants.ids), size=int(elsewhere.sum()))
    amount = _amounts(customer, merchants, idx, rng)
    occasional_big = rng.random(n) < 0.01
    amount[occasional_big] *= rng.uniform(3, 10, size=int(occasional_big.sum()))

    # A quarter of customers travel once: a few days of spending at merchants in another country.
    if rng.random() < 0.25:
        abroad = [c for c in COUNTRIES if c != customer.country]
        trip_country = rng.choice(abroad)
        trip_start = int(rng.integers(0, days)) * DAY
        trip = (ts >= trip_start) & (ts < trip_start + int(rng.integers(3, 11)) * DAY)
        local = np.flatnonzero(merchants.countries == trip_country)
        idx[trip] = rng.choice(local, size=int(trip.sum()))
        amount[trip] = _amounts(customer, merchants, idx[trip], rng)

    # One in ten customers replaces their phone during the period, which is a new device that is not fraud.
    device = np.full(n, f"{customer.id}-d0", dtype=object)
    if rng.random() < 0.10:
        device[ts >= int(rng.integers(0, days)) * DAY] = f"{customer.id}-d1"

    rows.add(customer, merchants, ts, idx, amount, device)


def _fraud(
    customer: Customer, merchants: Merchants, days: int, rng: np.random.Generator, rows: _Rows, incident: int
) -> None:
    pattern = str(rng.choice(PATTERNS))
    start = int(rng.integers(3 * DAY, days * DAY))
    stolen_device = f"x{incident:05d}"
    abroad = merchants.countries != customer.country

    if pattern == "card_testing":
        # Many tiny charges seconds apart at one digital-goods merchant, checking the card works.
        k = int(rng.integers(5, 16))
        ts = start + np.cumsum(rng.integers(5, 90, size=k))
        merchant = rng.choice(np.flatnonzero(np.isin(merchants.categories, CARD_TESTING_CATEGORIES)))
        idx = np.full(k, merchant)
        amount = rng.uniform(0.5, 3.0, size=k)
        if rng.random() < 0.3:  # sometimes the tester cashes out straight away
            ts = np.append(ts, ts[-1] + int(rng.integers(600, 3600)))
            idx = np.append(idx, rng.choice(np.flatnonzero(merchants.categories == "electronics")))
            amount = np.append(amount, customer.typical_amount * rng.uniform(5, 15))
        rows.add(customer, merchants, ts, idx, amount, stolen_device, pattern)

    elif pattern == "foreign_high_value":
        # A cloned card used abroad: same card, so no new device, but a country and amount out of character.
        k = int(rng.integers(1, 4))
        ts = start + np.cumsum(rng.integers(300, 3 * HOUR, size=k))
        candidates = np.flatnonzero(abroad & np.isin(merchants.categories, HIGH_VALUE_CATEGORIES))
        idx = rng.choice(candidates, size=k)
        amount = customer.typical_amount * rng.uniform(4, 25, size=k)
        rows.add(customer, merchants, ts, idx, amount, f"{customer.id}-d0", pattern)

    elif pattern == "new_merchant_night":
        # A large purchase in the small hours at a merchant this customer has never used.
        k = int(rng.integers(1, 3))
        night = (start // DAY) * DAY + int(rng.integers(0, 4 * HOUR))
        ts = night + np.cumsum(rng.integers(60, 1800, size=k))
        candidates = np.setdiff1d(np.flatnonzero(~abroad), customer.favourites)
        idx = rng.choice(candidates, size=k)
        amount = customer.typical_amount * rng.uniform(5, 20, size=k)
        rows.add(customer, merchants, ts, idx, amount, f"{customer.id}-d0", pattern)

    else:  # account_takeover
        # Someone else logs in from their own device and shops online across several new merchants.
        k = int(rng.integers(2, 6))
        ts = start + np.cumsum(rng.integers(120, 2 * HOUR, size=k))
        idx = rng.choice(np.flatnonzero(np.isin(merchants.categories, ONLINE_CATEGORIES)), size=k)
        amount = customer.typical_amount * rng.uniform(2, 8, size=k)
        rows.add(customer, merchants, ts, idx, amount, stolen_device, pattern)
