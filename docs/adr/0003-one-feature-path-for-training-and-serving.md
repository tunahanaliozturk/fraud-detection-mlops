# 3. One feature code path for training and serving

Status: Accepted, 2026-10-06

## Context

Fraud features are per-customer aggregates over time: transactions in the last hour, the usual amount, whether the
merchant is new. Two failures are common. Training computes them with vectorised group-by code that quietly includes
the current row or later rows (leakage), and serving reimplements them by hand and gets them slightly different
(training/serving skew). Both inflate offline metrics and stay invisible until production.

## Decision

Features are computed by `CustomerHistory.features(txn)` from the history as it stands, and only then does
`CustomerHistory.add(txn)` add the transaction. Training replays the whole file in arrival order through that pair
(`FeatureStore.replay`); the service calls the same two methods once per request. Arrival order is
`(ts, transaction_id)`, and the generator assigns ids in that order.

## Consequences

- No transaction can see itself or anything after it, by construction. Tests prove it from the outside: deleting or
  rewriting every row after a cut-off leaves earlier features unchanged, shuffling the input changes nothing, and a
  hand-built history gives the expected counts (`tests/test_features.py`). Swapping the two calls in `replay` makes
  two of those tests fail; we tried it.
- A Python loop is slower than a vectorised pipeline: 723,121 transactions take about 15 s on the laptop named in the
  README. Fine at this size, not at a billion rows.
- The first 14 days are a warm-up that only builds history. Training on them would teach the model a cold start
  that live traffic does not have.
- The service holds every customer's history in memory (ADR 0007).

## Alternatives

- **Pandas rolling windows with `closed="left"` and shifted expanding statistics.** Faster, but every feature needs
  its own leakage argument and serving still needs a second implementation. Would win at a much larger scale, with a
  test comparing it row by row against this loop.
- **A feature store product such as Feast.** The right answer in a bank, where the online store is shared by many
  models. Too much infrastructure for a single-model repository.
