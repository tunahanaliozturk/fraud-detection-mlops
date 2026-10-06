# fraud-detection-mlops

[![ci](https://github.com/tunahanaliozturk/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/tunahanaliozturk/fraud-detection-mlops/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Card-payment fraud detection built the way a regulated bank needs it: leak-free features, a threshold chosen on
money, reasons for every decision, a model registry with a promotion gate, and drift monitoring.

On a held-out, later time window the LightGBM model cuts the expected cost of fraud by 75.2% against doing nothing
(EUR 27,130 instead of EUR 109,593), where a logistic regression baseline manages 20.8%, and the scoring API answers in
about 3 ms at the median. The data is synthetic, so treat those as measurements of the pipeline rather than of
real-world fraud ([why, and what that hides](docs/adr/0002-synthetic-data.md)).

## The problem

A bank sees a few fraudulent card payments among thousands of genuine ones. Stopping a payment costs something too:
an analyst's time and an annoyed customer. So the job is not "maximise accuracy" (flagging nothing scores 99.5%)
but "lose the least money", while explaining every stop, proving the model never peeked at the future, and noticing
when the world it was trained on changes.

## Quick start

Python 3.14 and [uv](https://docs.astral.sh/uv/). About two minutes on the laptop below, no network after `uv sync`.

```sh
uv sync --locked
uv run fraud generate     # 723,121 synthetic transactions, 0.49% fraud, seeded
uv run fraud train        # LightGBM + baseline, logged to MLflow (SQLite), promoted to champion if cheaper
uv run fraud export       # champion -> models/champion
uv run fraud serve        # http://127.0.0.1:8000, try requests.http
```

Then `uv run fraud evaluate` for the full report and `uv run fraud drift --scenario shifted` to watch the drift
alarm go off. CI runs these commands on every push and checks the results
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

```sh
curl -s -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d '{
  "transaction_id": "demo-2", "timestamp": "2026-05-02T03:12:00+02:00", "customer_id": "c00042",
  "customer_country": "NL", "merchant_id": "m01234", "merchant_country": "US", "amount": 1890.0,
  "device_id": "unknown-phone"}'
```

The response, with the score rounded:

```json
{"transaction_id": "demo-2", "score": 0.9994, "decision": "decline", "model_version": "1",
 "reasons": [
   {"feature": "device_is_new", "direction": "increases_risk", "contribution": 6.8539,
    "text": "The purchase came from a device not seen before"},
   {"feature": "amount_zscore", "direction": "increases_risk", "contribution": 3.7591,
    "text": "The amount is far from this cardholder's usual spending"},
   {"feature": "is_foreign", "direction": "increases_risk", "contribution": 2.9784,
    "text": "The merchant is abroad"}]}
```

## How it fits together

```
 fraud generate            fraud train                                    fraud export        fraud serve
 ──────────────            ───────────                                    ────────────        ───────────
 seeded customers,   ──>   replay history in arrival order          ──>   champion alias ──>  model.lgb + policy.json
 merchants, 4 fraud        (features from strictly earlier rows)          -> 3 plain files    loaded once
 patterns                  │                                                                  │
 data/*.parquet            time split: 14d warm-up | train | valid | test                     replay history into
                           │                                                                  per-customer state
                           LightGBM (early stop on valid PR-AUC)                              │
                           + logistic regression baseline                                     POST /score:
                           │                                                                  validate (pydantic)
                           review threshold = min expected cost on valid                      features -> add to history
                           decline threshold = 90% precision on valid                         LightGBM score -> band
                           │                                                                  SHAP -> 3 reason codes
                           MLflow (sqlite): params, metrics, model,
                           policy, model card -> new registered version       fraud drift
                           │                                                  ───────────
                           promote to champion only if test cost is           PSI per feature and score,
                           strictly lower than the champion's                 train window vs live window,
                                                                              calibration buckets
```

## Results

All numbers below come from `uv run fraud evaluate` on the default data (full output in
[docs/results/evaluation.md](docs/results/evaluation.md)), and reproduce to the byte: training is deterministic.
Test window: 2026-04-09 to 2026-05-01, 126,611 transactions, 665 fraud. Thresholds were chosen on the validation
window before it and applied unchanged.

| model | PR-AUC | recall at 90% precision | expected cost | saving vs flagging nothing |
|---|---|---|---|---|
| **LightGBM** | **0.911** | **0.812** | **EUR 27,130** | **75.2%** |
| logistic regression | 0.643 | 0.514 | EUR 86,848 | 20.8% |

The cost model: a missed fraud costs its amount, a review costs EUR 5, stopping a genuine customer costs another
EUR 10 (`--review-cost`, `--friction-cost`). The cost curve on the test window:

| threshold | flagged | fraud caught | missed EUR | false alarms | total cost EUR |
|---|---|---|---|---|---|
| 0.001 | 3,050 | 654 | 1,680 | 2,396 | 40,890 |
| 0.005 | 1,648 | 633 | 8,298 | 1,015 | 26,688 |
| **0.0107 (chosen on validation)** | 1,242 | 613 | 14,630 | 629 | **27,130** |
| 0.05 | 783 | 576 | 26,030 | 207 | 32,015 |
| 0.25 | 591 | 538 | 36,994 | 53 | 40,479 |
| 0.5 (the default people reach for) | 545 | 516 | 45,106 | 29 | 48,121 |

The curve is flat between 0.005 and 0.02 and steep outside it. With hindsight the cheapest threshold on the test
window is 0.0071 at EUR 24,708, so the one picked on validation costs 9.8% more: that gap is the price of not
choosing on the data you report. A threshold of 0.5 would cost 77% more than the chosen one.

Decision bands: approve below 0.0107 (125,369 payments, 52 of them fraud), review up to 0.2417 (650 payments, 11.5%
fraud), decline above (592 payments, 90.9% fraud).

**Serving latency**, `load/latency.py`, 5,000 sequential requests over one keep-alive connection, three runs on fresh
processes: p50 2.77 to 3.15 ms, p99 7.7 to 18.5 ms. Laptop: Intel Core Ultra 7 255H, 32 GB, Windows 11, Python
3.14.6, client and server on the same machine
([docs/results/latency.md](docs/results/latency.md)). Start-up, including replaying 723,121 transactions into
customer histories: 4.5 s.

**Drift**, `fraud drift`: on the unchanged live window every PSI stays under 0.03. With `--scenario shifted` (amounts
up 50%, a third of purchases moved abroad) `is_foreign` reaches 0.559 and the model score 0.500, both alerts
([docs/results/drift-shifted.md](docs/results/drift-shifted.md)).

## The decisions worth reading

**One feature path for training and serving.** Every feature is computed by `CustomerHistory.features(txn)` from the
history *before* the transaction, and only then is the transaction added. Training replays the file through that
pair; the service calls the same pair per request. Leakage and training/serving skew are ruled out by construction,
and the tests prove it from outside: deleting or rewriting every row after a cut-off leaves earlier features
identical ([ADR 3](docs/adr/0003-one-feature-path-for-training-and-serving.md), `tests/test_features.py`).

**Imbalance is a cost question.** The model trains unweighted and the threshold minimises expected euros on
validation. A sweep over class weights backs that on validation and shows test disagreeing by 15%, which we report
rather than hide ([ADR 4](docs/adr/0004-imbalance-through-cost-not-weights.md),
[sweep](docs/results/pos-weight-sweep.md)).

All ADRs:
[1 Python 3.14 and dependencies](docs/adr/0001-python-3-14-and-the-dependency-set.md) ·
[2 synthetic data](docs/adr/0002-synthetic-data.md) ·
[3 one feature path](docs/adr/0003-one-feature-path-for-training-and-serving.md) ·
[4 imbalance through cost](docs/adr/0004-imbalance-through-cost-not-weights.md) ·
[5 reason codes from SHAP](docs/adr/0005-reason-codes-from-shap.md) ·
[6 registry and promotion](docs/adr/0006-mlflow-registry-and-promotion-on-cost.md) ·
[7 serving from a bundle](docs/adr/0007-serving-from-an-exported-bundle.md) ·
[8 drift with PSI](docs/adr/0008-drift-monitoring-with-psi.md)

More: the [model card](docs/model-card.md) with metrics by segment, the [data and fraud patterns](docs/data.md),
[operations and runbook](docs/operations.md), [requests.http](requests.http).

## Why the banking parts matter

- **Reason codes.** An analyst needs to know why a payment is in the queue, and a customer whose card was stopped is
  owed an explanation, much like an adverse-action notice on a credit decision. Every response carries the three
  largest SHAP contributions in plain words, and a test checks the wording never contradicts the sign.
- **Model risk review.** Every version in the registry has its data split, costs, parameters, metrics, policy and
  model card attached, and the reason it was promoted is written on it. The model card reports performance by
  segment so that uneven treatment is visible: a genuine payment abroad is flagged 37 times as often as one at home.
- **Monitoring before labels.** Chargebacks arrive weeks late. PSI on inputs and scores, and the calibration check,
  give a signal on day one.

## Tests and gates

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests                 # strict
uv run pytest                         # 41 tests, offline, warnings are errors, about 20 s warm
uv run python tools/license_audit.py  # every installed distribution, permissive or the build fails
```

What the tests pin down: generator determinism; point-in-time correctness (future rows deleted, rewritten, input
shuffled); the threshold search against a hand-worked example and against brute force; PSI against a hand-worked
value; the API contract (422 on ten kinds of bad input, the response shape, idempotent retries); reason codes
consistent with SHAP signs and SHAP values summing to the model output; and a tiny end-to-end run that trains,
registers, refuses to promote an equal model, exports, serves and raises a drift alert.

CI also exports the locked requirements and runs `pip-audit` on them, and builds the Docker image, starts it against
the exported model and scores a request.

## Docker

The image holds code only; train and export on the host, then mount the model and the history read-only.

```sh
docker build -t fraud-detection-mlops .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" fraud-detection-mlops
```

Non-root user, base image pinned by digest, a Python one-liner as the health check because the slim image has no
curl.

## Limitations

- **Synthetic data.** Four scripted patterns, a fraudster who never adapts, labels that are complete and instant.
  The scores are optimistic. Nothing here has met real card data.
- **One test window**, three weeks, 665 fraud transactions. No confidence intervals yet; some segments rest on fewer
  than 50 frauds.
- **Promotion overfits the test window** when many candidates go through it; the class-weight sweep showed it
  happening ([ADR 6](docs/adr/0006-mlflow-registry-and-promotion-on-cost.md)).
- **Scores under-predict in the middle** (0.022 predicted against 0.083 observed between 0.01 and 0.05). Not
  calibrated yet.
- **One process, in-memory histories.** About 250 scores a second, histories lost on restart, late-arriving
  transactions not handled ([ADR 7](docs/adr/0007-serving-from-an-exported-bundle.md)).
- **No authentication** on the API; it expects to sit behind a gateway.

## Licence

MIT. Every dependency is permissively licensed at every depth, checked in CI.
