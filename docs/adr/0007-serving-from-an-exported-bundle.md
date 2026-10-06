# 7. Serving from an exported bundle, with in-memory histories

Status: Accepted, 2026-10-06

## Context

The service has to score one transaction in a few milliseconds with the same features training used (ADR 0003),
explain the score, and survive a retried request. It also has to start without a tracking server and without
running code it loaded from disk.

## Decision

- `fraud export` copies the champion's `model.lgb` (LightGBM text format), `policy.json`, `model_card.md` and
  `version.json` into `models/champion`. `fraud serve` loads that directory once at start-up. No pickle, no MLflow.
- At start-up the service replays the transaction file into a `FeatureStore`, one history per customer.
- Each request computes features, then adds the transaction to the history, then predicts and explains. One lock
  around the whole request keeps the read-then-update of a history atomic.
- Responses are remembered by `transaction_id` (the last 100,000). A retry gets the identical answer and is not
  counted twice in velocity features (`test_a_retried_request_gets_the_same_answer_and_counts_once`).
- Handlers are plain `def`, so FastAPI runs the CPU work on its thread pool, not on the event loop.

## Consequences

- Start-up takes 4.5 s for 723,121 transactions on the README laptop. Median latency is about 3 ms
  (`docs/results/latency.md`).
- One lock means one request at a time per process, about 250 a second. More throughput needs per-customer locks or
  more processes, and more processes need a shared history store.
- Histories live in process memory and are lost on restart; the replay rebuilds them only from the file it was given.
  Transactions scored since start-up are gone after a restart.
- A late transaction (timestamp before the customer's last one) is scored against a history that already contains
  later activity. Real systems handle this with event-time windows; this one does not.

## Alternatives

- **Load the model from the registry at start-up.** One step fewer, but the image would need MLflow, the registry
  would need to be reachable, and the artefact paths stored in SQLite are absolute host paths.
- **Pickle the model and policy together.** Simpler loading, but unpickling runs code, and a model file is exactly the
  kind of artefact that travels between teams.
- **Redis or another external store for histories.** Needed for more than one process. Adds a moving part the
  quick start would have to run.
