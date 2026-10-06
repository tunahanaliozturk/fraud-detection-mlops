# Security

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" on this repository, not in a public
issue. You should hear back within a week.

## What the code does about it

- **No secrets anywhere.** Nothing in the project takes a password, token or key. `.env` files are ignored by git in
  case one ever appears.
- **No outbound traffic.** MLflow's usage telemetry is switched off in the package itself
  (`MLFLOW_DISABLE_TELEMETRY`, set in `fraud_detection/__init__.py`), the tracking store is a local SQLite file, and
  the tests run offline.
- **Models load without running code.** The service reads LightGBM's text model format and two JSON files. No pickle
  is loaded at serving time (ADR 0007).
- **Input is validated at the boundary.** Identifiers are limited to 64 characters of letters, digits, `_` and `-`;
  countries to two capital letters; amounts to positive finite numbers up to one million; timestamps must carry an
  offset. Unknown fields are refused, so a caller cannot slip a label or an internal field in.
- **Retries are idempotent.** A repeated `transaction_id` returns the stored response and does not change any
  customer's history.
- **The image runs as a non-root user** (uid 10001), without pip, on a base image pinned by digest, with the model
  and data mounted read-only. CI audits the locked dependencies with `pip-audit` and the licences of every installed
  distribution.

## What it does not do

- **No authentication or rate limiting** on the API. It is meant to sit behind a gateway that has both.
- **Responses are not signed**, and the retry cache lives in process memory.
- **Data at rest is not encrypted.** The generated data is synthetic; real card data would need encryption, access
  control and PCI DSS scope decisions that are out of scope here.
- **Reason codes reveal what the model looks at.** That is the point for analysts and customers, and also a hint for
  a fraudster. A real deployment would decide which reasons a customer sees and which only an analyst sees.
