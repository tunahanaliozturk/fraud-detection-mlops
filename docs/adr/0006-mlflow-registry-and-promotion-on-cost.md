# 6. MLflow on SQLite, and promotion only on lower expected cost

Status: Accepted, 2026-10-06

## Context

Model risk management needs every model that could reach production to be traceable: the data it saw, its
parameters, its metrics, the exact artefact, and why it was deployed. It also needs deployment to be a deliberate
gate, not "whatever trained last".

## Decision

Each `fraud train` logs one MLflow run to a local SQLite file (`sqlite:///mlflow.db`, no server): all parameters
including the split dates and costs, validation and test metrics for LightGBM and the baseline, the model in
LightGBM's native text format, the decision policy (`policy.json`) and the evaluation report as `model_card.md`. It
registers a new version of `card-fraud`. The `champion` alias moves to the new version only if its expected cost on
the current test window, at its own validation-chosen threshold, is strictly lower than the champion's on the same
window and the same costs. The promotion reason is written to the version as a tag.

## Consequences

- Retraining the same configuration does not move the alias (equal cost is not better), so the champion does not
  churn (`test_train_promote_export_and_serve`).
- The champion is compared on today's data and today's costs, not on the numbers it reported when it was trained.
- Deployment reads only the alias, through `fraud export`, which copies three files out of the registry. The service
  never imports MLflow.
- Known weakness, measured: running many candidates through this gate turns the test window into a selection set.
  During the weight sweep (`docs/results/pos-weight-sweep.md`) it promoted weight 20, which validation did not
  prefer. A bank would add a fresh out-of-time window per promotion decision, a minimum improvement larger than the
  noise (a bootstrap interval on cost), and a human sign-off.
- If the feature list changes, the champion cannot score the new matrix. There is no check for that yet.

## Alternatives

- **A tracking server with a database and object store.** What a team would run. Nothing here needs it, and the
  quick start would need Docker.
- **MLflow stages (Staging/Production).** Deprecated in favour of aliases.
- **Promote on PR-AUC.** Ranking quality is not what the bank pays for. Two models with the same PR-AUC can differ in
  cost when one misses the large frauds.
