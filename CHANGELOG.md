# Changelog

## Unreleased

First version.

- `fraud generate`: seeded synthetic card transactions with four fraud patterns, 0.49% fraud by default.
- Point-in-time features from one code path shared by training and serving.
- `fraud train`: LightGBM and a logistic regression baseline on a time split, a cost-minimising review threshold and a
  precision-based decline threshold, everything logged to MLflow on SQLite, promotion to `champion` only on lower cost.
- `fraud evaluate`: PR-AUC, recall at fixed precision, the cost curve, decision bands, recall by fraud pattern,
  metrics by segment, calibration and global SHAP importance.
- `fraud export` and `fraud serve`: a FastAPI scoring service with three SHAP reason codes per decision and
  idempotent retries.
- `fraud drift`: PSI per feature and on the score, a calibration check, and a synthetic drift scenario.
- Model card, ADRs 1 to 8, operations runbook, Docker image and CI.
