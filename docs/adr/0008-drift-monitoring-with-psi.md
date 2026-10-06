# 8. Drift monitoring with PSI per feature and on the score

Status: Accepted, 2026-10-06

## Context

Fraud labels arrive weeks late (chargebacks), so a model can degrade long before its recall can be measured. What
can be measured at once is whether the inputs and the scores still look like training. Model risk reviewers know the
population stability index (PSI) and its conventional bands.

## Decision

`fraud drift` compares the training window with the live (test) window: PSI for each feature and for the model score,
reported as stable below 0.10, watch from 0.10, alert from 0.25. Continuous features are binned at reference deciles.
Features with ten or fewer distinct values (flags, small counts) get one bin per value, because decile edges collapse
a 0/1 feature into one bin and hide any shift. Empty bins get a floor of 0.0001. The same command prints a calibration
check: mean score against observed fraud rate, overall and per score bucket. `--scenario shifted` applies a synthetic
shift to the live window (amounts up 50%, a third of purchases moved abroad) to show the alert firing;
`--fail-on-alert` makes the exit status usable in a scheduled job.

## Consequences

- On the generated data the live window is stable (largest PSI 0.022, `merchant_is_new`). The shifted scenario raises
  alerts on `is_foreign` (0.559) and the score (0.500), and watch on `amount_log` (0.134) and `amount_zscore` (0.144)
  (`docs/results/drift-*.md`, `test_the_drift_scenario_raises_alerts_that_the_quiet_one_does_not`).
- A 50% price rise only reaches "watch" on the amount itself, because log amounts are wide. PSI is a coarse alarm.
- The calibration check uses labels, which in reality are delayed. Here they are instant, which flatters it.
- PSI thresholds are convention, not derived from this model's cost. A bank would tie them to a measured effect on
  expected cost.

## Alternatives

- **Kolmogorov-Smirnov tests per feature.** With 100,000 rows almost every difference is "significant", so the p-value
  alarms constantly. PSI measures size, not significance.
- **A drift library (Evidently and similar).** Richer reports, but heavy dependencies for twenty lines of maths a
  reviewer can check by hand (`test_psi_on_a_hand_worked_example`).
