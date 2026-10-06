# Positive-class weight sweep, 2026-10-06

The question for ADR 0004: should LightGBM up-weight the rare fraud rows (`scale_pos_weight`), or train unweighted and
let the cost-based threshold do the trading?

Same data as the README (`uv run fraud generate`, seed 42), a separate tracking database so the sweep does not touch
the real registry:

```sh
mkdir sweep && cd sweep
for w in 1 5 20 50; do
  uv run fraud train --data ../data/transactions.parquet --tracking-uri sqlite:///sweep.db --pos-weight $w
done
```

Validation-window numbers are `valid_cost` and `valid_pr_auc` from each MLflow run; test numbers are from the printed
report. Hardware does not matter here; training is deterministic.

| pos_weight | validation cost, EUR | validation PR-AUC | review threshold | test cost, EUR | test PR-AUC |
|---|---|---|---|---|---|
| **1** | **22,144** | **0.931** | 0.0107 | 27,130 | **0.911** |
| 5 | 23,472 | 0.921 | 0.0325 | 26,516 | 0.902 |
| 20 | 22,785 | 0.920 | 0.0460 | **23,079** | 0.901 |
| 50 | 23,377 | 0.922 | 0.3805 | 26,489 | 0.901 |

Weight 1 is best on the window we are allowed to choose on (validation), on both cost and PR-AUC. Weight 20 is best
on the test window by 15%. We kept 1: picking the setting that wins on the test window would make the test window a
second validation set, and the swing between the two windows (about EUR 4,000) is the size of a handful of large
missed frauds, which is noise at 665 fraud transactions.

What the sweep also showed: the promotion rule, which compares candidate and champion on the test window, promoted
weight 5 and then weight 20 during this sweep. That is the selection effect described in ADR 0006, seen for real.
