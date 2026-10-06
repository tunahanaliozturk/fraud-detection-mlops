# Operations

How to run the service and the model lifecycle, what to watch, and what to do when something looks wrong.

## Configuration

Everything is a command-line flag; there are no environment variables to set and no secrets.

| command and flag | default | unit | raising it | lowering it |
|---|---|---|---|---|
| `train --review-cost` | 5.0 | EUR per flagged payment | fewer reviews, more missed fraud | more reviews |
| `train --friction-cost` | 10.0 | EUR per genuine payment stopped | fewer false alarms, more missed fraud | more customers stopped |
| `train --decline-precision` | 0.9 | precision on validation | smaller, purer decline band | more automatic declines, more of them wrong |
| `train --pos-weight` | 1.0 | LightGBM `scale_pos_weight` | see ADR 0004 and the sweep | |
| `train --tracking-uri`, `evaluate`, `export`, `drift` | `sqlite:///mlflow.db` | | | |
| `serve --model-dir` | `models/champion` | | | |
| `serve --history` | `data/transactions.parquet` | | longer start-up, more memory | velocity and novelty features start cold |
| `serve --host`, `--port` | 127.0.0.1, 8000 | | | |
| `drift --scenario` | `none` | `none` or `shifted` | | |
| `drift --fail-on-alert` | off | exit status 1 on any PSI of 0.25 or more | | |

Costs are recorded with every model version. Changing them means retraining, because the thresholds are derived from
them on the validation window.

## Lifecycle

```sh
uv run fraud generate          # or point --data at your own file with the same columns
uv run fraud train             # logs a version; moves `champion` only if it is cheaper on the test window
uv run fraud evaluate          # the champion's full report
uv run fraud export            # champion -> models/champion
uv run fraud serve             # loads models/champion once, replays history, listens on 127.0.0.1:8000
uv run fraud drift --fail-on-alert
```

Browse runs with `uvx mlflow ui --backend-store-uri sqlite:///mlflow.db` if you want the UI; nothing in the project
needs it.

## What to watch

| signal | where | healthy | act when |
|---|---|---|---|
| `/health` | HTTP 200, `model_version` | the exported version | not 200, or the wrong version after a deploy |
| p99 latency of `/score` | your proxy or APM | under 20 ms on the README laptop | twice the baseline for 15 minutes |
| decision mix | count of `decision` per hour | review about 0.5%, decline about 0.5% of traffic | either doubles or halves: drift or an upstream data change |
| feature and score PSI | `fraud drift` daily | all under 0.10 | any at 0.25 or above |
| calibration | `fraud drift` output | mean score near the observed fraud rate | they differ by a factor of two |
| review queue precision | analysts' outcomes | about 11% fraud in review, 90% in decline | decline precision under 80% |

## Runbook

### `/score` returns 422 for traffic that used to work

Confirm: the response body lists the field and the rule (for example a timestamp without an offset, a country not in
two capital letters, an extra field). Act: fix the caller. The schema forbids unknown fields on purpose, so a renamed
field upstream fails loudly instead of being ignored. Do not relax `extra="forbid"` to make the errors go away.

### Decline rate jumps after a deploy

Confirm: `/health` shows the new version; `fraud drift` on recent traffic shows no input alert. Then it is the model or
its policy, not the traffic. Act: re-export the previous version (move the alias back in MLflow:
`MlflowClient().set_registered_model_alias("card-fraud", "champion", "<previous>")`, then `fraud export`) and restart.
Verify the decision mix returns to its old level within the hour.

### Drift alert on one feature

Confirm: is the shift in the world or in the pipe? Compare the raw column upstream (amounts, country codes) with last
week. A unit change or a new country code format looks exactly like drift. Act: if the pipe, fix it and do not
retrain. If the world (prices, travel season), check the score PSI and the calibration lines; retrain only when the
score moves too. Do not retrain on every feature alert: the shifted scenario alerts on `is_foreign` while the model
may still rank fine.

### Service restarts and velocity features look cold

Confirm: transactions scored since the last start are not in `--history`. Act: point `--history` at a file that
includes recent traffic. Known limitation, see below.

### Start-up is slow or runs out of memory

Confirm: the history file size. 723,121 transactions take 4.5 s and fit easily; the store grows with customers and
their merchants, devices and countries. Act: trim history to the window the features need (the z-score uses all of
it; everything else uses 24 hours or novelty sets), or move histories to an external store.

## Known limitations

- One process, one lock: about 250 scores a second, sequential. More processes need a shared history store.
- Histories live in memory. A restart forgets everything scored since the history file was written.
- Late transactions (timestamp earlier than the customer's latest) are scored against a history that already
  contains the later ones.
- The retry cache holds the last 100,000 responses, in memory only.
- Promotion compares candidates on the same test window every time, so repeated candidates overfit it (ADR 0006).
- No authentication on the API. Put it behind the bank's gateway; it is an internal service.
- Labels in the data are instant. In production the calibration check and any recall number lag by the chargeback
  window.
