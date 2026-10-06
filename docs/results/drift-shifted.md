Reference: training window, 383,984 transactions. Live: test window, 126,611 transactions. Scenario: shifted.

| input | PSI | status |
|---|---|---|
| amount_log | 0.134 | watch |
| hour | 0.000 | stable |
| is_night | 0.000 | stable |
| is_foreign | 0.559 | alert |
| country_is_new | 0.004 | stable |
| merchant_is_new | 0.022 | stable |
| device_is_new | 0.000 | stable |
| txn_count_1h | 0.000 | stable |
| txn_count_24h | 0.000 | stable |
| amount_sum_24h_log | 0.079 | stable |
| minutes_since_last | 0.000 | stable |
| amount_zscore | 0.144 | watch |
| model score | 0.500 | alert |

PSI under 0.10 is stable, 0.10 to 0.25 is worth watching, 0.25 and above is an alert.

Calibration on the live window: mean score 0.0072, observed fraud rate 0.0053.

| score range | transactions | mean score | observed fraud rate |
|---|---|---|---|
| 0.0000 to 0.0010 | 112590 | 0.0001 | 0.0001 |
| 0.0010 to 0.0100 | 9121 | 0.0032 | 0.0030 |
| 0.0100 to 0.0500 | 2894 | 0.0229 | 0.0204 |
| 0.0500 to 0.1000 | 685 | 0.0700 | 0.0350 |
| 0.1000 to 0.2500 | 416 | 0.1554 | 0.0721 |
| 0.2500 to 0.5000 | 216 | 0.3578 | 0.1019 |
| 0.5000 to 0.7500 | 150 | 0.6263 | 0.3600 |
| 0.7500 to 1.0000 | 539 | 0.9655 | 0.8219 |

ALERT: is_foreign, model score
