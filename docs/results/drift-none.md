Reference: training window, 383,984 transactions. Live: test window, 126,611 transactions. Scenario: none.

| input | PSI | status |
|---|---|---|
| amount_log | 0.000 | stable |
| hour | 0.000 | stable |
| is_night | 0.000 | stable |
| is_foreign | 0.000 | stable |
| country_is_new | 0.014 | stable |
| merchant_is_new | 0.022 | stable |
| device_is_new | 0.000 | stable |
| txn_count_1h | 0.000 | stable |
| txn_count_24h | 0.000 | stable |
| amount_sum_24h_log | 0.000 | stable |
| minutes_since_last | 0.000 | stable |
| amount_zscore | 0.000 | stable |
| model score | 0.007 | stable |

PSI under 0.10 is stable, 0.10 to 0.25 is worth watching, 0.25 and above is an alert.

Calibration on the live window: mean score 0.0046, observed fraud rate 0.0053.

| score range | transactions | mean score | observed fraud rate |
|---|---|---|---|
| 0.0000 to 0.0010 | 123561 | 0.0000 | 0.0001 |
| 0.0010 to 0.0100 | 1782 | 0.0034 | 0.0213 |
| 0.0100 to 0.0500 | 485 | 0.0223 | 0.0825 |
| 0.0500 to 0.1000 | 108 | 0.0713 | 0.1481 |
| 0.1000 to 0.2500 | 84 | 0.1541 | 0.2619 |
| 0.2500 to 0.5000 | 46 | 0.3708 | 0.4783 |
| 0.5000 to 0.7500 | 33 | 0.6294 | 0.7273 |
| 0.7500 to 1.0000 | 512 | 0.9830 | 0.9609 |

No input crossed the alert threshold.
