# Champion: version 1

Test window: 2026-04-09 21:16 to 2026-05-01, 126,611 transactions, 665 fraud (0.53%). Thresholds were chosen on the validation window (2026-03-19 onwards) and applied unchanged here. Costs: review EUR 5, false alarm EUR 15, missed fraud its amount.

## Model against baseline

| model | PR-AUC | recall @ 90% precision | recall @ 50% precision | threshold (from validation) | cost, EUR | saving vs flag nothing |
|---|---|---|---|---|---|---|
| LightGBM | 0.911 | 0.812 | 0.922 | 0.0107 | 27,130 | 75.2% |
| Logistic regression | 0.643 | 0.514 | 0.624 | 0.1888 | 86,848 | 20.8% |

Flagging nothing costs EUR 109,593: every fraud goes through.

## Cost curve (LightGBM)

| threshold | flagged | fraud caught | fraud missed | missed EUR | false alarms | total cost EUR |
|---|---|---|---|---|---|---|
| 0.0010 | 3050 | 654 | 11 | 1,680 | 2396 | 40,890 |
| 0.0050 | 1648 | 633 | 32 | 8,298 | 1015 | 26,688 |
| 0.0100 | 1268 | 616 | 49 | 13,924 | 652 | 26,784 |
| 0.0107 (chosen) | 1242 | 613 | 52 | 14,630 | 629 | 27,130 |
| 0.0200 | 1017 | 602 | 63 | 17,533 | 415 | 26,768 |
| 0.0500 | 783 | 576 | 89 | 26,030 | 207 | 32,015 |
| 0.1000 | 675 | 560 | 105 | 29,278 | 115 | 33,803 |
| 0.2500 | 591 | 538 | 127 | 36,994 | 53 | 40,479 |
| 0.5000 | 545 | 516 | 149 | 45,106 | 29 | 48,121 |

With hindsight, the cheapest threshold on this window is 0.0071 at EUR 24,708. The threshold chosen on validation costs 9.8% more.

## Decision bands

Review from 0.0107, decline from 0.2417.

| decision | transactions | fraud | fraud rate |
|---|---|---|---|
| approve | 125369 | 52 | 0.04% |
| review | 650 | 75 | 11.54% |
| decline | 592 | 538 | 90.88% |

## Recall by fraud pattern

| fraud pattern | transactions | recall at the review threshold |
|---|---|---|
| account_takeover | 167 | 80.8% |
| card_testing | 368 | 100.0% |
| foreign_high_value | 84 | 79.8% |
| new_merchant_night | 46 | 93.5% |

## Segments

| segment | transactions | fraud | PR-AUC | recall | false positive rate |
|---|---|---|---|---|---|
| amount_band = < 10 | 9972 | 357 | 1.000 | 100.0% | 0.08% |
| amount_band = 10 to 100 | 87131 | 48 | 0.434 | 62.5% | 0.14% |
| amount_band = 100 to 500 | 27365 | 197 | 0.753 | 87.8% | 1.46% |
| amount_band = >= 500 | 2143 | 63 | 0.736 | 84.1% | 4.90% |
| merchant_location = abroad | 11335 | 450 | 0.907 | 92.4% | 4.50% |
| merchant_location = home country | 115276 | 215 | 0.920 | 91.6% | 0.12% |
| customer_country = BE | 13312 | 97 | 0.956 | 95.9% | 0.58% |
| customer_country = DE | 12251 | 58 | 0.947 | 98.3% | 0.83% |
| customer_country = NL | 101048 | 510 | 0.900 | 90.8% | 0.45% |

## Calibration

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

## Global importance

| feature | mean abs SHAP (log-odds) |
|---|---|
| minutes_since_last | 0.719 |
| amount_zscore | 0.719 |
| merchant_is_new | 0.612 |
| is_foreign | 0.343 |
| hour | 0.231 |
| amount_log | 0.208 |
| amount_sum_24h_log | 0.160 |
| txn_count_24h | 0.115 |
| txn_count_1h | 0.050 |
| device_is_new | 0.023 |
| country_is_new | 0.010 |
| is_night | 0.007 |

