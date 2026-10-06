# Model card: card-fraud, version 1

Written for a model risk reviewer. Every number is from `uv run fraud evaluate` on the default data, saved in
`docs/results/evaluation.md`; the same report is logged to MLflow with every model version as `model_card.md`.

## Intended use

Score a card payment at authorisation time and put it in one of three bands: approve, send to an analyst for review,
or decline. Each score carries three reason codes for the analyst and for the customer conversation.

Not intended for: decisions about a customer rather than a payment (credit, account closure, AML reporting), any
use on real card data before the steps under "Before real use", or as a ranking of countries or merchants by risk.

## Model

LightGBM gradient-boosted trees, binary objective, unweighted classes, early stopping on validation PR-AUC (355
trees). Twelve features, each computed from the cardholder's own history strictly before the payment (`docs/data.md`,
ADR 0003). The baseline is a logistic regression on the same standardised features.

The decision policy is part of the model version (`policy.json`): review from a score of 0.0107, the threshold that
minimised expected cost on the validation window with a review costing EUR 5 and a false alarm EUR 15 (review plus
customer friction), a missed fraud costing its amount. Decline from 0.2417, the lowest score at which validation
precision reached 90%.

## Data

Synthetic, seeded, 723,121 transactions over 120 days, 0.49% fraud in four scripted patterns (ADR 0002). Split by
time, never at random: 14 days of history warm-up, then training (383,984 transactions), validation from 2026-03-19
(127,999) and test from 2026-04-09 21:16 (126,611, 665 fraud). Thresholds come from validation and are applied
unchanged to test.

## Performance on the test window

| | LightGBM | logistic regression |
|---|---|---|
| PR-AUC | 0.911 | 0.643 |
| recall at 90% precision | 0.812 | 0.514 |
| expected cost at its own validation threshold | EUR 27,130 | EUR 86,848 |
| saving against flagging nothing (EUR 109,593) | 75.2% | 20.8% |

| band | transactions | fraud | fraud rate |
|---|---|---|---|
| approve | 125,369 | 52 | 0.04% |
| review | 650 | 75 | 11.5% |
| decline | 592 | 538 | 90.9% |

Recall at the review threshold by pattern: card testing 100%, new merchant at night 93.5%, account takeover 80.8%,
foreign high value 79.8%.

## Performance by segment

| segment | transactions | fraud | PR-AUC | recall | false positive rate |
|---|---|---|---|---|---|
| amount under EUR 10 | 9,972 | 357 | 1.000 | 100.0% | 0.08% |
| EUR 10 to 100 | 87,131 | 48 | 0.434 | 62.5% | 0.14% |
| EUR 100 to 500 | 27,365 | 197 | 0.753 | 87.8% | 1.46% |
| EUR 500 and over | 2,143 | 63 | 0.736 | 84.1% | 4.90% |
| merchant abroad | 11,335 | 450 | 0.907 | 92.4% | 4.50% |
| merchant in home country | 115,276 | 215 | 0.920 | 91.6% | 0.12% |
| cards issued in BE | 13,312 | 97 | 0.956 | 95.9% | 0.58% |
| cards issued in DE | 12,251 | 58 | 0.947 | 98.3% | 0.83% |
| cards issued in NL | 101,048 | 510 | 0.900 | 90.8% | 0.45% |

What a reviewer should take from this table:

- **Travellers carry the false alarms.** A genuine payment abroad is flagged 4.5% of the time, against 0.12% at home:
  about 37 times as often. In a real portfolio that lands on customers who travel, and needs a decision about whether
  the friction is acceptable or whether travel notices and device history should offset it.
- **Mid-sized fraud is the weak spot.** Between EUR 10 and 100, recall is 62.5% and PR-AUC 0.434: the amount is
  unremarkable, so the model has less to go on. Only 48 fraud transactions sit in that band, so the number itself is
  uncertain.
- **Large genuine purchases are flagged often** (4.9% above EUR 500). That is the cost-based threshold doing what it
  was asked: a missed EUR 2,000 fraud costs more than many reviews.
- **Small amounts look perfect** only because card testing is the most scripted pattern in the generator.
- Differences by issuing country are small and come from the generator, not from anything about those countries.

## Calibration

Scores are usable as rough probabilities at the extremes and under-predict in the middle: between 0.01 and 0.05 the
mean score is 0.022 and the observed fraud rate 0.083; between 0.05 and 0.10, 0.071 against 0.148. Above 0.75 they
match (0.983 against 0.961). The expected-cost threshold is chosen on observed outcomes, so it is not affected, but
anyone reading a score as "a 2% chance" would be wrong by a factor of about four. Isotonic calibration on the
validation window is the obvious next step.

## Explanations

Per decision: the three largest SHAP contributions with direction and fixed wording (ADR 0005). Globally, by mean
absolute SHAP: time since the previous transaction (0.719), amount against the customer's usual (0.719), new merchant
(0.612), merchant abroad (0.343), hour (0.231). Device novelty ranks low (0.023) despite driving card testing and
account takeover, because velocity features already separate those patterns; this is a correlation effect worth
knowing before reading global importance as "what matters".

SHAP describes the model, not cause and effect. "The merchant is abroad" means the model leaned on that feature for
this payment.

## Limitations

- Synthetic data: scores are optimistic and no real fraud pattern has been tested (ADR 0002).
- Labels are complete and instant. Real chargebacks arrive weeks later and some fraud is never reported, which biases
  training towards the fraud that gets noticed.
- One test window of three weeks, 665 fraud transactions. Segment numbers rest on as few as 48 frauds.
- The champion was chosen by comparing costs on the same test window it is reported on (ADR 0006).
- No adversary: fraudsters adapt to whatever the model flags.
- No fairness analysis beyond the segments above. The data has no protected attributes, and real data would need
  proxies checked (country, merchant category, device type).

## Before real use

1. Train and validate on real, properly labelled data, with label delay modelled and an out-of-time test window that
   no selection decision has touched.
2. Bootstrap confidence intervals for every headline and segment number.
3. Calibrate the scores, then re-derive the thresholds with costs agreed with fraud operations and customer service.
4. Shadow mode next to the current system, comparing alerts and analyst outcomes, before any customer is affected.
5. Independent model validation, documented under the bank's model risk policy, including the reason-code wording
   reviewed by legal and customer communications.
6. Monitoring with owners and actions: PSI and score drift daily (`fraud drift --fail-on-alert`), realised precision
   in the review queue weekly, recall once chargebacks mature.
