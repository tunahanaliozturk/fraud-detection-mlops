# The synthetic data

`uv run fraud generate` writes `data/transactions.parquet`. Defaults: 4,000 customers, 1,500 merchants, 120 days from
2026-01-01, seed 42. That gives 723,121 transactions, 3,516 of them fraud (0.49%). Same arguments, same rows.
Why synthetic at all: ADR 0002.

## Columns

| column | meaning |
|---|---|
| `transaction_id` | 1, 2, 3... in arrival order |
| `ts` | timestamp; treated as the cardholder's local time |
| `customer_id`, `customer_country` | the cardholder and the country the card was issued in (NL 80%, BE 10%, DE 10%) |
| `merchant_id`, `merchant_category`, `merchant_country` | ten categories, ten countries |
| `amount` | euros, at least 0.50 |
| `device_id` | the device or card the payment came from |
| `is_fraud` | the label |
| `fraud_pattern` | which pattern produced a fraud row; empty for genuine rows. Never a feature. |

## Genuine spending

Each customer gets a typical amount (log-normal around EUR 30), a spread, a daily rate (about 1.5 purchases a day on
average), a peak hour between 11:00 and 17:00 with a wide tail into the night, and 8 to 24 favourite merchants in
their home country. 88% of purchases go to favourites, the rest to any merchant in any country. Amounts scale by
category: a jeweller sees five times the typical amount, a digital-goods shop under half.

The noise that keeps fraud from being trivial:

- a quarter of customers take one trip abroad of 3 to 10 days, spending at new merchants in a new country;
- one in ten replaces their phone, so a new device appears without fraud;
- 1% of purchases are 3 to 10 times larger than usual;
- some genuine spending lands between midnight and 06:00.

## The four fraud patterns

One customer in five suffers one incident, at a random time after day 3. Each pattern is something an analyst would
recognise and a model could learn from the features.

| pattern | what happens | features that should notice |
|---|---|---|
| `card_testing` | 5 to 15 charges of EUR 0.50 to 3.00, seconds to a minute and a half apart, at one digital-goods merchant, from a new device. 30% end with one large electronics purchase. | `txn_count_1h`, `minutes_since_last`, `device_is_new`, `merchant_is_new` |
| `foreign_high_value` | 1 to 3 purchases at electronics, jewellery or travel merchants abroad, 4 to 25 times the usual amount. Same device: a cloned card. | `is_foreign`, `country_is_new`, `amount_zscore` |
| `new_merchant_night` | 1 or 2 purchases between midnight and 04:00 at a home-country merchant outside the customer's favourites, 5 to 20 times the usual amount. | `hour`, `is_night`, `merchant_is_new`, `amount_zscore` |
| `account_takeover` | a new device, then 2 to 5 online purchases minutes to two hours apart, at 2 to 8 times the usual amount. | `device_is_new`, `merchant_is_new`, `txn_count_24h` |

## Features

Twelve, all computed from the customer's history strictly before the transaction (ADR 0003):
`amount_log`, `hour`, `is_night`, `is_foreign`, `country_is_new`, `merchant_is_new`, `device_is_new`,
`txn_count_1h`, `txn_count_24h`, `amount_sum_24h_log`, `minutes_since_last` (capped at 30 days) and `amount_zscore`
(log amount against the customer's own mean and standard deviation, 0 until there are 5 earlier purchases).

Left out on purpose: merchant or category fraud rates (target encoding is easy to leak and would encode the
generator's choices), and the customer and merchant ids themselves.

## What this data cannot tell you

The fraudster never adapts, labels are complete and arrive instantly, every fraud follows one of four scripts, and the
incident rate per customer is far higher than any bank sees. Treat every metric in this repository as a measurement
of the pipeline, not as a forecast of production performance.
