# 2. Synthetic data instead of a public dataset

Status: Accepted, 2026-10-06

## Context

A public repository needs data anyone can download and redistribute, and features a person can explain. The well
known card-fraud datasets each fail one of those:

- The ULB credit card set on Kaggle has 28 anonymised PCA components plus time and amount. Nothing in it can become
  a reason code a customer understands, and none of the point-in-time feature work can be shown on it.
- The IEEE-CIS (Vesta) competition data is bound by competition rules that restrict use outside the competition.
- PaySim is CC BY-SA 4.0 (share-alike) and simulates mobile money, not card payments.

## Decision

We generate the data (`fraud_detection/generate.py`): one seeded numpy generator, customers with their own typical
amount, spread, waking hours and favourite merchants, and four injected fraud patterns described in `docs/data.md`.
Legitimate behaviour is deliberately noisy: a quarter of customers travel abroad, one in ten changes phone, 1% of
purchases are unusually large, and some spending happens at night.

## Consequences

- The data is reproducible to the row (`test_the_same_seed_gives_identical_rows`), MIT like the code, and has
  readable columns that make reason codes meaningful.
- Scores are optimistic. The patterns are cleaner than real fraud, the fraudster does not adapt, and labels are
  complete and instant. A PR-AUC of 0.911 here says the pipeline works, not that a bank would see 0.911.
- 20% of customers suffer one incident in four months, far above reality, so that the test window holds enough fraud
  (665 transactions) to measure anything. The transaction-level fraud rate is still realistic at 0.49%.
- Any country effect the model learns is an artefact: the fraud patterns pick foreign countries uniformly.

## Alternatives

- **ULB credit card data.** Real, but the features cannot be explained or engineered. Would win for a pure
  modelling benchmark.
- **PaySim.** Open, but share-alike and the wrong payment type. Would win for a mobile money project.
