# 4. Class imbalance handled by a cost-based threshold, not by re-weighting

Status: Accepted, 2026-10-06

## Context

Fraud is 0.49% of transactions. The usual responses are to re-weight or resample the rare class, or to train on the
data as it is and choose the operating point afterwards. A bank cares about money and analyst time, not about a
symmetric loss.

## Decision

LightGBM trains unweighted (`scale_pos_weight = 1`) with early stopping on validation PR-AUC. The imbalance is handled
at decision time: the review threshold is the one that minimises expected cost on the validation window, where a
missed fraud costs its amount, a reviewed transaction costs EUR 5 and a false alarm costs another EUR 10 of customer
friction (`metrics.cost_minimising_threshold`; `--review-cost` and `--friction-cost` change them). Accuracy is not
reported anywhere.

## Consequences

- The sweep in `docs/results/pos-weight-sweep.md` backs it on the window we choose on: weight 1 has the lowest
  validation cost (EUR 22,144) and the highest validation PR-AUC (0.931) of weights 1, 5, 20 and 50.
- The cost is real: on the test window weight 20 was 15% cheaper (EUR 23,079 against 27,130). We read that as noise
  at 665 fraud transactions and did not choose on the test window, but a reviewer could fairly argue it.
- Unweighted scores stay close to probabilities, which the calibration check relies on. They are not perfect: scores
  between 0.01 and 0.05 under-predict (mean 0.022, observed 0.083 on the test window).
- The threshold belongs to the costs. Change the review cost and the threshold moves, without retraining.

## Alternatives

- **Re-weighting or oversampling (SMOTE and friends).** Moves the boundary inside the model, distorts the scores as
  probabilities, and still needs a threshold. Would win if the output were consumed as a hard label.
- **Amount-weighted training,** each fraud weighted by its euros. Aligns the loss with the cost directly. Worth trying
  next, with its own validation sweep.
- **A fixed threshold of 0.5.** Costs EUR 48,121 on the test window against 27,130 for the chosen one (cost curve in
  `docs/results/evaluation.md`).
