# 5. Reason codes from SHAP values

Status: Accepted, 2026-10-06

## Context

In a bank a fraud score is not enough. An analyst reviewing a flagged payment needs to see why. A customer whose card
was declined is owed an explanation, much as credit decisions owe an adverse-action reason. A model risk reviewer
wants evidence that the model leans on features that make sense, globally and case by case.

## Decision

Every score comes with three reason codes: the three features with the largest absolute SHAP value for that
transaction, each with its direction (`increases_risk` or `decreases_risk`), its contribution in log-odds and a
sentence in plain words from a fixed table (`explain.WORDING`). SHAP values come from `shap.TreeExplainer`. Global
importance is the mean absolute SHAP value over 10,000 test transactions, printed by `fraud evaluate`.

## Consequences

- The wording is chosen by direction, not by value, so a reason never contradicts the sign of its number
  (`test_reason_directions_match_the_sign_of_the_shap_values`, and a unit test with hand-set contributions).
- The values add up: SHAP values plus the expected value equal the model's raw score to 1e-6
  (`test_shap_values_add_up_to_the_model_output`).
- Reasons include risk-lowering factors. An approved transaction still gets reasons, and a reviewer sees what spoke
  for the customer as well as against.
- For a LightGBM model, `TreeExplainer` hands the computation to LightGBM's own TreeSHAP (`pred_contrib=True`).
  The `shap` package therefore buys a standard, recognisable interface at the price of numba and llvmlite in the
  image. Dropping it would be a one-line change in `explain.py` and would take llvmlite and numba (207 MB together,
  measured with `du` in the built image) out of the image.
- SHAP explains the model, not the world. "The merchant is abroad" means the model leaned on that feature; it does
  not mean being abroad causes fraud. The model card says so.

## Alternatives

- **LightGBM `pred_contrib` directly.** Same numbers, fewer dependencies. Would win if image size or start-up time
  mattered more than a recognisable explainer API.
- **Global feature importance only.** Cheaper, but cannot explain a single decision, which is the case that matters.
- **Rule-based reasons** (if the amount z-score is above 3, say "unusual amount"). Readable, but they can disagree
  with what the model actually used.
