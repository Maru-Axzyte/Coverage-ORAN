# Contextual weights: two-stage update

The default `contextual` learner now solves two CVXPY problems at each existing
weight-update event. Run `main.py` as before. No radio parameters, GA settings,
MILP budgets, reproduction, CMA updates, or Deb comparison rules are changed.
`--weight-method fixed` still bypasses learning.

## Objective

Let `w(c) = B @ c` and `r = [normalized power, global PRB load, bottleneck]`.
The resource fitness remains `-w(c) @ r`.

1. Minimize the weighted exact-preference hinge loss `L_pref(B)` alone.
2. Minimize `0.5 * weighted_mean((error_delta @ B @ c)**2)` plus the existing
   history and temporal penalties (`0.08/2` and `0.20/2` times squared Frobenius
   distances), subject to keeping `L_pref` at the first-stage optimum within
   numerical tolerance.

`error_delta = (predicted_r_b - exact_r_b) - (predicted_r_a - exact_r_a)`.
The primary objective is the existing context-weighted hinge, not an unweighted
loss. Prediction comparisons are context-weighted separately. Simplex
constraints hold at all normalized context corners. The aggregate preference
loss cap does not protect every individual pair when the optimum is nonzero.

The first-stage reference is its numerically attained hinge value. The reported
cap allows `1e-7` numerical tolerance, not an operational relaxation. Both
solver statuses are logged; results are checked for finite values, simplex
feasibility and the measured hinge cap before committing. A failed solve keeps
the previous matrix and history average. No clipping of the fitted matrix is
used to hide a failed constraint check.

## Forecast provenance

Actual surrogate resource predictions, their frozen context and prediction
batch are captured before MILP replaces predictions with exact results.
Warm-up seeds and exact-cache lookups are not forecasts. The collector copies
arrays so subsequent mutation cannot change historical evidence.

Calibration pairs require two pre-MILP forecasts from the same frozen batch
and context, with equal exact served count and service score. They need not
have a Pareto preference label. V is metadata, not a learning filter.
The existing 80-observation look-back and 64/256 pair limits are reused for the
new calibration buffer; exact-preference selection is unchanged. Unique exact
observations are still deduplicated. No extra MILP evaluations are requested.

No calibration evidence means a zero *objective term*, not synthetic zero-error
records: diagnostics report absent MSE as `null`. Stage 2 then selects by the
existing regularizers within the protected preference set. The existing
minimum of three preference pairs remains required. New calibration evidence
can trigger an update even if no new exact preference pairs were formed.

## Diagnostics

`weight_learning_history` and the existing result JSON include:

- `learning_objective`: `lexicographic_preference_then_prediction`;
- `stage1_status`, `stage1_pair_loss`, `mean_pair_loss`;
- `preference_loss_limit`, `preference_tolerance`;
- `prediction_observations`, `prediction_pairs`, `new_prediction_pairs`;
- `prediction_mse_before`, `prediction_mse_after`: weighted in-buffer comparison
  MSE under the old/new matrices (training diagnostics, not a generalization claim);
- `prequential_prediction_mse`: MSE on newly formed pairs under the matrix
  frozen before this update (audited subset only, not the whole population).

Console lines show `prediction_pairs`, `pref=before->after`, and, when available,
`pred_MSE=before->after`. The new objective need not reduce prediction MSE alone:
regularization and primary-loss protection can constrain that improvement.

This update selects more reliably predicted scores among weights compatible
with observed preferences. It does not identify an operator's optimal unseen
power/PRB trade-off, guarantee that weights move, or prove placement optimality.
The exact-preference labels remain Pareto-only; calibration data are not new
trade-off preference labels. A larger weight change is not a success criterion.

## Verification

`python -m unittest discover -p "test_*.py" -v`

The added tests use synthetic resource data and a fake allocation evaluator;
they verify the two-stage loss cap, calibration, frozen forecast capture,
same-context pairing, cache deduplication, and transactional failure handling.
No full GA/MILP experiment is needed to run these new tests.
