"""Convex, regularised learning of resource-preference weights.

This module deliberately does *not* allocate users or recompute radio links.
Those remain the responsibilities of :mod:`env` and :mod:`milp`.  Given
MILP-validated preference pairs, it learns the resource scalarisation

    cost(r) = beta * p_bar + mu * n_bar + eta * z_plus (+ lambda_HO * h_bar)

on the probability simplex.  The default loss is the Lipschitz pairwise hinge
loss used by regularised pairwise-ranking theory.  ``squared_hinge`` is kept
only as the earlier, more outlier-sensitive formulation for ablation tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

import cvxpy as cp
import numpy as np


LossKind = Literal["hinge", "squared_hinge"]


@dataclass(frozen=True)
class PreferencePair:
    """One exact preference: ``better`` should cost less than ``worse``."""

    better: np.ndarray
    worse: np.ndarray
    margin: float = 0.02
    confidence: float = 1.0


@dataclass(frozen=True)
class PreferenceFit:
    weights: np.ndarray
    status: str
    objective_value: float
    mean_pair_loss: float
    loss_kind: LossKind


@dataclass(frozen=True)
class ContextualPreferencePair:
    """MILP-confirmed resource preference under one shared system context."""

    better: np.ndarray
    worse: np.ndarray
    context: np.ndarray
    margin: float = 0.01


@dataclass(frozen=True)
class ContextualPreferenceFit:
    vartheta: np.ndarray
    status: str
    mean_pair_loss: float
    stage1_status: str | None = None
    stage1_pair_loss: float | None = None
    preference_loss_limit: float | None = None
    preference_tolerance: float | None = None
    prediction_mse_before: float | None = None
    prediction_mse_after: float | None = None


@dataclass(frozen=True)
class ContextualPredictionPair:
    """Difference of two pre-MILP residuals under the SAME frozen context.

    error_delta = (rhat_b - r_b) - (rhat_a - r_a). No preference label
    is inferred from this residual or from the current resource weights.
    """

    error_delta: np.ndarray
    context: np.ndarray


def fit_contextual_pairwise_weights(
    pairs: Sequence[ContextualPreferencePair], *,
    prior: np.ndarray,
    previous: np.ndarray,
    prior_strength: float = 0.08,
    temporal_strength: float = 0.20,
    solver: str = "CLARABEL",
    pair_weights: Sequence[float] | None = None,
) -> ContextualPreferenceFit:
    """Learn ``w(c) = vartheta @ c`` by a convex pairwise hinge problem.
    """
    prior = np.asarray(prior, dtype=float)
    previous = np.asarray(previous, dtype=float)
    context_size = prior.shape[1]

    design: list[np.ndarray] = []
    margins: list[float] = []
    for pair in pairs:
        better = np.asarray(pair.better, dtype=float)
        worse = np.asarray(pair.worse, dtype=float)
        context = np.asarray(pair.context, dtype=float)
        design.append(np.outer(worse - better, context).ravel())
        margins.append(float(pair.margin))

    matrix = np.vstack(design)
    margin_vector = np.asarray(margins)
    importance = np.ones(len(pairs)) if pair_weights is None else np.asarray(pair_weights, dtype=float)
    importance = importance / importance.sum()
    vartheta = cp.Variable(prior.shape, name="contextual_vartheta")
    slack = cp.Variable(len(pairs), nonneg=True, name="contextual_pair_slack")
    constraints = [
        cp.sum(vartheta[:, 0]) == 1.0,
        *(cp.sum(vartheta[:, column]) == 0.0 for column in range(1, context_size)),
        slack >= margin_vector - matrix @ cp.reshape(vartheta, (prior.size,), order="C"),
    ]
    for features in product((0.0, 1.0), repeat=context_size - 1):
        constraints.append(vartheta @ np.asarray((1.0, *features)) >= 0.0)
    objective = cp.Minimize(
        importance @ slack
        + prior_strength * cp.sum_squares(vartheta - prior)
        + temporal_strength * cp.sum_squares(vartheta - previous)
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, warm_start=True)
    except cp.error.SolverError:
        problem.solve(solver="OSQP", warm_start=True)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or vartheta.value is None:
        raise RuntimeError(f"Contextual weight fitting failed with status {problem.status}")
    fitted = np.asarray(vartheta.value, dtype=float)
    residual = np.maximum(0.0, margin_vector - matrix @ fitted.ravel())
    return ContextualPreferenceFit(fitted, str(problem.status), float(importance @ residual))


def fit_history_contextual_weights(
    pairs: Sequence[ContextualPreferencePair], *,
    average: np.ndarray,
    previous: np.ndarray,
    pair_weights: Sequence[float],
    lambda_avg: float = 0.08,
    lambda_time: float = 0.20,
    solver: str = "CLARABEL",
    prediction_pairs: Sequence[ContextualPredictionPair] = (),
    prediction_weights: Sequence[float] | None = None,
) -> ContextualPreferenceFit:
    """Lexicographic fit: minimum preference hinge, then prediction error.

    Stage 1 minimizes the existing weighted exact-preference hinge ALONE.
    Stage 2 minimizes half weighted prediction MSE plus the unchanged
    lambda/2 history and temporal penalties, subject to preserving stage 1
    within numerical tolerance. The cap protects aggregate hinge, not every
    individual pair when its optimum is nonzero.

    No predictions at warm-up: stage 2 uses only the regularizers, not fake
    zero residual observations. Both solves are transactional to the caller.
    """
    average = np.asarray(average, dtype=float)
    previous = np.asarray(previous, dtype=float)
    importance = np.asarray(pair_weights, dtype=float)
    if (not pairs or importance.shape != (len(pairs),)
            or not np.all(np.isfinite(importance)) or np.any(importance < 0.)
            or importance.sum() <= 0.):
        raise RuntimeError("Two-stage weights need positive finite preference evidence")
    importance = importance / importance.sum()
    design = np.asarray([np.outer(p.worse-p.better, p.context).ravel() for p in pairs])
    margins = np.asarray([p.margin for p in pairs], dtype=float)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(margins)):
        raise RuntimeError("Non-finite preference evidence")

    error_design = np.empty((0, average.size))
    error_importance = np.empty(0)
    if prediction_pairs:
        error_design = np.asarray([
            np.outer(p.error_delta, p.context).ravel() for p in prediction_pairs])
        error_importance = (np.ones(len(prediction_pairs)) if prediction_weights is None
                            else np.asarray(prediction_weights, dtype=float))
        if (error_importance.shape != (len(prediction_pairs),)
                or not np.all(np.isfinite(error_design))
                or not np.all(np.isfinite(error_importance))
                or np.any(error_importance < 0.)):
            raise RuntimeError("Invalid prediction-comparison evidence")
        if error_importance.sum() > 0.:
            error_importance = error_importance / error_importance.sum()
        else:
            error_design = np.empty((0, average.size))
            error_importance = np.empty(0)

    dimension = average.shape[1]
    corners = np.asarray([[1., *v] for v in product((0., 1.), repeat=dimension-1)])
    matrix = cp.Variable(average.shape, name="two_stage_contextual_weights")
    flat = cp.reshape(matrix, (average.size,), order="C")
    hinge = importance @ cp.pos(margins-design @ flat)
    constraints = [cp.sum(matrix[:, 0]) == 1., corners @ matrix.T >= 0.]
    constraints.extend(cp.sum(matrix[:, j]) == 0. for j in range(1, dimension))

    def measured_hinge(value):
        return float(importance @ np.maximum(0., margins-design @ value.ravel()))

    def valid_simplex(value):
        if value is None or value.shape != average.shape or not np.all(np.isfinite(value)):
            return False
        outputs = corners @ value.T
        return (np.min(outputs) >= -1e-7
                and np.allclose(outputs.sum(axis=1), 1., atol=1e-7, rtol=0.))

    def solve_checked(problem, loss_limit=None):
        # Numerical tolerances only: not a tunable relaxation of WE or service.
        for backend in dict.fromkeys((solver, "CLARABEL", "OSQP")):
            options = ({"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
                       if backend == "CLARABEL" else
                       {"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 100000}
                       if backend == "OSQP" else {})
            try:
                problem.solve(solver=backend, warm_start=True, **options)
            except cp.error.SolverError:
                continue
            value = None if matrix.value is None else np.asarray(matrix.value, dtype=float)
            if (problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and valid_simplex(value)
                    and (loss_limit is None or measured_hinge(value) <= loss_limit)):
                return value.copy(), str(problem.status)
        raise RuntimeError("Two-stage weight solve failed numerical feasibility checks")

    first, first_status = solve_checked(cp.Problem(cp.Minimize(hinge), constraints))
    first_loss = measured_hinge(first)
    tolerance = 1e-7
    # Reserve half the numerical budget for solver residuals; independently
    # reject the second solution if the measured hinge exceeds the full cap.
    protected = constraints + [hinge <= first_loss + tolerance/2.]
    prediction_loss = (0.5*cp.sum(cp.multiply(error_importance, cp.square(error_design @ flat)))
                       if len(error_importance) else cp.Constant(0.))
    objective = cp.Minimize(
        prediction_loss
        + lambda_avg/2.*cp.sum_squares(matrix-average)
        + lambda_time/2.*cp.sum_squares(matrix-previous)
    )
    fitted, status = solve_checked(cp.Problem(objective, protected), first_loss+tolerance)

    def prediction_mse(value):
        return (float(error_importance @ np.square(error_design @ value.ravel()))
                if len(error_importance) else None)

    return ContextualPreferenceFit(
        fitted, status, measured_hinge(fitted), first_status, first_loss,
        first_loss+tolerance, tolerance,
        prediction_mse(previous), prediction_mse(fitted),
    )


def resource_cost(weights: np.ndarray, resource: np.ndarray) -> float:
    """Lower is operationally better; all resource components are normalised."""
    return float(np.asarray(weights, dtype=float) @ np.asarray(resource, dtype=float))


def fit_pairwise_weights(
    pairs: Sequence[PreferencePair], *,
    prior: np.ndarray,
    previous: np.ndarray | None = None,
    prior_strength: float = 0.08,
    temporal_strength: float = 0.20,
    loss_kind: LossKind = "hinge",
    min_weight: float = 0.0,
    solver: str = "CLARABEL",
) -> PreferenceFit:
    """Fit simplex-constrained weights from exact pairwise preferences.

    The empirical pairwise loss is divided by the sum of confidences so that
    adding many correlated pairs does not silently change the regularisation
    scale.  A nonzero quadratic prior/temporal term makes the objective
    strongly convex in the unconstrained directions.
    """

    prior = np.asarray(prior, dtype=float).reshape(-1)
    prior = np.maximum(prior, 0.0)
    prior /= prior.sum()

    previous_value = prior if previous is None else np.asarray(previous, dtype=float).reshape(-1)
    previous_value = np.maximum(previous_value, 0.0)
    if previous_value.sum() <= 0:
        previous_value = prior.copy()
    else:
        previous_value /= previous_value.sum()

    dimension = prior.size
    deltas = []
    margins = []
    confidences = []
    for pair in pairs:
        # cost(better) + margin <= cost(worse)
        deltas.append(np.asarray(pair.worse - pair.better, dtype=float))
        margins.append(float(pair.margin))
        confidences.append(float(pair.confidence))
    delta = np.vstack(deltas)
    margin = np.asarray(margins)
    confidence = np.asarray(confidences)
    normaliser = float(confidence.sum())

    weights = cp.Variable(dimension, nonneg=True, name="resource_weights")
    slack = cp.Variable(len(pairs), nonneg=True, name="pairwise_hinge_slack")
    constraints = [cp.sum(weights) == 1.0, weights >= min_weight]
    constraints += [slack >= margin - delta @ weights]

    if loss_kind == "hinge":
        empirical_loss = cp.sum(cp.multiply(confidence, slack)) / normaliser
    else:
        empirical_loss = cp.sum(cp.multiply(confidence, cp.square(slack))) / normaliser
    objective = cp.Minimize(
        empirical_loss
        + prior_strength * cp.sum_squares(weights - prior)
        + temporal_strength * cp.sum_squares(weights - previous_value)
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, warm_start=True)
    except cp.error.SolverError:
        # OSQP is a useful fallback for the QP form when CLARABEL is absent.
        problem.solve(solver="OSQP", warm_start=True)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or weights.value is None:
        raise RuntimeError(f"Pairwise weight fitting failed with status {problem.status}")

    fitted = np.asarray(weights.value, dtype=float).reshape(-1)
    fitted = np.maximum(fitted, 0.0)
    fitted /= fitted.sum()
    residual = np.maximum(0.0, margin - delta @ fitted)
    if loss_kind == "hinge":
        mean_pair_loss = float(confidence @ residual / normaliser)
    else:
        mean_pair_loss = float(confidence @ np.square(residual) / normaliser)
    return PreferenceFit(
        weights=fitted,
        status=str(problem.status),
        objective_value=float(problem.value),
        mean_pair_loss=mean_pair_loss,
        loss_kind=loss_kind,
    )
