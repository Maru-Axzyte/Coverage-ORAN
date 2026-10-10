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
) -> ContextualPreferenceFit:
    """The merged note: weighted hinge + lambda/2 Frobenius penalties.

    ``average`` is the arithmetic mean of B0 and successful past fits, frozen
    before this solve. Each pair retains its historical context. The original
    fitting API keeps its old penalty convention for legacy experiments.
    """
    return fit_contextual_pairwise_weights(
        pairs, prior=average, previous=previous, pair_weights=pair_weights,
        prior_strength=lambda_avg / 2.0, temporal_strength=lambda_time / 2.0,
        solver=solver,
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
