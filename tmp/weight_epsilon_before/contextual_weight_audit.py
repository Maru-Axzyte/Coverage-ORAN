"""Small, auditable extension of the existing contextual hinge learner.

No new resource preference is invented. Labels remain exact, equal-service
Pareto comparisons, which cannot identify the best non-dominated trade-off.
Contexts are captured before exact evaluation; repeated cached placements do
not create extra evidence. Prequential metrics are measured BEFORE fitting.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
import cvxpy as cp

from preference_weights import ContextualPreferencePair, fit_history_contextual_weights


@dataclass(frozen=True)
class WeightObservation:
    key: tuple
    resource: np.ndarray
    served: int
    service: float
    context: np.ndarray


class AuditedContextualWeights:
    context_names = ("intercept", "served_fraction", "total_power_ratio",
                     "global_prb_ratio", "peak_uav_power_ratio")

    def __init__(self, initial_weights):
        weights = np.asarray(initial_weights, dtype=float)
        if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.min(weights) < 0 or weights.sum() <= 0:
            raise ValueError("Three finite nonnegative prior weights are required")
        self.prior = np.zeros((3, len(self.context_names)))
        self.prior[:, 0] = weights / weights.sum()
        self.matrix = self.prior.copy()
        self.average_matrix = self.prior.copy()  # Includes B0 exactly once.
        self.successful_updates = 0
        self.observations = []
        self.seen = set()
        self.pending = []
        self.pairs = []
        self.duplicates = 0

    def observe(self, key, resource, served, service, violation, context):
        resource = np.asarray(resource, dtype=float).copy()
        context = np.asarray(context, dtype=float).copy()
        if (resource.shape != (3,) or context.shape != (5,)
                or not np.all(np.isfinite(resource)) or not np.all(np.isfinite(context))
                or not np.isfinite(service) or not np.isfinite(violation)):
            raise ValueError("Invalid exact observation")
        if abs(context[0] - 1) > 1e-9 or np.min(context[1:]) < 0 or np.max(context[1:]) > 1:
            raise ValueError("Context must lie in the declared unit box")
        if violation > 1e-12:
            return
        if key in self.seen:
            self.duplicates += 1
            return
        self.seen.add(key)
        resource.setflags(write=False)
        context.setflags(write=False)
        item = WeightObservation(key, resource, int(served), float(service), context)
        for other in self.observations[-80:]:
            if other.served != served or abs(other.service - service) > 1e-9:
                continue
            distance = float(np.linalg.norm(resource - other.resource))
            if distance < 0.002:
                continue
            first = np.all(resource <= other.resource + 1e-9) and np.any(resource < other.resource - 1e-9)
            second = np.all(other.resource <= resource + 1e-9) and np.any(other.resource < resource - 1e-9)
            if not first and not second:
                continue
            better, worse = (resource, other.resource) if first else (other.resource, resource)
            self.pending.append((distance, ContextualPreferencePair(
                better=better.copy(), worse=worse.copy(), context=context.copy(),
                margin=min(0.02, max(0.003, 0.10 * distance)),
            )))
        self.observations.append(item)

    @staticmethod
    def context_importance(pairs, current_context):
        """varpi = 1 - mean squared context distance; exclude intercept.

        No positive floor is invented. Opposite corners may have zero weight;
        an entirely zero-weight batch must abstain rather than divide by zero.
        """
        current = np.asarray(current_context, dtype=float)
        if (current.shape != (5,) or not np.all(np.isfinite(current))
                or abs(current[0] - 1.) > 1e-9
                or np.any(current[1:] < 0) or np.any(current[1:] > 1)):
            raise ValueError("Current context must be [1, four features in [0,1]]")
        return np.asarray([1. - np.mean((current[1:] - p.context[1:]) ** 2)
                           for p in pairs])

    def update(self, current_context=None):
        if current_context is None:
            current_context = (self.observations[-1].context if self.observations
                               else np.array([1., 0., 0., 0., 0.]))
        # Validate before consuming pending observations.
        self.context_importance([], current_context)
        average_before = self.average_matrix.copy()
        previous = self.matrix.copy()
        # Same pair-window and per-update limits as the previous implementation.
        self.pending.sort(key=lambda entry: entry[0], reverse=True)
        new_pairs = [pair for _, pair in self.pending[:64]]
        self.pending = []
        predicted = [float((self.matrix @ p.context) @ (p.worse - p.better)) for p in new_pairs]
        pre_loss = (float(np.mean([max(0., p.margin - score) for p, score in zip(new_pairs, predicted)]))
                    if new_pairs else None)
        self.pairs = (self.pairs + new_pairs)[-256:]
        importance = self.context_importance(self.pairs, current_context)
        status = "no_new_pairs" if len(self.pairs) >= 3 and not new_pairs else "insufficient_exact_pairs"
        train_loss = None
        if len(self.pairs) >= 3 and new_pairs:
            if importance.sum() <= 0:
                status = "zero_context_weight"
            else:
                try:
                    fit = fit_history_contextual_weights(
                        self.pairs, average=average_before, previous=previous,
                        pair_weights=importance,
                    )
                    if fit.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        raise ValueError("Weight solve did not succeed")
                    fitted = np.asarray(fit.vartheta, dtype=float)
                    if fitted.shape != self.matrix.shape:
                        raise ValueError("Fitted matrix has wrong shape")
                    corners = np.asarray([[1., *v] for v in product((0., 1.), repeat=4)])
                    outputs = corners @ fitted.T
                    if (not np.all(np.isfinite(outputs)) or np.min(outputs) < -1e-6
                            or not np.allclose(outputs.sum(axis=1), 1., atol=1e-6, rtol=0)):
                        raise ValueError("Fitted matrix violates simplex constraints")
                    self.matrix = fitted.copy()
                    self.successful_updates += 1
                    self.average_matrix = average_before + (
                        self.matrix - average_before) / (self.successful_updates + 1)
                    status, train_loss = fit.status, fit.mean_pair_loss
                except (ValueError, RuntimeError, cp.error.SolverError) as error:
                    status = f"fit_failed: {type(error).__name__}: {error}"
        contexts = np.asarray([p.context for p in self.pairs])
        return {
            "status": status, "new_pairs": len(new_pairs), "training_pairs": len(self.pairs),
            "mean_pair_loss": train_loss, "prequential_hinge": pre_loss,
            "prequential_correct": sum(score > 1e-12 for score in predicted),
            "prequential_pairs": len(predicted),
            "context_rank": int(np.linalg.matrix_rank(contexts)) if self.pairs else 0,
            "context_dimension": 5, "unique_exact_observations": len(self.observations),
            "duplicate_observations_ignored": self.duplicates,
            "successful_updates": self.successful_updates,
            "average_matrix_before": average_before.tolist(),
            "average_matrix": self.average_matrix.tolist(),
            "previous_matrix": previous.tolist(),
            "lambda_avg": 0.08, "lambda_time": 0.20,
            "regularizer_convention": "lambda/2 * squared_Frobenius_norm",
            "pair_context_weights": importance.tolist(),
            "pair_context_weight_sum": float(importance.sum()),
            "context_distance_dimensions": 4,
            "preference_source": "exact_equal_service_pareto_dominance",
            "tradeoff_labels": 0,
            "identifiability_note": "Dominance-only labels do not identify optimal trade-off weights.",
        }
