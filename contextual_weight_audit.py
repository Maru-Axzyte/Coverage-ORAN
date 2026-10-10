"""Small, auditable extension of the existing contextual hinge learner.

No new resource preference is invented. Labels remain exact, equal-service
Pareto comparisons, which cannot identify the best non-dominated trade-off.
Contexts are captured before exact evaluation; repeated cached placements do
not create extra evidence. Prequential metrics are measured BEFORE fitting.
Violation is retained as metadata only; WE, not this learner, handles feasibility.
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
    violation: float


class AuditedContextualWeights:
    context_names = ("intercept", "served_fraction", "total_power_ratio",
                     "global_prb_ratio", "peak_uav_power_ratio")

    def __init__(self, initial_weights):
        weights = np.asarray(initial_weights, dtype=float)
        self.prior = np.zeros((3, len(self.context_names)))
        self.prior[:, 0] = weights / weights.sum()
        self.matrix = self.prior.copy()
        self.average_matrix = self.prior.copy()  # Includes B0 exactly once.
        self.successful_updates = 0
        self.observations = []
        self.seen = set()
        self.pairs = []
        self._paired_observations = set()
        self.duplicates = 0

    def observe(self, key, resource, served, service, violation, context):
        resource = np.asarray(resource, dtype=float).copy()
        context = np.asarray(context, dtype=float).copy()
        if key in self.seen:
            self.duplicates += 1
            return
        self.seen.add(key)
        resource.setflags(write=False)
        context.setflags(write=False)
        # V is diagnostic metadata, not a filter or a resource-preference label.
        self.observations.append(WeightObservation(
            key, resource, int(served), float(service), context, float(violation)))

    def _collect_pairs(self):
        pending = []
        for index, item in enumerate(self.observations):
            for other in self.observations[max(0, index-80):index]:
                if (item.key in self._paired_observations
                        and other.key in self._paired_observations):
                    continue
                if other.served != item.served or abs(other.service-item.service) > 1e-9:
                    continue
                resource = item.resource
                distance = float(np.linalg.norm(resource-other.resource))
                if distance < 0.002:
                    continue
                first = np.all(resource <= other.resource+1e-9) and np.any(resource < other.resource-1e-9)
                second = np.all(other.resource <= resource+1e-9) and np.any(other.resource < resource-1e-9)
                if not first and not second:
                    continue
                better, worse = (resource, other.resource) if first else (other.resource, resource)
                pending.append((distance, ContextualPreferencePair(
                    better=better.copy(), worse=worse.copy(), context=item.context.copy(),
                    margin=min(0.02, max(0.003, 0.10*distance)))))
        self._paired_observations.update(item.key for item in self.observations)
        return sorted(pending, key=lambda entry: entry[0], reverse=True)[:64]

    @staticmethod
    def context_importance(pairs, current_context):
        """varpi = 1 - mean squared context distance; exclude intercept.

        No positive floor is invented. Opposite corners may have zero weight;
        an entirely zero-weight batch must abstain rather than divide by zero.
        """
        current = np.asarray(current_context, dtype=float)
        return np.asarray([1. - np.mean((current[1:] - p.context[1:]) ** 2)
                           for p in pairs])

    def update(self, current_context=None):
        if current_context is None:
            current_context = (self.observations[-1].context if self.observations
                               else np.array([1., 0., 0., 0., 0.]))
        average_before = self.average_matrix.copy()
        previous = self.matrix.copy()
        pending = self._collect_pairs()
        new_pairs = [pair for _, pair in pending]
        predicted = [float((self.matrix @ p.context) @ (p.worse - p.better)) for p in new_pairs]
        pre_loss = (float(np.mean([max(0., p.margin - score) for p, score in zip(new_pairs, predicted)]))
                    if new_pairs else None)
        self.pairs = (self.pairs + new_pairs)[-256:]
        importance = self.context_importance(self.pairs, current_context)
        changed = bool(new_pairs)
        status = "no_new_pairs" if len(self.pairs) >= 3 and not changed else "insufficient_exact_pairs"
        train_loss = None
        if len(self.pairs) >= 3 and changed:
            if importance.sum() <= 0:
                status = "zero_context_weight"
            else:
                try:
                    fit = fit_history_contextual_weights(
                        self.pairs, average=average_before, previous=previous,
                        pair_weights=importance,
                    )
                    fitted = np.asarray(fit.vartheta, dtype=float)
                    usable = (fit.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                              and fitted.shape == self.matrix.shape)
                    if usable:
                        corners = np.asarray([[1., *v] for v in product((0., 1.), repeat=4)])
                        outputs = corners @ fitted.T
                        usable = (np.all(np.isfinite(outputs)) and np.min(outputs) >= -1e-6
                                  and np.allclose(outputs.sum(axis=1), 1., atol=1e-6, rtol=0))
                    if usable:
                        self.matrix = fitted.copy()
                        self.successful_updates += 1
                        self.average_matrix = average_before + (
                            self.matrix - average_before) / (self.successful_updates + 1)
                        status, train_loss = fit.status, fit.mean_pair_loss
                    else:
                        status = "fit_failed: invalid_solver_result"
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
            "violation_filter": False,
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
