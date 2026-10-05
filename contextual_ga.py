"""Shared GA reproduction and context-dependent resource weights.

This base has no survival loop. JointTopKMixin provides the only active WE loop.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from ga import ArchiveRecord, CSAEA, Individual
from contextual_weight_audit import AuditedContextualWeights

class ContextualCSAEA(CSAEA):
    def __init__(self, *args, weight_method="contextual", **kwargs):
        super().__init__(*args, **kwargs)
        if weight_method not in ("fixed", "contextual"):
            raise ValueError("Unknown resource-weight method")
        self.weight_method = weight_method
        self.weight_learner = (AuditedContextualWeights(self._resource_weights)
                               if weight_method == "contextual" else None)
        self.war_history = []
        self._brood_families = []
        self.exact_trace = []
        # A neutral prior, not an allegedly learned matrix. CVXPY updates it
        # from exact MILP pairs every update_period generations.
        self.vartheta_prior = np.zeros((3, 5 if self.weight_learner else 4), dtype=float)
        self.vartheta_prior[:, 0] = self._resource_weights
        self.vartheta = self.vartheta_prior.copy()
        self.weight_learning_history: list[dict[str, Any]] = []
        self._pending_weight_observations = []

    def evaluate_exact(self, item):
        context_before = self._context().copy() if self.weight_learner else None
        super().evaluate_exact(item)
        if self.weight_learner:
            # Joint evaluation finalizes archive V after this base method returns.
            # Keep the pre-evaluation context, consume the finalized record at update.
            self._pending_weight_observations.append((self.archive[-1], context_before))
        # Actual cache misses, not archive length, measure expensive MILP work.
        self.exact_trace.append({
            "unique_solves": int(getattr(self.milp_evaluator, "unique_solves", len(self.archive))),
            "served_ues": len(item.coverage), "service_score": float(item.service_score),
            "violation": float(item.violation), "power_w": float(item.power_est_w),
            "prbs": float(item.prbs_est), "generation": int(self.current_generation),
            "bottleneck_pressure": float(item.bottleneck_pressure),
            "idle_redeployment_pressure": float(item.idle_redeployment_pressure),
        })

    def _context(self) -> np.ndarray:
        """The previous exact incumbent supplies one common comparison context."""
        if not self.archive:
            return np.array([1.0] + [0.0] * (4 if self.weight_learner else 3))
        best_record = max(
            self.archive,
            key=lambda record: (
                record.violation <= 1e-12,
                len(record.coverage), self._service_score(record.coverage),
                -record.prbs, -record.power_w,
            ),
        )
        context = [
            1.0,
            len(best_record.coverage) / len(self.ues),
            np.clip(best_record.power_w / self.power_budget_w, 0.0, 1.0),
            np.clip(best_record.prbs / self.prb_budget, 0.0, 1.0),
        ]
        if self.weight_learner:
            peak_power, _, _ = self._exact_bottleneck_pressure(best_record.milp_result)
            context.append(float(np.clip(peak_power, 0., 1.)))
        return np.asarray(context, dtype=float)

    def _get_contextual_weights(self) -> np.ndarray:
        weights = np.maximum(self.vartheta @ self._context(), 0.0)
        total = float(np.sum(weights))
        return weights / total if total > 1e-12 else self.vartheta_prior[:, 0].copy()

    def _resource_vector(self, record: ArchiveRecord) -> np.ndarray:
        _power_load, _prb_load, bottleneck = self._exact_bottleneck_pressure(record.milp_result)
        return np.array([
            record.power_w / self.power_budget_w,
            record.prbs / self.prb_budget,
            bottleneck,
        ], dtype=float)

    def _refresh_resource_scores(self, items: Sequence[Individual]) -> None:
        """Avoid comparing cached individuals scored under different weights."""
        for item in items:
            item.fitness = self._resource_fitness(
                item.power_est_w, item.prbs_est, item.bottleneck_pressure,
                item.idle_redeployment_pressure,
            )

    def update_contextual_model(self, stage: str, items: Sequence[Individual] = ()) -> None:
        """Fit vartheta from unambiguous exact resource-dominance pairs.

        A label is admitted only when two MILP-evaluated placements have equal
        service and one uses no more of *every* resource (strictly less of at
        least one).  Non-dominated trade-offs are deliberately left unlabeled:
        MILP observations alone cannot reveal whether the operator values a
        watt or a PRB more highly, and inventing that order would make the
        learned weights circular.
        V and epsilon do not filter training pairs; they belong to WE comparison.
        """
        if self.weight_learner:
            for record, context in self._pending_weight_observations:
                self.weight_learner.observe(
                    tuple(np.asarray(record.genome).ravel()), self._resource_vector(record),
                    len(record.coverage), self._service_score(record.coverage), record.violation,
                    context,
                )
            self._pending_weight_observations.clear()
        diagnostics = (self.weight_learner.update(self._context()) if self.weight_learner else {
            "status": "fixed_prior", "new_pairs": 0, "training_pairs": 0,
            "mean_pair_loss": None, "preference_source": "none",
        })
        if self.weight_learner:
            self.vartheta = self.weight_learner.matrix.copy()
        self._resource_weights = self._get_contextual_weights()
        self._refresh_resource_scores(items)
        self.weight_learning_history.append({
            **diagnostics, "stage": stage, "archive_size": len(self.archive),
            "context": self._context().tolist(), "weights": self._resource_weights.tolist(),
            "vartheta": self.vartheta.tolist(), "weight_method": self.weight_method,
        })
        evidence_log = (f"observations={diagnostics['unique_exact_observations']}; V_filter=off; "
                        if self.weight_learner else "")
        print(f"CVXPY {stage}: {diagnostics['status']}; {self.weight_method}; "
              f"{evidence_log}pairs {diagnostics['training_pairs']}, "
              f"weights={np.round(self._resource_weights, 3).tolist()}", flush=True)

    def _population_target(self, generation: int) -> int:
        """Expansion -> war -> consolidation schedule from the WE note."""
        initial = self.settings.population_size
        capacity = self.settings.population_capacity or initial
        progress = generation / max(self.settings.generations, 1)
        if progress <= 0.40:
            fraction = progress / 0.40
            target = initial + fraction * (capacity - initial)
        elif progress <= 0.75:
            fraction = (progress - 0.40) / 0.35
            target = capacity - fraction * (capacity - max(initial, capacity // 2))
        else:
            fraction = (progress - 0.75) / 0.25
            middle = max(initial, capacity // 2)
            target = middle - fraction * (middle - max(initial, capacity // 5))
        return int(np.clip(round(target), initial, capacity))

    @staticmethod
    def _individual_key(item: Individual) -> tuple[float, ...]:
        genome = item.evaluated_genome if item.evaluated_genome is not None else item.genome
        return tuple(np.round(np.asarray(genome, dtype=float), 8).ravel())

    def _independent_brood(
        self, parents: Sequence[Individual], budget: int,
    ) -> list[Individual]:
        """Create independent siblings from a distinct pair of configurations.

        Parent quality already affects tournament selection.  Letting strong
        pairs also produce more children would apply selection pressure twice
        and can collapse diversity before joint top-K selection has evaluated it.
        """
        if budget <= 0:
            return []
        children: list[Individual] = []
        self._brood_families = []
        while len(children) < budget:
            first = self._tournament(parents)
            alternatives = [p for p in parents
                            if self._individual_key(p) != self._individual_key(first)]
            if not alternatives:
                # A collapsed pool cannot supply a distinct physical parent.
                # Mutation can still restore diversity; do not loop forever.
                alternatives = [p for p in parents if p is not first] or [first]
            second = self._tournament(alternatives)
            family = [self._individual_key(first), self._individual_key(second)]
            family_budget = min(self.settings.children_per_pair, budget - len(children))
            produced = 0
            while produced < family_budget:
                # Always the ORIGINAL two parents, never the previous sibling.
                one, two = self._sbx(first, second)
                for child in (one, two):
                    child = self._mutate(child)
                    self.evaluate_surrogate(child)
                    children.append(child)
                    family.append(self._individual_key(child))
                    produced += 1
                    if produced >= family_budget:
                        break
            self._brood_families.append(family)
        return children

