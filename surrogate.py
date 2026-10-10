"""Frozen transition-kernel predictions, residuals and exact-label replacement.

y = [P_u/Pmax_u, N_u/Nsys_max, C, S] in the configured shared PRB pool.
No provisional allocation is solved here.
"""
from __future__ import annotations
from statistics import NormalDist
import math
import numpy as np

def _resource_statistics(y, power_limits, prb_limits, system_prbs, weights):
    """Reconstruct shared PRB usage from normalized per-UAV allocations.

    prb_limits are coordinate scales, not separate PRB reservations.
    Only total PRB usage contributes to PRB pressure and violation.
    """
    y = np.asarray(y, dtype=float)
    u = len(power_limits)
    p, n = np.maximum(y[..., :u], 0.), np.maximum(y[..., u:2*u], 0.)
    pbar = p @ power_limits / np.sum(power_limits)
    nbar = n @ prb_limits / system_prbs
    z = np.maximum(np.max(p, axis=-1), nbar)
    cost = weights[0]*pbar + weights[1]*nbar + weights[2]*z
    violation = (np.maximum(p-1., 0.).sum(axis=-1)
                 + np.maximum(nbar-1., 0.))
    return pbar, nbar, z, cost, violation

class JointSurrogateMixin:
    def _context(self):
        frozen = getattr(self, "_frozen_context", None)
        return frozen.copy() if frozen is not None else super()._context()

    def _joint_key(self, item):
        # Include the raw genotype: two repaired placements can have different
        # bound violations. The underlying MILP evaluator caches the phenotype.
        return tuple(np.round(np.asarray(item.genome), 7).ravel())

    def _joint_record_y(self, record):
        _, prbs, powers = self._exact_uav_loads(record.milp_result)
        return np.r_[powers/self._uav_power_budgets, prbs/self._uav_prb_budgets,
                     len(record.coverage)/len(self.ues), self._service_score(record.coverage)]

    def _joint_exact_y(self, item):
        _, prbs, powers = self._exact_uav_loads(item.exact_result)
        return np.r_[powers/self._uav_power_budgets, prbs/self._uav_prb_budgets,
                     item.served_fraction, item.service_score]

    def _exact_bottleneck_pressure(self, result):
        _, prbs, powers = self._exact_uav_loads(result)
        p, n = float(np.max(powers/self._uav_power_budgets)), float(np.sum(prbs)/self.prb_budget)
        return p, n, max(p, n)

    def _resource_fitness(self, power_w, prbs, bottleneck_pressure, idle_redeployment_pressure=0.):
        return -float(self._resource_weights @ np.array([
            power_w/self.power_budget_w, prbs/self.prb_budget, bottleneck_pressure]))

    def _joint_feature(self, anchor, genome):
        return np.r_[((anchor.genome-self.bounds.lower)/self.bounds.span).ravel(),
                     self._joint_record_y(anchor), ((genome-anchor.genome)/self.bounds.span).ravel()]

    def _joint_weights(self, feature, rows):
        if not rows:
            return np.array([], dtype=int), np.array([]), 1., 0.
        d, m = 4*self.num_uavs, 2*self.num_uavs+2
        ls, la = self.settings.transition_state_bandwidth, self.settings.transition_action_bandwidth
        diff = np.array([row["feature"] for row in rows])-feature
        # Normalize blocks as well as physical coordinates: dimensionality
        # alone must not give geometry an arbitrarily larger weight than m.
        distances = ((diff[:, :d]**2).mean(axis=1)/ls**2
                     + (diff[:, d:d+m]**2).mean(axis=1)/ls**2
                     + (diff[:, d+m:]**2).mean(axis=1)/la**2)
        ids = np.argsort(distances, kind="stable")[:max(1, self.settings.transition_neighbors)]
        w = np.exp(-.5*(distances[ids]-distances[ids[0]]))
        w /= w.sum()
        novelty = float(1.-np.exp(-.5*distances[ids[0]]))
        return ids, w, novelty, float(1./(w @ w))

    def _joint_stats(self, y):
        return _resource_statistics(y, self._uav_power_budgets, self._uav_prb_budgets,
                                    self.prb_budget, self._resource_weights)

    def evaluate_surrogate(self, item):
        key = self._joint_key(item)
        if key in self.joint_exact:
            restored = self.joint_exact[key].clone()
            item.__dict__.update(restored.__dict__)
            item.fitness = self._resource_fitness(item.power_est_w, item.prbs_est, item.bottleneck_pressure)
            y = self._joint_exact_y(item)
            self.joint_predictions[key] = {"mean": y, "covariance": np.zeros((len(y), len(y))),
                "geometry_violation": item.violation, "service_upper": y[-2:].copy()}
            return item
        genome = self._repair(item.genome)
        archive = self.archive if self._frozen_archive is None else self._frozen_archive
        if not archive:
            raise RuntimeError("MILP warm-up is required before prediction")
        anchor = item.anchor_record
        if anchor is None:
            anchor = min(archive, key=lambda a: np.sum(((genome-a.genome)/self.bounds.span)**2))
        feature = self._joint_feature(anchor, genome)
        base = self._joint_record_y(anchor)
        rows = self.joint_records if self._frozen_records is None else self._frozen_records
        ids, w, novelty, neff = self._joint_weights(feature, rows)
        raw_mean = base.copy()
        potential, _, _ = self._coverage_gain(genome)
        upper = np.array([len(potential)/len(self.ues), self._service_score(potential)])
        if len(ids):
            raw_mean += w @ np.array([rows[i]["delta"] for i in ids])
        else:
            # Transfer admission fraction only; no provisional UE association.
            raw_mean[-2] = upper[0]*len(anchor.coverage)/max(1, len(anchor.potential_coverage))
            raw_mean[-1] = upper[1]*self._service_score(anchor.coverage)/max(
                self._service_score(anchor.potential_coverage), 1e-12)
        dimension = len(base)
        bias = np.zeros(dimension)
        calibrated = len(rows) >= self.min_residual_samples
        if calibrated:
            errors = np.array([rows[i]["raw_error"] for i in ids])
            bias = w @ errors
            centered = errors-bias
            covariance = ((centered.T*w) @ centered)/max(1.-float(w @ w), 1e-6)
            covariance += .03**2*(1.+novelty**2+1./max(neff, 1.))*np.eye(dimension)
        else:
            covariance = .2**2*(1.+novelty**2)*np.eye(dimension)
        mean = raw_mean+bias
        mean[:-2] = np.maximum(mean[:-2], 0.)
        mean[-2:] = np.clip(mean[-2:], 0., upper)
        terms = self._violation(item.genome, genome, 0., 0.)
        geometry = terms.total
        _, _, z, cost, overload = self._joint_stats(mean)
        u = self.num_uavs
        terms.power = float(np.maximum(mean[:u]-1., 0.).sum())
        terms.prb = float(overload-terms.power)
        item.evaluated_genome, item.anchor_record = genome, anchor
        item.coverage = potential
        item.power_est_w = float(mean[:u] @ self._uav_power_budgets)
        item.prbs_est = float(mean[u:2*u] @ self._uav_prb_budgets)
        item.served_fraction, item.service_score = map(float, mean[-2:])
        item.power_ratio = float(max(mean[:u]))
        item.prb_ratio = item.prbs_est/self.prb_budget
        item.bottleneck_pressure, item.fitness = float(z), -float(cost)
        item.violation, item.violation_terms = terms.total, terms.as_dict()
        item.uncertainty = float(np.sqrt(np.trace(covariance)/dimension))
        item.exact_result = None
        self.joint_predictions[key] = {"mean": mean.copy(), "raw_mean": raw_mean.copy(),
            "covariance": covariance, "feature": feature, "before": base,
            "geometry_violation": geometry, "service_upper": upper,
            "calibrated": calibrated, "residual_samples": len(rows)}
        return item

    def evaluate_exact(self, item):
        key = self._joint_key(item)
        if key in self.joint_exact:
            return self.evaluate_surrogate(item)
        prediction = self.joint_predictions.get(key)
        super().evaluate_exact(item)
        payload = item.exact_result.get("milp", item.exact_result)
        if not payload.get("solver_optimal", False):
            raise RuntimeError("Joint top-K needs a solver_optimal MILP label; no exact label was accepted")
        y = self._joint_exact_y(item)
        # Per-UAV power and shared PRB terms use exactly the same primitives
        # as sampled ranks (the allocator itself enforces these constraints).
        terms = self._violation(item.genome, item.evaluated_genome, 0., 0.)
        _, _, _, _, overload = self._joint_stats(y)
        terms.power = float(np.maximum(y[:self.num_uavs]-1., 0.).sum())
        terms.prb = float(overload-terms.power)
        item.violation, item.violation_terms = terms.total, terms.as_dict()
        self.archive[-1].violation = item.violation
        if prediction is not None and "feature" in prediction:
            residual = y-prediction["mean"]
            record = {"feature": prediction["feature"].copy(), "before": prediction["before"].copy(),
                "after": y.copy(), "delta": y-prediction["before"],
                "raw_error": y-prediction["raw_mean"], "error": residual,
                "source": item.source, "generation": self.current_generation, "key": key}
            self._pending_records.append(record)
            exact_j = float(self._joint_stats(y)[3])
            pred_j = float(self._joint_stats(prediction["mean"])[3])
            grad = self._resource_j_gradient(prediction["mean"])
            std = math.sqrt(max(0., float(grad @ prediction["covariance"] @ grad)))
            half = NormalDist().inv_cdf((1.+self.nominal_confidence)/2.)*std
            covered = (abs(exact_j-pred_j) <= half) if prediction["calibrated"] else None
            rescued = (self._joint_quality(item) < self._check_cut
                       if self._check_role == "audit" and self._check_cut is not None else None)
            self.joint_topk_checks.append({"generation": self.current_generation,
                "source": item.source, "role": self._check_role,
                "predicted_j": pred_j, "exact_j": exact_j, "error_j": exact_j-pred_j,
                "interval_lower": pred_j-half, "interval_upper": pred_j+half,
                "interval_covered": covered, "calibrated": prediction["calibrated"],
                "residual_samples": prediction["residual_samples"],
                "audited_false_drop": rescued, "outside_topk_rescue": rescued,
                "error_primitives": residual.tolist()})
        self.joint_exact[key] = item.clone()
        self.evaluate_surrogate(item)  # deterministic exact replacement
        return item

    def _resource_j_gradient(self, y):
        # Delta-method diagnostic for max-load J; not a simultaneous or
        # finite-sample coverage guarantee, particularly near tied maxima.
        u = self.num_uavs
        grad = np.zeros(len(y))
        grad[:u] = self._resource_weights[0]*self._uav_power_budgets/self.power_budget_w
        grad[u:2*u] = self._resource_weights[1]*self._uav_prb_budgets/self.prb_budget
        nbar = float(y[u:2*u] @ self._uav_prb_budgets/self.prb_budget)
        peak = max(float(np.max(y[:u])), nbar)
        maxima = np.flatnonzero(np.isclose(y[:u], peak, rtol=0., atol=1e-12))
        shared_active = bool(np.isclose(nbar, peak, rtol=0., atol=1e-12))
        share = self._resource_weights[2]/(len(maxima)+int(shared_active))
        grad[maxima] += share
        if shared_active:
            grad[u:2*u] += share*self._uav_prb_budgets/self.prb_budget
        return grad

    def _joint_freeze(self):
        self._frozen_context = super()._context().copy()
        self._frozen_archive = tuple(self.archive)
        self._frozen_records = tuple(self.joint_records)
        self.joint_predictions.clear()

    def _joint_commit(self):
        self.joint_records.extend(self._pending_records)
        self._pending_records.clear()
        window = max(self.min_residual_samples, 10*self.settings.transition_neighbors)
        self.joint_records = self.joint_records[-window:]
        self._frozen_context = self._frozen_archive = self._frozen_records = None

