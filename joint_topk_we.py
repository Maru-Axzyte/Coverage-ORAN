"""WE top-K selection and the joint GA/CMA/MILP generation loop.

Each candidate contains every UAV. Predicted candidates may survive, but only
MILP-verified candidates can become the reported incumbent.
"""
from __future__ import annotations
import math
import numpy as np
from ga import Individual
from cma import JointCMA
from surrogate import JointSurrogateMixin, _resource_statistics

def topk_probabilities(items, predictions, key, target, samples, rng,
                       power_limits, prb_limits, system_prbs, weights, epsilon):
    """Sample primitives, derive (V,-C,-S,J), then rank the WHOLE pool.

    Conditional draws across candidates are independent; within a candidate
    its full covariance is retained. Exact candidates have no sampling noise.
    """
    count = len(items)
    target = min(max(int(target), 0), count)
    v, c, s, j = [np.empty((samples, count)) for _ in range(4)]
    for i, item in enumerate(items):
        prediction = predictions[key(item)]
        mean = prediction["mean"]
        # Draw even for an exact point so a repeated seed preserves the
        # random numbers for every OTHER point after exact replacement.
        noise = rng.standard_normal((samples, len(mean)))
        if item.exact_result is not None:
            draw = np.broadcast_to(mean, noise.shape)
        else:
            cov = prediction["covariance"]
            vals, vecs = np.linalg.eigh((cov + cov.T)*.5)
            factor = vecs * np.sqrt(np.maximum(vals, 0.))
            draw = mean + noise @ factor.T
            draw[:, :-2] = np.maximum(draw[:, :-2], 0.)
            draw[:, -2:] = np.clip(draw[:, -2:], 0., prediction["service_upper"])
        _, _, _, j[:, i], overload = _resource_statistics(
            draw, power_limits, prb_limits, system_prbs, weights)
        v[:, i] = (item.violation if item.exact_result is not None
                   else prediction["geometry_violation"] + overload)
        c[:, i], s[:, i] = draw[:, -2], draw[:, -1]
    feasible = v <= max(epsilon, 1e-12)
    # Last lexsort key has highest priority. Index only breaks exact ties.
    index = np.broadcast_to(np.arange(count), (samples, count))
    order = np.lexsort((index, j, -s, -c, np.where(feasible, 0., v), ~feasible), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, count+1), order.shape), axis=1)
    return np.mean(ranks <= target, axis=0), np.mean(ranks, axis=0), ranks

def choose_milp_indices(items, probabilities, mean_ranks, target, budget,
                        generation, audit_period, rng, provisional_keys=None):
    """Disjoint elite/boundary/audit sets sharing ONE generation budget.

    provisional_keys is a list of deterministic quality keys, used only to
    break ties in the provisional ordering, not as a second fitness.
    """
    unknown = [i for i, item in enumerate(items) if item.exact_result is None]
    groups = {"elite": [], "boundary": [], "audit": []}
    budget = min(max(int(budget), 0), len(unknown))
    if not budget:
        return groups
    order = sorted(range(len(items)), key=lambda i: (
        -probabilities[i], mean_ranks[i],
        () if provisional_keys is None else provisional_keys[i], i))
    top = set(order[:target])
    elite = next((i for i in order if i in top and i in unknown), None)
    if elite is not None:
        groups["elite"].append(elite)
    remaining = [i for i in unknown if i != elite]
    outside = [i for i in remaining if i not in top]
    if (audit_period > 0 and generation % audit_period == 0
            and outside and budget > len(groups["elite"])):
        audit = int(rng.choice(outside))
        groups["audit"].append(audit)
        remaining.remove(audit)
    remaining.sort(key=lambda i: (-probabilities[i]*(1-probabilities[i]),
                                  abs(mean_ranks[i]-target), mean_ranks[i], i))
    slots = budget - len(groups["elite"]) - len(groups["audit"])
    groups["boundary"] = remaining[:slots]
    return groups

class JointTopKMixin(JointSurrogateMixin):
    def __init__(self, *args, milp_budget=4, rank_samples=64, cma_pre_samples=8,
                 audit_period=5, min_residual_samples=20, nominal_confidence=.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_milp_budget = int(milp_budget)
        self.rank_samples, self.cma_pre_samples = int(rank_samples), int(cma_pre_samples)
        self.audit_period, self.min_residual_samples = int(audit_period), int(min_residual_samples)
        self.nominal_confidence = nominal_confidence
        self.joint_exact, self.joint_predictions = {}, {}
        self.joint_records, self.joint_topk_history, self.joint_topk_checks = [], [], []
        self.joint_cma_updates = []
        self._frozen_context = None
        self._frozen_archive, self._frozen_records = None, None
        self._pending_records = []
        self._check_role, self._check_cut = None, None

    def _joint_quality(self, item, epsilon=None):
        eps = self.current_eps if epsilon is None else epsilon
        feasible = item.violation <= max(eps, 1e-12)
        cost = -self._resource_fitness(item.power_est_w, item.prbs_est, item.bottleneck_pressure)
        return (not feasible, 0. if feasible else item.violation,
                -item.served_fraction, -item.service_score, cost)

    def _joint_snapshot(self, best, generation):
        return {"generation": generation, "fitness": -self._joint_quality(best)[-1],
            "violation": best.violation, "served_ues": len(best.coverage),
            "served_fraction": best.served_fraction, "service_score": best.service_score,
            "power_w": best.power_est_w, "prbs": best.prbs_est,
            "power_ratio": best.power_ratio, "prb_ratio": best.prb_ratio,
            "bottleneck_pressure": best.bottleneck_pressure,
            "idle_redeployment_pressure": best.idle_redeployment_pressure,
            "exact_evaluations": len(self.archive),
            "unique_milp_solves": int(getattr(self.milp_evaluator, "unique_solves", len(self.archive))),
            "genome": best.evaluated_genome.tolist(), "epsilon": self.current_eps,
            "resource_weights": self._resource_weights.tolist()}

    def run(self):
        lower, span = np.asarray(self.bounds.lower), self.bounds.span
        initial = self.settings.population_size
        cap = self.settings.population_capacity or 2*initial
        pop = [Individual(self._repair(g), source="initial") for g in self.initial_genomes[:initial]]
        while len(pop) < initial:
            genome = lower+self.rng.random((self.num_uavs, 4))*span
            genome[:, :2] = lower[:2]+.5*span[:2]
            pop.append(Individual(genome, source="initial"))
        for item in pop[:min(initial, max(1, self.settings.initial_milp_seeds))]:
            self.evaluate_exact(item)
        self._joint_commit()
        self._joint_freeze()
        for item in pop:
            self.evaluate_surrogate(item)
        eps0 = float(np.quantile([x.violation for x in pop], .8))
        self.current_eps = eps0
        self._joint_commit()
        # Update weights once the initial evaluation batch is complete.
        self.update_contextual_model("warmup", pop)
        best = min(self.joint_exact.values(), key=lambda x: self._joint_quality(x, 0.)).clone()
        snapshots = [self._joint_snapshot(best, 0)]
        cma = JointCMA.from_genome(best.evaluated_genome, lower, span)
        cma_origins, pending_cma = {}, {}
        for generation in range(1, self.settings.generations+1):
            self.current_generation = generation
            self.current_eps = eps0*max(0., 1.-generation/max(1., .7*self.settings.generations))**2
            if generation > 1 and (generation-1) % max(1, self.settings.update_period) == 0:
                self.update_contextual_model(f"WE-{generation-1}", pop)
            self._joint_freeze()
            # Reserve reproductive space so reaching the cap never stops all
            # variation. The COMPLETE parent+child pool remains <= cap.
            birth_slots = min(cap-2, max(initial, self.cma_pre_samples+2))
            target = min(self._population_target(generation), cap-birth_slots)
            target = max(2, target)
            for item in pop:
                self.evaluate_surrogate(item)
            if len(pop) > cap-birth_slots:
                pop = sorted(pop, key=self._joint_quality)[:cap-birth_slots]
                if all(self._joint_key(x) != self._joint_key(best) for x in pop):
                    pop[-1] = best.clone()
            available = cap-len(pop)
            cma_count = min(self.cma_pre_samples, max(0, available-2))
            brood_count = min(max(initial, target-len(pop)), available-cma_count)
            brood = self._independent_brood(pop, brood_count)
            cma_children = cma.sample(cma_count, lower, span, self.num_uavs, self.rng)
            for item in cma_children:
                item.anchor_record = best.anchor_record
                cma_origins[self._joint_key(item)] = cma.updates
                self.evaluate_surrogate(item)
            unique = {}
            for item in [*pop, *brood, *cma_children]:
                key = self._joint_key(item)
                if key not in unique or item.exact_result is not None:
                    unique[key] = item
            pool = list(unique.values())
            target = min(target, len(pool))
            rank_seed = int(self.rng.integers(0, 2**32))
            def rank_pool():
                return topk_probabilities(pool, self.joint_predictions, self._joint_key,
                    target, self.rank_samples, np.random.default_rng(rank_seed),
                    self._uav_power_budgets, self._uav_prb_budgets, self.prb_budget,
                    self._resource_weights, self.current_eps)
            pi, mean_ranks, _ = rank_pool()
            quality = [self._joint_quality(x) for x in pool]
            provisional = sorted(range(len(pool)), key=lambda i: (-pi[i], mean_ranks[i], quality[i], i))
            cut = quality[provisional[target-1]]
            groups = choose_milp_indices(pool, pi, mean_ranks, target, self.joint_milp_budget,
                generation, self.audit_period, self.rng, quality)
            before_exact = len(self.joint_exact)
            for role, indices in groups.items():
                for i in indices:
                    self._check_role, self._check_cut = role, cut
                    self.evaluate_exact(pool[i])
                    if self._better_exact_incumbent(pool[i], best):
                        best = pool[i].clone()
                    key = self._joint_key(pool[i])
                    if cma_origins.get(key) == cma.updates:
                        pending_cma[key] = pool[i].clone()
            self._check_role = self._check_cut = None
            pi, mean_ranks, _ = rank_pool()
            order = sorted(range(len(pool)), key=lambda i: (
                -pi[i], mean_ranks[i], self._joint_quality(pool[i]), i))
            # The strict incumbent is protected even while population epsilon
            # permits exploratory configurations with a nonzero violation.
            elite_count = min(target, max(1, math.ceil(self.settings.elite_fraction*target)))
            elites = sorted((x for x in pool if x.exact_result is not None),
                            key=lambda x: self._joint_quality(x, 0.))
            selected = {self._joint_key(best): best.clone()}
            for item in elites:
                if len(selected) >= elite_count:
                    break
                selected.setdefault(self._joint_key(item), item)
            for i in order:
                if len(selected) >= target:
                    break
                selected.setdefault(self._joint_key(pool[i]), pool[i])
            pop = list(selected.values())
            if len(pending_cma) >= 2:
                cma.update(list(pending_cma.values()), self._joint_quality, lower, span)
                self.joint_cma_updates.append({"generation": generation,
                    "update": cma.updates, "verified_samples": len(pending_cma),
                    "sigma": cma.sigma, "mean": cma.mean.tolist(),
                    "path_sigma": cma.path_sigma.tolist(), "path_c": cma.path_c.tolist(),
                    "covariance": cma.covariance.tolist()})
                pending_cma.clear()
            active_keys = {self._joint_key(x) for x in pop}
            cma_origins = {k: v for k, v in cma_origins.items() if k in active_keys and v == cma.updates}
            self._joint_commit()
            row = {"generation": generation, "population": len(pop), "population_target": target,
                "pool_size": len(pool), "offspring": len(brood)+len(cma_children),
                "archive_size": len(self.archive), "epsilon": self.current_eps,
                "milp_validations": len(self.joint_exact)-before_exact,
                "elite_checks": len(groups["elite"]), "boundary_checks": len(groups["boundary"]),
                "audit_checks": len(groups["audit"]), "residual_samples": len(self.joint_records),
                "cma_sigma": cma.sigma, "mean_topk_probability": float(np.mean(pi)),
                "best_surrogate_fitness": max(x.fitness for x in pop)}
            self.history.append(row.copy())
            self.joint_topk_history.append(row)
            snapshots.append(self._joint_snapshot(best, generation))
            if generation == 1 or generation % 10 == 0 or generation == self.settings.generations:
                print(f"Joint WE {generation}/{self.settings.generations}: served {len(best.coverage)}/{len(self.ues)}, "
                      f"V={best.violation:.4g}, PRB {best.prbs_est:.0f}/{self.prb_budget:.0f}, "
                      f"P {best.power_est_w:.3f} W; MILP {len(self.archive)}, pool {len(pool)}", flush=True)
        self.current_eps = 0.
        best_genome = best.evaluated_genome.copy()
        # Runner contract indexes the final result by its physical placement.
        self.joint_exact[tuple(np.round(best_genome, 7).ravel())] = best.clone()
        return {"best_genome": best_genome, "best_milp_result": best.exact_result,
            "best_violation": best.violation, "best_violation_terms": best.violation_terms.copy(),
            "archive_size": len(self.archive), "joint_transition_window_size": len(self.joint_records),
            "history": self.history, "exact_incumbent_history": snapshots,
            "trajectory": snapshots, "weight_learning_history": self.weight_learning_history,
            "war_method": "joint-topk", "war_history": [], "exact_trace": self.exact_trace,
            "joint_topk_history": self.joint_topk_history, "joint_topk_checks": self.joint_topk_checks,
            "joint_cma_updates": self.joint_cma_updates}

