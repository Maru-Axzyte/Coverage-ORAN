"""GA data structures, evaluation helpers, parent selection, SBX and mutation.

``milp_evaluator`` receives a genome with rows ``[x, y, h, theta_deg]`` and
returns ``total_power_w``, ``total_num_prbs`` (or ``total_prbs``), and
optionally ``served_ue_ids``. The active generation loop is in joint_topk_we.py;
run main.py to assemble the complete optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, Callable, Mapping, Sequence
import math
import numpy as np

import config
import env

MilpEvaluator = Callable[[np.ndarray], Mapping[str, Any]]
GenomeRepair = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class Bounds:
    lower: tuple[float, float, float, float]
    upper: tuple[float, float, float, float]
    def __post_init__(self) -> None:
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError("Each lower bound must be below its upper bound")
        if self.lower[2] <= 0 or self.lower[3] <= 0 or self.upper[3] >= 90:
            raise ValueError("Require h > 0 and 0 < theta < 90")
    @property
    def span(self) -> np.ndarray:
        return np.asarray(self.upper) - np.asarray(self.lower)
    def clip(self, genome: np.ndarray) -> np.ndarray:
        return np.clip(genome, np.asarray(self.lower), np.asarray(self.upper))

@dataclass(frozen=True)
class GASettings:
    population_size: int = 200
    population_capacity: int | None = 1000
    generations: int = 400
    initial_milp_seeds: int = 8
    update_period: int = 10
    clusters: int = 4
    sbx_eta: float = 18.0
    sbx_uav_blocks: bool = False
    children_per_pair: int = 2
    mutation_probability: float = 0.12
    mutation_sigma: float = 0.08
    elite_fraction: float = 0.15
    light_violation_epsilon: float = 0.10
    light_violation_fraction: float = 0.10
    beta_power: float = 1 / 3
    mu_prb: float = 1 / 3
    eta_bottleneck: float = 1 / 3
    bottleneck_target: float = 0.85
    idle_uav_redeployment_weight: float = 0.10
    unserved_infill_variants: int = 4
    surrogate_service_tie_epsilon: float = 0.02
    omega_uncertainty: float = 0.25
    omega_overlap: float = 0.75
    jaccard_weight: float = 0.65
    radio_similarity_weight: float = 0.20
    topology_similarity_weight: float = 0.15
    transition_state_bandwidth: float = 1.0
    transition_action_bandwidth: float = 0.5
    transition_neighbors: int = 12
    transition_risk_kappa: float = 0.5
    # This is not part of V(x) in the supplied theory.  Keep it opt-in for
    # experiments that impose a separate operational spacing constraint.
    include_separation_constraint: bool = False
    min_uav_separation_m: float = 250.0
    use_war_elimination: bool = True
    random_seed: int | None = 2026

@dataclass
class ArchiveRecord:
    coverage: frozenset[int]  # Exact MILP-served UE ids.
    potential_coverage: frozenset[int]  # Geometric/radio UE set used by surrogate.
    descriptor: np.ndarray
    power_w: float
    prbs: float
    required_rate_bps: float
    mean_gain: float
    milp_result: Mapping[str, Any]
    fitness: float = 0.0
    violation: float = 0.0
    genome: np.ndarray | None = None

@dataclass
class TransitionRecord:
    """One MILP-confirmed full-configuration move, never a hypothetical sibling."""
    state: np.ndarray
    action: np.ndarray
    before: np.ndarray
    after: np.ndarray
    source: str

@dataclass
class ViolationBreakdown:
    """Hard geometry/resource terms; allocation quality is ranked separately."""
    theta: float = 0.0
    height: float = 0.0
    empty: float = 0.0
    power: float = 0.0
    prb: float = 0.0
    separation: float = 0.0  # Optional extension; zero in the paper model.

    @property
    def total(self) -> float:
        return self.theta + self.height + self.empty + self.power + self.prb + self.separation

    def as_dict(self) -> dict[str, float]:
        return {
            "V_theta": self.theta,
            "V_h": self.height,
            # This is deliberately not called an "idle-UAV" violation: it
            # only says a UAV has no UE in its geometric footprint.  Exact
            # MILP allocation (and the soft redeployment pressure) handles
            # whether it actually serves a UE.
            "V_geometric_void": self.empty,
            "V_empty_legacy": self.empty,
            "V_power": self.power,
            "V_PRB": self.prb,
            "V_separation_optional": self.separation,
            "V_total": self.total,
        }


@dataclass
class Individual:
    genome: np.ndarray
    # ``genome`` is the raw GA genotype.  This is the only way V_theta and V_h
    # remain observable after SBX/mutation.  ``evaluated_genome`` is its
    # clipped/snapped physical phenotype supplied to env.py and the MILP.
    evaluated_genome: np.ndarray | None = None
    coverage: frozenset[int] = field(default_factory=frozenset)
    power_est_w: float = 0.0
    prbs_est: float = 0.0
    served_fraction: float = 0.0
    service_score: float = 0.0
    power_ratio: float = 0.0
    prb_ratio: float = 0.0
    bottleneck_pressure: float = 0.0
    idle_redeployment_pressure: float = 0.0
    fitness: float = float("-inf")
    violation: float = float("inf")
    uncertainty: float = 1.0
    overlap_ratio: float = 0.0
    violation_terms: dict[str, float] = field(default_factory=dict)
    exact_result: Mapping[str, Any] | None = None
    anchor_record: ArchiveRecord | None = None
    source: str = "initial"
    def clone(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(),
            evaluated_genome=(None if self.evaluated_genome is None else self.evaluated_genome.copy()),
            coverage=self.coverage,
            power_est_w=self.power_est_w, prbs_est=self.prbs_est,
            served_fraction=self.served_fraction, service_score=self.service_score,
            power_ratio=self.power_ratio,
            prb_ratio=self.prb_ratio, bottleneck_pressure=self.bottleneck_pressure,
            idle_redeployment_pressure=self.idle_redeployment_pressure,
            fitness=self.fitness, violation=self.violation,
            uncertainty=self.uncertainty, overlap_ratio=self.overlap_ratio,
            violation_terms=self.violation_terms.copy(),
            exact_result=self.exact_result,
            anchor_record=self.anchor_record, source=self.source,
        )

class CSAEA:
    """Shared GA operators and oracle/archive helpers; no standalone run loop."""
    def __init__(self, num_uavs: int, ues: Sequence[Any], bounds: Bounds,
                 power_budget_w: float, prb_budget: float,
                 milp_evaluator: MilpEvaluator,
                 genome_repair: GenomeRepair | None = None,
                 initial_genomes: Sequence[np.ndarray] | None = None,
                 settings: GASettings = GASettings(), carrier_hz: float = config.f_c,
                 antenna_efficiency: float = 1.0, path_loss_exponent: float = 2.0,
                 phys: config.PhysConstant | None = None,
                 uav_power_budgets_w: Sequence[float] | None = None,
                 uav_prb_budgets: Sequence[float] | None = None) -> None:
        if num_uavs <= 0 or not ues or power_budget_w <= 0 or prb_budget <= 0:
            raise ValueError("UAV count, UEs, and budgets must be positive")
        self.num_uavs, self.ues, self.bounds = num_uavs, list(ues), bounds
        self._ue_by_id = {self._ue(ue)[0]: self._ue(ue) for ue in self.ues}
        self.power_budget_w, self.prb_budget = power_budget_w, prb_budget
        self._uav_power_budgets = np.asarray(
            ([power_budget_w / num_uavs] * num_uavs if uav_power_budgets_w is None else uav_power_budgets_w),
            dtype=float,
        )
        self._uav_prb_budgets = np.asarray(
            ([prb_budget] * num_uavs if uav_prb_budgets is None else uav_prb_budgets),
            dtype=float,
        )
        if (self._uav_power_budgets.shape != (num_uavs,)
                or self._uav_prb_budgets.shape != (num_uavs,)
                or np.any(self._uav_power_budgets <= 0)
                or np.any(self._uav_prb_budgets <= 0)):
            raise ValueError("Per-UAV power and PRB budgets must be positive vectors of length num_uavs")
        resource_weight_sum = (
            settings.beta_power + settings.mu_prb + settings.eta_bottleneck
        )
        if resource_weight_sum <= 0 or not 0 <= settings.bottleneck_target <= 1:
            raise ValueError("Resource weights must have positive mass and bottleneck target must be in [0, 1]")
        self._resource_weights = np.asarray([
            settings.beta_power, settings.mu_prb, settings.eta_bottleneck,
        ]) / resource_weight_sum
        self.milp_evaluator, self.settings = milp_evaluator, settings
        if settings.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if settings.children_per_pair < 2 or settings.sbx_eta <= 0:
            raise ValueError("children_per_pair >= 2 and sbx_eta > 0 required")
        if (settings.population_capacity is not None
                and settings.population_capacity < settings.population_size + 1):
            raise ValueError("population_capacity must leave room for active parents and at least one child")
        self.genome_repair = genome_repair
        self.initial_genomes = [] if initial_genomes is None else [
            np.asarray(genome, dtype=float).copy() for genome in initial_genomes
        ]
        self.carrier_hz, self.antenna_efficiency = carrier_hz, antenna_efficiency
        self.path_loss_exponent = path_loss_exponent
        self.phys = phys or config.PhysConstant(
            f_c=carrier_hz, eta_ant=antenna_efficiency, a=path_loss_exponent
        )
        self.rng = np.random.default_rng(settings.random_seed)
        self.archive: list[ArchiveRecord] = []
        self.transitions: list[TransitionRecord] = []
        self.history: list[dict[str, float]] = []
        # Epsilon-Deb: adaptive violation tolerance (decays over generations)
        self.current_eps: float = 1.0
        self.current_generation: int = 0

    @property
    def _total_demand_bps(self) -> float:
        return sum(record[3] for record in self._ue_by_id.values())

    def _service_score(self, served: frozenset[int]) -> float:
        return sum(self._ue_by_id[ue_id][3] for ue_id in served) / max(self._total_demand_bps, 1e-20)

    def _resource_fitness(
        self, power_w: float, prbs: float, bottleneck_pressure: float,
        idle_redeployment_pressure: float = 0.0,
    ) -> float:
        """Negative tertiary cost, consulted only after feasibility and service.

        ``idle_redeployment_pressure`` is nonzero only if UEs remain rejected.
        It therefore breaks otherwise equivalent solutions in favour of a
        deployment that gives a currently idle UAV a useful chance to serve;
        it is not a hard requirement that every UAV always be active.
        """
        resource_cost = float(self._resource_weights @ np.asarray([
            power_w / self.power_budget_w,
            prbs / self.prb_budget,
            bottleneck_pressure,
        ]))
        return -(resource_cost + self.settings.idle_uav_redeployment_weight * idle_redeployment_pressure)

    def _idle_redeployment_pressure(
        self, served_count: int, per_uav_ue_count: np.ndarray,
    ) -> float:
        """Soft activity signal: idle UAVs matter only while demand is unmet."""
        if served_count >= len(self.ues):
            return 0.0
        return float(np.count_nonzero(per_uav_ue_count == 0) / self.num_uavs)

    def _exact_uav_loads(
        self, result: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact (UE count, PRB, power) per UAV from a MILP solution."""
        payload = result.get("milp", result)
        ue_count = np.zeros(self.num_uavs, dtype=float)
        prbs = np.zeros(self.num_uavs, dtype=float)
        for choice in payload.get("uav_mode_choices", ()):
            index = int(choice.get("u", choice.get("uav_id", -1)))
            if 0 <= index < self.num_uavs:
                ue_count[index] += 1.0
                prbs[index] += float(choice["num_prbs"])
        powers = np.asarray(payload.get("power_by_uav_w", ()), dtype=float)
        if powers.shape != (self.num_uavs,):
            powers = np.zeros(self.num_uavs, dtype=float)
            for choice in payload.get("uav_mode_choices", ()):
                index = int(choice.get("u", choice.get("uav_id", -1)))
                if 0 <= index < self.num_uavs:
                    powers[index] += float(choice.get("power_w", 0.0))
        return ue_count, prbs, powers

    def _exact_bottleneck_pressure(self, result: Mapping[str, Any]) -> tuple[float, float, float]:
        payload = result.get("milp", result)
        _ue_count, prbs, powers = self._exact_uav_loads(result)
        
        # Áp lực Công suất Cục bộ (UAV nào gánh nặng nhất)
        power_load = float(np.max(powers / self._uav_power_budgets))
        
        # Áp lực Băng thông Toàn cục (Global PRB Pooling)
        total_prb_budget = float(self.prb_budget)
        prb_load = float(np.sum(prbs) / total_prb_budget) if total_prb_budget > 0 else 0.0
        
        pressure = max(0.0, max(power_load, prb_load) - self.settings.bottleneck_target)
        return power_load, prb_load, pressure

    def _primitive_exact(self, record: ArchiveRecord) -> np.ndarray:
        """Normalized MILP labels: per-UAV power, shared PRBs, K, demand, idle."""
        ue_count, _prbs, powers = self._exact_uav_loads(record.milp_result)
        demand = self._service_score(record.coverage)
        idle = self._idle_redeployment_pressure(len(record.coverage), ue_count)
        return np.concatenate((
            powers / self._uav_power_budgets,
            [record.prbs / self.prb_budget,
             len(record.coverage) / len(self.ues), demand, idle],
        ))

    def _transition_state(self, record: ArchiveRecord) -> np.ndarray:
        """Pre-action geometry z and exact operating state m, dimensionless."""
        if record.genome is None:
            raise ValueError("Exact archive record has no evaluated genome")
        geometry = ((record.genome - self.bounds.lower) / self.bounds.span).ravel()
        primitive = self._primitive_exact(record)
        pressure = max(float(np.max(primitive[:self.num_uavs])), primitive[self.num_uavs])
        radio = np.concatenate((record.descriptor[:3],
                                [(record.descriptor[3] + 30.0) / 30.0]))
        return np.concatenate((
            geometry, radio, primitive,
            [max(0.0, pressure - self.settings.bottleneck_target),
             min(1.0, record.violation)],
        ))

    def _transition_prediction(
        self, anchor: ArchiveRecord, genome: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Kernel regression of observed exact *deltas*, with novelty uncertainty."""
        base = self._primitive_exact(anchor)
        action = ((genome - anchor.genome) / self.bounds.span).ravel()
        if not self.transitions:
            # Bootstrap from an exact state; no fictitious UE association.
            return base.copy(), 1.0
        state = self._transition_state(anchor)
        ell_s = self.settings.transition_state_bandwidth
        ell_a = self.settings.transition_action_bandwidth
        if ell_s <= 0 or ell_a <= 0:
            raise ValueError("Transition kernel bandwidths must be positive")
        distances = np.asarray([
            np.sum((state - rec.state) ** 2) / (ell_s ** 2)
            + np.sum((action - rec.action) ** 2) / (ell_a ** 2)
            for rec in self.transitions
        ])
        nearest = np.argsort(distances)[:max(1, self.settings.transition_neighbors)]
        local = distances[nearest]
        weights = np.exp(-0.5 * (local - local.min()))
        weights /= weights.sum()
        deltas = np.asarray([
            self.transitions[int(idx)].after - self.transitions[int(idx)].before
            for idx in nearest
        ])
        delta = weights @ deltas
        # Remote records provide weak evidence; shrink their effect toward no change.
        novelty = 1.0 - math.exp(-0.5 * float(local.min()))
        estimate = base + (1.0 - novelty) * delta
        dispersion = np.sqrt(weights @ ((deltas - delta) ** 2))
        uncertainty = float(np.mean(dispersion) + novelty)
        return estimate, uncertainty

    def _repair(self, genome: np.ndarray) -> np.ndarray:
        """Map a raw GA genotype to the physical placement used by env/MILP."""
        repaired = self.bounds.clip(np.asarray(genome, dtype=float))
        if self.genome_repair is not None:
            repaired = np.asarray(self.genome_repair(repaired.copy()), dtype=float)
        if repaired.shape != (self.num_uavs, 4):
            raise ValueError("genome_repair must return shape (num_uavs, 4)")
        return self.bounds.clip(repaired)

    @staticmethod
    def _attr(ue: Any, *names: str) -> float:
        if isinstance(ue, Mapping):
            for name in names:
                if name in ue:
                    return float(ue[name])
        for name in names:
            if hasattr(ue, name):
                return float(getattr(ue, name))
        raise AttributeError(f"UE is missing one of {names}")

    def _ue(self, ue: Any) -> tuple[int, float, float, float]:
        if isinstance(ue, tuple):
            return int(ue[0]), float(ue[1]), float(ue[2]), float(ue[3])
        return (int(self._attr(ue, "id")), self._attr(ue, "x_m", "x"),
                self._attr(ue, "y_m", "y"),
                self._attr(ue, "req_data_rate", "rate_bps", "require_mbps"))

    def _coverage_gain(self, genome: np.ndarray) -> tuple[frozenset[int], float, float]:
        covered: set[int] = set()
        gains: list[float] = []
        rate_sum = 0.0
        for ue in self.ues:
            ue_id, x, y, rate = self._ue(ue)
            ue_model = env.UE(ue_id, x, y, rate / 1e6 if rate >= 1e4 else rate)
            best_gain = 0.0
            for x_u, y_u, h_u, theta in genome:
                placement = env.solution(-1, x_u, y_u, h_u, theta)
                if env.dis_2d(x_u, y_u, x, y) > placement.r_u:
                    continue
                gain = env.gain(placement, ue_model, self.phys)
                best_gain = max(best_gain, gain)
            if best_gain:
                covered.add(ue_id)
                gains.append(best_gain)
                rate_sum += rate * (1e6 if rate < 1e4 else 1.0)
        return frozenset(covered), rate_sum, float(np.mean(gains)) if gains else 0.0

    @staticmethod
    def _interval_violation(values: np.ndarray, lower: float, upper: float) -> float:
        """Sum of normalised lower/upper bound violations for one variable."""
        if not lower < upper:
            raise ValueError("A violation interval must have lower < upper")
        return float(np.sum(
            (np.maximum(0.0, lower - values) + np.maximum(0.0, values - upper))
            / (upper - lower)
        ))

    def _theta_violation(self, raw_genome: np.ndarray) -> float:
        return self._interval_violation(
            raw_genome[:, 3], self.bounds.lower[3], self.bounds.upper[3]
        )

    def _height_violation(self, raw_genome: np.ndarray) -> float:
        return self._interval_violation(
            raw_genome[:, 2], self.bounds.lower[2], self.bounds.upper[2]
        )

    def _separation_violation(self, genome: np.ndarray) -> float:
        minimum = self.settings.min_uav_separation_m
        if minimum <= 0:
            return 0.0
        violation = 0.0
        for first in range(len(genome) - 1):
            for second in range(first + 1, len(genome)):
                # 3D Euclidean distance: (x, y, h) to allow altitude layering
                dx = genome[first, 0] - genome[second, 0]
                dy = genome[first, 1] - genome[second, 1]
                dh = genome[first, 2] - genome[second, 2]
                distance = math.sqrt(dx * dx + dy * dy + dh * dh)
                violation += max(0.0, minimum - distance) / minimum
        return violation

    def _coverage_overlap(self, genome: np.ndarray) -> float:
        per_uav = self._per_uav_coverage(genome)
        total = sum(len(items) for items in per_uav)
        union = set().union(*per_uav) if per_uav else set()
        return max(0, total - len(union)) / max(total, 1)

    def _per_uav_coverage(self, genome: np.ndarray) -> list[set[int]]:
        per_uav: list[set[int]] = [set() for _ in range(len(genome))]
        for ue in self.ues:
            ue_id, x, y, _rate = self._ue(ue)
            for index, (x_u, y_u, h_u, theta) in enumerate(genome):
                if math.hypot(x_u - x, y_u - y) <= h_u * math.tan(math.radians(theta)):
                    per_uav[index].add(ue_id)
        return per_uav

    def _empty_uav_violation(self, genome: np.ndarray) -> float:
        """Disabled: allow UAVs to traverse empty zones toward dense clusters.

        The original penalty killed any UAV whose geometric footprint was
        momentarily empty, trapping UAVs at map edges.  MILP handles idle
        UAVs correctly, so this hard penalty is not needed.
        """
        return 0.0

    def _violation(
        self, raw_genome: np.ndarray, evaluated_genome: np.ndarray,
        power_w: float, prbs: float,
    ) -> ViolationBreakdown:
        
        optional_separation = (
            self._separation_violation(evaluated_genome)
            if self.settings.include_separation_constraint else 0.0
        )
        return ViolationBreakdown(
            theta=self._theta_violation(raw_genome),
            height=self._height_violation(raw_genome),
            empty=self._empty_uav_violation(evaluated_genome),
            power=0.0,
            prb=0.0,
            separation=optional_separation,
        )

    def _radio_topology_descriptor(self, genome: np.ndarray) -> np.ndarray:
        covered_demands = 0.0
        best_gains: list[float] = []
        best_los: list[float] = []
        total_demand = sum(item[3] for item in self._ue_by_id.values())
        for ue_id, x, y, demand in self._ue_by_id.values():
            ue = env.UE(ue_id, x, y, demand / 1e6 if demand >= 1e4 else demand)
            best_gain = 0.0
            best_probability = 0.0
            for x_u, y_u, h_u, theta in genome:
                placement = env.solution(-1, x_u, y_u, h_u, theta)
                if env.dis_2d(x_u, y_u, x, y) > placement.r_u:
                    continue
                link = env.air_to_ground_link(placement, ue, self.phys)
                if link.channel_gain > best_gain:
                    best_gain = link.channel_gain
                    best_probability = link.los_probability
            if best_gain > 0:
                covered_demands += demand
                best_gains.append(best_gain)
                best_los.append(best_probability)
        radio = np.asarray([
            len(best_gains) / len(self.ues),                        # Tỷ lệ phủ sóng (tỉ lệ UE)
            covered_demands / max(total_demand, 1e-20),             # Tỷ lệ đáp ứng mạng
            float(np.mean(best_los)) if best_los else 0.0,          # Xác suất LoS trực tiếp
            float(np.mean(np.log10(np.maximum(best_gains, 1e-30)))) if best_gains else -30.0, # Chất lượng kênh truyền tb
        ])
        canonical = np.asarray(sorted(genome.tolist(), key=lambda row: tuple(row)))
        topology = ((canonical - np.asarray(self.bounds.lower)) / self.bounds.span).ravel()  # Chuẩn hóa và đưa nó vào tọa độ 1D
        return np.concatenate((radio, topology))

    def evaluate_surrogate(self, item: Individual) -> Individual:
        """Predict exact output changes from confirmed transitions, not an allocation."""
        raw = np.asarray(item.genome, dtype=float)
        if raw.shape != (self.num_uavs, 4):
            raise ValueError("Each GA genome must have shape (num_uavs, 4)")
        genome = self._repair(raw)
        item.evaluated_genome = genome
        geometric_coverage, _, _ = self._coverage_gain(genome)
        item.overlap_ratio = self._coverage_overlap(genome)
        if not self.archive:
            raise RuntimeError("MILP warm-up must create Archive records first")
        anchor = item.anchor_record
        if anchor is None or anchor.genome is None:
            anchor = min(self.archive, key=lambda record: float(np.sum(
                ((genome - record.genome) / self.bounds.span) ** 2
            )))
        # Preserve the actual verified pre-state used for this prediction.
        # Otherwise a later exact evaluation silently loses the transition.
        item.anchor_record = anchor
        estimate, uncertainty = self._transition_prediction(anchor, genome)
        if not self.transitions:
            # Only during warm-up: transfer empirical admission rate from the
            # exact anchor to geometric potential, without assigning any UE.
            potential_anchor = max(len(anchor.potential_coverage), 1)
            estimate[-3] = len(geometric_coverage) / len(self.ues) * len(anchor.coverage) / potential_anchor
            potential_demand = sum(self._ue_by_id[k][3] for k in anchor.potential_coverage)
            exact_demand = sum(self._ue_by_id[k][3] for k in anchor.coverage)
            geometric_demand = sum(self._ue_by_id[k][3] for k in geometric_coverage)
            estimate[-2] = geometric_demand / max(self._total_demand_bps, 1e-20) * exact_demand / max(potential_demand, 1e-20)
        possible_demand = sum(self._ue_by_id[k][3] for k in geometric_coverage) / max(self._total_demand_bps, 1e-20)
        powers_ratio = np.maximum(estimate[:self.num_uavs], 0.0)
        prb_ratio = max(float(estimate[self.num_uavs]), 0.0)
        item.served_fraction = float(np.clip(estimate[-3], 0.0, len(geometric_coverage) / len(self.ues)))
        item.service_score = float(np.clip(estimate[-2], 0.0, possible_demand))
        item.coverage = geometric_coverage  # Potential only; not an invented MILP association.
        item.idle_redeployment_pressure = float(np.clip(estimate[-1], 0.0, 1.0))
        item.power_est_w = float(powers_ratio @ self._uav_power_budgets)
        item.prbs_est = prb_ratio * self.prb_budget
        item.power_ratio = float(np.max(powers_ratio))
        item.prb_ratio = prb_ratio
        item.bottleneck_pressure = max(0.0, max(item.power_ratio, prb_ratio) - self.settings.bottleneck_target)
        item.uncertainty = uncertainty
        terms = self._violation(raw, genome, item.power_est_w, item.prbs_est)
        risk = self.settings.transition_risk_kappa * min(uncertainty, 0.25)
        terms.power = float(np.sum(np.maximum(0.0, powers_ratio + risk - 1.0)))
        terms.prb = max(0.0, prb_ratio + risk - 1.0)
        item.violation, item.violation_terms = terms.total, terms.as_dict()
        item.fitness = self._resource_fitness(
            item.power_est_w, item.prbs_est, item.bottleneck_pressure,
            item.idle_redeployment_pressure,
        )
        item.exact_result = None
        return item

    def evaluate_exact(self, item: Individual) -> Individual:
        """Run the existing MILP once and use its real values in Archive."""
        raw_genome = np.asarray(item.genome, dtype=float)
        if raw_genome.shape != (self.num_uavs, 4):
            raise ValueError("Each GA genome must have shape (num_uavs, 4)")
        evaluated_genome = self._repair(raw_genome)
        result = self.milp_evaluator(evaluated_genome.copy())
        if "evaluated_genome" in result:
            evaluated = np.asarray(result["evaluated_genome"], dtype=float)
            if evaluated.shape != raw_genome.shape:
                raise ValueError("evaluated_genome must have the same shape as the GA genome")
            evaluated_genome = self._repair(evaluated)
        item.evaluated_genome = evaluated_genome
        coverage, rate_sum, mean_gain = self._coverage_gain(evaluated_genome)
        descriptor = self._radio_topology_descriptor(evaluated_genome)
        payload = result.get("milp", result)
        served = frozenset(int(x) for x in result.get(
            "served_ue_ids", [choice["ue_id"] for choice in payload.get("uav_mode_choices", ())]
        ))
        power = float(result["total_power_w"])
        prbs = float(result.get("total_num_prbs", result.get("total_prbs")))
        item.coverage, item.power_est_w, item.prbs_est = served, power, prbs
        item.served_fraction = len(served) / len(self.ues)
        item.service_score = self._service_score(served)
        item.power_ratio, item.prb_ratio, item.bottleneck_pressure = self._exact_bottleneck_pressure(result)
        exact_ue_count, _exact_prbs, _exact_powers = self._exact_uav_loads(result)
        item.idle_redeployment_pressure = self._idle_redeployment_pressure(
            len(served), exact_ue_count,
        )
        item.overlap_ratio = self._coverage_overlap(evaluated_genome)
        item.exact_result, item.uncertainty = result, 0.0
        terms = self._violation(raw_genome, evaluated_genome, power, prbs)
        terms.power = float(np.sum(np.maximum(0.0, _exact_powers / self._uav_power_budgets - 1.0)))
        terms.prb = max(0.0, prbs / self.prb_budget - 1.0)
        item.violation, item.violation_terms = terms.total, terms.as_dict()
        item.fitness = self._resource_fitness(
            power, prbs, item.bottleneck_pressure, item.idle_redeployment_pressure,
        )
        record = ArchiveRecord(
            served, coverage, descriptor, power, prbs, rate_sum, mean_gain, result,
            item.fitness, item.violation, evaluated_genome.copy(),
        )
        anchor = item.anchor_record
        if anchor is not None and anchor.genome is not None:
            action = ((evaluated_genome - anchor.genome) / self.bounds.span).ravel()
            if np.any(np.abs(action) > 1e-12):
                self.transitions.append(TransitionRecord(
                    state=self._transition_state(anchor),
                    action=action,
                    before=self._primitive_exact(anchor),
                    after=self._primitive_exact(record),
                    source=item.source,
                ))
        self.archive.append(record)
        item.anchor_record = record
        return item

    def _better(self, a: Individual, b: Individual) -> bool:
        """Admission-first epsilon-Deb comparator used by GA, WE and CMA.

        Resource weights are allowed to break ties only after physical
        feasibility and service.  This prevents a low-power solution from
        replacing an incumbent that serves more UEs.
        """
        tolerance = max(float(self.current_eps), 1e-12)
        a_feasible = a.violation <= tolerance
        b_feasible = b.violation <= tolerance
        if a_feasible != b_feasible:
            return a_feasible
        if not a_feasible and abs(a.violation - b.violation) > 1e-12:
            return a.violation < b.violation
        if abs(a.served_fraction - b.served_fraction) > 1e-12:
            return a.served_fraction > b.served_fraction
        if abs(a.service_score - b.service_score) > 1e-12:
            return a.service_score > b.service_score
        if abs(a.fitness - b.fitness) > 1e-12:
            return a.fitness > b.fitness
        return a.uncertainty < b.uncertainty

    def _better_exact_incumbent(self, a: Individual, b: Individual) -> bool:
        """Fixed ranking for MILP-confirmed incumbents; no generation epsilon.

        The evolving epsilon belongs to population exploration only.  It must
        not silently change which exact solution is reported as the best one.
        """
        if a.exact_result is None or b.exact_result is None:
            raise ValueError("Exact incumbent comparison requires two MILP results")
        a_feasible = a.violation <= 1e-12
        b_feasible = b.violation <= 1e-12
        if a_feasible != b_feasible:
            return a_feasible
        if not a_feasible and abs(a.violation - b.violation) > 1e-12:
            return a.violation < b.violation
        if abs(a.served_fraction - b.served_fraction) > 1e-12:
            return a.served_fraction > b.served_fraction
        if abs(a.service_score - b.service_score) > 1e-12:
            return a.service_score > b.service_score
        # Contextual weights may have changed since either exact call.  Score
        # both records using the *same current* weights before the tie-break.
        a_cost = self._resource_fitness(
            a.power_est_w, a.prbs_est, a.bottleneck_pressure,
            a.idle_redeployment_pressure,
        )
        b_cost = self._resource_fitness(
            b.power_est_w, b.prbs_est, b.bottleneck_pressure,
            b.idle_redeployment_pressure,
        )
        return a_cost > b_cost + 1e-12

    def _sorted(self, pop: Sequence[Individual]) -> list[Individual]:
        def compare(first: Individual, second: Individual) -> int:
            if self._better(first, second):
                return -1
            if self._better(second, first):
                return 1
            return 0
        return sorted(pop, key=cmp_to_key(compare))

    def _tournament(self, pop: Sequence[Individual]) -> Individual:
        if len(pop) == 1:
            return pop[0]
        i, j = self.rng.choice(len(pop), 2, replace=False)
        return pop[i] if self._better(pop[i], pop[j]) else pop[j]

    def _sbx(self, a: Individual, b: Individual) -> tuple[Individual, Individual]:
        shape = (self.num_uavs, 1) if self.settings.sbx_uav_blocks else a.genome.shape
        u, eta = self.rng.random(shape), self.settings.sbx_eta
        beta = np.where(u <= .5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
        first = .5*((1+beta)*a.genome + (1-beta)*b.genome)
        second = .5*((1-beta)*a.genome + (1+beta)*b.genome)
        # Keep raw offspring so C6/C7 can be measured by V(x).  The physical
        # projection is performed only inside evaluate_surrogate/exact.
        anchor_a = a.anchor_record if a.anchor_record is not None else b.anchor_record
        anchor_b = b.anchor_record if b.anchor_record is not None else a.anchor_record
        return (Individual(first, anchor_record=anchor_a, source="sbx"),
                Individual(second, anchor_record=anchor_b, source="sbx"))

    def _mutate(self, item: Individual) -> Individual:
        mask = self.rng.random(item.genome.shape) < self.settings.mutation_probability
        noise = self.rng.normal(0, self.settings.mutation_sigma, item.genome.shape) * self.bounds.span
        return Individual(item.genome + mask * noise,
                          anchor_record=item.anchor_record,
                          source=f"{item.source}+mutation")
