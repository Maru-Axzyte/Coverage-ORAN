"""GA data structures, evaluation helpers, parent selection, SBX and mutation.

``milp_evaluator`` receives a genome with rows ``[x, y, h, theta_deg]`` and
returns ``total_power_w``, ``total_num_prbs`` (or ``total_prbs``), and
optionally ``served_ue_ids``. The active generation loop is in joint_topk_we.py;
run main.py to assemble the complete optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

    @property
    def span(self) -> np.ndarray:
        return np.asarray(self.upper) - np.asarray(self.lower)
    def clip(self, genome: np.ndarray) -> np.ndarray:
        return np.clip(genome, np.asarray(self.lower), np.asarray(self.upper))

@dataclass(frozen=True)
class GASettings:
    """Settings consumed by the supported joint WE flow (main.py).

    population_size is the initial population; population_capacity is its
    concurrent parent/offspring ceiling. Weight values below are initial priors.
    """
    population_size: int = 400
    population_capacity: int | None = 4000
    generations: int = 400
    initial_milp_seeds: int = 80
    update_period: int = 10
    sbx_eta: float = 18.0
    sbx_uav_blocks: bool = True
    children_per_pair: int = 50
    mutation_probability: float = 0.12
    mutation_sigma: float = 0.08
    elite_fraction: float = 0.15
    beta_power: float = 1 / 3
    mu_prb: float = 1 / 3
    eta_bottleneck: float = 1 / 3
    transition_state_bandwidth: float = 1.0
    transition_action_bandwidth: float = 0.5
    transition_neighbors: int = 12
    # Preserve the active main.py profile; physical spacing comes from config.
    include_separation_constraint: bool = True
    min_uav_separation_m: float = config.uav_min_separation_m
    random_seed: int | None = config.seed

@dataclass
class ArchiveRecord:
    coverage: frozenset[int]  # Exact MILP-served UE ids.
    potential_coverage: frozenset[int]  # Geometric/radio UE set used by surrogate.
    power_w: float
    prbs: float
    milp_result: Mapping[str, Any]
    fitness: float = 0.0
    violation: float = 0.0
    genome: np.ndarray | None = None

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
            # Empty-footprint penalties are disabled. Retain these zero-valued
            # output keys for existing result readers; they do not affect V.
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
    """Shared GA operators and oracle/archive helpers; no standalone run loop.

    JointSurrogateMixin supplies the sole surrogate, resource fitness and
    bottleneck calculation used by the assembled optimizer.
    """
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
        resource_weight_sum = (
            settings.beta_power + settings.mu_prb + settings.eta_bottleneck
        )
        self._resource_weights = np.asarray([
            settings.beta_power, settings.mu_prb, settings.eta_bottleneck,
        ]) / resource_weight_sum
        self.milp_evaluator, self.settings = milp_evaluator, settings
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
        self.history: list[dict[str, float]] = []
        # Epsilon-Deb: adaptive violation tolerance (decays over generations)
        self.current_eps: float = 1.0
        self.current_generation: int = 0

    @property
    def _total_demand_bps(self) -> float:
        return sum(record[3] for record in self._ue_by_id.values())

    def _service_score(self, served: frozenset[int]) -> float:
        return sum(self._ue_by_id[ue_id][3] for ue_id in served) / max(self._total_demand_bps, 1e-20)

    def _idle_redeployment_pressure(
        self, served_count: int, per_uav_ue_count: np.ndarray,
    ) -> float:
        """Diagnostic only; idle UAV fraction is not a fitness penalty."""
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

    def _repair(self, genome: np.ndarray) -> np.ndarray:
        """Map a raw GA genotype to the physical placement used by env/MILP."""
        repaired = self.bounds.clip(np.asarray(genome, dtype=float))
        if self.genome_repair is not None:
            repaired = np.asarray(self.genome_repair(repaired.copy()), dtype=float)
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

    def evaluate_exact(self, item: Individual) -> Individual:
        """Run the existing MILP once and use its real values in Archive."""
        raw_genome = np.asarray(item.genome, dtype=float)
        evaluated_genome = self._repair(raw_genome)
        result = self.milp_evaluator(evaluated_genome.copy())
        if "evaluated_genome" in result:
            evaluated = np.asarray(result["evaluated_genome"], dtype=float)
            evaluated_genome = self._repair(evaluated)
        item.evaluated_genome = evaluated_genome
        coverage, _, _ = self._coverage_gain(evaluated_genome)
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
            served, coverage, power, prbs, result,
            item.fitness, item.violation, evaluated_genome.copy(),
        )
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
