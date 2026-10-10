import math

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import config
import env

import highspy as gp


@dataclass(frozen=True)
class ServiceMode:
    u: int
    k: int
    placement: int
    power_index: int
    num_prbs: int
    power_w: float
    rate_mbps: float
    sinr_linear: float

    @property
    def key(self) -> Tuple[int,int,int,int]:
        return (self.u, self.k, self.placement, self.power_index)

def pareto_filter_modes(modes: Sequence[ServiceMode])-> List[ServiceMode]:
    kept: List[ServiceMode] = []
    for candidate in modes:
        dominated = any(
            other.num_prbs <= candidate.num_prbs
            and other.power_w <= candidate.power_w + 1e-12
            and(
                other.num_prbs < candidate.num_prbs
                or other.power_w < candidate.power_w - 1e-12
            )
            for other in modes
            if other is not candidate
        )
        if not dominated:
            kept.append(candidate)

    return sorted(kept,key=lambda mode: (mode.num_prbs,mode.power_w))



class UAVMILP:
    def __init__(
        self,
        uavs: Sequence[env.UAV],
        ues: Sequence[env.UE],
        placements: Sequence[env.solution],
        phys: config.PhysConstant,
        settings: config.SolverSettings,
        fixed_placement_indices: Sequence[int] | None = None,
        reference_placement_indices: Sequence[int] | None = None,
        reference_total_power_w: Sequence[float] | None = None,
        system_prb_budget: int | None = None,
        orthogonal_prb_pool: bool = False,
    ):
        self.uavs = uavs
        self.ues = ues
        self.placements = list(placements)
        self.phys = phys
        self.settings = settings
        self.orthogonal_prb_pool = bool(orthogonal_prb_pool)
        self.system_prb_budget = int(
            config.system_max_prbs
            if system_prb_budget is None else system_prb_budget
        )
        self.fixed_placement_indices = (
            None
            if fixed_placement_indices is None
            else tuple(fixed_placement_indices)
        )
        self.reference_placement_indices = (
            None if reference_placement_indices is None else tuple(reference_placement_indices)
        )
        self.reference_total_power_w = (
            None if reference_total_power_w is None else tuple(float(value) for value in reference_total_power_w)
        )
        self.power_level_dbm = self.phys.power_levels_dbm
        self.power_level_w = list(self.phys.power_levels_w)

        self.service_modes: Dict[Tuple[int,int,int,int], ServiceMode] = {}
        self.modes_by_uk: Dict[Tuple[int,int], List[Tuple[int,int,int,int]]] = defaultdict(list)
        self.modes_by_uks: DefaultDict[Tuple[int,int,int], List[Tuple[int,int,int,int]]] = defaultdict(list)
        self.distance_2d: Dict[Tuple[int,int], float] = {}
        self.big_m: Dict[Tuple[int,int], float] = {}
        self.model: Any = None
        self.vars: Dict[str, Any] = {}

    def _minimum_prbs_for_power(
        self,
        power_w: float,
        max_prbs: int,
        channel_gain: float,
        require_mbps: float,
        interference_w: float = 0.0,
        ) -> Tuple[int, float, float] | None:
        for num_prbs in range(1, max_prbs + 1):
            rate, sinr = env.data_rate_mbps(
                power_w, num_prbs, channel_gain, self.phys, interference_w
            )
            if rate + 1e-9 >= require_mbps:
                return num_prbs, rate, sinr
        return None

    def _reference_interference_w(self, served_uav: int, ue_index: int) -> float:
        """Interference from the previous fixed-point MILP solution."""
        if self.orthogonal_prb_pool:
            return 0.0
        if self.reference_placement_indices is None or self.reference_total_power_w is None:
            return 0.0
        ue = self.ues[ue_index]
        return sum(
            self.reference_total_power_w[other] * env.gain(
                self.placements[self.reference_placement_indices[other]], ue, self.phys
            )
            for other in range(len(self.uavs)) if other != served_uav
        )



    def precompute_service_modes(self) -> None:
        self.service_modes.clear()
        self.modes_by_uk.clear()
        self.modes_by_uks.clear()
        self.distance_2d.clear()
        self.big_m.clear()

        # A GA exact evaluation fixes every UAV placement.  Link modes for
        # other grid cells can never be selected, so do not build them.
        active_placements = (
            set(self.fixed_placement_indices)
            if self.fixed_placement_indices is not None
            else set(range(len(self.placements)))
        )
        geometry: Dict[Tuple[int, int], Tuple[float, float, bool]] = {}
        for k, ue in enumerate(self.ues):
            for s, placement in enumerate(self.placements):
                distance = env.dis_2d(
                    placement.x_u, placement.y_u, ue.x_m, ue.y_m
                )
                self.distance_2d[(k, s)] = distance
                if s in active_placements:
                    covered = distance <= placement.r_u + 1e-9
                    geometry[(k, s)] = (
                        distance,
                        env.gain(placement, ue, self.phys),
                        covered,
                    )

        frontier_cache: Dict[
            Tuple[int, float, int, int], List[Tuple[int, int, float, float, float]]
        ] = {}
        for u, uav in enumerate(self.uavs):
            # With GA/CMA exact evaluation, every UAV is already fixed to one
            # placement.  Creating link modes for the other UAVs' placements
            # only bloats the MILP; those variables can never be selected.
            allowed_placements = (
                (self.fixed_placement_indices[u],)
                if self.fixed_placement_indices is not None else active_placements
            )
            for k, ue in enumerate(self.ues):
                interference_w = self._reference_interference_w(u, k)
                self.big_m[(u, k)] = max(
                    0.0,
                    max(
                        self.distance_2d[(k, s)] - placement.r_u
                        for s, placement in enumerate(self.placements)
                    ),
                )

                for s in allowed_placements:
                    _distance, channel_gain, covered = geometry[(k, s)]
                    if not covered:
                        continue

                    total_prb_budget = min(self.system_prb_budget, uav.max_prbs,
                                           self.phys.allocation_max_prbs)
                    cache_key = (
                        uav.max_prbs, round(uav.max_power_w, 12), k, s,
                        total_prb_budget, round(uav.max_power_w, 12), k, s,
                        round(interference_w, 24),
                    )
                    cached = frontier_cache.get(cache_key)
                    if cached is None:
                        modes: List[ServiceMode] = []
                        for power_index, power_w in enumerate(self.power_level_w):
                            if power_w > min(uav.max_power_w, self.phys.allocation_max_power_w) + 1e-12:
                                continue
                            result = self._minimum_prbs_for_power(
                                power_w,
                                total_prb_budget,
                                channel_gain,
                                ue.require_mbps,
                                interference_w,
                            )
                            if result is None:
                                continue
                            num_prbs, rate, sinr = result
                            modes.append(
                                ServiceMode(
                                    u=-1,
                                    k=k,
                                    placement=s,
                                    power_index=power_index,
                                    num_prbs=num_prbs,
                                    power_w=power_w,
                                    rate_mbps=rate,
                                    sinr_linear=sinr,
                                )
                            )
                        cached = [
                            (
                                mode.power_index,
                                mode.num_prbs,
                                mode.power_w,
                                mode.rate_mbps,
                                mode.sinr_linear,
                            )
                            for mode in pareto_filter_modes(modes)
                        ]
                        frontier_cache[cache_key] = cached

                    for power_index, num_prbs, power_w, rate, sinr in cached:
                        mode = ServiceMode(
                            u=u,
                            k=k,
                            placement=s,
                            power_index=power_index,
                            num_prbs=num_prbs,
                            power_w=power_w,
                            rate_mbps=rate,
                            sinr_linear=sinr,
                        )
                        self.service_modes[mode.key] = mode
                        self.modes_by_uk[(u, k)].append(mode.key)
                        self.modes_by_uks[(u, k, s)].append(mode.key)



    def build_model(self) -> Any:
        if gp is None:
            raise RuntimeError(
                "highspy is not installed; run `python -m pip install highspy`"
            )
        if not self.service_modes:
            self.precompute_service_modes()

        model = gp.Highs()
        for option, value in (
            ("time_limit", self.settings.time_limit_s),
            ("mip_rel_gap", self.settings.mip_gap),
            ("output_flag", bool(self.settings.log_to_console)),
            ("threads", self.settings.num_threads),
            ("parallel", "on" if self.settings.parallel_search else "off"),
        ):
            model.setOptionValue(option, value)

        # HiGHS drops tiny matrix coefficients and returns kWarning. The
        # highspy addConstr wrapper treats that warning as an exception.
        _, matrix_zero = model.getOptionValue("small_matrix_value")

        U = range(len(self.uavs))
        K = range(len(self.ues))
        S = range(len(self.placements))
        mode_keys = list(self.service_modes)

        pi = model.addVariables(
            K,
            type=gp.HighsVarType.kInteger,
            lb=0,
            ub=1,
            name=[f"pi_{k}" for k in K],
        )
        phi = model.addVariables(
            U,
            K,
            type=gp.HighsVarType.kInteger,
            lb=0,
            ub=1,
            name=[f"phi_{u}_{k}" for u in U for k in K],
        )
        n_prb = model.addVariables(
            U,
            K,
            type=gp.HighsVarType.kInteger,
            lb=0,
            ub=self.system_prb_budget,
            name=[f"n_prb_{u}_{k}" for u in U for k in K],
        )
        power = model.addVariables(
            U, K, name=[f"power_w_{u}_{k}" for u in U for k in K]
        )
        rate = model.addVariables(
            U, K, name=[f"rate_mbps_{u}_{k}" for u in U for k in K]
        )
        place = model.addVariables(
            U,
            S,
            type=gp.HighsVarType.kInteger,
            lb=0,
            ub=1,
            name=[f"place_{u}_{s}" for u in U for s in S],
        )
        mode_choice = model.addVariables(
            mode_keys,
            type=gp.HighsVarType.kInteger,
            lb=0,
            ub=1,
            name=[f"mode_{'_'.join(map(str, key))}" for key in mode_keys],
        )
        x_u = model.addVariables(U, name=[f"x_u_{u}" for u in U])
        y_u = model.addVariables(U, name=[f"y_u_{u}" for u in U])
        h_u = model.addVariables(U, name=[f"h_u_{u}" for u in U])
        theta_u = model.addVariables(U, name=[f"theta_u_{u}" for u in U])
        r_u = model.addVariables(U, lb=0, name=[f"r_u_{u}" for u in U])

        for u in U:
            model.addConstr(
                model.qsum(place[u, s] for s in S) == 1,
                name=f"one_place_{u}",
            )
            if self.fixed_placement_indices is not None:
                model.addConstr(
                    place[u, self.fixed_placement_indices[u]] == 1,
                    name=f"fixed_place_{u}",
                )
            model.addConstr(
                x_u[u]
                == model.qsum(
                    self.placements[s].x_u * place[u, s] for s in S
                ),
                name=f"define_x_u_{u}",
            )
            model.addConstr(
                y_u[u]
                == model.qsum(
                    self.placements[s].y_u * place[u, s] for s in S
                ),
                name=f"define_y_u_{u}",
            )
            model.addConstr(
                h_u[u]
                == model.qsum(
                    self.placements[s].h_u * place[u, s] for s in S
                ),
                name=f"Constr7_11_{u}",
            )
            model.addConstr(
                theta_u[u]
                == model.qsum(
                    self.placements[s].theta_u * place[u, s] for s in S
                ),
                name=f"Constr6_10_{u}",
            )
            model.addConstr(
                r_u[u]
                == model.qsum(
                    self.placements[s].r_u * place[u, s] for s in S
                ),
                name=f"define_r_u_{u}",
            )

        for (u, k, s), keys in self.modes_by_uks.items():
            model.addConstr(
                model.qsum(mode_choice[key] for key in keys) <= place[u, s],
                name=f"only_1_place_for_uav_{u}_{k}_{s}",
            )

        for k in K:
            model.addConstr(
                model.qsum(phi[u, k] for u in U) == pi[k],
                name=f"constr_1_{k}",
            )

        for u in U:
            for k in K:
                keys = self.modes_by_uk.get((u, k), [])
                model.addConstr(
                    phi[u, k] == model.qsum(mode_choice[key] for key in keys),
                    name=f"mode_defines_phi_{u}_{k}",
                )
                model.addConstr(
                    n_prb[u, k]
                    == model.qsum(
                        mode_choice[key] * self.service_modes[key].num_prbs
                        for key in keys
                    ),
                    name=f"mode_define_number_of_prb_{u}_{k}",
                )
                model.addConstr(
                    power[u, k]
                    == model.qsum(
                        self.service_modes[key].power_w * mode_choice[key]
                        for key in keys
                    ),
                    name=f"define_power_{u}_{k}",
                )
                model.addConstr(
                    rate[u, k]
                    == model.qsum(
                        self.service_modes[key].rate_mbps * mode_choice[key]
                        for key in keys
                    ),
                    name=f"define_rate_{u}_{k}",
                )
                model.addConstr(
                    n_prb[u, k] >= phi[u, k], name=f"min_n_prb_{u}_{k}"
                )
                total_prb_budget = self.system_prb_budget
                model.addConstr(
                    n_prb[u, k] <= total_prb_budget * phi[u, k],
                    name=f"max_n_prb_{u}_{k}",
                )
                feasible_powers = [
                    self.service_modes[key].power_w for key in keys
                ]
                min_power = min(feasible_powers, default=self.power_level_w[0])
                max_power = max(feasible_powers, default=self.power_level_w[-1])
                model.addConstr(
                    power[u, k] >= min_power * phi[u, k],
                    name=f"min_power_{u}_{k}",
                )
                model.addConstr(
                    power[u, k] <= max_power * phi[u, k],
                    name=f"max_power_{u}_{k}",
                )
                model.addConstr(
                    rate[u, k] >= self.ues[k].require_mbps * phi[u, k],
                    name=f"data_rate_constraint_{u}_{k}",
                )
                selected_position = model.qsum(
                    self.distance_2d[(k, s)] * place[u, s] for s in S
                    # Coincident weighted centres can leave ~1e-13 m residue.
                    # Omit only coefficients the solver would ignore anyway;
                    # do not change distances used by the physical link model.
                    if self.distance_2d[(k, s)] > matrix_zero
                )
                model.addConstr(
                    selected_position
                    <= r_u[u] + self.big_m[(u, k)] * (1 - phi[u, k]),
                    name=f"C8_Constraint_{u}_{k}",
                )

        for u in U:
            model.addConstr(
                model.qsum(power[u, k] for k in K)
                <= self.uavs[u].max_power_w,
                name=f"C4_Constraint_{u}",
            )
            # In the configured shared pool max_prbs=system_prb_budget, so
            # this bound is redundant, not a reserved 1/U share of the pool.
            model.addConstr(
                model.qsum(n_prb[u, k] for k in K)
                <= self.uavs[u].max_prbs,
                name=f"C9_Constraint_{u}",
            )

        # O-RAN Global PRB Pool Constraint
        total_prb_budget = self.system_prb_budget
        model.addConstr(
            model.qsum(n_prb[u, k] for u in U for k in K) <= total_prb_budget,
            name="Global_PRB_Constraint",
        )
        if (
            self.fixed_placement_indices is None
            and self.settings.symmetry_breaking
            and self._uavs_are_identical()
        ):
            for u in range(len(self.uavs) - 1):
                model.addConstr(
                    model.qsum(s * place[u, s] for s in S)
                    <= model.qsum(s * place[u + 1, s] for s in S),
                    name=f"symmetry_order_{u}",
                )

        total_served = model.qsum(pi[k] for k in K)
        if self.settings.minimize_power_tiebreak:
            total_num_prbs = model.qsum(n_prb[u, k] for u in U for k in K)
            total_power_w = model.qsum(power[u, k] for u in U for k in K)
            max_total_prbs = self.system_prb_budget
            max_total_power_w = sum(uav.max_power_w for uav in self.uavs)

            # Exact lexicographic order required by the GA comparator:
            #   admitted UE count -> admitted demand -> PRBs -> power.
            # Convert Mbps to integer demand units so the coefficients below
            # provably preserve that order in a single MILP objective.
            demand_bps = [max(1, int(round(ue.require_mbps * 1e6))) for ue in self.ues]
            demand_gcd = demand_bps[0]
            for value in demand_bps[1:]:
                demand_gcd = math.gcd(demand_gcd, value)
            demand_units = [value // demand_gcd for value in demand_bps]
            admitted_demand = model.qsum(demand_units[k] * pi[k] for k in K)
            max_demand_units = sum(demand_units)

            # Resource penalty lies in [0, 2*max_total_prbs + 1].  Therefore
            # one demand unit outweighs every possible resource improvement,
            # and one extra UE outweighs all demand/resource differences.
            resource_span = 2 * max_total_prbs + 1
            demand_weight = resource_span + 1
            count_weight = demand_weight * (max_demand_units + 1)
            model.setObjective(
                count_weight * total_served
                + demand_weight * admitted_demand
                - 2 * total_num_prbs
                - total_power_w / max_total_power_w,
                gp.ObjSense.kMaximize,
            )
        else:
            model.setObjective(total_served, gp.ObjSense.kMaximize)

        self.model = model
        self.vars = {
            "pi": pi,
            "phi": phi,
            "n_prb": n_prb,
            "power": power,
            "rate": rate,
            "place": place,
            "mode": mode_choice,
            "x_u": x_u,
            "y_u": y_u,
            "h_u": h_u,
            "theta_u": theta_u,
            "r_u": r_u,
        }
        return model

    def _uavs_are_identical(self) -> bool:
        first = (self.uavs[0].max_power_w, self.uavs[0].max_prbs)
        return all(
            (uav.max_power_w, uav.max_prbs) == first for uav in self.uavs[1:]
        )

    def solve(self, lp_path: str | Path | None = None) -> Dict[str, Any]:
        model = self.model if self.model is not None else self.build_model()
        if lp_path is not None:
            if model.writeModel(str(lp_path)) != gp.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS could not write model to {lp_path}")
        if model.run() == gp.HighsStatus.kError:
            raise RuntimeError("HiGHS failed while solving the model")
        if not model.getSolution().value_valid:
            status = model.getModelStatus()
            raise RuntimeError(
                f"HiGHS found no solution; status={int(status)} "
                f"({model.modelStatusToString(status)})"
            )
        return self.extract_solution()

    def solve_interference_aware(self, max_iterations: int = 6) -> Dict[str, Any]:
        """Fixed-point MILP with the shared interference-aware SINR in env.py.

        A direct binary-position/binary-association SINR denominator is MINLP.
        This outer loop keeps each inner HiGHS model linear by fixing the
        previous solution as the interference snapshot.
        """
        if self.orthogonal_prb_pool:
            result = self.solve()
            result["interference_iterations"] = 0
            result["interference_converged"] = True
            result["orthogonal_prb_pool"] = True
            return result
        current: UAVMILP = self
        result = current.solve()
        previous_signature: tuple[tuple[int, ...], tuple[float, ...]] | None = None
        for iteration in range(1, max_iterations + 1):
            positions = tuple(result["selected_placement_indices"])
            powers = tuple(round(value, 9) for value in result["power_by_uav_w"])
            signature = (positions, powers)
            if signature == previous_signature:
                result["interference_iterations"] = iteration - 1
                result["interference_converged"] = True
                return result
            previous_signature = signature
            current = UAVMILP(
                uavs=self.uavs, ues=self.ues, placements=self.placements,
                phys=self.phys, settings=self.settings,
                fixed_placement_indices=self.fixed_placement_indices,
                reference_placement_indices=positions,
                reference_total_power_w=powers,
                system_prb_budget=self.system_prb_budget,
                orthogonal_prb_pool=self.orthogonal_prb_pool,
            )
            result = current.solve()
        result["interference_iterations"] = max_iterations
        result["interference_converged"] = False
        return result

    def extract_solution(self) -> Dict[str, Any]:
        """Return active UAV-UE modes and the three numeric GA objectives.

        Idle UAVs have no service mode and therefore do not appear in
        ``uav_mode_choices``.
        """
        if self.model is None:
            raise RuntimeError("There is no solution to extract")
        solution = self.model.getSolution()
        if not solution.value_valid:
            raise RuntimeError("There is no solution to extract")

        col_value = solution.col_value
        mode_choice = self.vars["mode"]
        place = self.vars["place"]
        selected_modes = sorted(
            (
                mode
                for key, mode in self.service_modes.items()
                if col_value[int(mode_choice[key])] > 0.5
            ),
            key=lambda mode: mode.key,
        )
        uav_mode_choices = [
            {
                **asdict(mode),
                "uav_id": self.uavs[mode.u].id,
                "ue_id": self.ues[mode.k].id,
                "placement_id": self.placements[mode.placement].index,
            }
            for mode in selected_modes
        ]
        selected_placement_indices = [
            next(
                s for s in range(len(self.placements))
                if col_value[int(place[u, s])] > 0.5
            )
            for u in range(len(self.uavs))
        ]
        power_by_uav = [0.0] * len(self.uavs)
        for mode in selected_modes:
            power_by_uav[mode.u] += mode.power_w

        return {
            "uav_mode_choices": uav_mode_choices,
            # Consumers can distinguish a proved optimum from a feasible
            # time-limit incumbent before using it as a calibration label.
            "solver_status": self.model.modelStatusToString(self.model.getModelStatus()),
            "solver_optimal": self.model.getModelStatus() == gp.HighsModelStatus.kOptimal,
            "solver_mip_gap": float(self.model.getInfo().mip_gap),
            "num_ues_served": len({mode.k for mode in selected_modes}),
            "total_num_prbs": sum(mode.num_prbs for mode in selected_modes),
            "total_power_w": sum(
                (mode.power_w for mode in selected_modes), start=0.0
            ),
            "selected_placement_indices": selected_placement_indices,
            "power_by_uav_w": power_by_uav,
            "system_prb_budget": self.system_prb_budget,
            "orthogonal_prb_pool": self.orthogonal_prb_pool,
        }
