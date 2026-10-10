from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable
import math
import numpy as np

import config


def dbm_to_wat(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


def wat_to_dbm(power_w: float) -> float:
    return 10.0 * math.log10(power_w) + 30.0


def dis_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def coverage_radius(height_m: float, theta_deg: float) -> float:
    return height_m * math.tan(math.radians(theta_deg))


# Al-Hourani LoS parameters: (a, b).  The loss values are in dB.
A2G_ENVIRONMENT_PRESETS: dict[str, tuple[float, float]] = {
    "Suburban": (4.88, 0.43),
    "Urban": (9.61, 0.16),
    "Dense Urban": (12.08, 0.11),
    "High-rise Urban": (27.23, 0.08),
}
# Mean excessive losses from the supplied environment model (dB).
N_LOS_DB = 1.0
N_NLOS_DB = 20.0
# Compatibility names matching the supplied _env.py.
n_LOS = N_LOS_DB
n_NLOS = N_NLOS_DB


@dataclass(frozen=True)
class AirToGroundParameters:
    """Al-Hourani probabilistic LoS/NLoS channel parameters."""
    environment: str = "Urban"
    los_a: float | None = None
    los_b: float | None = None
    excess_loss_los_db: float = N_LOS_DB
    excess_loss_nlos_db: float = N_NLOS_DB
    rsrp_threshold_dbm: float = -100.0

    def __post_init__(self) -> None:
        preset_a, preset_b = A2G_ENVIRONMENT_PRESETS[self.environment]
        object.__setattr__(self, "los_a", preset_a if self.los_a is None else self.los_a)
        object.__setattr__(self, "los_b", preset_b if self.los_b is None else self.los_b)


def default_air_to_ground_parameters() -> AirToGroundParameters:
    return AirToGroundParameters(environment=config.a2g_environment)


@dataclass(frozen=True)
class AirToGroundLink:
    distance_2d_m: float
    distance_3d_m: float
    elevation_deg: float
    los_probability: float
    path_loss_db: float
    antenna_gain_linear: float
    channel_gain: float


def los_probability(elevation_deg: float, channel: AirToGroundParameters | None = None) -> float:
    """Al-Hourani P(LoS) as a function of elevation angle in degrees."""
    channel = channel or default_air_to_ground_parameters()
    return 1.0 / (1.0 + channel.los_a * math.exp(
        -channel.los_b * (elevation_deg - channel.los_a)
    ))


@dataclass
class UAV:
    id: int
    x: float
    y: float
    theta_max: float = 60.0
    z: float = 250.0
    p_tx_dBm: float = 43.0
    f_c: float = config.f_c
    num_antenna: int = 5
    bandwidth: float = 20e6
    num_RB: int = 60
    max_power_w: float | None = None
    max_prbs: int | None = None

    def __post_init__(self) -> None:
        if self.max_power_w is None:
            self.max_power_w = dbm_to_wat(self.p_tx_dBm)
        if self.max_prbs is None:
            self.max_prbs = self.num_RB
        self.radius = coverage_radius(self.z, self.theta_max)

    def get_position(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=float)


@dataclass
class UE:
    id: int
    x_m: float
    y_m: float
    require_mbps: float = 2.0
    min_sinr_db: float = -3.0

    @property
    def x(self) -> float:
        return self.x_m

    @property
    def y(self) -> float:
        return self.y_m

    @property
    def req_data_rate(self) -> float:
        return self.require_mbps * 1e6


@dataclass(frozen=True)
class solution:
    """One discrete MILP placement: x, y, h, theta."""
    index: int
    x_u: float
    y_u: float
    h_u: float
    theta_u: float
    r_u: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_u", coverage_radius(self.h_u, self.theta_u))


Placement = solution


@dataclass(frozen=True)
class Scenario:
    uavs: tuple[UAV, ...]
    ues: tuple[UE, ...]
    placements: tuple[solution, ...]


def make_placements(
    x_values: Iterable[float], y_values: Iterable[float],
    heights_m: Iterable[float], theta_degrees: Iterable[float],
) -> list[solution]:
    return [
        solution(index, float(x), float(y), float(h), float(theta))
        for index, (x, y, h, theta) in enumerate(
            product(x_values, y_values, heights_m, theta_degrees)
        )
    ]


def air_to_ground_link(
    placement: solution, ue: UE, phys: config.PhysConstant,
    channel: AirToGroundParameters | None = None,
) -> AirToGroundLink:
    """Shared probabilistic LoS/NLoS link budget for every model component."""
    channel = channel or default_air_to_ground_parameters()
    d_2d = dis_2d(placement.x_u, placement.y_u, ue.x_m, ue.y_m)
    distance = max(math.hypot(d_2d, placement.h_u), phys.d_0)
    elevation_deg = math.degrees(math.atan2(placement.h_u, max(d_2d, 1e-9)))
    probability = los_probability(elevation_deg, channel)
    fspl_db = 20.0 * math.log10(4.0 * math.pi * phys.f_c * distance / phys.C)
    path_loss_db = fspl_db + probability * channel.excess_loss_los_db + (
        1.0 - probability
    ) * channel.excess_loss_nlos_db
    antenna_gain = 2.0 * phys.eta_ant / (1.0 - math.cos(math.radians(placement.theta_u)))
    channel_gain = antenna_gain * 10.0 ** (-path_loss_db / 10.0)
    return AirToGroundLink(
        d_2d, distance, elevation_deg, probability, path_loss_db,
        antenna_gain, channel_gain,
    )


def gain(
    placement: solution, ue: UE, phys: config.PhysConstant,
    channel: AirToGroundParameters | None = None,
) -> float:
    """Shared channel gain with probabilistic Al-Hourani LoS/NLoS loss."""
    return air_to_ground_link(placement, ue, phys, channel).channel_gain


def rsrp_dbm(
    transmit_power_dbm: float, num_prbs: int, placement: solution, ue: UE,
    phys: config.PhysConstant, channel: AirToGroundParameters | None = None,
) -> float:
    """Per-PRB RSRP using the same A2G path loss and antenna gain."""
    link = air_to_ground_link(placement, ue, phys, channel)
    return transmit_power_dbm - 10.0 * math.log10(num_prbs) + 10.0 * math.log10(
        link.antenna_gain_linear
    ) - link.path_loss_db


def noise_per_prb_w(phys: config.PhysConstant) -> float:
    """Thermal-noise power on one PRB, including the receiver noise figure."""
    return phys.k_B * phys.T * phys.Bandwidth_BRP * phys.noise_figure_linear


def sinr_linear(
    power_per_prb_w: float, desired_gain: float, interference_w: float,
    phys: config.PhysConstant,
) -> float:
    """SINR on one PRB; ``power_per_prb_w`` is not total link power."""
    return power_per_prb_w * desired_gain / max(
        noise_per_prb_w(phys) + interference_w, 1e-30
    )


def spectral_efficiency_bps_hz(sinr: float) -> float:
    return math.log2(1.0 + sinr)


def data_rate_mbps(
    power_w: float, num_prbs: int, channel_gain: float, phys: config.PhysConstant,
    interference_w: float = 0.0,
) -> tuple[float, float]:
    """Rate when ``power_w`` is total link power spread over its PRBs."""
    if num_prbs == 0:
        return 0.0, 0.0
    power_per_prb_w = power_w / num_prbs
    sinr = sinr_linear(power_per_prb_w, channel_gain, interference_w, phys)
    return num_prbs * phys.Bandwidth_BRP * spectral_efficiency_bps_hz(sinr) / 1e6, sinr


def link_requirements(
    placement: solution, ue: UE, phys: config.PhysConstant, target_sinr_db: float,
    interference_w: float = 0.0, channel: AirToGroundParameters | None = None,
) -> tuple[int, float] | None:
    """Minimum integer PRBs and power for a covered UE under shared SINR."""
    if dis_2d(placement.x_u, placement.y_u, ue.x_m, ue.y_m) > placement.r_u + 1e-9:
        return None
    return resource_requirements(
        ue.require_mbps, target_sinr_db, gain(placement, ue, phys, channel),
        phys, interference_w,
    )


def resource_requirements(
    required_mbps: float, target_sinr_db: float, channel_gain: float,
    phys: config.PhysConstant, interference_w: float = 0.0,
) -> tuple[int, float] | None:
    """PRBs/power from already-computed channel metrics under shared SINR."""
    if required_mbps <= 0 or channel_gain <= 0 or interference_w < 0:
        return None
    target_sinr = 10 ** (target_sinr_db / 10.0)
    efficiency = spectral_efficiency_bps_hz(target_sinr)
    prbs = math.ceil(required_mbps * 1e6 / (phys.Bandwidth_BRP * efficiency))
    # Total link power = number of PRBs times required power on one PRB.
    power_w = prbs * target_sinr * (
        noise_per_prb_w(phys) + interference_w
    ) / channel_gain
    return prbs, power_w


def build_hotspot_ues(
    num_ues: int, length_m: float, width_m: float, demands_mbps: Iterable[float],
    num_hotspots: int = 5, hotspot_std_m: float = 100.0, seed: int = config.seed,
) -> tuple[UE, ...]:
    """Create reproducible clustered UE demand while keeping all UEs in bounds."""
    demands = np.asarray(tuple(demands_mbps), dtype=float)
    rng = np.random.default_rng(seed)
    hotspots = rng.uniform([0.0, 0.0], [length_m, width_m], size=(num_hotspots, 2))
    hotspot_ids = rng.integers(0, num_hotspots, size=num_ues)
    points = hotspots[hotspot_ids] + rng.normal(0.0, hotspot_std_m, size=(num_ues, 2))
    points = np.clip(points, [0.0, 0.0], [length_m, width_m])
    return tuple(
        UE(index, float(point[0]), float(point[1]), float(rng.choice(demands)))
        for index, point in enumerate(points)
    )


def build_random_scenario(
    phys: config.PhysConstant, num_uavs: int = config.num_uavs,
    num_ues: int = config.num_ues, length_m: float = config.length_m,
    width_m: float = config.width_m, grid_size: int = config.grid_size,
    seed: int = config.seed, ue_distribution: str = "uniform",
    num_hotspots: int = 5, hotspot_std_m: float = 100.0,
    prb_budgets: Iterable[int] | None = None,
    power_budgets_w: Iterable[float] | None = None,
) -> Scenario:
    """Create a reproducible default scenario ready for both MILP and GA."""
    rng = np.random.default_rng(seed)
    side = math.ceil(math.sqrt(num_uavs))
    start_x = np.linspace(length_m / (2 * side), length_m - length_m / (2 * side), side)
    start_y = np.linspace(width_m / (2 * side), width_m - width_m / (2 * side), side)
    start_positions = list(product(start_x, start_y))
    if prb_budgets is None:
        # Individual upper bounds share one system pool; no equal split is reserved.
        prb_limits = (config.uav_max_prb,) * num_uavs
    else:
        prb_limits = tuple(int(value) for value in prb_budgets)
    power_limits = (
        tuple(float(value) for value in power_budgets_w)
        if power_budgets_w is not None else (config.uav_max_power_w,) * num_uavs
    )
    uavs = tuple(
        UAV(
            i, float(start_positions[i][0]), float(start_positions[i][1]),
            max(phys.Theta_U), max(phys.H_U), p_tx_dBm=float(phys.P_max_dbm),
            max_power_w=power_limits[i], max_prbs=prb_limits[i],
        )
        for i in range(num_uavs)
    )
    demands = np.asarray(config.required_mbs, dtype=float)
    if ue_distribution == "uniform":
        ues = tuple(
            UE(i, float(rng.uniform(0, length_m)), float(rng.uniform(0, width_m)),
               float(rng.choice(demands)))
            for i in range(num_ues)
        )
    elif ue_distribution == "hotspot":
        ues = build_hotspot_ues(
            num_ues, length_m, width_m, demands, num_hotspots, hotspot_std_m, seed
        )
    placements = tuple(make_placements(
        np.linspace(0, length_m, grid_size + 1), np.linspace(0, width_m, grid_size + 1),
        phys.H_U, phys.Theta_U,
    ))
    return Scenario(uavs, ues, placements)


@dataclass(frozen=True)
class LinkMetrics:
    distance_2d_m: np.ndarray
    distance_3d_m: np.ndarray
    elevation_deg: np.ndarray
    los_probability: np.ndarray
    path_loss_db: np.ndarray
    antenna_gain_linear: np.ndarray
    channel_gain: np.ndarray
    rsrp_dbm: np.ndarray


class RadioEnvironment:
    """
    Ma trận sử dụng có dạng [uav_index, ue_index]
    """
    def __init__(
        self, num_uavs: int, num_ues: int, phys: config.PhysConstant | None = None,
        channel: AirToGroundParameters | None = None,
        orthogonal_prb_pool: bool = config.global_orthogonal_prb_pool,
    ) -> None:
        self.num_uavs, self.num_ues = num_uavs, num_ues
        self.phys = phys or config.PhysConstant()
        self.channel = channel or default_air_to_ground_parameters()
        self.orthogonal_prb_pool = bool(orthogonal_prb_pool)
        shape = (num_uavs, num_ues)
        self.r_uav_ue = np.zeros(shape)
        self.dis = np.zeros(shape)
        self.theta = np.zeros(shape)
        self.los_probability = np.zeros(shape)
        self.pl = np.zeros(shape)
        self.rsrp = np.full(shape, -np.inf)
        self.gain = np.zeros(shape)
        self.interference = np.zeros(shape)
        self.sinr = np.zeros(shape)
        self.sinr_db = np.full(shape, -np.inf)
        self.spectral_efficiency = np.zeros(shape)
        self.data_rate = np.zeros(shape)
        self.serving_matrix = np.zeros(shape, dtype=bool)
        self.association = np.full(num_ues, -1, dtype=int)


    def calc_basic_parameter(self, uavs: Sequence[UAV], ues: Sequence[UE]) -> LinkMetrics:
        """Calculate shared geometry, probabilistic LoS/NLoS loss, RSRP and gain."""
        for u, uav in enumerate(uavs):
            placement = solution(-1, uav.x, uav.y, uav.z, uav.theta_max)
            for k, ue in enumerate(ues):
                link = air_to_ground_link(placement, ue, self.phys, self.channel)
                self.r_uav_ue[u, k] = link.distance_2d_m
                self.dis[u, k] = link.distance_3d_m
                self.theta[u, k] = link.elevation_deg
                self.los_probability[u, k] = link.los_probability
                self.pl[u, k] = link.path_loss_db
                self.gain[u, k] = link.channel_gain
                self.rsrp[u, k] = rsrp_dbm(
                    uav.p_tx_dBm, uav.num_RB, placement, ue, self.phys, self.channel
                )
                self.serving_matrix[u, k] = (
                    link.distance_2d_m <= uav.radius
                    and self.rsrp[u, k] >= self.channel.rsrp_threshold_dbm
                )
        return LinkMetrics(
            self.r_uav_ue.copy(), self.dis.copy(), self.theta.copy(),
            self.los_probability.copy(), self.pl.copy(), self.antenna_gain_linear,
            self.gain.copy(), self.rsrp.copy(),
        )

    @property
    def antenna_gain_linear(self) -> np.ndarray:
        """Recover antenna factor from the diagnostic channel gain and path loss."""
        return self.gain / np.maximum(10 ** (-self.pl / 10), 1e-30)

    def associate_strongest_rsrp(self) -> np.ndarray:
        """Associate each UE with the strongest eligible UAV."""
        eligible = np.where(self.serving_matrix, self.rsrp, -np.inf)
        best = np.argmax(eligible, axis=0)
        valid = np.isfinite(eligible[best, np.arange(self.num_ues)])
        self.association = np.where(valid, best, -1)
        return self.association.copy()

    def calc_interference(
        self, power_matrix_w: np.ndarray, served_uav_idx: int, served_ue_idx: int,
    ) -> float:
        power = np.asarray(power_matrix_w, dtype=float)
        if self.orthogonal_prb_pool:
            self.interference[served_uav_idx, served_ue_idx] = 0.0
            return 0.0
        total = 0.0
        for u in range(self.num_uavs):
            if u != served_uav_idx:
                total += power[u].sum() * self.gain[u, served_ue_idx]
        self.interference[served_uav_idx, served_ue_idx] = total
        return total

    def calc_sinr(
        self, power_matrix_w: np.ndarray, association: np.ndarray | None = None,
        prb_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        association = self.association if association is None else np.asarray(association)
        power = np.asarray(power_matrix_w, dtype=float)
        prbs = np.asarray(prb_matrix, dtype=float)
        self.interference.fill(0.0)
        self.sinr.fill(0.0)
        self.sinr_db.fill(-np.inf)
        for k, u in enumerate(association):
            if u < 0:
                continue
            interference = self.calc_interference(power, int(u), k)
            if prbs[u, k] <= 0:
                continue
            value = sinr_linear(
                power[u, k] / prbs[u, k], self.gain[u, k], interference, self.phys
            )
            self.sinr[u, k] = value
            self.sinr_db[u, k] = 10 * math.log10(max(value, 1e-30))
        return self.sinr.copy()

    def calc_data_rate(
        self, prb_matrix: np.ndarray, association: np.ndarray | None = None,
    ) -> np.ndarray:
        association = self.association if association is None else np.asarray(association)
        prbs = np.asarray(prb_matrix, dtype=float)
        self.spectral_efficiency = np.log2(1 + self.sinr)
        self.data_rate = prbs * self.phys.Bandwidth_BRP * self.spectral_efficiency
        for k, u in enumerate(association):
            if u < 0:
                self.data_rate[:, k] = 0.0
        return self.data_rate.copy()

    def minimum_prbs(
        self, uav_index: int, ue_index: int, power_w: float, target_rate_mbps: float,
        interference_w: float = 0.0,
    ) -> int | None:
        if self.gain[uav_index, ue_index] <= 0:
            return None
        # ``power_w`` is the whole link budget, so increasing n also reduces
        # the signal power on each PRB.  A one-PRB rate cannot be multiplied
        # by n in this model.
        for count in range(1, config.system_max_prbs + 1):
            rate_mbps, _sinr = data_rate_mbps(
                power_w, count, self.gain[uav_index, ue_index],
                self.phys, interference_w,
            )
            if rate_mbps + 1e-9 >= target_rate_mbps:
                return count
        return None

    def summary(self) -> dict[str, float]:
        associated = int(np.count_nonzero(self.association >= 0))
        active = self.association >= 0
        rates = [self.data_rate[u, k] for k, u in enumerate(self.association) if u >= 0]
        return {
            "associated_ues": associated,
            "mean_los_probability": float(self.los_probability.mean()),
            "mean_path_loss_db": float(self.pl.mean()),
            "mean_sinr_db": float(np.mean([self.sinr_db[u, k] for k, u in enumerate(self.association) if u >= 0])) if active.any() else float("-inf"),
            "sum_rate_mbps": float(sum(rates) / 1e6),
        }


Env = RadioEnvironment
