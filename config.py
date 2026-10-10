import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Cấu hình tần số Sub-6 GHz (FR1)
N0_dbm_hz = -174       # Mật độ nhiễu nhiệt (dBm/Hz)
NF_db = 9.0            # Hệ số nhiễu của điện thoại (Noise Figure)

# Băng thông hữu ích của một PRB trong thí nghiệm UAV này.  Giá trị 360 kHz
# được giữ ở một nơi duy nhất để cả env.py, MILP và các biểu đồ dùng cùng
# giả thiết vô tuyến.
W_r = 360000

bandwidth_per_RB = W_r

noise_power_dBm = N0_dbm_hz + 10 * np.log10(W_r) + NF_db
noise_power_W = 10 ** ((noise_power_dBm - 30) / 10)

# Chuyển -174 dBm/Hz sang Watt/Hz (1 mW = 10^-3 W, nên -174 dBm = -204 dBW)
N0_W_hz = 10 ** (-204 / 10) 
noise_figure_linear = 10 ** (NF_db / 10)

# Công suất nhiễu (Watt) = N0 * B * NF
noise_power_RB = N0_W_hz * bandwidth_per_RB * noise_figure_linear

num_antennaA_UAV = 5

length_m = 1000
width_m = 1000
num_uavs = 5
num_ues = 50
eta_ant = 1.0
d_0 = 1.0
f_c = 3.5e9
C = 3.0e8
k_B = 1.380649e-23
T = 290.0
Bandwidth_BRP = W_r
a = 2.0
F = NF_db
# Al-Hourani air-to-ground propagation preset used everywhere through env.py.
# Valid values: Suburban, Urban, Dense Urban, High-rise Urban.
a2g_environment = "Urban"
# Shared orthogonal pool: UAVs borrow freely; their combined allocation <= 135.
global_orthogonal_prb_pool = True
total_system_prbs = 135
system_max_prbs = total_system_prbs  # compatibility name; do not set separately
# Redundant individual upper bound, NOT an independently reserved quota.
uav_max_prb = total_system_prbs
uav_max_power_w = 2.0
power_step_w = 0.1
allocation_max_prbs = 10       # per UAV-UE link, not per UAV
allocation_max_power_w = 1.0   # total power of one link across all its PRBs
uav_min_separation_m = 250.0
required_mbs = [5, 10, 15]
grid_size = 10  # number of intervals per axis (11 grid points including ends)
seed = 2026
precompute_only = False
write_lp = None
output_json = Path("ga_mil_sol_1.json")

time_limit_s = math.inf
mip_gap = 0.02
log_to_console = True
# Exact lexicographic tie-break: maximise served UEs, then minimise PRBs,
# then minimise power. This is shared by the HiGHS MILP and CVXPY MILP.
minimize_power_tiebreak = True
symmetry_breaking = True
num_threads = 8
parrallel_search = True
# Exact WE labels need zero requested MIP gap. The override is explicit here,
# not hidden in a runner; threads/time limit still follow SolverSettings.
exact_mip_gap = 0.0
exact_log_to_console = False

@dataclass(frozen=True)
class PhysConstant:
    eta_ant: float = eta_ant
    d_0: float = d_0
    f_c: float = f_c
    C: float = C
    k_B: float = k_B
    T: float = T
    Bandwidth_BRP: float = Bandwidth_BRP
    a: float = a
    F: float = F

    Theta_U: Tuple[float, ...] = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
    H_U: Tuple[float, ...] = (250.0, 260.0, 270.0, 280.0, 290.0, 300.0)
    P_max_w: float = uav_max_power_w
    power_step_w: float = power_step_w
    allocation_max_prbs: int = allocation_max_prbs
    allocation_max_power_w: float = allocation_max_power_w
    # Optional compatibility for explicit historical dBm experiments only.
    # With PhysConstant() the grid is 0.1, 0.2, ... 2.0 W, NOT 3 dB steps.
    P_min_dbm: float | None = None
    P_max_dbm: float | None = None
    power_level_step_db: float | None = None

    def __post_init__(self):
        legacy = any(v is not None for v in
                     (self.P_min_dbm, self.P_max_dbm, self.power_level_step_db))
        maximum = (self.P_max_w if self.P_max_dbm is None
                   else 10 ** ((self.P_max_dbm - 30) / 10))
        object.__setattr__(self, "P_max_w", maximum)
        object.__setattr__(self, "P_max_dbm", 10 * math.log10(maximum) + 30)
        if legacy:
            object.__setattr__(self, "P_min_dbm", 12.0 if self.P_min_dbm is None else self.P_min_dbm)
            object.__setattr__(self, "power_level_step_db", 3.0 if self.power_level_step_db is None else self.power_level_step_db)

    @property
    def power_levels_w(self) -> Tuple[float, ...]:
        if self.power_level_step_db is not None:
            return tuple(10 ** ((p - 30) / 10) for p in self.power_levels_dbm)
        count = round(self.P_max_w / self.power_step_w)
        return tuple(round(i * self.power_step_w, 12) for i in range(1, count + 1))

    @property
    def noise_figure_linear(self) -> float:
        return 10 ** (self.F / 10.0)

    @property
    def power_levels_dbm(self) -> Tuple[float, ...]:
        if self.power_level_step_db is None:
            return tuple(10 * math.log10(p) + 30 for p in self.power_levels_w)
        levels: list[float] = []
        value = float(self.P_min_dbm)
        while value < self.P_max_dbm - 1e-12:
            levels.append(value)
            value += self.power_level_step_db
        if not levels or not math.isclose(levels[-1], self.P_max_dbm):
            levels.append(float(self.P_max_dbm))
        return tuple(levels)


@dataclass(frozen=True)
class SolverSettings:
    time_limit_s: float = time_limit_s
    mip_gap: float = mip_gap
    log_to_console: bool = log_to_console
    minimize_power_tiebreak: bool = minimize_power_tiebreak
    symmetry_breaking: bool = symmetry_breaking
    num_threads: int = num_threads
    parallel_search: bool = parrallel_search


def exact_solver_settings() -> SolverSettings:
    """Central, explicit solver policy for MILP-verified WE/CMA labels."""
    return SolverSettings(mip_gap=exact_mip_gap, log_to_console=exact_log_to_console)
