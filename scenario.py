"""Construct the configured environment; physical formulas remain in env.py."""
import config
import env

def configured_scenario(*, seed=config.seed, ue_distribution="uniform", quick=False,
                        num_ues=None):
    """Build physics and environment from config without starting GA or MILP."""
    phys = config.PhysConstant()
    count, users = (3, 15) if quick else (config.num_uavs, config.num_ues)
    if num_ues is not None:
        users = int(num_ues)
    scenario = env.build_random_scenario(
        phys, num_uavs=count, num_ues=users,
        length_m=config.length_m, width_m=config.width_m,
        grid_size=config.grid_size, seed=seed, ue_distribution=ue_distribution,
        prb_budgets=(config.uav_max_prb,) * count,
        power_budgets_w=(config.uav_max_power_w,) * count,
    )
    return phys, scenario
