"""Assemble the active GA + contextual weights + joint WE/CMA optimizer."""
from dataclasses import replace
import config
from ga import Bounds, GASettings
from contextual_ga import ContextualCSAEA
from joint_topk_we import JointTopKMixin


class JointTopKCSAEA(JointTopKMixin, ContextualCSAEA):
    """The single supported optimization flow; component order is intentional."""


def build_ga(*, uavs, ues, phys, evaluator, population, population_capacity,
             generations, seed, weight_method="contextual", joint_milp_budget=4,
             rank_samples=64, cma_pre_samples=8, audit_period=5,
             min_residual_samples=20, nominal_confidence=.95):
    defaults = GASettings()
    settings = replace(
        defaults, population_size=population, population_capacity=population_capacity,
        generations=generations, initial_milp_seeds=min(defaults.initial_milp_seeds, population),
        clusters=min(defaults.clusters, population), use_war_elimination=True,
        include_separation_constraint=True, min_uav_separation_m=config.uav_min_separation_m,
        random_seed=seed, sbx_uav_blocks=True, children_per_pair=10,
    )
    return JointTopKCSAEA(
        milp_budget=joint_milp_budget, rank_samples=rank_samples,
        cma_pre_samples=cma_pre_samples, audit_period=audit_period,
        min_residual_samples=min_residual_samples, nominal_confidence=nominal_confidence,
        weight_method=weight_method, num_uavs=len(uavs), ues=ues,
        bounds=Bounds((0., 0., min(phys.H_U), min(phys.Theta_U)),
                      (float(config.length_m), float(config.width_m), max(phys.H_U), max(phys.Theta_U))),
        power_budget_w=sum(uav.max_power_w for uav in uavs), prb_budget=config.system_max_prbs,
        milp_evaluator=evaluator, initial_genomes=None, phys=phys,
        uav_power_budgets_w=[uav.max_power_w for uav in uavs],
        uav_prb_budgets=[uav.max_prbs for uav in uavs], settings=settings,
    )
