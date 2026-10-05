"""Run this file in VS Code: configured joint WE/CMA + MILP and all figures.

Importing this module does not start a simulation or open a window.
Physics/scenario values come from config.py; GA defaults come from GASettings.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np

import config
from ga import GASettings
from scenario import configured_scenario
from evaluation import make_exact_evaluator
from optimizer import build_ga
from plots import (PlotSettings, draw_figure, draw_all_steps_figure,
                   draw_we_service_resource_figure, draw_surrogate_screening_quality,
                   show_saved_figures)


def build_parser():
    defaults = GASettings()
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Joint WE + CMA, MILP validation and 2-D figures")
    parser.add_argument("--population", type=int, default=defaults.population_size)
    parser.add_argument("--population-capacity", type=int, default=defaults.population_capacity)
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument("--weight-method", choices=("contextual", "fixed"), default="contextual")
    parser.add_argument("--joint-milp-budget", type=int, default=4)
    parser.add_argument("--rank-samples", type=int, default=64)
    parser.add_argument("--cma-population", type=int, default=8)
    parser.add_argument("--interval-confidence", type=float, default=.95)
    parser.add_argument("--interval-min-samples", type=int, default=20)
    parser.add_argument("--interval-audit-period", type=int, default=5)
    parser.add_argument("--snapshot-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--ue-distribution", choices=("uniform", "hotspot"), default="uniform")
    parser.add_argument("--quick", action="store_true",
                        help="Only reduce the scenario to 3 UAV / 15 UE; keep algorithm settings.")
    parser.add_argument("--figure", type=Path, default=root/"note_we_2d_convergence.png")
    parser.add_argument("--output", type=Path, default=root/"note_we_2d_convergence.json")
    parser.set_defaults(show=True)
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--no-open", action="store_true", help="Do not open PNGs in the Windows viewer.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.population < 2 or (args.population_capacity is not None
                               and args.population_capacity < args.population+1):
        parser.error("Population must be >= 2; capacity must leave space for offspring")
    if args.generations < 0 or args.snapshot_period <= 0:
        parser.error("generations must be nonnegative; snapshot-period must be positive")
    if args.joint_milp_budget < 1 or args.rank_samples < 1 or args.cma_population < 0:
        parser.error("MILP/rank budgets must be positive and CMA sample count nonnegative")
    if not 0. < args.interval_confidence < 1. or args.interval_min_samples < 2 or args.interval_audit_period < 0:
        parser.error("Invalid interval calibration/audit settings")
    phys, scenario = configured_scenario(seed=args.seed, ue_distribution=args.ue_distribution, quick=args.quick)
    uavs, ues = scenario.uavs, scenario.ues
    plot_settings = PlotSettings.from_config(num_uavs=len(uavs), num_ues=len(ues))
    evaluator, _cache = make_exact_evaluator(uavs, ues, phys)
    ga = build_ga(
        uavs=uavs, ues=ues, phys=phys, evaluator=evaluator, population=args.population,
        population_capacity=args.population_capacity, generations=args.generations,
        seed=args.seed, weight_method=args.weight_method, joint_milp_budget=args.joint_milp_budget,
        rank_samples=args.rank_samples, cma_pre_samples=args.cma_population,
        audit_period=args.interval_audit_period, min_residual_samples=args.interval_min_samples,
        nominal_confidence=args.interval_confidence,
    )
    print(f"Config: {len(uavs)} UAV / {len(ues)} UE; {config.length_m}x{config.width_m} m; "
          f"PRB system={config.system_max_prbs}, per UAV={config.uav_max_prb}; "
          f"power UAV={config.uav_max_power_w:g} W; "
          f"h={min(phys.H_U):g}..{max(phys.H_U):g} m, theta={min(phys.Theta_U):g}..{max(phys.Theta_U):g} deg",
          flush=True)
    print(f"WE settings: initial population={ga.settings.population_size}, "
          f"capacity={ga.settings.population_capacity}, generations={ga.settings.generations}, "
          f"initial MILP seeds={ga.settings.initial_milp_seeds}, update period={ga.settings.update_period}, "
          f"clusters={ga.settings.clusters}, children/pair={ga.settings.children_per_pair}", flush=True)
    print("[1/2] Running joint WE + CMA with MILP validation...", flush=True)
    result = ga.run()
    final = ga.joint_exact[tuple(np.round(result["best_genome"], 7).ravel())]
    history = result["exact_incumbent_history"]
    print("[2/2] Saving 2-D convergence, resource and surrogate-quality figures...", flush=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeline = args.figure.with_name(f"{args.figure.stem}_all_steps.png")
    resources = args.figure.with_name(f"{args.figure.stem}_we_service_resources.png")
    quality = args.figure.with_name(f"{args.figure.stem}_surrogate_screening_quality.png")
    draw_figure(ues=ues, phys=phys, evaluator=evaluator, we_history=history,
                output=args.figure, show=False, settings=plot_settings)
    draw_all_steps_figure(ues=ues, phys=phys, evaluator=evaluator, we_history=history,
                          output=timeline, snapshot_period=args.snapshot_period, settings=plot_settings)
    draw_we_service_resource_figure(history, resources, settings=plot_settings)
    draw_surrogate_screening_quality(checks=result["joint_topk_checks"],
        generations=[r["generation"] for r in history], output=quality,
        rolling_checks=20, nominal_confidence=args.interval_confidence,
        audit_label="Audited outside-topK rescue vs provisional cut")
    summary = {
        "assumptions": {
            "uavs": len(uavs), "ues": len(ues), "system_max_prbs": config.system_max_prbs,
            "area_m": [config.length_m, config.width_m], "ue_distribution": args.ue_distribution,
            "continuous_height_bounds_m": [min(phys.H_U), max(phys.H_U)],
            "continuous_theta_bounds_deg": [min(phys.Theta_U), max(phys.Theta_U)],
            "generations": args.generations, "active_population": ga.settings.population_size,
            "population_capacity": ga.settings.population_capacity,
            "initial_milp_seeds": ga.settings.initial_milp_seeds,
            "update_period": ga.settings.update_period, "clusters": ga.settings.clusters,
            "per_uav_prb_budgets": [u.max_prbs for u in uavs], "prb_bandwidth_hz": phys.Bandwidth_BRP,
            "uav_max_power_w": config.uav_max_power_w, "ue_demands_mbps": list(config.required_mbs),
            "seed": args.seed,
            "initialization": "all UAVs and initial individuals share area-centre x/y; h and theta uniform within bounds",
            "parameter_source": "config.py", "carrier_hz": phys.f_c, "noise_figure_db": phys.F,
            "power_step_w": phys.power_step_w, "power_levels_w": list(phys.power_levels_w),
            "allocation_max_prbs": phys.allocation_max_prbs, "allocation_max_power_w": phys.allocation_max_power_w,
            "grid_intervals_per_axis": config.grid_size, "orthogonal_prb_pool": config.global_orthogonal_prb_pool,
            "solver_mip_gap": config.exact_mip_gap, "solver_num_threads": config.num_threads,
            "war_method": "joint-topk", "weight_method": args.weight_method,
            "note_we_enabled": True, "joint_topk_enabled": True,
            "sbx_uav_blocks": ga.settings.sbx_uav_blocks, "children_per_pair": ga.settings.children_per_pair,
        },
        # Keep result keys so existing analysis of saved joint runs still works.
        "csaea_we_before_cma": history[-1], "hybrid_we_exact_cma": history[-1],
        "we_incumbent_history": history, "we_population_history": result["history"],
        "war_history": [], "exact_trace": ga.exact_trace,
        "unique_milp_solves": evaluator.unique_solves, "milp_solve_seconds": evaluator.solve_seconds,
        "weight_learning_history": ga.weight_learning_history, "learned_vartheta": ga.vartheta.tolist(),
        "cma_history": [], "final_milp": final.exact_result,
        "all_steps_figure": str(timeline), "we_service_resource_figure": str(resources),
        "note": "Reported incumbents are evaluated by the fixed-placement MILP; CMA operates inside WE generations.",
        "joint_topk": {
            "milp_budget_per_generation": args.joint_milp_budget, "rank_samples": args.rank_samples,
            "cma_pre_samples": args.cma_population, "audit_period": args.interval_audit_period,
            "history": result["joint_topk_history"], "checks": result["joint_topk_checks"],
            "cma_updates": result["joint_cma_updates"], "quality_figure": str(quality),
            "cost": "J = beta*pbar + mu*nbar + eta*max_u(power_load_u, prb_load_u)",
            "audit_interpretation": "Exact audit quality is compared with the frozen provisional cut; not a population false-drop probability.",
        },
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Final: served {len(final.coverage)}/{len(ues)}, V={final.violation:.4g}, "
          f"PRB={final.prbs_est:g}/{config.system_max_prbs}, power={final.power_est_w:.3f} W")
    for label, path in [("Placement", args.figure), ("Timeline", timeline),
                        ("Service/resources", resources), ("Surrogate quality", quality), ("JSON", args.output)]:
        print(f"{label}: {path.resolve()}")
    figures = (args.figure, resources, quality)
    if args.show:
        show_saved_figures(figures)
    if not args.no_open and os.name == "nt":
        for path in figures:
            os.startfile(path.resolve())
    return summary


if __name__ == "__main__":
    main()
