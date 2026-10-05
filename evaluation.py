"""Cached fixed-placement MILP oracle used by the optimizer and plots."""
from __future__ import annotations
import time
from typing import Any, Sequence
import numpy as np
import config
import env
from milp import UAVMILP

def make_exact_evaluator(
    uavs: Sequence[env.UAV], ues: Sequence[env.UE], phys: config.PhysConstant,
) -> tuple[Any, dict[tuple[float, ...], dict[str, Any]]]:
    """Create the real fixed-placement MILP oracle with a lossless cache."""
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(genome: np.ndarray) -> dict[str, Any]:
        genome = np.asarray(genome, dtype=float)
        key = tuple(np.round(genome, decimals=7).ravel())
        if key not in cache:
            started = time.perf_counter()
            placements = [env.solution(index, *map(float, row)) for index, row in enumerate(genome)]
            result = UAVMILP(
                uavs=uavs,
                ues=ues,
                placements=placements,
                phys=phys,
                settings=config.exact_solver_settings(),
                fixed_placement_indices=tuple(range(len(uavs))),
                system_prb_budget=config.system_max_prbs,
                orthogonal_prb_pool=config.global_orthogonal_prb_pool,
            ).solve()
            cache[key] = {
                "total_power_w": float(result["total_power_w"]),
                "total_num_prbs": int(result["total_num_prbs"]),
                "served_ue_ids": sorted({int(item["ue_id"]) for item in result["uav_mode_choices"]}),
                "milp": result,
                "evaluated_genome": genome.tolist(),
            }
            evaluate.solve_seconds += time.perf_counter() - started
            evaluate.unique_solves += 1
        return cache[key]

    evaluate.unique_solves = 0
    evaluate.solve_seconds = 0.0
    return evaluate, cache
