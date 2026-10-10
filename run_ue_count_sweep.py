"""Run in VS Code: 20/30/40/50/60/70 UEs, 50 random maps each.

Uses the existing main.py algorithm defaults, config.py physics and current
GASettings. Does not change any of those files or perform work on import.
Each map gets a new optimizer, learner and MILP cache. Completed runs are saved
and resumed automatically. --plot-only redraws saved results without optimizing.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

import config
from evaluation import make_exact_evaluator
from ga import GASettings
from main import build_parser as main_parser
from optimizer import build_ga
from plots import draw_served_ue_bar_chart
from scenario import configured_scenario


UE_COUNTS = (20, 30, 40, 50, 60, 70)
MAPS_PER_CASE = 50
ROOT = Path(__file__).resolve().parent


def json_value(value):
    """Portable JSON, including the configured infinite solver time limit."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_json(path, payload):
    save_text(path, json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False))


def scenario_seed(base_seed, count, map_index):
    # Stable across restarts, unique stream per (UE count, map index).
    return int(np.random.SeedSequence([int(base_seed), int(count), int(map_index)])
               .generate_state(1, dtype=np.uint32)[0])


def experiment_manifest(defaults):
    # Hash the implementation too: resume must not mix old/new algorithms.
    sources = sorted(path for path in ROOT.glob("*.py") if not path.name.startswith("test_"))
    settings = {
        "format_version": 1,
        "ue_counts": UE_COUNTS,
        "maps_per_case": MAPS_PER_CASE,
        "base_map_seed": config.seed,
        "algorithm_seed": defaults.seed,
        "seed_policy": "different map seed; fixed algorithm seed; fresh learner/cache per run",
        "ue_distribution": defaults.ue_distribution,
        "demand_policy": "unchanged env.py random choice from config.required_mbs",
        "metric": "len(final MILP-verified incumbent coverage), not generation average",
        "num_uavs": config.num_uavs,
        "area_m": [config.length_m, config.width_m],
        "system_prbs": config.system_max_prbs,
        "per_uav_prb_bound": config.uav_max_prb,
        "per_uav_power_w": config.uav_max_power_w,
        "ue_demand_levels_mbps": config.required_mbs,
        "orthogonal_prb_pool": config.global_orthogonal_prb_pool,
        "physics": asdict(config.PhysConstant()),
        "solver": asdict(config.exact_solver_settings()),
        "ga_settings": asdict(GASettings()),
        "runner_settings": {name: getattr(defaults, name) for name in (
            "population", "population_capacity", "generations", "weight_method",
            "joint_milp_budget", "rank_samples", "cma_population", "interval_audit_period",
            "interval_min_samples", "interval_confidence")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    return json_value(settings)


def run_one(count, map_index, map_seed, defaults, log_path):
    phys, scenario = configured_scenario(
        seed=map_seed, ue_distribution=defaults.ue_distribution, num_ues=count,
    )
    evaluator, _ = make_exact_evaluator(scenario.uavs, scenario.ues, phys)
    ga = build_ga(
        uavs=scenario.uavs, ues=scenario.ues, phys=phys, evaluator=evaluator,
        population=defaults.population, population_capacity=defaults.population_capacity,
        generations=defaults.generations, seed=defaults.seed,
        weight_method=defaults.weight_method, joint_milp_budget=defaults.joint_milp_budget,
        rank_samples=defaults.rank_samples, cma_pre_samples=defaults.cma_population,
        audit_period=defaults.interval_audit_period,
        min_residual_samples=defaults.interval_min_samples,
        nominal_confidence=defaults.interval_confidence,
    )
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = ga.run()
    final = ga.joint_exact[tuple(np.round(result["best_genome"], 7).ravel())]
    payload = final.exact_result.get("milp", final.exact_result)
    if not payload.get("solver_optimal", False):
        raise RuntimeError("Final allocation was not MILP-verified; no result was counted")
    return {
        "status": "complete", "num_ues": count, "map_index": map_index,
        "map_seed": map_seed, "algorithm_seed": defaults.seed,
        "served_ues": len(final.coverage), "served_ue_ids": sorted(final.coverage),
        "solver_optimal": True,
        "final_violation": float(final.violation),
        "total_power_w": float(final.power_est_w), "total_prbs": float(final.prbs_est),
        "elapsed_seconds": time.perf_counter()-started,
        "unique_milp_solves": evaluator.unique_solves,
        "milp_solve_seconds": evaluator.solve_seconds,
        "final_genome": result["best_genome"].tolist(),
        "final_weights": ga._resource_weights.tolist(),
        "ues": [asdict(ue) for ue in scenario.ues],
    }


def read_result(path, count, map_index, expected_seed):
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("status") != "complete" or row.get("num_ues") != count
            or row.get("map_index") != map_index or row.get("map_seed") != expected_seed
            or row.get("solver_optimal") is not True
            or not isinstance(row.get("served_ues"), int)
            or not 0 <= row["served_ues"] <= count):
        raise RuntimeError(f"Invalid or mismatched saved result: {path}")
    return row


def summarize(records, counts, maps_per_case):
    rows = []
    for count in counts:
        case = [r for r in records if r["num_ues"] == count]
        if len({r["map_index"] for r in case}) != len(case):
            raise RuntimeError(f"Duplicate maps for {count} UEs")
        complete = len(case) == maps_per_case
        values = [r["served_ues"] for r in case]
        rows.append({
            "num_ues": count, "completed_maps": len(case), "required_maps": maps_per_case,
            "mean_served_ues": float(np.mean(values)) if complete else None,
            "std_served_ues": float(np.std(values, ddof=1)) if complete and len(values) > 1 else None,
            "min_served_ues": min(values) if complete else None,
            "max_served_ues": max(values) if complete else None,
            "feasible_maps": sum(r["final_violation"] <= 1e-12 for r in case),
        })
    return rows


def write_tables(output_dir, records, manifest):
    summary = summarize(records, manifest["ue_counts"], manifest["maps_per_case"])
    save_json(output_dir/"summary.json", summary)
    fields = list(summary[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)
    save_text(output_dir/"summary.csv", stream.getvalue())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results_ue_count_sweep")
    parser.add_argument("--plot-only", action="store_true", help="Read saved maps and redraw; no GA/MILP")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening a plot window")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir/"manifest.json"
    defaults = main_parser().parse_args([])
    if args.plot_only:
        if not manifest_path.exists():
            parser.error("No saved experiment in this output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = experiment_manifest(defaults)
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved != manifest:
                parser.error("Config/code changed since this experiment. Use a new --output-dir "
                             "to avoid mixing results, or --plot-only to view old results.")
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                parser.error("Output directory is nonempty without a manifest; choose another directory")
            save_json(manifest_path, manifest)

    records = []
    total = len(manifest["ue_counts"])*manifest["maps_per_case"]
    print(f"Experiment: {total} runs; {manifest['num_uavs']} UAV; "
          f"{manifest['runner_settings']['generations']} generations per run.", flush=True)
    print("Completed maps are resumed. Demand generation stays unchanged; GA seed is fixed.", flush=True)
    for count in manifest["ue_counts"]:
        for map_index in range(1, manifest["maps_per_case"]+1):
            seed = scenario_seed(manifest["base_map_seed"], count, map_index)
            stem = f"ue_{count:03d}_map_{map_index:03d}"
            path = output_dir/"runs"/(stem+".json")
            if path.exists():
                row = read_result(path, count, map_index, seed)
                print(f"[{len(records)+1}/{total}] Resume: {count} UEs, map {map_index}", flush=True)
            elif args.plot_only:
                continue
            else:
                log_path = output_dir/"logs"/(stem+".log")
                print(f"[{len(records)+1}/{total}] Running {count} UEs, map {map_index}; "
                      f"seed={seed}\n  Progress log: {log_path}", flush=True)
                row = run_one(count, map_index, seed, defaults, log_path)
                save_json(path, row)
                print(f"  Served {row['served_ues']}/{count}; "
                      f"V={row['final_violation']:.4g}; {row['elapsed_seconds']:.1f}s", flush=True)
                gc.collect()
            records.append(row)
            write_tables(output_dir, records, manifest)

    summary = write_tables(output_dir, records, manifest)
    print("\nUE count | Completed maps | Mean served UEs")
    for row in summary:
        mean = "pending" if row["mean_served_ues"] is None else f"{row['mean_served_ues']:.2f}"
        print(f"{row['num_ues']:8d} | {row['completed_maps']:3d}/{row['required_maps']:<9d} | {mean}")
    if any(row["mean_served_ues"] is None for row in summary):
        parser.error("Experiment incomplete: partial maps are not presented as a 50-map mean")
    if any(row["feasible_maps"] < row["required_maps"] for row in summary):
        print("WARNING: Some final placements have nonzero V. Counts reflect verified allocation, "
              "not a claim that every placement is feasible; see feasible_maps in summary.csv.")
    figure = output_dir/"mean_served_ues.png"
    draw_served_ue_bar_chart(summary, figure, maps_per_case=manifest["maps_per_case"],
                            show=not args.no_show)
    print(f"\nFigure: {figure}\nTable: {output_dir/'summary.csv'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
