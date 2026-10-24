"""Grouped means of saved results; no optimization and no Zamani data.

All completed, comparable-environment runs are used separately per method.
Reference fitness uses fixed equal weights. Missing WE peak-power loads yield
identification bounds, never an invented midpoint or an optimizer rerun.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from main import DEFAULT_WE, ROOT, fingerprint


METHODS = ("WE", "PSO-MILP", "Zhong GA-MILP", "Random Search-MILP")
LABELS = ("WE", "PSO–MILP", "Zhong GA–MILP", "Random Search–MILP")
COLORS = ("#168563", "#7651aa", "#287bb5", "#dc9728")
PLOT_METHODS = ("WE", "Zhong GA-MILP", "Random Search-MILP")
STYLES = {method: (label, color) for method, label, color in zip(METHODS, LABELS, COLORS)}
LOADS = (20, 30, 40, 50, 60, 70)
WEIGHTS = np.full(3, 1 / 3)
OUTPUT = Path(__file__).resolve().parent / "mean_bar_figures"


def collect():
    source_hashes, excluded, pending, rows = {}, [], [], []
    bank = {}

    def read(path):
        raw = path.read_bytes()
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    manifest = read(DEFAULT_WE / "manifest.json")
    limits = np.full(manifest["num_uavs"], manifest["per_uav_power_w"], dtype=float)
    pool = float(manifest["system_prbs"])

    def add(method, path, we, *, served, ids, feasible, power, prbs,
            payload=None, weights=None):
        if power is not None and not np.isfinite(power):
            raise RuntimeError(f"Non-finite power in {path}")
        powers = (payload or {}).get("power_by_uav_w")
        if powers is not None:
            powers = np.asarray(powers, dtype=float)
            if powers.shape != limits.shape or not np.isfinite(powers).all():
                raise RuntimeError(f"Invalid per-UAV loads in {path}")
            if abs(powers.sum() - power) > 1e-7:
                raise RuntimeError(f"Per-UAV loads disagree with total in {path}")
        f_lo = f_hi = None
        if feasible and power is not None and prbs is not None:
            pbar, nbar = power / limits.sum(), prbs / pool
            if powers is not None:
                z_lo = z_hi = max(float(np.max(powers / limits)), nbar)
            else:
                # max normalized power >= weighted average normalized power.
                # For a feasible allocation it is <= min(1, P_total/min Pmax).
                z_lo = max(pbar, nbar)
                z_hi = max(min(1., power / float(limits.min())), nbar)
            f_lo = -float(WEIGHTS @ [pbar, nbar, z_hi])
            f_hi = -float(WEIGHTS @ [pbar, nbar, z_lo])
            if f_lo > f_hi + 1e-10:
                raise RuntimeError(f"Reversed fitness bounds in {path}")
        rows.append(dict(
            method=method, num_ues=we["num_ues"], map_index=we["map_index"],
            map_seed=we["map_seed"], map_fingerprint=fingerprint(we["ues"]),
            feasible=bool(feasible), served_ues=int(served),
            coverage_percent=100 * served / we["num_ues"] if feasible else 0.,
            power_w=float(power) if feasible and power is not None else None,
            prbs=float(prbs) if feasible and prbs is not None else None,
            f_ref_lower=f_lo, f_ref_upper=f_hi,
            f_ref_exact=f_lo is not None and abs(f_lo - f_hi) <= 1e-10,
            peak_power_load=(float(np.max(powers / limits)) if powers is not None else None),
            original_weights=weights,
            served_ue_ids=sorted(map(int, ids)), source=str(path.relative_to(ROOT))))

    for path in sorted((DEFAULT_WE / "runs").glob("*.json")):
        we = read(path)
        if we.get("status") != "complete":
            pending.append(str(path.relative_to(ROOT)))
            continue
        if not we.get("solver_optimal"):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="WE label not optimal"))
            continue
        bank[(we["num_ues"], we["map_seed"])] = we
        add("WE", path, we, served=we["served_ues"], ids=we["served_ue_ids"],
            feasible=we["final_violation"] <= 1e-12,
            power=we["total_power_w"], prbs=we["total_prbs"],
            weights=we.get("final_weights"))
    print(f"WE: {len(rows)} completed records", flush=True)

    def resolve(path, source):
        we = bank.get((source["num_ues"], source["map_seed"]))
        if we is None or fingerprint(we["ues"]) != fingerprint(source["ues"]):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="No matching saved WE map"))
            return None
        fields = ("algorithm_seed", "served_ues", "total_power_w", "total_prbs")
        if (any(abs(float(we[k]) - float(source[k])) > 1e-8 for k in fields)
                or set(we["served_ue_ids"]) != set(source["served_ue_ids"])):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="Different historical WE reference"))
            return None
        return we

    def check_environment(path, environment):
        powers = environment.get("uav_power_budgets_w")
        if powers is None:
            powers = [environment["per_uav_power_w"]] * environment["num_uavs"]
        if (not np.allclose(powers, limits, atol=1e-12, rtol=0)
                or environment["system_prbs"] != pool
                or not environment.get("orthogonal_prb_pool", False)):
            raise RuntimeError(f"Resource environment differs in {path}")

    paths = set((ROOT / "baselines").glob("results*/**/pso_result.json"))
    for folder in ROOT.glob("compare_*"):
        paths.update((folder / "pso").glob("**/pso_result.json"))
    for path in sorted(paths):
        if "interrupted" in path.parts:
            continue
        data = read(path)
        if data.get("status") != "complete":
            pending.append(str(path.relative_to(ROOT)))
            continue
        settings, environment = data["settings"], data["environment"]
        we = bank.get((settings["num_ues"], settings["map_seed"]))
        if we is None or fingerprint(we["ues"]) != fingerprint(environment["ues"]):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="Standalone PSO map outside saved suite"))
            continue
        if (settings["seed"] != we["algorithm_seed"]
                or settings["milp_budget"] != we["unique_milp_solves"]):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="Different PSO seed/budget"))
            continue
        check_environment(path, environment)
        exact = data["final_milp"]
        payload = exact.get("milp", exact)
        if not payload.get("solver_optimal"):
            excluded.append(dict(source=str(path.relative_to(ROOT)), reason="PSO final allocation not optimal"))
            continue
        add("PSO-MILP", path, we, served=data["served_ues"], ids=exact["served_ue_ids"],
            feasible=data["final_violation"] <= 1e-12,
            power=exact["total_power_w"], prbs=exact["total_num_prbs"], payload=payload)
    print(f"PSO: {sum(r['method'] == 'PSO-MILP' for r in rows)} records before deduplication", flush=True)

    folders = [ROOT / "zhong_ga/results"]
    folders.extend(folder / "ga" for folder in ROOT.glob("compare_*"))
    sources = [("Zhong GA-MILP", folder) for folder in folders]
    sources.append(("Random Search-MILP", ROOT / "classic_milp_baselines/results/random"))
    for method, folder in sources:
        for path in sorted((folder / "runs").glob("*.json")):
            data = read(path)
            if data.get("status") != "complete":
                pending.append(str(path.relative_to(ROOT)))
                continue
            we = resolve(path, data["source_we"])
            if we is None:
                continue
            check_environment(path, data["environment"])
            if method == "Zhong GA-MILP":
                best = data.get("best")
                exact = best["exact"] if best else {}
                feasible = best is not None and best["violation"] <= 1e-12
                served = best["served_ues"] if best else 0
            else:
                exact, served = data["final_milp"], data["served_ues"]
                feasible = data["final_violation"] <= 1e-12
            payload = exact.get("milp", exact)
            if exact and not payload.get("solver_optimal"):
                excluded.append(dict(source=str(path.relative_to(ROOT)), reason="Final allocation not optimal"))
                continue
            add(method, path, we, served=served, ids=exact.get("served_ue_ids", []),
                feasible=feasible, power=exact.get("total_power_w"),
                prbs=exact.get("total_num_prbs"), payload=payload)
        print(f"Read {method}: {folder.relative_to(ROOT)}", flush=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["num_ues"], row["map_seed"], row["map_fingerprint"])].append(row)
    final = []
    fields = ("served_ues", "feasible", "power_w", "prbs", "served_ue_ids", "f_ref_lower", "f_ref_upper")
    for duplicates in grouped.values():
        first = duplicates[0]
        if any(any(row[k] != first[k] for k in fields) for row in duplicates[1:]):
            excluded.append(dict(source=[r["source"] for r in duplicates], reason="Conflicting duplicate runs"))
            continue
        first["duplicate_sources"] = [r["source"] for r in duplicates[1:]]
        final.append(first)
    return final, dict(source_hashes=source_hashes, excluded=excluded, pending=pending,
                       system_prbs=pool, power_limits_w=limits.tolist())


def aggregate(rows):
    result = []
    for load in LOADS:
        for method in METHODS:
            group = [r for r in rows if r["num_ues"] == load and r["method"] == method]
            if not group:
                continue
            mean = lambda key: float(np.mean([r[key] for r in group if r[key] is not None]))
            n_resource = sum(r["power_w"] is not None and r["prbs"] is not None for r in group)
            result.append(dict(method=method, num_ues=load, n_completed=len(group),
                n_feasible=sum(r["feasible"] for r in group), n_resource=n_resource,
                mean_served_ues=mean("served_ues"),
                mean_coverage_percent=mean("coverage_percent"),
                mean_power_w=mean("power_w") if n_resource else None,
                mean_prbs=mean("prbs") if n_resource else None,
                mean_f_ref_lower=mean("f_ref_lower") if n_resource else None,
                mean_f_ref_upper=mean("f_ref_upper") if n_resource else None,
                n_exact_f_ref=sum(r["f_ref_exact"] for r in group)))
    return result


def draw_panel(ax, summary, metric):
    lookup = {(r["num_ues"], r["method"]): r for r in summary}
    x, width = np.arange(len(LOADS)), .24
    fields = {"coverage": ("mean_coverage_percent", "Mean served UE coverage (%)", "Coverage", (0, 114)),
              "power": ("mean_power_w", "Mean total transmit power (W)", "Transmit power", (0, 11.5)),
              "prbs": ("mean_prbs", "Mean total PRBs", "PRB use", (0, 155)),
              "fitness": ("mean_f_ref_lower", "Reference resource fitness f (higher is better)",
                          "Common reference fitness · equal weights", (-1.22, .13))}
    field, ylabel, title, ylim = fields[metric]
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe5eb", linewidth=.7)
    for m, method in enumerate(PLOT_METHODS):
        _, color = STYLES[method]
        for i, load in enumerate(LOADS):
            row = lookup.get((load, method))
            if row is None or row[field] is None:
                continue
            xpos = x[i] + (m - (len(PLOT_METHODS)-1)/2) * width
            value = row[field]
            n = row["n_completed"] if metric == "coverage" else row["n_resource"]
            fmt = (f"{value:.1f}" if metric in ("coverage", "prbs") else
                   f"{value:.2f}" if metric == "power" else f"{value:.3f}")
            if metric == "fitness" and row["n_exact_f_ref"] < n:
                lo, hi = row["mean_f_ref_lower"], row["mean_f_ref_upper"]
                ax.bar(xpos, hi, width=width*.92, color=color, alpha=.22, edgecolor=color)
                ax.bar(xpos, hi-lo, bottom=lo, width=width*.92,
                       facecolor="none", edgecolor=color, hatch="////", linewidth=1.)
                ax.text(xpos, hi/2, f"{lo:.3f}–{hi:.3f}", ha="center", va="center",
                        fontsize=7.3, rotation=90, color="#125b45", weight="semibold")
            else:
                ax.bar(xpos, value, width=width*.92, color=color, edgecolor="white", linewidth=.6)
                ax.text(xpos, value/2, fmt, ha="center", va="center",
                        fontsize=7.3, rotation=90, weight="semibold",
                        color="#30220d" if method == "Random Search-MILP" else "white")
    ax.set_xticks(x, [str(k) for k in LOADS])
    ax.set(xlabel="Number of UEs", ylabel=ylabel, title=title, ylim=ylim,
           xlim=(-.6, len(LOADS)-.4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def draw(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "svg.fonttype": "none"})
    legend = [Patch(facecolor=STYLES[m][1], label=STYLES[m][0]) for m in PLOT_METHODS]
    note = ("Available completed maps averaged separately per method (unpaired means).\n"
            "Power/PRB comparisons must be read with coverage; lower resource use may serve fewer UEs.")
    fitness_note = ("Reference f = -(P/10 + PRBs/135 + max(peak UAV power/2, PRBs/135))/3.\n"
                    "WE hatching: valid range of the mean from missing per-UAV power, not a confidence interval or a point estimate.")
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.8))
    for ax, metric in zip(axes.flat, ("coverage", "power", "prbs", "fitness")):
        draw_panel(ax, summary, metric)
    fig.suptitle("Mean final results by UE load", fontsize=18, weight="bold", y=.985)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .956),
               ncol=len(PLOT_METHODS), frameon=False, fontsize=11)
    fig.text(.06, .027, note + "\n" + fitness_note, fontsize=9, va="bottom", color="#435160")
    fig.subplots_adjust(top=.88, bottom=.19, hspace=.40, wspace=.17)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"mean_metrics_overview.{suffix}", dpi=190, facecolor="white")
    plt.close(fig)
    for metric in ("coverage", "power", "prbs", "fitness"):
        fig, ax = plt.subplots(figsize=(14, 6.6))
        draw_panel(ax, summary, metric)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .94),
                   ncol=len(PLOT_METHODS), frameon=False, fontsize=10)
        fig.text(.08, .025, note + ("\n" + fitness_note if metric == "fitness" else ""),
                 fontsize=8.5, va="bottom", color="#435160")
        fig.subplots_adjust(top=.82, bottom=.31 if metric == "fitness" else .20, left=.08, right=.98)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"mean_{metric}_bars.{suffix}", dpi=200, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--from-cache", action="store_true", help="Redraw the exported snapshot without rereading running experiments")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        snapshot = json.loads((output / "source_snapshot.json").read_text(encoding="utf-8"))
        rows, metadata = snapshot["rows"], snapshot["metadata"]
    else:
        rows, metadata = collect()
        (output / "source_snapshot.json").write_text(
            json.dumps(dict(rows=rows, metadata=metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = aggregate(rows)
    if any(r["method"] not in METHODS for r in rows):
        raise RuntimeError("Unexpected method in export")
    for row in summary:
        if (row["mean_f_ref_lower"] is not None
                and row["mean_f_ref_lower"] > row["mean_f_ref_upper"] + 1e-10):
            raise RuntimeError("Invalid aggregate fitness bounds")
    (output / "summary.json").write_text(json.dumps(dict(summary=summary,
        aggregation="Unpaired arithmetic means of available completed saved runs; no optimization rerun",
        fitness="Fixed equal-weight reference, not native learned-weight fitness",
        weights=WEIGHTS.tolist(), metadata=metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    draw(summary, output)
    for load in LOADS:
        print(load, {r["method"]: r["n_completed"] for r in summary if r["num_ues"] == load}, flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
