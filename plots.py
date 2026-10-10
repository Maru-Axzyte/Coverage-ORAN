"""All placement, convergence, resource and surrogate-quality figures."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import config
import env

@dataclass(frozen=True)
class PlotSettings:
    length_m: float
    width_m: float
    num_uavs: int
    num_ues: int
    system_prbs: int
    prb_budgets: tuple
    power_budget_w: float
    ue_demands_mbps: tuple

    @classmethod
    def from_config(cls, *, num_uavs=None, num_ues=None):
        u = config.num_uavs if num_uavs is None else num_uavs
        k = config.num_ues if num_ues is None else num_ues
        return cls(float(config.length_m), float(config.width_m), u, k,
                   config.system_max_prbs, (config.uav_max_prb,)*u,
                   config.uav_max_power_w, tuple(config.required_mbs))

def snapshot_metrics(record: dict[str, Any], *, settings=None) -> str:
    settings = settings or PlotSettings.from_config()
    return f"served {record['served_ues']}/{settings.num_ues}\nPRB {record['prbs']:.0f}/{settings.system_prbs} | P {record['power_w']:.2f} W"

def draw_snapshot(ax: Any, *, title: str, genome: np.ndarray, exact: dict[str, Any], ues: Sequence[env.UE], phys: config.PhysConstant, settings=None) -> None:
    settings = settings or PlotSettings.from_config()
    milp = exact['milp']
    served = set(exact['served_ue_ids'])
    assignments = {int(choice['ue_id']): int(choice['uav_id']) for choice in milp['uav_mode_choices']}
    uav_prbs = np.zeros(len(genome), dtype=int)
    uav_power = np.zeros(len(genome), dtype=float)
    uav_ues = np.zeros(len(genome), dtype=int)
    for choice in milp['uav_mode_choices']:
        index = int(choice['u'])
        uav_prbs[index] += int(choice['num_prbs'])
        uav_power[index] += float(choice['power_w'])
        uav_ues[index] += 1
    palette = plt.cm.tab10(np.arange(len(genome)))
    demand_colors = {5.0: '#4cd137', 10.0: '#fbc531', 15.0: '#e84118'}
    for ue in ues:
        req = float(ue.require_mbps)
        ue_color = demand_colors.get(req, '#7f8fa6')
        if ue.id in served:
            uav_idx = assignments[ue.id]
            uav_color = palette[uav_idx % len(palette)]
            uav_x, uav_y = genome[uav_idx][:2]
            ax.plot([ue.x_m, uav_x], [ue.y_m, uav_y], color=uav_color, alpha=0.3, linewidth=1.2, zorder=2)
            ax.scatter(ue.x_m, ue.y_m, s=36 + 3 * req, facecolor=ue_color, edgecolor=uav_color, linewidth=1.5, zorder=5)
        else:
            ax.scatter(ue.x_m, ue.y_m, marker='x', s=50 + 3 * req, linewidth=2.0, color=ue_color, zorder=4)
    for index, (x, y, h, theta) in enumerate(genome):
        radius = env.coverage_radius(float(h), float(theta))
        ax.add_patch(Circle((x, y), radius, facecolor=palette[index], edgecolor=palette[index], alpha=0.1, linewidth=1.2, zorder=1))
        ax.scatter(x, y, marker='^', s=104, color=palette[index], edgecolor='black', linewidth=0.55, zorder=6)
        ax.annotate(f'U{index + 1}: {uav_ues[index]} UE\n{uav_prbs[index]} PRB | {uav_power[index]:.2f}/{settings.power_budget_w:.1f} W', (x, y), xytext=(-6 if x > 0.72 * settings.length_m else 6, -6 if y > 0.86 * settings.width_m else 6), textcoords='offset points', fontsize=6.6, weight='bold', zorder=7, ha='right' if x > 0.72 * settings.length_m else 'left', va='top' if y > 0.86 * settings.width_m else 'bottom', bbox={'boxstyle': 'round,pad=.14', 'facecolor': 'white', 'alpha': 0.72, 'edgecolor': 'none'})
    ax.set_title(title, fontsize=10, weight='bold')
    ax.text(0.02, 0.03, snapshot_metrics({'served_ues': len(served), 'prbs': exact['total_num_prbs'], 'power_w': exact['total_power_w']}, settings=settings), transform=ax.transAxes, fontsize=8, va='bottom', bbox={'boxstyle': 'round,pad=.28', 'facecolor': 'white', 'alpha': 0.86, 'edgecolor': '#b0b0b0'})
    ax.set_xlim(0, settings.length_m)
    ax.set_ylim(0, settings.width_m)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.17)

def draw_figure(*, ues: Sequence[env.UE], phys: config.PhysConstant, evaluator: Any, we_history: list[dict[str, Any]], output: Path, show: bool, settings=None) -> None:
    settings = settings or PlotSettings.from_config()
    ga_indices = [0, len(we_history) // 2, len(we_history) - 1]
    snapshots = [we_history[index] for index in ga_indices]
    titles = ['MILP seed (generation 0)', f"C-SAEA + WE (generation {int(we_history[ga_indices[1]]['generation'])})", f"C-SAEA + WE (generation {int(we_history[ga_indices[2]]['generation'])})"]
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.86), hspace=0.24, wspace=0.14)
    map_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    chart = figure.add_subplot(grid[1, :])
    for ax, title, snapshot in zip(map_axes, titles, snapshots):
        genome = np.asarray(snapshot['genome'], dtype=float)
        draw_snapshot(ax, title=title, genome=genome, exact=evaluator(genome), ues=ues, phys=phys, settings=settings)
    for ax in map_axes[len(snapshots):]:
        ax.axis('off')
    curves: list[tuple[str, list[dict[str, Any]], str, str]] = [('Joint WE + CMA', we_history, '-', '#187a48')]
    for name, history, style, color in curves:
        values = np.maximum.accumulate([row.get('served_fraction', 0.0) for row in history])
        chart.plot([row['exact_evaluations'] for row in history], values, linestyle=style, linewidth=2.1, marker='o', markersize=3.2, color=color, label=name)
    chart.set_title('MILP-verified service convergence', fontsize=10, weight='bold')
    chart.set_xlabel('Exact MILP validations')
    chart.set_ylabel('Best exact admitted-UE fraction')
    chart.grid(alpha=0.25)
    chart.legend(fontsize=7.2, loc='best')
    chart.text(0.03, -0.26, 'All maps and curve points use env.py LoS/NLoS + orthogonal-PRB SNR and the exact HiGHS MILP.\nSurrogate values never appear as final evidence.', transform=chart.transAxes, fontsize=7.6, va='top')
    figure.suptitle(
        f"{settings.num_uavs} UAV / {settings.num_ues} UE: C-SAEA + War Elimination + joint CMA\n"
        f"{settings.system_prbs} shared system PRBs, {phys.Bandwidth_BRP / 1000.0:g} kHz per PRB, "
        f"{settings.power_budget_w} W per UAV, UE demand {list(settings.ue_demands_mbps)} Mbps",
        weight='bold', y=0.985)
    figure.savefig(output, dpi=230, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(figure)

def draw_all_steps_figure(*, ues: Sequence[env.UE], phys: config.PhysConstant, evaluator: Any, we_history: list[dict[str, Any]], output: Path, snapshot_period: int, settings=None) -> None:
    """A 2-D panel for each requested joint WE/CMA generation milestone."""
    settings = settings or PlotSettings.from_config()
    all_steps = [(f"WE generation {int(item['generation'])}", item)
                 for item in we_history if int(item['generation']) % snapshot_period == 0]
    columns = 4
    rows = int(np.ceil(len(all_steps) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.0 * rows))
    for ax, (title, item) in zip(np.ravel(axes), all_steps):
        genome = np.asarray(item['genome'], dtype=float)
        draw_snapshot(ax, title=title, genome=genome, exact=evaluator(genome), ues=ues, phys=phys, settings=settings)
    for ax in np.ravel(axes)[len(all_steps):]:
        ax.axis('off')
    figure.suptitle(
        f"MILP-verified incumbent every {snapshot_period} WE generations, with joint CMA\n"
        f"{settings.num_uavs} UAV / {settings.num_ues} UE, {settings.system_prbs} PRBs, "
        f"{phys.Bandwidth_BRP / 1000.0:g} kHz/PRB, {settings.power_budget_w} W/UAV",
        weight='bold', y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=210, bbox_inches='tight')
    plt.close(figure)

def draw_we_service_resource_figure(we_history: Sequence[dict[str, Any]], output: Path, *, settings=None) -> None:
    """Plot exact service and normalized resource use at every WE generation.

    ``we_history`` contains only incumbent values confirmed by the fixed-
    placement MILP.  The two resource ratios therefore use the same budgets
    as the optimizer, rather than surrogate estimates.
    """
    settings = settings or PlotSettings.from_config()
    generations = np.asarray([row['generation'] for row in we_history], dtype=int)
    served = np.asarray([row['served_ues'] for row in we_history], dtype=float)
    mean_power_load = np.asarray([row['power_w'] / (settings.num_uavs * settings.power_budget_w) for row in we_history], dtype=float)
    mean_prb_load = np.asarray([row['prbs'] / settings.system_prbs for row in we_history], dtype=float)
    figure, served_axis = plt.subplots(figsize=(10.5, 5.8))
    resource_axis = served_axis.twinx()
    service_line, = served_axis.plot(generations, served, color='#187a48', linewidth=2.4, marker='o', markersize=3.2, label='Served UE (exact MILP)')
    power_line, = resource_axis.plot(generations, mean_power_load, color='#c44e52', linewidth=2.0, label='Mean normalized power $\\bar p$')
    prb_line, = resource_axis.plot(generations, mean_prb_load, color='#4c72b0', linewidth=2.0, label='Normalized system PRBs $\\bar n$')
    resource_axis.axhline(1.0, color='0.45', linestyle=':', linewidth=1.2, label='Resource limit')
    served_axis.set_xlabel('WE generation')
    served_axis.set_ylabel('Number of served UEs', color='#187a48')
    served_axis.set_ylim(0, max(float(settings.num_ues), float(np.max(served))) * 1.05)
    served_axis.tick_params(axis='y', labelcolor='#187a48')
    resource_axis.set_ylabel('Normalized resource load', color='#3b4a66')
    resource_top = max(1.05, 1.05 * float(np.max([mean_power_load, mean_prb_load])))
    resource_axis.set_ylim(0, resource_top)
    served_axis.grid(alpha=0.25)
    served_axis.set_title('MILP-verified service and resource use over WE generations', weight='bold')
    lines = [service_line, power_line, prb_line]
    served_axis.legend(lines, [line.get_label() for line in lines], loc='best')
    served_axis.text(0.01, -0.2, '$\\bar p_t=\\sum_u P_u(t)/(U P_u^{\\max})$,  $\\bar n_t=\\sum_u N_u(t)/N_{\\rm sys}^{\\max}$; lower resource load is better.', transform=served_axis.transAxes, fontsize=9, va='top')
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)

def show_saved_figures(paths: Sequence[Path]) -> None:
    """Open saved PNGs in Matplotlib windows after all computations finish."""
    for path in paths:
        image = plt.imread(path)
        figure, axis = plt.subplots(figsize=(15, 10))
        axis.imshow(image)
        axis.set_title(path.name, weight="bold")
        axis.axis("off")
        figure.tight_layout()
    plt.show()

def generation_quality(checks, generations):
    """Use every audited prediction in each generation, with no cross-generation window.

    Only predictions with a subsequent MILP check can have a measured error.
    Interval and outside-topK audit fractions have their own denominators.
    Missing evidence is NaN, never a zero error or a zero rescue rate.
    """
    by_generation = {}
    for check in checks:
        by_generation.setdefault(check["generation"], []).append(check)
    rows = []
    for generation in generations:
        current = by_generation.get(generation, [])
        covered = [r["interval_covered"] for r in current if r.get("interval_covered") is not None]
        audited = [r["outside_topk_rescue"] for r in current if r.get("outside_topk_rescue") is not None]
        rows.append({"generation": generation,
            "mae": float(np.mean([abs(r["error_j"]) for r in current])) if current else np.nan,
            "coverage": float(np.mean(covered)) if covered else np.nan,
            "rescue": float(np.mean(audited)) if audited else np.nan,
            "checks": len(current), "interval_checks": len(covered), "audit_checks": len(audited)})
    return rows

def draw_surrogate_screening_quality(*, checks, generations, output,
                                     nominal_confidence=.95,
                                     audit_label="Audited outside-topK rescue vs provisional cut"):
    rows = generation_quality(checks, generations)
    g = [r["generation"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(g, [r["mae"] for r in rows], color="#4c72b0", linewidth=1.5,
                 marker=".", markersize=3,
                 label=r"Per-generation MAE: mean $|J_{MILP}-\widehat J|$")
    axes[0].set_ylabel("Absolute resource-cost error")
    axes[0].set_title("Surrogate and screening quality (all MILP-checked predictions per generation)",
                      weight="bold")
    axes[1].plot(g, [r["coverage"] for r in rows], color="#187a48", linewidth=1.5,
                 marker=".", markersize=3,
                 label="Observed interval coverage (calibrated checks only)")
    # Audit generations are sparse. Mark isolated observations; do not connect
    # across generations without audit evidence or silently carry values forward.
    axes[1].plot(g, [r["rescue"] for r in rows], color="#c44e52", linestyle="none",
                 marker="x", markersize=5, label=audit_label)
    axes[1].axhline(nominal_confidence, color="#187a48", linestyle=":",
                    label=f"Nominal confidence ({nominal_confidence:.0%})")
    axes[1].set_ylim(-.04, 1.04)
    axes[1].set_ylabel("Observed fraction")
    axes[1].set_xlabel("WE generation")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="best", fontsize=9)
    count_ranges = "; ".join(
        f"{label}: {min((r[key] for r in rows), default=0)}-{max((r[key] for r in rows), default=0)}"
        for key, label in (("checks", "MILP checks"), ("interval_checks", "interval checks"),
                           ("audit_checks", "outside-topK audits")))
    fig.text(.02, .015, "Each point uses only its own generation, not the full unverified pool. "
             "Missing evidence is a gap.\n"
             f"Samples per generation ({count_ranges}). Intervals are nominal; audit cut is provisional.",
             fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_served_ue_bar_chart(summary, output, *, maps_per_case=50, show=True):
    """One bar per UE count; only complete, independently restarted map runs.

    This helper only renders supplied results. It never runs GA or MILP.
    """
    rows = sorted(summary, key=lambda row: row["num_ues"])
    if not rows or any(row["completed_maps"] != maps_per_case
                       or row["mean_served_ues"] is None for row in rows):
        raise RuntimeError("The bar chart requires all maps in every UE-count case")
    counts = [row["num_ues"] for row in rows]
    means = [row["mean_served_ues"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(rows))
    bars = ax.bar(positions, means, width=.65, color="#3979ad", edgecolor="white")
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=5, fontsize=11)
    ax.set_xticks(positions, [str(count) for count in counts])
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Mean number of served UEs")
    ax.set_ylim(0., max(counts)*1.10)
    ax.set_title(f"Final MILP-verified served UEs\nMean over {maps_per_case} random maps per case")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

