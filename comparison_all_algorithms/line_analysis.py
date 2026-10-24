import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from main import ROOT, METHODS, COLORS, identity, matched_service


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'figures'
WE_SOURCE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'
WE_PATTERN = re.compile(r'Joint WE \d+/\d+: served (\d+)/(\d+), V=([\deE.+-]+),.*?MILP (\d+)')
OTHER_PATTERN = re.compile(r'MILP (\d+)/(\d+): served=(\d+)/(\d+), V=([\deE.+-]+)')


def load_rows():
    with (OUTPUT / 'results_long.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ('num_ues', 'map_index', 'map_seed', 'served', 'plotted_served'):
            row[field] = int(row[field])
        row['feasible'] = row['feasible'] == 'True'
        for field in ('demand_mbps', 'power_w', 'prbs'):
            row[field] = float(row[field]) if row[field] else None
    return rows


def parse_progress(text, method):
    events = []
    pattern = WE_PATTERN if method == 'WE' else OTHER_PATTERN
    for match in pattern.finditer(text):
        if method == 'WE':
            served, total, violation, calls = match.groups()
        else:
            calls, budget, served, total, violation = match.groups()
        event = int(calls), int(served), int(total), float(violation)
        if events and event[0] < events[-1][0]:
            events = []
        events.append(event)
    return events


def observed_curve(events, budget):
    x, y, best = [0], [0.], 0.
    for calls, served, total, violation in events:
        if calls > budget:
            continue
        if violation <= 1e-12:
            best = max(best, served/total)
        x.append(calls)
        y.append(best)
    return np.asarray(x), np.asarray(y)


def carry_forward(x, y, grid):
    return y[np.searchsorted(x, grid, side='right')-1]


def log_path(row):
    name = f"ue_{row['num_ues']:03d}_map_{row['map_index']:03d}.log"
    if row['method'] == 'WE':
        return WE_SOURCE / 'logs' / name
    source = ROOT / row['source']
    base = source.parent.parent.parent if row['method'] == 'PSO-MILP' else source.parent.parent
    return base / 'logs' / name


def draw_progress(rows):
    methods = METHODS[:3]
    curves, exported = {}, []
    for count in sorted({r['num_ues'] for r in rows}):
        by_method = {m: {identity(r): r for r in rows
                         if r['method'] == m and r['num_ues'] == count} for m in methods}
        common = set(by_method['WE'])
        for method in methods[1:]:
            common &= set(by_method[method])
        accepted = []
        for key in sorted(common):
            entries = {}
            for method in methods:
                row = by_method[method][key]
                path = log_path(row)
                if not path.exists():
                    break
                events = parse_progress(path.read_text(encoding='utf-8-sig'), method)
                if not events:
                    break
                entries[method] = (row, events)
            if len(entries) == len(methods):
                budgets = [int(entries[m][0]['we_milp_calls']) for m in methods]
                if len(set(budgets)) == 1 and all(entries[m][1][-1][0] >= budgets[0] for m in methods):
                    accepted.append((key, budgets[0], entries))
        if not accepted:
            continue
        maximum = min(budget for key, budget, entries in accepted)
        grid = np.unique(np.r_[np.arange(0, maximum+1, 40), maximum])
        means = {}
        for method in methods:
            series = []
            for key, budget, entries in accepted:
                x, y = observed_curve(entries[method][1], maximum)
                series.append(carry_forward(x, y, grid))
            means[method] = np.mean(series, axis=0)
            for calls, coverage in zip(grid, means[method]):
                exported.append(dict(num_ues=count, method=method, paired_maps=len(accepted),
                                     logged_evaluations=int(calls), mean_recorded_feasible_coverage=float(coverage)))
        curves[count] = grid, means, len(accepted)
    if not curves:
        return []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, (count, (grid, means, size)) in zip(axes.flat, curves.items()):
        for method, color in zip(methods, COLORS):
            ax.step(grid, means[method], where='post', label=method, color=color, linewidth=1.8)
        ax.set(title=f'{count} UE | same {size} maps for all three methods',
               xlabel='Logged MILP evaluation counter', ylabel='Best recorded feasible served fraction', ylim=(-.02, 1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower right')
    for ax in list(axes.flat)[len(curves):]:
        ax.axis('off')
    fig.suptitle('Search progress on paired maps: recorded feasible service', fontsize=15, weight='bold')
    fig.text(.025, .02,
             'WE: periodic incumbent logs (every 10 generations, plus generation 1). PSO/GA: per-evaluation logs.\n'
             'Step-hold only; WE can understate progress between checkpoints. Zero means no feasible result observed yet, not proven infeasibility.\n'
             'Counter is taken from logs, not asserted to be unique cache misses. Grid MILP has a different budget/progress format and is not mixed into this axis.',
             fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(OUTPUT / 'line_service_progress.png', dpi=170)
    plt.close(fig)
    with (OUTPUT / 'line_progress.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    return exported


def paired_metrics(rows):
    results = []
    for count in sorted({r['num_ues'] for r in rows}):
        we = {identity(r): r for r in rows if r['method'] == 'WE' and r['num_ues'] == count}
        for method in METHODS[1:]:
            pairs = [(we[identity(r)], r) for r in rows
                     if r['method'] == method and r['num_ues'] == count and identity(r) in we]
            matched = [(a, b) for a, b in pairs if matched_service(a, b)]
            if not pairs:
                continue
            row = dict(num_ues=count, baseline=method, paired_maps=len(pairs),
                       matched_resource_maps=len(matched),
                       service_gain_pp=float(100*np.mean([(a['plotted_served']-b['plotted_served'])/count for a,b in pairs])))
            for metric in ('power_w', 'prbs'):
                a = float(np.mean([a[metric] for a,b in matched])) if matched else None
                b = float(np.mean([b[metric] for a,b in matched])) if matched else None
                row[f'we_{metric}'] = a
                row[f'baseline_{metric}'] = b
                row[f'{metric}_saving_percent'] = 100*(b-a)/b if b else None
            results.append(row)
    return results


def draw_advantages(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fields = ('service_gain_pp', 'power_w_saving_percent', 'prbs_saving_percent')
    titles = ('Feasible service difference', 'Power saving at matched service', 'PRB saving at matched service')
    labels = ('WE minus baseline (percentage points)', '100 x (baseline - WE) / baseline (%)',
              '100 x (baseline - WE) / baseline (%)')
    for ax, field, title, label in zip(axes, fields, titles, labels):
        for j, (method, color, marker) in enumerate(zip(METHODS[1:], COLORS[1:], ('^','D','s'))):
            group = [r for r in results if r['baseline'] == method]
            valid = [r for r in group if r[field] is not None]
            if not valid:
                continue
            x = [r['num_ues'] for r in group]
            y = [r[field] if r[field] is not None else np.nan for r in group]
            ax.plot(x, y, color=color, marker=marker, linewidth=1.7, label=method)
            for row in valid:
                n = row['paired_maps'] if field == 'service_gain_pp' else row['matched_resource_maps']
                ax.annotate(f'n={n}', (row['num_ues'],row[field]),
                            xytext=(0, 9 if j%2 else -13), textcoords='offset points',
                            ha='center', fontsize=7, color=color)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set(title=title, xlabel='UE count', ylabel=label, xticks=[20,30,40,50,60,70])
        ax.margins(y=.18)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Where does WE have an observed advantage? Positive values favor WE', fontsize=14, weight='bold')
    fig.text(.02,.025,
             'Service: all completed maps shared by WE and that baseline. Resources: feasible, equal served count AND admitted demand only.\n'
             'n labels give map counts; each baseline/load can use a different cohort. Lines connect load-level summaries, not evolution generations.\n'
             'Grid MILP results are time-limited incumbents, not certified optima. These are descriptive results, not significance/global superiority claims.',
             fontsize=8.5)
    fig.tight_layout(rect=(0,.13,1,.93))
    fig.savefig(OUTPUT / 'line_advantages_by_load.png', dpi=180)
    plt.close(fig)


def draw_resource_levels(results):
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for row_index, method in enumerate(METHODS[1:3]):
        group = [r for r in results if r['baseline'] == method]
        for ax, metric, label in zip(axes[row_index], ('power_w','prbs'), ('Mean transmit power (W)','Mean system PRBs')):
            for prefix, color, marker, name in [('we',COLORS[0],'o','WE'),
                                               ('baseline',COLORS[row_index+1],'^',method)]:
                ax.plot([r['num_ues'] for r in group],
                        [r[f'{prefix}_{metric}'] if r[f'{prefix}_{metric}'] is not None else np.nan for r in group],
                        color=color, marker=marker, linewidth=1.8, label=name)
            for row in group:
                if row[f'we_{metric}'] is not None:
                    ax.annotate(f"n={row['matched_resource_maps']}",
                                (row['num_ues'],row[f'we_{metric}']), xytext=(0,8),
                                textcoords='offset points', ha='center', fontsize=8)
            ax.set(title=f'WE vs {method}: matched service only', xlabel='UE count', ylabel=label,
                   xticks=[20,30,40,50,60,70])
            ax.margins(y=.16)
            ax.legend()
            ax.grid(alpha=.2)
    fig.suptitle('Resource-load curves without sacrificing service in the compared pairs', fontsize=14, weight='bold')
    fig.text(.02,.015,'Same served count and admitted demand within each pair; cohorts may differ across methods/load. Missing points are not zero.\n'
             'n = matched map count. One algorithm seed per map; descriptive means, not confidence intervals.', fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.94))
    fig.savefig(OUTPUT / 'line_resource_levels.png', dpi=180)
    plt.close(fig)


def main():
    rows = load_rows()
    results = paired_metrics(rows)
    draw_advantages(results)
    draw_resource_levels(results)
    progress = draw_progress(rows)
    (OUTPUT / 'line_analysis.json').write_text(json.dumps(dict(load_metrics=results,
        progress_points=len(progress)),indent=2),encoding='utf-8')
    for row in results:
        print(f"{row['num_ues']} UE | {row['baseline']} | service gain={row['service_gain_pp']:.2f} pp | "
              f"power saving={row['power_w_saving_percent']} | PRB saving={row['prbs_saving_percent']} | "
              f"n={row['paired_maps']}/{row['matched_resource_maps']}")
    print('Saved three line-chart figures. No optimization was run.')


if __name__ == '__main__':
    main()
