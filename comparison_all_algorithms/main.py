import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'
METHODS = ('WE', 'PSO-MILP', 'Zhong GA-MILP', 'Zamani Grid-MILP')
COLORS = ('#187a48', '#7250ae', '#2679bd', '#e78a1b')
MARKERS = ('o', '^', 'D', 's')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fingerprint(users):
    users = sorted(users, key=lambda u: u['id'])
    values = [(int(u['id']), float(u['x_m']), float(u['y_m']), float(u['require_mbps']))
              for u in users]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def demand(users, served):
    lookup = {int(u['id']): float(u['require_mbps']) for u in users}
    return sum(lookup[int(k)] for k in set(served))


def make_row(method, we, served, ids, feasible, power, prbs, path, seconds=None,
             optimal=None, calls=None):
    return dict(method=method, num_ues=int(we['num_ues']), map_index=int(we['map_index']),
                map_seed=int(we['map_seed']), algorithm_seed=int(we['algorithm_seed']),
                map_fingerprint=fingerprint(we['ues']), served=int(served), feasible=bool(feasible),
                plotted_served=int(served) if feasible else 0,
                demand_mbps=demand(we['ues'], ids), served_ue_ids=sorted(map(int, ids)),
                power_w=None if power is None else float(power),
                prbs=None if prbs is None else float(prbs), seconds=seconds,
                solver_optimal=optimal, milp_calls=calls,
                we_milp_calls=we.get('unique_milp_solves'), source=str(path.relative_to(ROOT)))


def identity(row):
    return row['num_ues'], row['map_index'], row['map_seed'], row['map_fingerprint']


def matched_service(a, b):
    return (identity(a) == identity(b) and a['feasible'] and b['feasible']
            and a['served'] == b['served']
            and abs(a['demand_mbps'] - b['demand_mbps']) <= 1e-9)


def collect(we_root):
    bank, rows, skipped, pending = {}, [], [], []
    for path in sorted((we_root / 'runs').glob('*.json')):
        we = read(path)
        if we.get('status') != 'complete':
            continue
        key = int(we['num_ues']), int(we['map_seed'])
        bank[key] = we
        rows.append(make_row('WE', we, we['served_ues'], we['served_ue_ids'],
                             we['final_violation'] <= 1e-12, we['total_power_w'],
                             we['total_prbs'], path, we.get('elapsed_seconds'),
                             we.get('solver_optimal'), we.get('unique_milp_solves')))

    def resolve(path, source_we):
        we = bank.get((int(source_we['num_ues']), int(source_we['map_seed'])))
        if we is None or fingerprint(we['ues']) != fingerprint(source_we['ues']):
            skipped.append(dict(source=str(path), reason='No identical saved WE map'))
            return None
        if (we['algorithm_seed'] != source_we['algorithm_seed']
                or we['served_ues'] != source_we['served_ues']
                or set(we['served_ue_ids']) != set(source_we['served_ue_ids'])
                or abs(we['total_power_w']-source_we['total_power_w']) > 1e-8
                or abs(we['total_prbs']-source_we['total_prbs']) > 1e-8):
            skipped.append(dict(source=str(path), reason='Different historical WE reference'))
            return None
        return we

    pso_paths = set((ROOT / 'baselines').glob('results*/**/pso_result.json'))
    for folder in ROOT.glob('compare_*'):
        pso_paths.update((folder / 'pso').glob('**/pso_result.json'))
    for path in sorted(pso_paths):
        if 'interrupted' in path.parts:
            continue
        result = read(path)
        if result.get('status') != 'complete':
            pending.append(str(path.relative_to(ROOT)))
            continue
        settings = result['settings']
        we = bank.get((int(settings['num_ues']), int(settings['map_seed'])))
        if we is None or fingerprint(we['ues']) != fingerprint(result['environment']['ues']):
            skipped.append(dict(source=str(path), reason='PSO run outside the paired saved-map suite'))
            continue
        if (settings['seed'] != we['algorithm_seed']
                or settings['milp_budget'] != we['unique_milp_solves']):
            skipped.append(dict(source=str(path), reason='Different PSO seed or requested MILP budget'))
            continue
        exact = result['final_milp']
        rows.append(make_row('PSO-MILP', we, result['served_ues'], exact['served_ue_ids'],
                             result['final_violation'] <= 1e-12, exact['total_power_w'],
                             exact['total_num_prbs'], path, result.get('elapsed_seconds'),
                             exact.get('milp', exact).get('solver_optimal'),
                             result.get('unique_milp_solves')))

    ga_folders = [ROOT / 'zhong_ga/results']
    ga_folders.extend(folder / 'ga' for folder in ROOT.glob('compare_*'))
    sources = [('Zhong GA-MILP', folder) for folder in ga_folders]
    sources.append(('Zamani Grid-MILP', ROOT / 'zamani_grid_milp/results'))
    for method, folder in sources:
        for path in sorted((folder / 'runs').glob('*.json')):
            result = read(path)
            if result.get('status') != 'complete':
                pending.append(str(path.relative_to(ROOT)))
                continue
            we = resolve(path, result['source_we'])
            if we is None:
                continue
            best = result.get('best')
            if method == 'Zhong GA-MILP':
                exact = best['exact'] if best else {}
                feasible = best is not None and best['violation'] <= 1e-12
                ids = exact.get('served_ue_ids', [])
                power, prbs = exact.get('total_power_w'), exact.get('total_num_prbs')
                optimal = exact.get('milp', exact).get('solver_optimal')
            else:
                feasible = best is not None and bool(best.get('feasible'))
                ids = best.get('served_ue_ids', []) if best else []
                power = best.get('total_power_w') if best else None
                prbs = best.get('total_prbs') if best else None
                optimal = result.get('solver_optimal')
            rows.append(make_row(method, we, best['served_ues'] if best else 0, ids,
                                 feasible, power, prbs, path, result.get('elapsed_seconds'),
                                 optimal, result.get('unique_milp_solves')))

    groups = {}
    for row in rows:
        groups.setdefault((row['method'], identity(row)), []).append(row)
    final = []
    for candidates in groups.values():
        first = candidates[0]
        metrics = ('served', 'feasible', 'demand_mbps', 'power_w', 'prbs', 'served_ue_ids')
        if any(any(row[k] != first[k] for k in metrics) for row in candidates[1:]):
            skipped.append(dict(source=[r['source'] for r in candidates],
                                reason='Conflicting repeated runs; excluded rather than selecting the best'))
            continue
        first['duplicate_sources'] = [r['source'] for r in candidates[1:]]
        final.append(first)
    return final, skipped, pending


def draw_count(rows, count, output, common=False):
    subset = [r for r in rows if r['num_ues'] == count]
    by_method = {m: {identity(r): r for r in subset if r['method'] == m} for m in METHODS}
    keys = set(by_method['WE'])
    if common:
        for method in METHODS[1:]:
            keys &= set(by_method[method])
    keys = sorted(keys, key=lambda k: k[1])
    if not keys:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    x = np.arange(len(keys))
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        series = [by_method[method].get(key) for key in keys]
        y = [r['plotted_served'] if r else np.nan for r in series]
        n = sum(r is not None for r in series)
        axes[0].plot(x, y, marker=marker, color=color, linewidth=1.4,
                     markersize=4, label=f'{method} (n={n})')
    axes[0].set_xticks(x, [str(key[1]) for key in keys], rotation=90, fontsize=7)
    axes[0].set(xlabel=f'{count}-UE map ID (missing results left blank)',
                ylabel='Feasible served UEs; completed failure = 0', ylim=(-1, count+2),
                title='Same frozen UE coordinates and demands')
    axes[0].legend(fontsize=8)
    for ax, metric, label in zip(axes[1:], ('power_w', 'prbs'), ('Power (W)', 'PRBs')):
        extent = []
        for method, color, marker in zip(METHODS[1:], COLORS[1:], MARKERS[1:]):
            pairs = [(by_method['WE'][key], by_method[method][key]) for key in keys
                     if key in by_method[method]
                     and matched_service(by_method['WE'][key], by_method[method][key])]
            a = [we[metric] for we, other in pairs]
            b = [other[metric] for we, other in pairs]
            ax.scatter(a, b, color=color, marker=marker, s=45,
                       alpha=.8, label=f'{method} (n={len(pairs)})')
            extent.extend(a+b)
        if extent:
            lo, hi = min(extent), max(extent)
            pad = max((hi-lo)*.08, .2 if metric == 'power_w' else 1.)
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', color='gray', linewidth=1)
            ax.set_xlim(lo-pad, hi+pad)
            ax.set_ylim(lo-pad, hi+pad)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(.5, .5, 'No matched-service pairs', ha='center', transform=ax.transAxes)
        ax.set(xlabel=f'WE {label}', ylabel=f'Baseline {label}',
               title='Equal served count + total admitted demand')
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    title = 'Common completed maps of all four methods' if common else 'All available completed saved-map results'
    fig.suptitle(f'{count} UE: {title}', fontsize=14, weight='bold')
    fig.text(.01, .015,
             'Missing/running maps are not zero. Resources: pairwise matched service, not necessarily identical served UE sets.\n'
             'Below resource diagonal: baseline uses less. Zamani/Zhong are adapted implementations. '
             'PSO/GA use MILP-call budgets; Grid uses historical WE time budgets; no controlled speed claim.',
             fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .93))
    suffix = '_common' if common else ''
    path = output / f'all_methods_{count:03d}ue{suffix}.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_reports(rows, skipped, pending, output):
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r['num_ues'], r['map_index'], METHODS.index(r['method'])))
    with (output / 'results_long.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]) if ordered else ['method'])
        writer.writeheader()
        writer.writerows(ordered)
    summary = []
    for count in sorted({r['num_ues'] for r in rows}):
        we = {identity(r): r for r in rows if r['method'] == 'WE' and r['num_ues'] == count}
        for method in METHODS[1:]:
            group = [r for r in rows if r['method'] == method and r['num_ues'] == count]
            pairs = [(we[identity(r)], r) for r in group if identity(r) in we]
            matched = [(a, b) for a, b in pairs if matched_service(a, b)]
            if not pairs:
                continue
            summary.append(dict(num_ues=count, baseline=method, paired_maps=len(pairs),
                we_mean_served=float(np.mean([a['plotted_served'] for a, b in pairs])),
                baseline_mean_served=float(np.mean([b['plotted_served'] for a, b in pairs])),
                matched_service_maps=len(matched),
                we_wins=sum(a['plotted_served'] > b['plotted_served'] for a, b in pairs),
                ties=sum(a['plotted_served'] == b['plotted_served'] for a, b in pairs),
                baseline_wins=sum(a['plotted_served'] < b['plotted_served'] for a, b in pairs)))
    (output / 'summary.json').write_text(json.dumps(dict(pairwise=summary, pending=pending,
        excluded=skipped, notes=['Means use the same maps within each method pair, not between different pairs.',
        'Only completed runs are plotted; incomplete maps are missing, never scored zero.']),
        ensure_ascii=False, indent=2), encoding='utf-8')
    for row in summary:
        print(f"{row['num_ues']} UE | {row['baseline']} | maps={row['paired_maps']} | "
              f"WE={row['we_mean_served']:.2f}, baseline={row['baseline_mean_served']:.2f} | "
              f"matched resources={row['matched_service_maps']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-we', type=Path, default=DEFAULT_WE)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'figures')
    args = parser.parse_args()
    rows, skipped, pending = collect(args.source_we)
    write_reports(rows, skipped, pending, args.output_dir)
    for count in sorted({r['num_ues'] for r in rows}):
        for common in (False, True):
            path = draw_count(rows, count, args.output_dir, common)
            if path:
                print(path, flush=True)
    print(f'Excluded {len(skipped)} records; pending {len(pending)}. No optimization was run.', flush=True)


if __name__ == '__main__':
    main()
