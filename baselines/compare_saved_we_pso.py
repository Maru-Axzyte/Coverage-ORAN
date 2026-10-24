import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'
DEFAULT_OUTPUT = ROOT / 'baselines/results_we_pso_20'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def source_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def validate_source(manifest):
    mismatches = []
    for name in ('config.py', 'env.py', 'milp.py', 'evaluation.py'):
        raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        hashes = {hashlib.sha256(raw).hexdigest(),
                  hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}
        if manifest['source_sha256'].get(name) not in hashes:
            mismatches.append(name)
    return mismatches


def worker(args):
    import matplotlib
    matplotlib.use('Agg')
    import env
    import config
    import optimizer
    from ga import GASettings
    import run_pso as pso
    from scenario import configured_scenario

    manifest = read(args.source / 'manifest.json')
    row = read(args.worker)
    settings = GASettings(**manifest['ga_settings'])
    optimizer.GASettings = lambda: settings
    pso.GASettings = lambda: settings

    def frozen_scenario(**kwargs):
        phys, scene = configured_scenario(**kwargs)
        return phys, replace(scene, ues=tuple(env.UE(**ue) for ue in row['ues']))

    pso.configured_scenario = frozen_scenario
    destination = args.output / 'pso' / Path(args.worker).stem
    partial = destination / 'pso_result.json'
    if partial.exists():
        if read(partial).get('status') == 'complete':
            return 0
        backup = destination / 'interrupted'
        backup.mkdir(exist_ok=True)
        partial.replace(backup / f'pso_result_{__import__("time").time_ns()}.json')
    pso.main(['--swarm', str(args.swarm), '--milp-budget', str(row['unique_milp_solves']),
              '--seed', str(row['algorithm_seed']), '--map-seed', str(row['map_seed']),
              '--num-ues', str(row['num_ues']), '--ue-distribution', manifest['ue_distribution'],
              '--weight-method', manifest['runner_settings']['weight_method'],
              '--output-dir', str(destination), '--no-show'])
    return 0


def summarize(selection, output):
    rows = []
    for entry in selection['maps']:
        we = read(output / 'inputs' / entry['file'])
        result_path = output / 'pso' / Path(entry['file']).stem / 'pso_result.json'
        if not result_path.exists():
            continue
        pso = read(result_path)
        if pso.get('status') != 'complete':
            continue
        if pso['environment']['ues'] != we['ues']:
            print(f'UE data mismatch: {result_path}', flush=True)
            return None
        exact = pso['final_milp']
        ids = set(exact['served_ue_ids'])
        demand = {u['id']: u['require_mbps'] for u in we['ues']}
        we_demand = sum(demand[k] for k in we['served_ue_ids'])
        pso_demand = sum(demand[k] for k in ids)
        feasible_we = we['final_violation'] <= 1e-12
        feasible_pso = pso['final_violation'] <= 1e-12
        matched = (feasible_we and feasible_pso and we['served_ues'] == pso['served_ues']
                   and abs(we_demand-pso_demand) <= 1e-9)
        rows.append(dict(num_ues=we['num_ues'], map_index=we['map_index'],
            map_seed=we['map_seed'], we_served=we['served_ues'], pso_served=pso['served_ues'],
            we_feasible=feasible_we, pso_feasible=feasible_pso,
            we_demand_mbps=we_demand, pso_demand_mbps=pso_demand,
            we_prbs=we['total_prbs'], pso_prbs=exact['total_num_prbs'],
            we_power_w=we['total_power_w'], pso_power_w=exact['total_power_w'],
            we_milp_calls=we['unique_milp_solves'], pso_milp_calls=pso['unique_milp_solves'],
            we_seconds=we['elapsed_seconds'], pso_seconds=pso['elapsed_seconds'],
            resource_comparable=matched,
            pso_minus_we_served=pso['served_ues']-we['served_ues'],
            pso_minus_we_power_w=exact['total_power_w']-we['total_power_w'] if matched else None,
            pso_minus_we_prbs=exact['total_num_prbs']-we['total_prbs'] if matched else None))
    save(output / 'comparison.json', {'completed': len(rows), 'selected': len(selection['maps']),
        'timing_note': 'WE historical timing may be from another machine; not a controlled speed comparison.',
        'scope': 'Two complete systems, not an isolated WE vs PSO operator ablation.', 'rows': rows})
    if not rows:
        return rows
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    counts = sorted({r['num_ues'] for r in rows})
    x = np.arange(len(counts))
    for method, offset, color in [('we', -.18, '#187a48'), ('pso', .18, '#7250ae')]:
        groups = [[r[f'{method}_served'] if r[f'{method}_feasible'] else 0
                   for r in rows if r['num_ues'] == n] for n in counts]
        axes[0].bar(x+offset, [np.mean(g) for g in groups], .36, label=method.upper(), color=color)
    axes[0].set_xticks(x, [f'{n}\n(n={sum(r["num_ues"] == n for r in rows)})' for n in counts])
    axes[0].set(xlabel='UE count (completed paired maps)', ylabel='Mean feasible served UEs; failure = 0')
    axes[0].legend()
    matched = [r for r in rows if r['resource_comparable']]
    for ax, metric, unit in [(axes[1], 'power_w', 'Power (W)'), (axes[2], 'prbs', 'PRBs')]:
        if matched:
            a = [r[f'we_{metric}'] for r in matched]
            b = [r[f'pso_{metric}'] for r in matched]
            ax.scatter(a, b)
            lo, hi = min(a+b), max(a+b)
            ax.plot([lo-0.1, hi+0.1], [lo-0.1, hi+0.1], '--', color='gray')
        ax.set(xlabel=f'WE {unit}', ylabel=f'PSO {unit}',
               title=f'Equal count + demand, feasible: {len(matched)} maps\nBelow diagonal: PSO uses less')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'Paired WE / PSO: {len(rows)}/{len(selection["maps"])} selected maps completed')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--selection-seed', type=int, default=20261013)
    parser.add_argument('--swarm', type=int, default=40)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.plot_only:
        summarize(read(args.output / 'selection.json'), args.output)
        print(args.output / 'comparison.png')
        return 0
    manifest = read(args.source / 'manifest.json')
    mismatches = validate_source(manifest)
    if mismatches:
        print(f'Physical model or MILP source differs from saved WE: {mismatches}. No experiment started.')
        return 1
    files = sorted((args.source / 'runs').glob('*.json'))
    eligible = [(f, read(f)) for f in files]
    eligible = [(f, r) for f, r in eligible if r.get('status') == 'complete' and r.get('solver_optimal')]
    if not 1 <= args.samples <= len(eligible) or args.swarm < 2:
        print('Invalid sample count or swarm size.')
        return 1
    fingerprint = {str(p.relative_to(ROOT)): source_hash(p) for p in
                   [*sorted(ROOT.glob('*.py')), ROOT/'baselines/run_pso.py', Path(__file__).resolve()]
                   if not p.name.startswith('test_')}
    identity = dict(samples=args.samples, selection_seed=args.selection_seed, swarm=args.swarm,
                    source_manifest=manifest, code=fingerprint,
                    source_results={f.name: source_hash(f) for f, _ in eligible})
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / 'selection.json'
    if selection_path.exists():
        selection = read(selection_path)
        if selection['identity'] != identity:
            print('Comparison inputs/code changed. Use a new --output directory; existing results kept.')
            return 1
    else:
        rng = np.random.default_rng(args.selection_seed)
        counts = sorted({r['num_ues'] for _, r in eligible})
        groups = {n: [pair for pair in eligible if pair[1]['num_ues'] == n] for n in counts}
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < args.samples:
            for n in rng.permutation(counts):
                if groups[n] and len(chosen) < args.samples:
                    chosen.append(groups[n].pop())
        maps = []
        for path, row in sorted(chosen, key=lambda p: (p[1]['num_ues'], p[1]['map_index'])):
            save(args.output / 'inputs' / path.name, row)
            maps.append(dict(file=path.name, num_ues=row['num_ues'], map_index=row['map_index'],
                             milp_budget=row['unique_milp_solves']))
        selection = dict(identity=identity, sampling='Random stratified subset, near-balanced across UE counts', maps=maps)
        save(selection_path, selection)
    print(f'Selected {len(selection["maps"])} maps; WE is reused, PSO uses each map\'s WE MILP budget.', flush=True)
    for entry in selection['maps']:
        print(f'  {entry["file"]}: {entry["milp_budget"]} MILP calls', flush=True)
    if args.prepare_only:
        return 0
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    save(args.output / 'process.json', {'pid': os.getpid(), 'status': 'running'})
    for index, entry in enumerate(selection['maps'], 1):
        destination = args.output / 'pso' / Path(entry['file']).stem / 'pso_result.json'
        if destination.exists() and read(destination).get('status') == 'complete':
            print(f'[{index}/{len(selection["maps"])}] Skip completed {entry["file"]}', flush=True)
            continue
        log_path = logs / (Path(entry['file']).stem + '.log')
        print(f'[{index}/{len(selection["maps"])}] Running {entry["file"]}; log: {log_path}', flush=True)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                   str((args.output/'inputs'/entry['file']).resolve()), '--source', str(args.source.resolve()),
                   '--output', str(args.output.resolve()), '--swarm', str(args.swarm)]
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            save(args.output/'process.json', {'pid': os.getpid(), 'status': 'failed', 'log': str(log_path)})
            print(f'Stopped; inspect {log_path}. Completed maps are preserved.', flush=True)
            return 1
        rows = summarize(selection, args.output)
        if rows:
            row = rows[-1]
            print(f'WE {row["we_served"]}, PSO {row["pso_served"]}; '
                  f'power {row["we_power_w"]:.2f}/{row["pso_power_w"]:.2f} W; '
                  f'PRB {row["we_prbs"]}/{row["pso_prbs"]}', flush=True)
    summarize(selection, args.output)
    save(args.output/'process.json', {'pid': os.getpid(), 'status': 'complete'})
    print(f'Finished: {args.output / "comparison.png"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
