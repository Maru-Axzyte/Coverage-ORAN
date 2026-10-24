import argparse
import csv
import gc
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'
PAPER = 'https://doi.org/10.1186/s13638-020-01804-3'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def prepare(args):
    manifest = read(args.source / 'manifest.json')
    for name in ('config.py', 'env.py', 'milp.py', 'evaluation.py', 'scenario.py'):
        raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        if manifest['source_sha256'].get(name) not in {
                hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}:
            print(f'{name} differs from the saved WE environment. Use matching source/configuration.')
            return None
    if not manifest['orthogonal_prb_pool']:
        print('This baseline implements the shared orthogonal-PRB environment.')
        return None
    candidates = []
    for path in sorted((args.source / 'runs').glob('ue_070_map_*.json')):
        row = read(path)
        if row.get('status') == 'complete' and row.get('solver_optimal') and row.get('num_ues') == 70:
            candidates.append(path)
    if not 1 <= args.samples <= len(candidates) or args.grid_intervals < 1:
        print(f'Choose 1 to {len(candidates)} maps and a positive grid interval count.')
        return None
    if args.time_limit is not None and args.time_limit <= 0:
        print('--time-limit must be positive.')
        return None
    identity = dict(samples=args.samples, selection_seed=args.selection_seed,
                    grid_intervals=args.grid_intervals, time_limit=args.time_limit,
                    source_manifest=manifest, sources={p.name: digest(p) for p in candidates},
                    code={p.name: digest(p) for p in (HERE / 'main.py', HERE / 'model.py')})
    path = args.output_dir / 'selection.json'
    if path.exists():
        selection = read(path)
        if args.prepare_only and not list((args.output_dir / 'runs').glob('*.json')):
            previous = {k: v for k, v in selection['identity'].items() if k != 'code'}
            current = {k: v for k, v in identity.items() if k != 'code'}
            if previous == current:
                selection['identity'] = identity
                save(path, selection)
        if selection['identity'] != identity:
            print('Settings, code or inputs changed. Use a new --output-dir; existing results are preserved.')
            return None
    else:
        chosen = sorted(random.Random(args.selection_seed).sample(candidates, args.samples))
        selection = dict(identity=identity, maps=[p.name for p in chosen], paper=PAPER,
                         method='Adapted Zamani Section 3.2 candidate-location MILP',
                         sampling='Uniform without replacement among all completed saved 70-UE maps')
        for p in chosen:
            save(args.output_dir / 'inputs' / p.name, read(p))
        save(path, selection)
    for name in selection['maps']:
        if read(args.output_dir / 'inputs' / name) != read(args.source / 'runs' / name):
            print(f'Frozen input mismatch: {name}')
            return None
    levels = manifest['physics']
    size = (args.grid_intervals + 1) ** 2 * len(levels['H_U']) * len(levels['Theta_U'])
    print(f'Selected {len(selection["maps"])} of {len(candidates)} saved 70-UE maps; seed={args.selection_seed}')
    print('Map IDs: ' + ', '.join(str(read(args.output_dir / 'inputs' / n)['map_index']) for n in selection['maps']))
    print(f'Grid: {args.grid_intervals + 1} x {args.grid_intervals + 1} x '
          f'{len(levels["H_U"])} heights x {len(levels["Theta_U"])} angles = {size} candidates')
    print('Time budget per map: ' + (f'{args.time_limit:g} s' if args.time_limit is not None
          else 'saved WE elapsed time; includes preprocessing and model building'))
    return selection


def run_map(name, output):
    import numpy as np
    import highspy
    import config
    import env
    from milp import UAVMILP
    from model import build_model, extract

    selection = read(output / 'selection.json')
    identity = selection['identity']
    manifest = identity['source_manifest']
    row = read(output / 'inputs' / name)
    started = time.perf_counter()
    destination = output / 'runs' / name
    settings_data = manifest['solver']
    physics = dict(manifest['physics'])
    if physics['power_level_step_db'] is None:
        physics['P_min_dbm'] = physics['P_max_dbm'] = None
    phys = config.PhysConstant(**physics)
    user_list = [env.UE(**ue) for ue in row['ues']]
    number, power, prbs = manifest['num_uavs'], manifest['per_uav_power_w'], manifest['per_uav_prb_bound']
    uav = env.UAV(0, 0., 0., max_power_w=power, max_prbs=prbs)
    settings = config.SolverSettings(time_limit_s=math.inf, mip_gap=0., log_to_console=True,
        minimize_power_tiebreak=True, symmetry_breaking=True,
        num_threads=settings_data['num_threads'], parallel_search=settings_data['parallel_search'])
    intervals = identity['grid_intervals']
    placements = env.make_placements(np.linspace(0., manifest['area_m'][0], intervals + 1),
        np.linspace(0., manifest['area_m'][1], intervals + 1), phys.H_U, phys.Theta_U)
    ga = manifest['ga_settings']
    separation = ga['min_uav_separation_m'] if ga['include_separation_constraint'] else 0.
    budget = identity['time_limit'] if identity['time_limit'] is not None else row['elapsed_seconds']
    result = dict(method=selection['method'], paper=PAPER, status='running', source_we=row,
                  grid=dict(intervals=intervals, candidates=len(placements), heights=list(phys.H_U),
                            angles=list(phys.Theta_U)), time_budget_seconds=budget,
                  budget_policy='Total preprocessing + model building + solver wall time',
                  objective='Max served count, then admitted demand, then min PRBs, then min transmit power',
                  adaptations=['Fixed UAV count and shared orthogonal PRBs', 'Existing channel and discrete link modes',
                               'Service-first objective; no RRH/fronthaul/hover-power model',
                               'Antenna angle is an additional candidate coordinate',
                               'Existing 3D UAV separation retained'], best=None)
    save(destination, result)
    print(f'Preparing {len(placements)} candidate placements for {len(user_list)} UEs, {number} UAVs', flush=True)
    print(f'Time budget {budget:.2f} s; WE reference {row["served_ues"]}/70, '
          f'{row["total_power_w"]:.2f} W, {row["total_prbs"]} PRBs', flush=True)
    precompute = time.perf_counter()
    oracle = UAVMILP([uav], user_list, placements, phys, settings,
                     system_prb_budget=manifest['system_prbs'], orthogonal_prb_pool=True)
    oracle.precompute_service_modes()
    modes = list(oracle.service_modes.values())
    result['precompute_seconds'] = time.perf_counter() - precompute
    print(f'Physical modes ready: {len(modes)} Pareto link modes; '
          f'{result["precompute_seconds"]:.2f} s', flush=True)
    del oracle
    gc.collect()
    built = build_model(placements, modes, user_list, number, power, prbs, manifest['system_prbs'],
                        separation, settings, output / 'logs' / (Path(name).stem + '.highs.log'))
    if built is None:
        result.update(status='failed', stop_reason='model_rejected', elapsed_seconds=time.perf_counter() - started)
        save(destination, result)
        return 1
    model, modes, dimensions = built
    result.update(dimensions)
    remaining = max(0., budget - (time.perf_counter() - started))
    print(f'Model: {dimensions["variables"]} binaries, {dimensions["constraints"]} rows, '
          f'{dimensions["nonzeros"]} nonzeros; remaining solver time {remaining:.2f} s', flush=True)
    model.setOptionValue('time_limit', remaining)
    solve_started = time.perf_counter()
    try:
        solver_return = model.run()
    except KeyboardInterrupt:
        result.update(status='interrupted', elapsed_seconds=time.perf_counter() - started)
        save(destination, result)
        return 130
    info = model.getInfo()
    status = model.getModelStatus()
    best = extract(model, placements, modes, number, user_list, power, prbs,
                   manifest['system_prbs'], separation, phys)
    dual = finite(info.mip_dual_bound) if info.valid else None
    upper_count = None
    if dual is not None:
        units = sorted(dimensions['demand_units'])
        prefix = [0.]
        for value in units:
            prefix.append(prefix[-1] + value)
        possible = [c for c in range(len(user_list) + 1)
                    if dimensions['count_weight'] * c + dimensions['demand_weight'] * prefix[c]
                    - (2 * manifest['system_prbs'] + 1) <= dual + 1e-5]
        upper_count = max(possible, default=0)
    solver_error = solver_return == highspy.HighsStatus.kError
    result.update(status='failed' if solver_error else 'complete', best=best,
        solver_status=model.modelStatusToString(status),
        solver_optimal=status == highspy.HighsModelStatus.kOptimal,
        solver_error=solver_error,
        solver_mip_gap=finite(info.mip_gap),
        solver_primal_bound=finite(info.objective_function_value) if best is not None else None,
        solver_dual_bound=dual, grid_served_upper_bound=upper_count,
        grid_served_count_proven=bool(best is not None and upper_count == best['served_ues']),
        mip_nodes=int(info.mip_node_count), solve_seconds=time.perf_counter() - solve_started,
        elapsed_seconds=time.perf_counter() - started,
        stop_reason='solver_finished' if remaining > 0 else 'preprocessing_exhausted_time_budget',
        bound_scope='Only this discrete candidate MILP; not an upper bound for continuous WE placements')
    save(destination, result)
    if best is None:
        print(f'No independently validated integer incumbent; status={result["solver_status"]}', flush=True)
    else:
        print(f'WE {row["served_ues"]}/70, Grid-MILP {best["served_ues"]}/70; '
              f'power {row["total_power_w"]:.2f}/{best["total_power_w"]:.2f} W; '
              f'PRB {row["total_prbs"]}/{best["total_prbs"]}', flush=True)
        print(f'Status={result["solver_status"]}; MIP gap={result["solver_mip_gap"]}; '
              f'grid served upper bound={upper_count}; total={result["elapsed_seconds"]:.2f} s', flush=True)
    return 1 if result['solver_error'] else 0


def summarize(output, selection):
    rows = []
    for name in selection['maps']:
        path = output / 'runs' / name
        if not path.exists():
            continue
        result = read(path)
        if result['status'] != 'complete':
            continue
        we, best = result['source_we'], result['best']
        demands = {u['id']: u['require_mbps'] for u in we['ues']}
        we_demand = sum(demands[k] for k in we['served_ue_ids'])
        feasible_we = we['final_violation'] <= 1e-12
        comparable = bool(feasible_we and best and we['served_ues'] == best['served_ues']
                          and abs(we_demand - best['admitted_demand_mbps']) <= 1e-9)
        row = dict(num_ues=70, map_index=we['map_index'], map_seed=we['map_seed'],
            we_served=we['served_ues'] if feasible_we else 0, grid_served=best['served_ues'] if best else 0,
            we_feasible=feasible_we, grid_feasible=best is not None,
            we_demand_mbps=we_demand, grid_demand_mbps=best['admitted_demand_mbps'] if best else 0.,
            we_power_w=we['total_power_w'], grid_power_w=best['total_power_w'] if best else None,
            we_prbs=we['total_prbs'], grid_prbs=best['total_prbs'] if best else None,
            resource_comparable=comparable,
            same_served_ue_set=bool(best and set(best['served_ue_ids']) == set(we['served_ue_ids'])),
            we_seconds=we['elapsed_seconds'], grid_seconds=result['elapsed_seconds'],
            precompute_seconds=result['precompute_seconds'], build_seconds=result['build_seconds'],
            solve_seconds=result['solve_seconds'], time_budget_seconds=result['time_budget_seconds'],
            solver_status=result['solver_status'], solver_optimal=result['solver_optimal'],
            mip_gap=result['solver_mip_gap'], grid_served_upper_bound=result['grid_served_upper_bound'],
            grid_served_count_proven=result['grid_served_count_proven'], mip_nodes=result['mip_nodes'],
            grid_candidates=result['grid']['candidates'], binary_variables=result['variables'])
        rows.append(row)
    if not rows:
        print('No finished map results yet.', flush=True)
        return
    matched = [r for r in rows if r['resource_comparable']]
    averages = dict(we_served=sum(r['we_served'] for r in rows) / len(rows),
                    grid_served=sum(r['grid_served'] for r in rows) / len(rows),
                    matched_service_maps=len(matched))
    for metric in ('power_w', 'prbs'):
        for method in ('we', 'grid'):
            averages[f'matched_{method}_{metric}'] = (sum(r[f'{method}_{metric}'] for r in matched)
                                                       / len(matched) if matched else None)
    save(output / 'comparison.json', dict(completed=len(rows), selected=len(selection['maps']),
        rows=rows, means=averages, paper=PAPER,
        scope='Adapted candidate-location MILP versus historical WE; not a full paper reproduction',
        timing_note='WE historical timings may use different hardware; no controlled speed claim',
        bound_note='MIP bounds/gaps refer only to the discretized objective and candidate set'))
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = list(range(len(rows)))
    axes[0].plot(x, [r['we_served'] for r in rows], 'o-', label='WE')
    axes[0].plot(x, [r['grid_served'] for r in rows], 's-', label='Grid MILP')
    axes[0].set_xticks(x, [str(r['map_index']) for r in rows], rotation=90)
    axes[0].set(xlabel='70-UE map ID', ylabel='Feasible served UEs; no incumbent = 0')
    axes[0].legend()
    for ax, metric, label in [(axes[1], 'power_w', 'Power (W)'), (axes[2], 'prbs', 'PRBs')]:
        if matched:
            a, b = [r[f'we_{metric}'] for r in matched], [r[f'grid_{metric}'] for r in matched]
            lo, hi = min(a + b), max(a + b)
            ax.scatter(a, b)
            ax.plot([lo - .1, hi + .1], [lo - .1, hi + .1], '--', color='gray')
        ax.set(xlabel=f'WE {label}', ylabel=f'Grid MILP {label}',
               title=f'Equal served count + demand: {len(matched)} maps')
    for ax in axes:
        ax.grid(alpha=.2)
    figure.suptitle(f'Adapted Zamani grid MILP: {len(rows)}/{len(selection["maps"])} maps')
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)
    print(f'Mean served over {len(rows)} maps: WE={averages["we_served"]:.3f}/70, '
          f'Grid-MILP={averages["grid_served"]:.3f}/70; matched-service maps={len(matched)}', flush=True)


def main():
    import config

    parser = argparse.ArgumentParser(description='Adapted Zamani candidate-location MILP on saved 70-UE maps')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--selection-seed', type=int, default=20261016)
    parser.add_argument('--grid-intervals', type=int, default=config.grid_size)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source, args.output_dir = args.source.resolve(), args.output_dir.resolve()
    if args.worker:
        return run_map(args.worker, args.output_dir)
    if args.plot_only:
        summarize(args.output_dir, read(args.output_dir / 'selection.json'))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'run.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('This output folder already has an active run.')
            return 1
        try:
            selection = prepare(args)
            if selection is None:
                return 1
            if args.prepare_only:
                print('Prepared inputs. No optimization started.')
                return 0
            (args.output_dir / 'logs').mkdir(exist_ok=True)
            for index, name in enumerate(selection['maps'], 1):
                result_path = args.output_dir / 'runs' / name
                if result_path.exists() and read(result_path).get('status') == 'complete':
                    print(f'Skip completed: {name}', flush=True)
                    continue
                log_path = args.output_dir / 'logs' / (Path(name).stem + '.log')
                print(f'[{index}/{len(selection["maps"])}] Running {name}; log: {log_path}', flush=True)
                command = [sys.executable, '-B', '-u', str(HERE / 'main.py'), '--worker', name,
                           '--output-dir', str(args.output_dir)]
                environment = dict(os.environ, PYTHONIOENCODING='utf-8')
                with log_path.open('a', encoding='utf-8') as stream:
                    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                            bufsize=1, env=environment) as process:
                        try:
                            for line in process.stdout:
                                print(line, end='', flush=True)
                                stream.write(line)
                                stream.flush()
                            code = process.wait()
                        except KeyboardInterrupt:
                            process.terminate()
                            process.wait()
                            print('Stopped. Completed maps are preserved; unfinished map restarts next run.')
                            return 130
                if code:
                    print(f'Map stopped with exit code {code}. See {log_path}')
                    return code
                summarize(args.output_dir, selection)
            summarize(args.output_dir, selection)
            return 0
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
