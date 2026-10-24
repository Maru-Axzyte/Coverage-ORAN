import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import COUNTS, CORE, METHODS, RunLock, digest, read, save, token
from recorded_data import (DATA_POLICY, OUTPUT, execution_policy, forbidden_source,
                           recorded_maps, require_current, selected_recorded_maps)

RUNNERS = {'random': 'classic_milp_baselines/main.py', 'ga': 'classic_milp_baselines/main.py',
           'pso': 'baselines/run_pso.py', 'zhong': 'zhong_ga/main.py',
           'grid': 'zamani_grid_milp/main.py'}


def signature():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in
            [*(ROOT / name for name in CORE), *(ROOT / name for name in sorted(set(RUNNERS.values()))),
             ROOT / 'zamani_grid_milp/model.py', HERE / 'main.py', HERE / 'common.py',
             HERE / 'worker.py', HERE / 'recorded_data.py']}


def normalized(method, row, result):
    if method in ('zhong', 'grid'):
        best = result.get('best')
        if best is None:
            return dict(served_ues=0, violation=None, power_w=None, prbs=None,
                        elapsed_seconds=result.get('elapsed_seconds'), incumbent_available=False)
        exact = best.get('exact', best)
        return dict(served_ues=best['served_ues'], violation=best['violation'],
                    power_w=exact['total_power_w'], prbs=exact.get('total_prbs', exact.get('total_num_prbs')),
                    elapsed_seconds=result['elapsed_seconds'], incumbent_available=True)
    exact = result['final_milp']
    return dict(served_ues=result['served_ues'], violation=result['final_violation'],
                power_w=result.get('total_power_w', exact['total_power_w']),
                prbs=result.get('total_prbs', exact['total_num_prbs']),
                elapsed_seconds=result['elapsed_seconds'], incumbent_available=True)


def prepare(output, seed):
    from main import build_parser
    from run_ue_count_sweep import experiment_manifest
    manifest = experiment_manifest(build_parser().parse_args([]))
    current = dict(version=4, data_policy=DATA_POLICY, counts=COUNTS, selection_seed=seed,
                   execution_policy=execution_policy(manifest), code=signature(), environment=manifest)
    path = output / 'selection.json'
    if path.exists():
        selection = read(path)
        require_current(selection)
        if selection['identity'] != json.loads(json.dumps(current)):
            raise RuntimeError('Code/config changed. Preserve results and use a new --output-dir.')
        for name in selection['maps']:
            row = read(output / 'inputs' / name)
            if row.get('input_origin') != DATA_POLICY or token(row) != selection['inputs'][name]:
                raise RuntimeError(f'Recorded input changed: {name}')
        return selection
    if output.exists() and any(p.name not in ('.run.lock', 'source_availability.json') for p in output.iterdir()):
        raise RuntimeError('Nonempty output without a current manifest; choose a new directory.')
    _, _, availability = recorded_maps(manifest)
    save(output / 'source_availability.json', availability)
    rows, provenance, availability = selected_recorded_maps(manifest, seed)
    save(output / 'source_availability.json', availability)
    selection = dict(identity=current, maps=list(rows), inputs={name: token(row) for name, row in rows.items()},
                     sources=provenance, map_policy='Keep completed normalized maps; seeded unused plan IDs fill the remainder; frozen inputs for all six methods')
    for name, row in rows.items():
        save(output / 'inputs' / name, row)
    save(path, selection)
    save(output / 'reuse_report.json', dict(reused=[], not_reused=[],
         note='Only restored normalized data is used. This study skips its own completed jobs.'))
    return selection


def summary(output, selection):
    require_current(selection)
    rows = []
    identity = token(selection['identity'])
    for method in METHODS:
        for name in selection['maps']:
            path = output / method / 'runs' / name
            if not path.exists():
                continue
            result = read(path)
            if result.get('status') != 'complete' or result.get('study_identity') != identity:
                continue
            source = read(output / 'inputs' / name)
            rows.append(dict(method=method, num_ues=source['num_ues'], map_index=source['map_index'],
                             map_seed=source['map_seed'], **normalized(method, source, result)))
    groups = []
    for method in METHODS:
        for count, required in COUNTS.items():
            group = [row for row in rows if row['method'] == method and row['num_ues'] == count]
            feasible = [r for r in group if r['violation'] is not None and r['violation'] <= 1e-7]
            groups.append(dict(method=method, num_ues=count, completed=len(group), required=required,
                feasible=len(feasible), mean_feasible_served_ues=(sum(r['served_ues'] for r in feasible)/len(group) if group else None),
                mean_power_w_feasible=(sum(r['power_w'] for r in feasible)/len(feasible) if feasible else None),
                mean_prbs_feasible=(sum(r['prbs'] for r in feasible)/len(feasible) if feasible else None)))
    save(output / 'summary.json', dict(rows=rows, groups=groups,
         note='Recorded normalized maps only; partial groups are labelled completed/required.'))
    for filename, values in (('comparison.csv', rows), ('summary.csv', groups)):
        if values:
            import io
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
            from sweep_checkpoint import atomic_bytes
            atomic_bytes(output / filename, ('\ufeff' + stream.getvalue()).encode('utf-8'))
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description='Five sequential baselines on 150 frozen normalized-plan maps')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--selection-seed', type=int, default=20261023)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('METHOD', 'MAP'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if forbidden_source(output):
        raise RuntimeError('The retired archive cannot be used as input or output.')
    if args.worker:
        from worker import run
        return run(output, *args.worker)
    with RunLock(output / '.run.lock'):
        selection = read(output / 'selection.json') if args.summary_only else prepare(output, args.selection_seed)
        groups = summary(output, selection)
        print('150 normalized-plan maps/method: 20/30/40/50/60/70 UE = 10/10/10/20/50/50 maps.', flush=True)
        print('Sequential; no archived maps, budgets or timing are read.', flush=True)
        for method in args.methods:
            done = sum(g['completed'] for g in groups if g['method'] == method)
            print(f'{method}: completed {done}/150', flush=True)
        if args.prepare_only or args.summary_only:
            return 0
        for method in args.methods:
            for index, name in enumerate(selection['maps'], 1):
                destination = output / method / 'runs' / name
                if destination.exists():
                    result = read(destination)
                    if result.get('study_identity') != token(selection['identity']):
                        raise RuntimeError(f'Result belongs to another experiment: {destination}')
                    if result.get('status') == 'complete':
                        print(f'Skip completed {method}/{name}', flush=True)
                        continue
                log = output / method / 'logs' / f'{Path(name).stem}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f'{method} [{index}/150] {name}; log: {log}', flush=True)
                command = [sys.executable, '-B', str(HERE / 'main.py'), '--output-dir', str(output), '--worker', method, name]
                try:
                    with log.open('a', encoding='utf-8') as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        code = process.wait()
                except KeyboardInterrupt:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait()
                    print('Stopped. Committed exact evaluations retained; rerun this command.', flush=True)
                    return 130
                summary(output, selection)
                if code:
                    print(f'Stopped with code {code}; inspect {log}. Rerun to resume.', flush=True)
                    return code
                if not destination.exists() or read(destination).get('status') != 'complete':
                    print(f'Worker ended without a completed result: {destination}', flush=True)
                    return 1
        print(f'Finished. Summary: {output / "summary.csv"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
