import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import COUNTS, METHODS, RunLock, digest, map_identity, read, save, token
from recorded_data import OUTPUT, forbidden_source, require_current
from worker import load


def configuration(manifest):
    return {k: v for k, v in manifest.items() if k != 'maps_per_case'}


def native_experiment(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode('utf-8')).hexdigest()


def we_identity(selection):
    return dict(study=token(selection['identity']), runner=digest(__file__))


def validate_study(output):
    launcher = load('we_study_launcher', 'baseline_150maps/main.py')
    selection = read(output / 'selection.json')
    require_current(selection)
    if selection['identity']['code'] != launcher.signature():
        raise RuntimeError('Study code changed; do not mix old/new results.')
    for name in selection['maps']:
        if token(read(output / 'inputs' / name)) != selection['inputs'][name]:
            raise RuntimeError(f'Frozen input changed: {name}')
    return selection


def copy_native_checkpoint(output, selection, name, row, provenance):
    destination = output / 'we' / 'checkpoints' / (Path(name).stem + '.pkl')
    if destination.exists():
        return None
    base = (ROOT / 'we_normalized_300maps').resolve()
    manifest = selection['identity']['environment']
    from run_ue_count_sweep import case_size, scenario_seed
    for candidate in provenance.get('checkpoints', []):
        source, manifest_path = Path(candidate['path']).resolve(), Path(candidate['manifest']).resolve()
        if (forbidden_source(source) or forbidden_source(manifest_path)
                or not source.is_relative_to(base) or not manifest_path.is_relative_to(base)):
            raise RuntimeError('Checkpoint escapes the normalized source')
        if not source.exists():
            continue
        old = read(manifest_path)
        if (configuration(old) != configuration(manifest)
                or row['map_index'] > case_size(old['maps_per_case'], row['num_ues'])
                or old['algorithm_seed'] != row['algorithm_seed']
                or scenario_seed(old['base_map_seed'], row['num_ues'], row['map_index']) != row['map_seed']):
            continue
        aliases = [native_experiment(old)]
        extension = source.parent.parent / 'extension.json'
        if extension.exists():
            metadata = read(extension)
            if (configuration(metadata['source_manifest']) != configuration(manifest)
                    or metadata['target_experiment'] != native_experiment(old)
                    or metadata['source_experiment'] != native_experiment(metadata['source_manifest'])):
                continue
            aliases.append(metadata['source_experiment'])
        with source.open('rb') as stream:
            payload = pickle.load(stream)
        expected = [dict(experiment=experiment, count=row['num_ues'],
                         map_index=row['map_index'], map_seed=row['map_seed']) for experiment in aliases]
        if payload.get('version') != 1 or payload.get('identity') not in expected:
            continue
        from sweep_checkpoint import atomic_bytes
        atomic_bytes(destination, source.read_bytes())
        save(destination.with_suffix('.origin.json'), dict(experiments=aliases, source=str(source),
             we_runner_identity=token(we_identity(selection))))
        return dict(map=name, path=str(source))
    return None


def prepare_we(output, selection):
    identity = we_identity(selection)
    marker = output / 'we' / 'manifest.json'
    if marker.exists() and read(marker) != identity:
        raise RuntimeError('WE runner/study changed since its previous run. Use a new output directory.')
    save(marker, identity)
    manifest = selection['identity']['environment']
    base = (ROOT / 'we_normalized_300maps').resolve()
    reused, incompatible, copied, pending = [], [], [], []
    for name in selection['maps']:
        row = read(output / 'inputs' / name)
        destination = output / 'we' / 'runs' / name
        if destination.exists():
            existing = read(destination)
            if existing.get('we_runner_identity') != token(identity):
                raise RuntimeError(f'WE result belongs to another study: {destination}')
            if existing.get('status') == 'complete':
                continue
        provenance = selection['sources'][name]
        source = Path(provenance['result']).resolve()
        source_manifest = Path(provenance['manifest']).resolve()
        if (forbidden_source(source) or forbidden_source(source_manifest)
                or not source.is_relative_to(base) or not source_manifest.is_relative_to(base)):
            raise RuntimeError('WE reuse is restricted to the restored normalized directory')
        if configuration(read(source_manifest)) != configuration(manifest):
            incompatible.append(dict(map=name, path=str(source), reason='different_code_or_settings'))
            continue
        if not source.exists():
            if provenance.get('kind') != 'pending':
                raise RuntimeError(f'A recorded completed result is missing: {source}')
            restored = copy_native_checkpoint(output, selection, name, row, provenance)
            if restored:
                copied.append(restored)
            pending.append(name)
            continue
        result = read(source)
        if result.get('status') != 'complete' or not result.get('solver_optimal'):
            incompatible.append(dict(map=name, path=str(source), reason='not_solver_confirmed'))
            continue
        if map_identity(result) != map_identity(row):
            incompatible.append(dict(map=name, path=str(source), reason='different_recorded_map'))
            continue
        result.update(study_identity=token(selection['identity']), we_runner_identity=token(identity),
                      reuse_provenance=dict(path=str(source), sha256=digest(source)))
        save(destination, result)
        source_log = source.parent.parent / 'logs' / (Path(name).stem + '.log')
        target_log = output / 'we' / 'logs' / source_log.name
        if source_log.exists() and not target_log.exists():
            target_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_log, target_log)
        reused.append(dict(map=name, path=str(source)))
    report_path = output / 'we' / 'reuse_report.json'
    previous = read(report_path) if report_path.exists() else dict(reused=[])
    report = dict(reused=previous['reused'] + reused,
                  copied_checkpoints=previous.get('copied_checkpoints', []) + copied,
                  pending=pending, not_reused=incompatible,
                  policy='Reuse real completed normalized results; resume/run selected remaining normalized plan IDs')
    save(report_path, report)
    return report


def restore_checkpoint(path, identity, optimizer, cache, evaluator):
    origin = path.with_suffix('.origin.json')
    aliases = read(origin)['experiments'] if origin.exists() else []
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    allowed = [identity, *(dict(identity, experiment=old) for old in aliases)]
    if payload.get('version') != 1 or payload.get('identity') not in allowed:
        raise RuntimeError('WE checkpoint belongs to a different map or experiment')
    optimizer.__dict__.update(payload['optimizer_state'])
    cache.update(payload['cache'])
    evaluator.unique_solves = payload['unique_milp_solves']
    evaluator.solve_seconds = payload['milp_solve_seconds']
    return payload['run_state'], payload['elapsed_seconds']


def run_one(output, name):
    import matplotlib
    matplotlib.use('Agg')
    import env
    import run_ue_count_sweep as sweep
    from main import build_parser
    selection = validate_study(output)
    if name not in selection['maps']:
        raise RuntimeError('Unknown WE map')
    identity = we_identity(selection)
    if read(output / 'we' / 'manifest.json') != identity:
        raise RuntimeError('WE manifest changed')
    row = read(output / 'inputs' / name)
    defaults = build_parser().parse_args([])
    defaults.seed = row['algorithm_seed']
    original = sweep.configured_scenario

    def frozen(**kwargs):
        phys, scene = original(**kwargs)
        return phys, replace(scene, ues=tuple(env.UE(**ue) for ue in row['ues']))

    original_restore = sweep.restore_checkpoint
    sweep.configured_scenario, sweep.restore_checkpoint = frozen, restore_checkpoint
    try:
        result = sweep.run_one(row['num_ues'], row['map_index'], row['map_seed'], defaults,
            output / 'we' / 'logs' / (Path(name).stem + '.log'),
            checkpoint_path=output / 'we' / 'checkpoints' / (Path(name).stem + '.pkl'),
            checkpoint_identity=dict(experiment=token(identity), count=row['num_ues'],
                                     map_index=row['map_index'], map_seed=row['map_seed']),
            checkpoint_every=10)
    finally:
        sweep.configured_scenario, sweep.restore_checkpoint = original, original_restore
    if map_identity(result) != map_identity(row):
        raise RuntimeError('WE returned different UE data/seed')
    result.update(study_identity=token(selection['identity']), we_runner_identity=token(identity))
    save(output / 'we' / 'runs' / name, result)
    return 0


def summarize_all(output, selection):
    import csv
    from sweep_checkpoint import atomic_bytes
    launcher = load('we_summary_launcher', 'baseline_150maps/main.py')
    launcher.summary(output, selection)
    rows = read(output / 'summary.json')['rows']
    for name in selection['maps']:
        path = output / 'we' / 'runs' / name
        if not path.exists():
            continue
        result = read(path)
        if result.get('status') != 'complete' or result.get('we_runner_identity') != token(we_identity(selection)):
            continue
        rows.append(dict(method='we', num_ues=result['num_ues'], map_index=result['map_index'],
            map_seed=result['map_seed'], served_ues=result['served_ues'], violation=result['final_violation'],
            power_w=result['total_power_w'], prbs=result['total_prbs'], elapsed_seconds=result['elapsed_seconds'],
            incumbent_available=True))
    groups = []
    for method in ('we', *METHODS):
        for count, required in COUNTS.items():
            group = [r for r in rows if r['method'] == method and r['num_ues'] == count]
            feasible = [r for r in group if r['violation'] is not None and r['violation'] <= 1e-7]
            groups.append(dict(method=method, num_ues=count, completed=len(group), required=required,
                feasible=len(feasible), mean_feasible_served_ues=(sum(r['served_ues'] for r in feasible)/len(group) if group else None),
                mean_power_w_feasible=(sum(r['power_w'] for r in feasible)/len(feasible) if feasible else None),
                mean_prbs_feasible=(sum(r['prbs'] for r in feasible)/len(feasible) if feasible else None)))
    save(output / 'all_methods_summary.json', dict(rows=rows, groups=groups))
    for filename, data in (('all_methods_comparison.csv', rows), ('all_methods_summary.csv', groups)):
        if data:
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
            atomic_bytes(output / filename, ('\ufeff' + buffer.getvalue()).encode('utf-8'))
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the main WE on exactly the baseline study\'s 150 maps')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--rerun-incompatible', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if forbidden_source(output):
        raise RuntimeError('The retired archive cannot be used as input or output.')
    if args.worker:
        return run_one(output, args.worker)
    with RunLock(output / '.run.lock'):
        selection = validate_study(output)
        report = prepare_we(output, selection)
        groups = summarize_all(output, selection)
        done = sum(g['completed'] for g in groups if g['method'] == 'we')
        print(f'WE: {done}/150 completed; exact same 150 maps as all five baselines.', flush=True)
        print(f'WE reuse report: {output / "we/reuse_report.json"}', flush=True)
        if args.prepare_only or args.summary_only:
            return 0
        if report['not_reused'] and not args.rerun_incompatible:
            print('Some old WE results are incompatible. Review the report; explicitly use --rerun-incompatible to recompute them.', flush=True)
            return 1
        for index, name in enumerate(selection['maps'], 1):
            destination = output / 'we' / 'runs' / name
            if destination.exists() and read(destination).get('status') == 'complete':
                print(f'WE [{index}/150] Skip completed {name}', flush=True)
                continue
            log = output / 'we' / 'logs' / (Path(name).stem + '.worker.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'WE [{index}/150] {name}; log: {log.with_name(Path(name).stem + ".log")}', flush=True)
            try:
                with log.open('a', encoding='utf-8') as stream:
                    process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                        '--output-dir', str(output), '--worker', name], stdout=stream, stderr=subprocess.STDOUT)
                    code = process.wait()
            except KeyboardInterrupt:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
                print('Stopped. WE checkpoints are saved every 10 completed generations; rerun this command.', flush=True)
                return 130
            summarize_all(output, selection)
            if code or not destination.exists() or read(destination).get('status') != 'complete':
                print(f'WE map stopped. Inspect {log}; rerun to resume.', flush=True)
                return code or 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
