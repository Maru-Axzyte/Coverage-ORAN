import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import run_ue_count_sweep as sweep


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def experiment_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode('utf-8')).hexdigest()


def configuration(manifest):
    return {key: value for key, value in manifest.items() if key != 'maps_per_case'}


def copy_if_missing(source, destination):
    if not source.exists() or destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'{destination.name}.{uuid.uuid4().hex}.tmp')
    shutil.copyfile(source, temporary)
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return True
        except PermissionError:
            if attempt == 7:
                print(f'Copy could not be committed; pending file retained: {temporary}', flush=True)
                raise
            time.sleep(min(.1 * 2**attempt, 2.))


def prepare_extension(source, output, current):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise RuntimeError('Choose a separate output directory; source results are preserved.')
    previous = read(source/'manifest.json')
    if configuration(previous) != configuration(current):
        raise RuntimeError('Source experiment and current code/settings differ beyond map counts. '
                           'No historical result was imported.')
    for count in current['ue_counts']:
        if sweep.case_size(previous['maps_per_case'], count) > sweep.case_size(current['maps_per_case'], count):
            raise RuntimeError('The target plan must contain every source map.')
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        if read(manifest_path) != current:
            raise RuntimeError('Target experiment settings/code changed. Existing results are preserved.')
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Target directory is nonempty without a manifest.')
        sweep.save_json(manifest_path, current)
    provenance = dict(source_directory=str(source), source_manifest=previous,
                      source_experiment=experiment_id(previous),
                      target_experiment=experiment_id(current),
                      policy='Same maps/code/settings; only number of maps is extended. No source files overwritten.')
    provenance_path = output/'extension.json'
    if provenance_path.exists() and read(provenance_path) != provenance:
        raise RuntimeError('The source experiment differs from this extension provenance.')
    if not provenance_path.exists():
        sweep.save_json(provenance_path, provenance)
    for folder in ('runs', 'logs', 'checkpoints'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    imported, checkpoints, records = 0, 0, []
    for count in current['ue_counts']:
        for index in range(1, sweep.case_size(current['maps_per_case'], count)+1):
            seed = sweep.scenario_seed(current['base_map_seed'], count, index)
            stem = f'ue_{count:03d}_map_{index:03d}'
            destination = output/'runs'/(stem+'.json')
            old_result = source/'runs'/(stem+'.json')
            if not destination.exists() and old_result.exists():
                sweep.read_result(old_result, count, index, seed)
                imported += int(copy_if_missing(old_result, destination))
            if destination.exists():
                row = sweep.read_result(destination, count, index, seed)
                if row.get('algorithm_seed') != current['algorithm_seed']:
                    raise RuntimeError(f'Algorithm seed mismatch: {destination}')
                records.append(row)
                copy_if_missing(source/'logs'/(stem+'.log'), output/'logs'/(stem+'.log'))
                continue
            if copy_if_missing(source/'checkpoints'/(stem+'.pkl'),
                               output/'checkpoints'/(stem+'.pkl')):
                checkpoints += 1
            copy_if_missing(source/'logs'/(stem+'.log'), output/'logs'/(stem+'.log'))
    sweep.write_tables(output, records, current)
    total = sum(sweep.case_size(current['maps_per_case'], count) for count in current['ue_counts'])
    print(f'All-map experiment: {total} maps; {current["runner_settings"]["generations"]} generations/map.', flush=True)
    print(f'Completed maps reused: {len(records)}; newly copied: {imported}; remaining: {total-len(records)}.', flush=True)
    print(f'Partial-map checkpoints copied: {checkpoints}. Output: {output}', flush=True)
    return provenance


def restore_extended_checkpoint(path, identity, optimizer, cache, evaluator, source_experiment):
    # Trusted checkpoints created by this project only; never load foreign pickle files.
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    old_identity = dict(identity, experiment=source_experiment)
    if payload.get('version') != 1 or payload.get('identity') not in (identity, old_identity):
        raise RuntimeError(f'Checkpoint does not match the source or target experiment/map: {path}')
    optimizer.__dict__.update(payload['optimizer_state'])
    cache.update(payload['cache'])
    evaluator.unique_solves = payload['unique_milp_solves']
    evaluator.solve_seconds = payload['milp_solve_seconds']
    return payload['run_state'], payload['elapsed_seconds']


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extend the normalized 85-map experiment to all 300 maps')
    parser.add_argument('--source-dir', type=Path, default=HERE/'results_85maps')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results_300maps')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1:
        parser.error('checkpoint-every must be positive')
    output = args.output_dir.resolve()
    if args.plot_only:
        return sweep.main(['--output-dir', str(output), '--no-show', '--plot-only'])
    current = sweep.experiment_manifest(sweep.main_parser().parse_args([]), maps_per_case=50)
    try:
        provenance = prepare_extension(args.source_dir, output, current)
    except RuntimeError as error:
        parser.error(str(error))
    if args.prepare_only:
        print('Preparation only. No GA/MILP experiment started.', flush=True)
        return 0
    original_restore = sweep.restore_checkpoint
    def restore(path, identity, optimizer, cache, evaluator):
        return restore_extended_checkpoint(path, identity, optimizer, cache, evaluator,
                                           provenance['source_experiment'])
    sweep.restore_checkpoint = restore
    try:
        return sweep.main(['--output-dir', str(output), '--no-show',
                           '--checkpoint-every', str(args.checkpoint_every)])
    finally:
        sweep.restore_checkpoint = original_restore


if __name__ == '__main__':
    main()
