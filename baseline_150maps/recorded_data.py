import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import COUNTS, ROOT, map_identity, read, select_maps, token


OUTPUT = Path(__file__).resolve().parent / 'results_normalized'
DATA_POLICY = 'normalized-resume-plan-v4'


def forbidden_source(path):
    parts = str(path).replace('\\', '/').split('/')
    return (any(p.casefold().startswith('ga_logs') for p in parts)
            or any(p.casefold().startswith('ga_logs') for p in Path(path).resolve().parts))


def configuration(manifest):
    return {k: v for k, v in manifest.items() if k != 'maps_per_case'}


def recorded_maps(manifest):
    base = (ROOT / 'we_normalized_300maps').resolve()
    if forbidden_source(base):
        raise RuntimeError('The normalized source must not resolve into the retired archive')
    rows, provenance, rejected, plans = {}, {}, [], []
    sources = sorted(base.glob('*/manifest.json'), key=lambda p: (p.parent.name != 'results_300maps', str(p)))
    for path in sources:
        if forbidden_source(path) or not path.resolve().is_relative_to(base):
            raise RuntimeError('Source escapes the approved normalized directory')
        source_manifest = read(path)
        if configuration(source_manifest) != configuration(manifest):
            rejected.append(dict(path=str(path), reason='different_code_or_settings'))
            continue
        plans.append(dict(manifest=str(path), settings=source_manifest))
        for file in sorted((path.parent / 'runs').glob('ue_*_map_*.json')):
            if forbidden_source(file) or not file.resolve().is_relative_to(base):
                raise RuntimeError('Result escapes the approved normalized directory')
            row = read(file)
            if row.get('status') != 'complete' or not row.get('solver_optimal'):
                continue
            count = row.get('num_ues')
            if count not in COUNTS or len(row.get('ues', [])) != count:
                raise RuntimeError(f'Invalid recorded UE array: {file}')
            expected_name = f'ue_{count:03d}_map_{row["map_index"]:03d}.json'
            if file.name != expected_name or len({u['id'] for u in row['ues']}) != count:
                raise RuntimeError(f'Invalid recorded map identity: {file}')
            if file.name in rows:
                if map_identity(rows[file.name]) != map_identity(row):
                    raise RuntimeError(f'Conflicting recorded map data: {file}')
                continue
            rows[file.name] = {**map_identity(row), 'input_origin': DATA_POLICY}
            provenance[file.name] = dict(result=str(file), manifest=str(path),
                                         input_sha256=token(map_identity(row)), kind='completed')
    availability = [dict(num_ues=n, available=sum(row['num_ues'] == n for row in rows.values()), required=k)
                    for n, k in COUNTS.items()]
    return rows, provenance, dict(source=str(base), groups=availability, rejected_manifests=rejected,
                                 plans=plans, policy='Keep completed maps; run missing IDs from the same normalized plan')


def selected_recorded_maps(manifest, seed):
    rows, provenance, report = recorded_maps(manifest)
    if not report['plans']:
        raise RuntimeError('No restored normalized plan matches current code/settings; no archive fallback is allowed.')
    from run_ue_count_sweep import case_size, scenario_seed
    from scenario import configured_scenario
    rng = np.random.default_rng(seed)
    chosen, selected_sources = {}, {}
    for count, required in COUNTS.items():
        completed = sorted(name for name, row in rows.items() if row['num_ues'] == count)
        if len(completed) > required:
            completed = sorted(rng.choice(completed, required, replace=False).tolist())
        for name in completed:
            chosen[name], selected_sources[name] = rows[name], provenance[name]
        pending = {}
        for plan in report['plans']:
            for index in range(1, case_size(plan['settings']['maps_per_case'], count)+1):
                name = f'ue_{count:03d}_map_{index:03d}.json'
                if name not in rows:
                    pending.setdefault(name, plan)
        missing = required - len(completed)
        if len(pending) < missing:
            raise RuntimeError(f'{count} UE: normalized plan has insufficient unused map IDs; no fallback used.')
        additions = sorted(rng.choice(sorted(pending), missing, replace=False).tolist()) if missing else []
        for name in additions:
            plan = pending[name]
            settings = plan['settings']
            index = int(Path(name).stem.rsplit('_', 1)[1])
            map_seed = scenario_seed(settings['base_map_seed'], count, index)
            _, scene = configured_scenario(seed=map_seed, num_ues=count,
                                           ue_distribution=settings['ue_distribution'])
            row = dict(num_ues=count, map_index=index, map_seed=map_seed,
                       algorithm_seed=settings['algorithm_seed'], ues=[asdict(ue) for ue in scene.ues],
                       input_origin=DATA_POLICY)
            chosen[name] = row
            folder = Path(plan['manifest']).parent
            selected_sources[name] = dict(kind='pending', result=str(folder/'runs'/name),
                manifest=plan['manifest'], input_sha256=token(map_identity(row)),
                checkpoints=[dict(path=str(Path(p['manifest']).parent/'checkpoints'/(Path(name).stem+'.pkl')),
                                  manifest=p['manifest']) for p in report['plans']])
        group = next(g for g in report['groups'] if g['num_ues'] == count)
        group.update(selected_completed=len(completed), selected_new=len(additions))
    names = sorted(chosen)
    return {name: chosen[name] for name in names}, {name: selected_sources[name] for name in names}, report


def execution_policy(manifest):
    initial = min(manifest['ga_settings']['initial_milp_seeds'], manifest['runner_settings']['population'])
    budget = initial + manifest['runner_settings']['generations'] * manifest['runner_settings']['joint_milp_budget']
    return dict(milp_budgets={method: budget for method in ('random', 'ga', 'pso', 'zhong')},
                budget_origin='Current configured settings, never retired result files',
                grid_time_policy='Same-map normalized WE result imported into this study')


def require_current(selection):
    if selection['identity'].get('data_policy') != DATA_POLICY:
        raise RuntimeError('This selection is not the approved normalized resume plan; previous outputs are preserved.')


def allocation_input(selection, method, row):
    require_current(selection)
    return dict(row, unique_milp_solves=int(selection['identity']['execution_policy']['milp_budgets'][method]))


def current_grid_budget(output, selection, row):
    from run_we import we_identity
    require_current(selection)
    path = Path(output) / 'we' / 'runs' / f'ue_{row["num_ues"]:03d}_map_{row["map_index"]:03d}.json'
    if not path.exists():
        raise RuntimeError('Grid needs a same-map normalized WE result imported by run_we.py; no archive timing is used.')
    result = read(path)
    if (result.get('status') != 'complete' or not result.get('solver_optimal')
            or result.get('we_runner_identity') != token(we_identity(selection))
            or any(result.get(k) != row[k] for k in ('num_ues', 'map_index', 'map_seed', 'algorithm_seed', 'ues'))):
        raise RuntimeError('Grid reference WE result does not match this recorded map/study')
    seconds = float(result['elapsed_seconds'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError('Invalid normalized WE runtime')
    return seconds
