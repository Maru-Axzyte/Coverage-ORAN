import importlib.util
import math
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

from common import CORE, ExactJournal, ROOT, digest, read, save, token
from recorded_data import allocation_input, current_grid_budget, require_current


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_journal(path, identity):
    import evaluation
    original = evaluation.make_exact_evaluator
    journal = ExactJournal(path, identity)

    def factory(*args, **kwargs):
        base, cache = original(*args, **kwargs)
        return journal.wrap(base), cache

    evaluation.make_exact_evaluator = factory
    return journal


def run(output, method, name):
    import matplotlib
    matplotlib.use('Agg')
    launcher = load('study_launcher', 'baseline_150maps/main.py')
    selection = read(output / 'selection.json')
    require_current(selection)
    if method not in launcher.METHODS or name not in selection['maps']:
        raise RuntimeError('Unknown method/map')
    if selection['identity']['code'] != launcher.signature():
        raise RuntimeError('Code changed after preparation')
    row = read(output / 'inputs' / name)
    if token(row) != selection['inputs'][name]:
        raise RuntimeError('Frozen map data changed')
    manifest = selection['identity']['environment']
    destination = output / method / 'runs' / name
    if destination.exists():
        result = read(destination)
        if result.get('study_identity') != token(selection['identity']):
            raise RuntimeError('Existing result belongs to a different experiment')
        if result.get('status') == 'complete':
            return 0
    identity = dict(study=token(selection['identity']), method=method, map=selection['inputs'][name])
    checkpoint = output / method / 'checkpoints' / Path(name).stem
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    timing_path = checkpoint.with_suffix('.timing.json')
    previous_elapsed = read(timing_path)['elapsed_seconds'] if timing_path.exists() else 0.
    started = time.perf_counter()

    def persist(path, value):
        value = dict(value)
        value['study_identity'] = token(selection['identity'])
        value['elapsed_seconds'] = previous_elapsed + time.perf_counter() - started
        value['resume_policy'] = 'Deterministic replay from saved exact MILP labels; committed solves are not repeated'
        save(path, value)
        save(timing_path, dict(elapsed_seconds=value['elapsed_seconds']))

    if method == 'grid':
        seconds = current_grid_budget(output, selection, row)
        return run_grid(output, row, manifest, identity, destination, checkpoint, persist, time_budget=seconds)
    row = allocation_input(selection, method, row)
    journal = install_journal(checkpoint.with_suffix('.sqlite'), identity)
    print(f'{method}/{name}: durable exact cache has {journal.db.execute("SELECT COUNT(*) FROM solves").fetchone()[0]} labels', flush=True)
    try:
        if method in ('random', 'ga'):
            runner = load('study_classic', 'classic_milp_baselines/main.py')
            runner.save = persist
            result = runner.run_map(method, row, manifest, SimpleNamespace(population=80, report_every=40), destination)
        elif method == 'zhong':
            runner = load('study_zhong', 'zhong_ga/main.py')
            runner.save = persist
            args = SimpleNamespace(population=100, crossover=.8, mutation=.01, max_generations=17000)
            result = runner.run_map(row, manifest, args, destination)
        else:
            from dataclasses import replace
            import env
            import optimizer
            from ga import GASettings
            from scenario import configured_scenario
            runner = load('study_pso', 'baselines/run_pso.py')
            defaults = GASettings(**manifest['ga_settings'])
            runner.GASettings = lambda: defaults
            optimizer.GASettings = lambda: defaults

            def frozen(**kwargs):
                phys, scenario = configured_scenario(**kwargs)
                return phys, replace(scenario, ues=tuple(env.UE(**ue) for ue in row['ues']))

            runner.configured_scenario = frozen
            work = output / method / 'work' / Path(name).stem
            work.mkdir(parents=True, exist_ok=True)
            partial = work / 'pso_result.json'
            if partial.exists():
                partial.replace(work / 'previous_partial.json')

            def pso_save(path, value):
                persist(path, value)
                persist(destination, value)

            runner.save_result = pso_save
            runner.main(['--swarm', '40', '--milp-budget', str(row['unique_milp_solves']),
                         '--seed', str(row['algorithm_seed']), '--map-seed', str(row['map_seed']),
                         '--num-ues', str(row['num_ues']), '--ue-distribution', manifest['ue_distribution'],
                         '--weight-method', manifest['runner_settings']['weight_method'],
                         '--output-dir', str(work), '--no-show'])
            result = read(destination)
        print(f'Checkpoint cache: replayed={journal.replayed}, new MILP={journal.new_solves}', flush=True)
        return 0 if result.get('status') == 'complete' else 1
    finally:
        save(timing_path, dict(elapsed_seconds=previous_elapsed + time.perf_counter() - started))
        journal.close()


def run_grid(output, row, manifest, identity, destination, checkpoint, persist, *, time_budget):
    import gc
    import highspy
    import numpy as np
    import config
    import env
    from milp import UAVMILP
    from sweep_checkpoint import atomic_bytes

    model_code = load('study_grid_model', 'zamani_grid_milp/model.py')
    signature = token(identity)
    modes_path = checkpoint.with_suffix('.modes.pkl')
    solution_path = checkpoint.with_suffix('.solution.pkl')
    started = time.perf_counter()
    physics = dict(manifest['physics'])
    if physics['power_level_step_db'] is None:
        physics['P_min_dbm'] = physics['P_max_dbm'] = None
    phys = config.PhysConstant(**physics)
    users = [env.UE(**ue) for ue in row['ues']]
    number, power, prbs = manifest['num_uavs'], manifest['per_uav_power_w'], manifest['per_uav_prb_bound']
    settings = config.SolverSettings(time_limit_s=math.inf, mip_gap=0., log_to_console=True,
        minimize_power_tiebreak=True, symmetry_breaking=True,
        num_threads=manifest['solver']['num_threads'], parallel_search=manifest['solver']['parallel_search'])
    budget = float(time_budget)
    previous = dict(identity=signature, elapsed_seconds=0., solution=None)
    if solution_path.exists():
        previous = pickle.loads(solution_path.read_bytes())
        if previous['identity'] != signature:
            raise RuntimeError('Grid checkpoint belongs to a different experiment')
    if modes_path.exists():
        stored = pickle.loads(modes_path.read_bytes())
        if stored['identity'] != signature:
            raise RuntimeError('Grid modes belong to a different experiment')
        placements, modes = stored['placements'], stored['modes']
        print('Loaded saved grid service modes.', flush=True)
    else:
        placements = env.make_placements(np.linspace(0., manifest['area_m'][0], config.grid_size + 1),
            np.linspace(0., manifest['area_m'][1], config.grid_size + 1), phys.H_U, phys.Theta_U)
        uav = env.UAV(0, 0., 0., max_power_w=power, max_prbs=prbs)
        oracle = UAVMILP([uav], users, placements, phys, settings,
                         system_prb_budget=manifest['system_prbs'], orthogonal_prb_pool=True)
        oracle.precompute_service_modes()
        modes = list(oracle.service_modes.values())
        atomic_bytes(modes_path, pickle.dumps(dict(identity=signature, placements=placements, modes=modes), protocol=pickle.HIGHEST_PROTOCOL))
        del oracle
        gc.collect()
    ga = manifest['ga_settings']
    separation = ga['min_uav_separation_m'] if ga['include_separation_constraint'] else 0.
    built = model_code.build_model(placements, modes, users, number, power, prbs, manifest['system_prbs'],
        separation, settings, output / 'grid' / 'logs' / f'{Path(destination).stem}.highs.log')
    if built is None:
        raise RuntimeError('Grid model rejected')
    model, modes, dimensions = built
    if previous['solution'] is not None:
        solution = highspy.HighsSolution()
        solution.col_value = previous['solution']
        solution.value_valid = True
        if model.setSolution(solution) == highspy.HighsStatus.kError:
            raise RuntimeError('Saved grid incumbent rejected')
    latest = previous['solution']
    saved_at = time.perf_counter()

    def checkpoint_solution(event=None, force=False):
        nonlocal latest, saved_at
        now = time.perf_counter()
        if event is not None and event.callback_type == highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution:
            latest = np.array(event.data_out.mip_solution, copy=True)
            force = True
        if force or now - saved_at >= 30.:
            atomic_bytes(solution_path, pickle.dumps(dict(identity=signature,
                elapsed_seconds=previous['elapsed_seconds'] + now - started, solution=latest), protocol=pickle.HIGHEST_PROTOCOL))
            saved_at = now

    model.cbMipImprovingSolution.subscribe(checkpoint_solution)
    model.cbMipLogging.subscribe(checkpoint_solution)
    model.HandleKeyboardInterrupt = True
    remaining = max(0., budget - previous['elapsed_seconds'] - (time.perf_counter() - started))
    model.setOptionValue('time_limit', remaining)
    print(f'Grid {row["num_ues"]} UE: {len(placements)} sites; remaining time {remaining:.1f}s', flush=True)
    checkpoint_solution(force=True)
    solve_started = time.perf_counter()
    try:
        returned = model.run()
    except KeyboardInterrupt:
        checkpoint_solution(force=True)
        persist(destination, dict(method='Adapted Zamani grid MILP', status='interrupted', best=None))
        return 130
    checkpoint_solution(force=True)
    status, info = model.getModelStatus(), model.getInfo()
    best = model_code.extract(model, placements, modes, number, users, power, prbs,
                              manifest['system_prbs'], separation, phys)
    if best is None and latest is not None:
        saved = highspy.HighsSolution()
        saved.col_value, saved.value_valid = latest, True
        best = model_code.extract(SimpleNamespace(getSolution=lambda: saved), placements, modes,
                                  number, users, power, prbs, manifest['system_prbs'], separation, phys)
    interrupted = status == highspy.HighsModelStatus.kInterrupt
    failed = returned == highspy.HighsStatus.kError
    finite = lambda v: float(v) if v is not None and math.isfinite(v) else None
    result = dict(method='Adapted Zamani grid MILP', source_we=row,
        status='interrupted' if interrupted else ('failed' if failed else 'complete'), best=best,
        objective='Max served count, then admitted demand, then min PRBs, then min transmit power',
        grid=dict(intervals=config.grid_size, candidates=len(placements)), time_budget_seconds=budget,
        solver_status=model.modelStatusToString(status), solver_optimal=status == highspy.HighsModelStatus.kOptimal,
        solver_mip_gap=finite(info.mip_gap), solver_dual_bound=finite(info.mip_dual_bound),
        solver_primal_bound=finite(info.objective_function_value) if best else None,
        solve_seconds=time.perf_counter() - solve_started, **dimensions)
    persist(destination, result)
    result = read(destination)
    result['resume_policy'] = 'Saved modes + incumbent warm start; branch-and-bound tree is rebuilt; original total time budget retained'
    save(destination, result)
    return 130 if interrupted else (1 if failed else 0)
