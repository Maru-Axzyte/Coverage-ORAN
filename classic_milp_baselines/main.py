import argparse
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from functools import cmp_to_key
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'
COUNTS = {20: 5, 30: 5, 40: 5, 50: 5, 60: 20, 70: 20}
CORE = ('config.py', 'env.py', 'milp.py', 'evaluation.py', 'scenario.py',
        'ga.py', 'contextual_ga.py', 'contextual_weight_audit.py',
        'preference_weights.py', 'surrogate.py', 'joint_topk_we.py', 'cma.py', 'optimizer.py')
IO_COMPATIBLE_RUNNERS = {'db72e25d67f98e7eb3de7898771d4799259f9d9a2bd0ae374aa737edc90604f5'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                print(f'Cannot replace locked result: {path}\n'
                      f'Previous result retained; new JSON retained at: {temporary}', flush=True)
                raise
            time.sleep(min(.1 * 2 ** attempt, 2.))


def compatible_runner_update(old, new, has_results):
    return ({k: v for k, v in old.items() if k != 'runner'} ==
            {k: v for k, v in new.items() if k != 'runner'} and
            (not has_results or old['runner'] in IO_COMPATIBLE_RUNNERS))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def matches_historical(name, raw, manifest):
    variants = [raw]
    if name == 'ga.py':
        historical = manifest['ga_settings']['generations']
        variants.append(re.sub(rb'generations: int = \d+',
                               f'generations: int = {historical}'.encode(), raw, count=1))
    accepted = {hashlib.sha256(value).hexdigest()
                for variant in variants for value in (variant, variant.replace(b'\n', b'\r\n'))}
    return manifest['source_sha256'].get(name) in accepted


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def select_maps(eligible, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for count, requested in COUNTS.items():
        group = sorted((p, row) for p, row in eligible if row['num_ues'] == count)
        if len(group) < requested:
            return None
        indices = rng.choice(len(group), requested, replace=False)
        selected.extend(group[int(i)] for i in indices)
    return sorted(selected, key=lambda pair: (pair[1]['num_ues'], pair[1]['map_index']))


def best_of(items, better):
    best = items[0]
    for item in items[1:]:
        if better(item, best):
            best = item
    return best


def ranked(items, better):
    def compare(a, b):
        return -1 if better(a, b) else (1 if better(b, a) else 0)
    return sorted(items, key=cmp_to_key(compare))


def optimize(method, row, manifest, args, destination, scoring, evaluator):
    from ga import Individual

    started = time.perf_counter()
    budget = int(row['unique_milp_solves'])
    size = min(args.population, budget)
    warmup = min(int(manifest['ga_settings']['initial_milp_seeds']), size)
    rng = scoring.rng
    lower, span = np.asarray(scoring.bounds.lower), scoring.bounds.span
    records, seen, history = [], {}, []
    population = []
    epsilon0 = None
    batch = 0
    status = 'running'
    result = dict(method=method, status=status, source_we=row,
                  environment=manifest, history=history, evaluations=[],
                  settings=dict(population=size, initial_milp_seeds=warmup,
                                algorithm_seed=row['algorithm_seed'], milp_budget=budget,
                                ga_settings=vars(scoring.settings)),
                  initialization='Shared centre x/y, random h/theta for warm-up; uniform full-domain thereafter',
                  scope='Shared exact scoring/weight learner; no surrogate, WE survival or CMA',
                  restart_policy='Completed maps skipped; interrupted map restarts with the same seed')

    def sample(centre=False):
        genome = lower + rng.random((scoring.num_uavs, 4)) * span
        if centre:
            genome[:, :2] = lower[:2] + .5 * span[:2]
        return Individual(genome, source='random')

    def finish_batch():
        nonlocal epsilon0
        if history and history[-1]['milp_calls'] == evaluator.unique_solves:
            if result['status'] != status:
                result.update(status=status, elapsed_seconds=time.perf_counter() - started)
                save(destination, result)
            return
        if epsilon0 is None:
            epsilon0 = float(np.quantile([item.violation for item in records], .8))
        progress = max(0., (evaluator.unique_solves - warmup) / max(1, budget - warmup))
        scoring.current_eps = epsilon0 * max(0., 1. - progress / .7) ** 2
        scoring.update_contextual_model(f'{method}-{evaluator.unique_solves}', records)
        scoring._refresh_resource_scores(records)
        best = best_of(records, scoring._better_exact_incumbent)
        history.append(dict(batch=batch, milp_calls=evaluator.unique_solves,
                            elapsed_seconds=time.perf_counter() - started,
                            served_ues=len(best.coverage), violation=best.violation,
                            service_score=best.service_score, power_w=best.power_est_w,
                            prbs=best.prbs_est, epsilon=scoring.current_eps,
                            weights=scoring._resource_weights.tolist()))
        result.update(status=status, final_genome=best.evaluated_genome.tolist(),
                      final_milp=best.exact_result, final_violation=best.violation,
                      served_ues=len(best.coverage), served_ue_ids=sorted(best.coverage),
                      total_power_w=best.power_est_w, total_prbs=best.prbs_est,
                      unique_milp_solves=evaluator.unique_solves,
                      milp_solve_seconds=evaluator.solve_seconds,
                      elapsed_seconds=time.perf_counter() - started,
                      weight_learning_history=scoring.weight_learning_history)
        save(destination, result)
        print(f'{method}: best {len(best.coverage)}/{row["num_ues"]}, '
              f'V={best.violation:.4g}, PRB={best.prbs_est:g}, P={best.power_est_w:.3f} W; '
              f'MILP {evaluator.unique_solves}/{budget}', flush=True)

    def evaluate(item):
        genome = scoring._repair(item.genome)
        token = tuple(np.round(genome, 7).ravel())
        if token in seen:
            return None
        print(f'MILP {evaluator.unique_solves + 1}/{budget}: solving...', flush=True)
        scoring.evaluate_exact(item)
        if not item.exact_result['milp'].get('solver_optimal', False):
            result.update(status='failed', stop_reason='unconfirmed_milp_optimum',
                          unresolved_result=item.exact_result)
            save(destination, result)
            return False
        seen[token] = item
        records.append(item)
        result['evaluations'].append(dict(milp_calls=evaluator.unique_solves,
            served_ues=len(item.coverage), service_score=item.service_score,
            violation=item.violation, power_w=item.power_est_w, prbs=item.prbs_est,
            elapsed_seconds=time.perf_counter() - started))
        print(f'MILP {evaluator.unique_solves}/{budget}: served={len(item.coverage)}/{row["num_ues"]}, '
              f'V={item.violation:.4g}, PRB={item.prbs_est:g}, P={item.power_est_w:.3f} W', flush=True)
        return item

    try:
        initial = [sample(i < warmup) for i in range(size)]
        while evaluator.unique_solves < budget:
            scoring.current_generation = batch
            children = []
            duplicate_attempts = 0
            target = min(size, budget - evaluator.unique_solves)
            while len(children) < target:
                if batch == 0:
                    candidates = [initial[len(children)]]
                elif method == 'random':
                    candidates = [sample()]
                else:
                    first, second = scoring._tournament(population), scoring._tournament(population)
                    candidates = [scoring._mutate(item) for item in scoring._sbx(first, second)]
                for item in candidates:
                    if len(children) >= target:
                        break
                    evaluated = evaluate(item)
                    if evaluated is False:
                        return result
                    if evaluated is None:
                        duplicate_attempts += 1
                        if batch == 0:
                            initial[len(children)] = sample(len(children) < warmup)
                        if duplicate_attempts >= 10000:
                            status = 'stalled'
                            result['stop_reason'] = 'repeated_placements_before_budget'
                            if records:
                                finish_batch()
                            return result
                        continue
                    children.append(evaluated)
                    if evaluator.unique_solves % args.report_every == 0:
                        finish_batch()
            finish_batch()
            if method == 'ga':
                if not population:
                    population = children
                else:
                    elite_count = min(len(population), max(1, round(size * scoring.settings.elite_fraction)))
                    elites = ranked(population, scoring._better)[:elite_count]
                    population = (elites + ranked(children, scoring._better))[:size]
            batch += 1
        status = 'complete'
        result['stop_reason'] = 'budget_exhausted'
        finish_batch()
    except KeyboardInterrupt:
        status = 'interrupted'
        if records:
            finish_batch()
        else:
            result.update(status=status)
            save(destination, result)
    return result


def run_map(method, row, manifest, args, destination):
    import config
    import env
    from evaluation import make_exact_evaluator
    from ga import Bounds, GASettings
    from optimizer import JointTopKCSAEA
    from scenario import configured_scenario

    phys, scenario = configured_scenario(seed=row['map_seed'], num_ues=row['num_ues'],
                                        ue_distribution=manifest['ue_distribution'])
    scenario = replace(scenario, ues=tuple(env.UE(**ue) for ue in row['ues']))
    evaluator, _ = make_exact_evaluator(scenario.uavs, scenario.ues, phys)
    settings = replace(GASettings(**manifest['ga_settings']), random_seed=row['algorithm_seed'])
    scoring = JointTopKCSAEA(num_uavs=len(scenario.uavs), ues=scenario.ues,
        bounds=Bounds((0., 0., min(phys.H_U), min(phys.Theta_U)),
                      (*manifest['area_m'], max(phys.H_U), max(phys.Theta_U))),
        power_budget_w=sum(u.max_power_w for u in scenario.uavs),
        prb_budget=config.system_max_prbs, milp_evaluator=evaluator, phys=phys,
        uav_power_budgets_w=[u.max_power_w for u in scenario.uavs],
        uav_prb_budgets=[u.max_prbs for u in scenario.uavs], settings=settings,
        weight_method=manifest['runner_settings']['weight_method'])
    return optimize(method, row, manifest, args, destination, scoring, evaluator)


def summarize(output, selection):
    rows = []
    for name in selection['maps']:
        we = read(output / 'inputs' / name)
        demand = {u['id']: u['require_mbps'] for u in we['ues']}
        for method in ('random', 'ga'):
            path = output / method / 'runs' / name
            if not path.exists() or read(path).get('status') != 'complete':
                continue
            result = read(path)
            ok = result['final_violation'] <= 1e-12
            we_ok = we['final_violation'] <= 1e-12
            total_demand = sum(demand[i] for i in result['served_ue_ids'])
            we_demand = sum(demand[i] for i in we['served_ue_ids'])
            matched = ok and we_ok and result['served_ues'] == we['served_ues'] and abs(total_demand-we_demand) < 1e-9
            rows.append(dict(method=method, num_ues=we['num_ues'], map_index=we['map_index'],
                map_seed=we['map_seed'], feasible=ok, we_feasible=we_ok,
                served_ues=result['served_ues'] if ok else 0, we_served=we['served_ues'] if we_ok else 0,
                demand_mbps=total_demand, we_demand_mbps=we_demand,
                power_w=result['total_power_w'], we_power_w=we['total_power_w'],
                prbs=result['total_prbs'], we_prbs=we['total_prbs'], resource_comparable=matched,
                same_served_ids=set(result['served_ue_ids']) == set(we['served_ue_ids']),
                milp_calls=result['unique_milp_solves'], we_milp_calls=we['unique_milp_solves'],
                elapsed_seconds=result['elapsed_seconds'], we_seconds=we['elapsed_seconds']))
    save(output / 'comparison.json', dict(rows=rows, selected_maps=len(selection['maps']),
        timing_note='Historical WE hardware/timing may differ; not a controlled speed comparison.'))
    if rows:
        with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for count in COUNTS:
            for method in ('random', 'ga'):
                group = [r for r in rows if r['num_ues'] == count and r['method'] == method]
                if group:
                    print(f'{count} UE, {method}, n={len(group)}: mean served '
                          f'{np.mean([r["served_ues"] for r in group]):.2f}; '
                          f'WE {np.mean([r["we_served"] for r in group]):.2f}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Random Search / standard GA with the shared MILP on 60 saved maps')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--method', choices=('both', 'random', 'ga'), default='both')
    parser.add_argument('--selection-seed', type=int, default=20261015)
    parser.add_argument('--population', type=int, default=80)
    parser.add_argument('--report-every', type=int, default=40)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.summary_only:
        summarize(output, read(output / 'selection.json'))
        return 0
    if args.population < 2 or args.report_every < 1:
        print('population >= 2 and report-every >= 1 are required.')
        return 1
    manifest = read(args.source / 'manifest.json')
    mismatches = []
    for name in CORE:
        raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        if not matches_historical(name, raw, manifest):
            mismatches.append(name)
    if mismatches:
        print('Current code differs from historical WE: ' + ', '.join(mismatches))
        print('No experiments started. Reconcile the environment before comparing.')
        return 1
    eligible = [(p.name, read(p)) for p in sorted((args.source / 'runs').glob('*.json'))]
    eligible = [(p, row) for p, row in eligible if row.get('status') == 'complete' and row.get('solver_optimal')]
    chosen = select_maps(eligible, args.selection_seed)
    if chosen is None:
        print('Not enough completed maps for the requested 5/5/5/5/20/20 selection.')
        return 1
    identity = dict(selection_seed=args.selection_seed, population=args.population,
                    report_every=args.report_every, source_manifest=manifest,
                    code={name: digest(ROOT / name) for name in CORE}, runner=digest(__file__),
                    inputs={name: digest(args.source / 'runs' / name) for name, _ in chosen})
    path = output / 'selection.json'
    if path.exists():
        selection = read(path)
        if (selection['identity'] != identity and compatible_runner_update(
                selection['identity'], identity, any(output.glob('*/runs/*.json')))):
            selection.setdefault('previous_runner_hashes', []).append(selection['identity']['runner'])
            selection['identity'] = identity
            save(path, selection)
            print('Accepted I/O-only runner update; maps, settings and completed results preserved.', flush=True)
        if selection['identity'] != identity:
            print('Inputs/code/settings changed. Use a new --output-dir to preserve old results.')
            return 1
        for name in selection['maps']:
            if digest(output / 'inputs' / name) != identity['inputs'][name]:
                print(f'Frozen input changed: {name}. No experiments started.')
                return 1
    else:
        selection = dict(identity=identity, maps=[name for name, _ in chosen], counts=COUNTS)
        for name, row in chosen:
            save(output / 'inputs' / name, row)
        save(path, selection)
    print(f'Selected {len(selection["maps"])} identical maps per method: {path}', flush=True)
    for count in COUNTS:
        print(f'{count} UE: ' + ', '.join(str(row['map_index']) for _, row in chosen if row['num_ues'] == count), flush=True)
    if args.prepare_only:
        return 0
    methods = ('random', 'ga') if args.method == 'both' else (args.method,)
    for method in methods:
        for index, name in enumerate(selection['maps'], 1):
            destination = output / method / 'runs' / name
            if destination.exists() and read(destination).get('status') == 'complete':
                print(f'Skip completed: {method}/{name}', flush=True)
                continue
            log = output / method / 'logs' / (Path(name).stem + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as stream:
                with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                    print(f'[{index}/{len(selection["maps"])}] {method}: {name}; log: {log}', flush=True)
                    result = run_map(method, read(output / 'inputs' / name), manifest, args, destination)
            if result['status'] != 'complete':
                print(f'Stopped: {result["status"]}. Completed maps are preserved.', flush=True)
                return 1
            summarize(output, selection)
    return 0


if __name__ == '__main__':
    sys.exit(main())
