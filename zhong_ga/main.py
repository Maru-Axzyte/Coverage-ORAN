import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def key(genome):
    return tuple(np.round(genome, 7).ravel())


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


def geometry_violation(genome, lower, upper, separation):
    span = np.maximum(upper - lower, 1e-12)
    value = np.sum((np.maximum(lower - genome, 0) + np.maximum(genome - upper, 0)) / span)
    if separation > 0:
        for i in range(len(genome)):
            distances = np.linalg.norm(genome[i + 1:, :3] - genome[i, :3], axis=1)
            value += np.maximum(0, separation - distances).sum() / separation
    return float(value)


def initialize(size, count, ues, lower, upper, separation, rng):
    xy = np.array([[u['x_m'], u['y_m']] for u in ues])
    population = []
    for _ in range(size):
        accepted = None
        for _ in range(100):
            genes = []
            for _ in range(count):
                for attempt in range(300):
                    gene = rng.uniform(lower, upper)
                    if attempt < 100 and len(xy) >= 3:
                        points = xy[rng.choice(len(xy), 3, replace=False)]
                        matrix = 2 * (points[1:] - points[0])
                        if abs(np.linalg.det(matrix)) < 1e-8:
                            continue
                        gene[:2] = np.linalg.solve(matrix, (points[1:] ** 2).sum(axis=1)
                                                  - (points[0] ** 2).sum())
                    trial = np.array([*genes, gene])
                    if geometry_violation(trial, lower, upper, separation) <= 1e-12:
                        genes.append(gene)
                        break
                else:
                    break
            if len(genes) == count:
                accepted = np.array(genes)
                break
        if accepted is None:
            return None
        population.append(accepted)
    return population


def evolve(population, fitness, rng, lower, upper, separation, crossover, mutation):
    competitor = int(rng.integers(len(population)))
    selected = [population[competitor if f < fitness[competitor] else i].copy()
                for i, f in enumerate(fitness)]
    indices = np.flatnonzero(rng.random(len(selected)) < crossover)
    rng.shuffle(indices)
    rejected = 0
    attempted = 0
    cut = len(selected[0]) // 2
    if cut:
        for a, b in zip(indices[::2], indices[1::2]):
            first, second = selected[a].copy(), selected[b].copy()
            first[cut:], second[cut:] = selected[b][cut:], selected[a][cut:]
            for i, child in ((a, first), (b, second)):
                attempted += 1
                if geometry_violation(child, lower, upper, separation) <= 1e-12:
                    selected[i] = child
                else:
                    rejected += 1
    for i in range(len(selected)):
        if rng.random() < mutation:
            child = selected[i].copy()
            child[int(rng.integers(len(child)))] = rng.uniform(lower, upper)
            attempted += 1
            if geometry_violation(child, lower, upper, separation) <= 1e-12:
                selected[i] = child
            else:
                rejected += 1
    return selected, attempted, rejected


def optimize(row, manifest, args, destination, evaluator, uavs, phys):
    started = time.perf_counter()
    rng = np.random.default_rng(row['algorithm_seed'])
    lower = np.array([0., 0., min(phys.H_U), min(phys.Theta_U)])
    upper = np.array([*manifest['area_m'], max(phys.H_U), max(phys.Theta_U)])
    settings = manifest['ga_settings']
    separation = settings['min_uav_separation_m'] if settings['include_separation_constraint'] else 0.
    budget = int(row['unique_milp_solves'])
    population = initialize(min(args.population, budget), len(uavs), row['ues'],
                            lower, upper, separation, rng)
    result = dict(method='Zhong-inspired GA-MILP', status='running', source_we=row,
                  settings=dict(population=args.population, crossover=args.crossover,
                                mutation=args.mutation, max_generations=args.max_generations,
                                algorithm_seed=row['algorithm_seed'], milp_budget=budget),
                  selection='One random competitor replaces strictly worse chromosomes',
                  fitness='Feasible served UE count only; no resource tie-break in evolution',
                  reporting_tie_break='First encountered among equal served counts',
                  environment=manifest, history=[], evaluations=[], geometry_attempts=0,
                  geometry_rejections=0, cache_hits=0)
    if population is None:
        result.update(status='failed', stop_reason='geometry_initialization_failed')
        save(destination, result)
        print('Initialization failed without relaxing any constraints.', flush=True)
        return result
    evaluated = {}
    best = None
    generations = range(args.max_generations + 1)
    print(f'GA: {len(population)} configurations, budget {budget} MILP, pc={args.crossover}, pm={args.mutation}', flush=True)
    print('Selection: random competitor; fitness: served UE count; no WE/CMA/weights.', flush=True)
    try:
        for generation in generations:
            fitness = []
            for genome in population:
                token = key(genome)
                if token not in evaluated:
                    if evaluator.unique_solves >= budget:
                        break
                    print(f'MILP {evaluator.unique_solves + 1}/{budget}: solving...', flush=True)
                    exact = evaluator(genome)
                    if not exact['milp'].get('solver_optimal', False):
                        result.update(status='failed', stop_reason='unconfirmed_milp_optimum',
                                      unresolved_result=exact)
                        save(destination, result)
                        print('MILP optimum unconfirmed; map stopped, not labelled infeasible.', flush=True)
                        return result
                    violation = geometry_violation(genome, lower, upper, separation)
                    powers = np.array(exact['milp']['power_by_uav_w'])
                    budgets = np.array([u.max_power_w for u in uavs])
                    violation += float(np.maximum(powers / budgets - 1., 0.).sum())
                    violation += max(0., exact['total_num_prbs'] / manifest['system_prbs'] - 1.)
                    record = dict(genome=genome.tolist(), exact=exact, violation=violation,
                                  served_ues=len(exact['served_ue_ids']), generation=generation,
                                  milp_calls=evaluator.unique_solves)
                    record['fitness'] = record['served_ues'] if violation <= 1e-12 else -1
                    evaluated[token] = record
                    result['evaluations'].append(record)
                    if record['fitness'] >= 0 and (best is None or record['fitness'] > best['fitness']):
                        best = record
                    print(f'MILP {evaluator.unique_solves}/{budget}: served={record["served_ues"]}/{row["num_ues"]}, '
                          f'V={violation:.4g}, PRB={exact["total_num_prbs"]}/{manifest["system_prbs"]}, '
                          f'P={exact["total_power_w"]:.3f} W', flush=True)
                else:
                    result['cache_hits'] += 1
                fitness.append(evaluated[token]['fitness'])
            result.update(best=best, unique_milp_solves=evaluator.unique_solves,
                          elapsed_seconds=time.perf_counter() - started,
                          milp_solve_seconds=evaluator.solve_seconds, generation=generation)
            result['history'].append(dict(generation=generation, milp_calls=evaluator.unique_solves,
                served_ues=best['served_ues'] if best else None,
                power_w=best['exact']['total_power_w'] if best else None,
                prbs=best['exact']['total_num_prbs'] if best else None,
                elapsed_seconds=result['elapsed_seconds']))
            finished = evaluator.unique_solves >= budget or generation == args.max_generations
            if finished:
                result.update(status='complete', stop_reason='budget_exhausted' if evaluator.unique_solves >= budget
                              else 'generation_limit_before_budget')
            if generation % 10 == 0 or finished:
                if best:
                    print(f'GA {generation}: best served {best["served_ues"]}/{row["num_ues"]}, '
                          f'V={best["violation"]:.4g}, PRB {best["exact"]["total_num_prbs"]}/{manifest["system_prbs"]}, '
                          f'P {best["exact"]["total_power_w"]:.3f} W; MILP {evaluator.unique_solves}/{budget}, '
                          f'population {len(population)}', flush=True)
            if finished or generation % 10 == 0:
                save(destination, result)
            if finished:
                return result
            population, attempted, rejected = evolve(population, fitness, rng, lower, upper,
                                                     separation, args.crossover, args.mutation)
            result['geometry_attempts'] += attempted
            result['geometry_rejections'] += rejected
    except KeyboardInterrupt:
        result.update(status='interrupted', unique_milp_solves=evaluator.unique_solves,
                      elapsed_seconds=time.perf_counter() - started)
        save(destination, result)
        print('Interrupted. Completed maps are retained; this map restarts next time.', flush=True)
        return result


def run_map(row, manifest, args, destination):
    import env
    from evaluation import make_exact_evaluator
    from scenario import configured_scenario

    phys, scenario = configured_scenario(seed=row['map_seed'], num_ues=row['num_ues'],
                                         ue_distribution=manifest['ue_distribution'])
    scenario = replace(scenario, ues=tuple(env.UE(**u) for u in row['ues']))
    evaluator, _ = make_exact_evaluator(scenario.uavs, scenario.ues, phys)
    return optimize(row, manifest, args, destination, evaluator, scenario.uavs, phys)


def summarize(output, selection):
    rows = []
    for entry in selection['maps']:
        path = output / 'runs' / entry
        if not path.exists():
            continue
        result = read(path)
        if result['status'] != 'complete':
            continue
        we, best = result['source_we'], result['best']
        demand = {u['id']: u['require_mbps'] for u in we['ues']}
        we_ok = we['final_violation'] <= 1e-12
        ga_ok = best is not None and best['violation'] <= 1e-12
        we_service = sum(demand[k] for k in we['served_ue_ids'])
        ga_service = sum(demand[k] for k in best['exact']['served_ue_ids']) if ga_ok else 0
        comparable = bool(we_ok and ga_ok and we['served_ues'] == best['served_ues']
                          and abs(we_service - ga_service) <= 1e-9)
        rows.append(dict(num_ues=we['num_ues'], map_index=we['map_index'], map_seed=we['map_seed'],
            we_feasible=we_ok, ga_feasible=ga_ok, we_served=we['served_ues'] if we_ok else 0,
            ga_served=best['served_ues'] if ga_ok else 0, we_power_w=we['total_power_w'],
            ga_power_w=best['exact']['total_power_w'] if ga_ok else None,
            we_prbs=we['total_prbs'], ga_prbs=best['exact']['total_num_prbs'] if ga_ok else None,
            we_demand_mbps=we_service, ga_demand_mbps=ga_service, resource_comparable=comparable,
            we_milp_calls=we['unique_milp_solves'], ga_milp_calls=result['unique_milp_solves'],
            same_budget_used=we['unique_milp_solves'] == result['unique_milp_solves'],
            ga_stop_reason=result['stop_reason'], we_seconds=we['elapsed_seconds'],
            ga_seconds=result['elapsed_seconds']))
    save(output / 'comparison.json', dict(completed=len(rows), selected=len(selection['maps']), rows=rows,
        scope='Adapted count-only GA versus complete historical WE system, not original paper reproduction.',
        timing_note='Historical WE timings may use different hardware; no speed superiority claim.',
        resource_note='GA has no outer resource objective; matched service resource figures are descriptive.'))
    if not rows:
        return
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = sorted({r['num_ues'] for r in rows})
    figure, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(counts))
    for method, shift, color in [('we', -.18, '#187a48'), ('ga', .18, '#7250ae')]:
        means = [np.mean([r[f'{method}_served'] for r in rows if r['num_ues'] == n]) for n in counts]
        ax.bar(positions + shift, means, .36, label=method.upper(), color=color)
    ax.set_xticks(positions, [f'{n}\n(n={sum(r["num_ues"] == n for r in rows)})' for n in counts])
    ax.set(xlabel='UE count (completed paired maps)', ylabel='Mean feasible served UEs; failure = 0',
           title=f'WE / Zhong-inspired GA-MILP: {len(rows)}/{len(selection["maps"])} maps')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)
    for n in counts:
        group = [r for r in rows if r['num_ues'] == n]
        print(f'Mean {n} UE ({len(group)} maps): WE={np.mean([r["we_served"] for r in group]):.2f}, '
              f'GA={np.mean([r["ga_served"] for r in group]):.2f}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent Zhong-inspired GA versus saved WE maps')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--selection-seed', type=int, default=20261013)
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--crossover', type=float, default=.8)
    parser.add_argument('--mutation', type=float, default=.01)
    parser.add_argument('--max-generations', type=int, default=17000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.plot_only:
        summarize(output, read(output / 'selection.json'))
        return 0
    if (args.population < 2 or args.max_generations < 1 or not 0 <= args.crossover <= 1
            or not 0 <= args.mutation <= 1):
        print('Invalid GA settings. No simulation started.')
        return 1
    manifest = read(args.source / 'manifest.json')
    names = ('config.py', 'env.py', 'milp.py', 'evaluation.py', 'scenario.py')
    for name in names:
        raw = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        hashes = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}
        if manifest['source_sha256'].get(name) not in hashes:
            print(f'{name} differs from the saved WE experiment. No simulation started.')
            return 1
    eligible = [(p, read(p)) for p in sorted((args.source / 'runs').glob('*.json'))]
    eligible = [(p, r) for p, r in eligible if r.get('status') == 'complete' and r.get('solver_optimal')]
    if not 1 <= args.samples <= len(eligible):
        print(f'Choose 1 to {len(eligible)} maps. No simulation started.')
        return 1
    identity = dict(samples=args.samples, selection_seed=args.selection_seed,
        population=args.population, crossover=args.crossover, mutation=args.mutation,
        max_generations=args.max_generations, source_manifest=manifest,
        code={name: digest(ROOT / name) for name in names}, runner=digest(Path(__file__)),
        source_results={p.name: digest(p) for p, _ in eligible})
    path = output / 'selection.json'
    if path.exists():
        selection = read(path)
        if selection['identity'] != identity and args.prepare_only and not list((output / 'runs').glob('*.json')):
            if all(selection['identity'][field] == identity[field] for field in
                   ('samples', 'selection_seed', 'source_manifest', 'source_results')):
                selection['identity'] = identity
                save(path, selection)
        if selection['identity'] != identity:
            print('Inputs/settings/code changed. Use a new --output-dir; old results are preserved.')
            return 1
    else:
        rng = np.random.default_rng(args.selection_seed)
        counts = sorted({r['num_ues'] for _, r in eligible})
        groups = {n: [(p, r) for p, r in eligible if r['num_ues'] == n] for n in counts}
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < args.samples:
            for n in rng.permutation(counts):
                if groups[n] and len(chosen) < args.samples:
                    chosen.append(groups[n].pop())
        chosen.sort(key=lambda pair: (pair[1]['num_ues'], pair[1]['map_index']))
        selection = dict(identity=identity, sampling='Near-balanced stratified random maps',
                         maps=[p.name for p, _ in chosen])
        for source, row in chosen:
            save(output / 'inputs' / source.name, row)
        save(path, selection)
    for entry in selection['maps']:
        frozen = output / 'inputs' / entry
        if digest(frozen) != identity['source_results'][entry]:
            if read(frozen) != read(args.source / 'runs' / entry):
                print(f'Frozen input changed: {entry}. No simulation started.')
                return 1
    print(f'Selected {len(selection["maps"])} of {len(eligible)} maps; selection saved at {path}', flush=True)
    for entry in selection['maps']:
        row = read(output / 'inputs' / entry)
        print(f'  {entry}: WE {row["served_ues"]}/{row["num_ues"]}; budget={row["unique_milp_solves"]}', flush=True)
    if args.prepare_only:
        return 0
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(selection['maps'], 1):
        destination = output / 'runs' / entry
        if destination.exists() and read(destination).get('status') == 'complete':
            print(f'Skip completed map: {entry}', flush=True)
            continue
        row = read(output / 'inputs' / entry)
        log_path = output / 'logs' / (Path(entry).stem + '.log')
        with log_path.open('a', encoding='utf-8') as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                print(f'\nMap {index}/{len(selection["maps"])}: {entry}; starting from generation 0', flush=True)
                print(f'Progress log: {log_path}', flush=True)
                try:
                    result = run_map(row, manifest, args, destination)
                except Exception:
                    traceback.print_exc()
                    print('Map failed; previous completed maps are preserved. Rerun to retry this map.', flush=True)
                    return 1
        if result['status'] != 'complete':
            return 1
        best = result['best']
        print(f'Finished {entry}: WE={row["served_ues"]}/{row["num_ues"]}, '
              f'GA={best["served_ues"] if best else 0}/{row["num_ues"]}; {result["stop_reason"]}', flush=True)
        summarize(output, selection)
    summarize(output, selection)
    print(f'Saved comparison: {output / "comparison.png"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
