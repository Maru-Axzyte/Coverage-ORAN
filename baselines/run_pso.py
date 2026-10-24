import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from evaluation import make_exact_evaluator
from ga import GASettings, Individual
from optimizer import build_ga
from plots import PlotSettings, draw_snapshot
from scenario import configured_scenario


def best_of(items, better):
    best = items[0]
    for item in items[1:]:
        if better(item, best):
            best = item
    return best


def save_result(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(path)


def draw_result(history, initial, final, ues, phys, output, show):
    settings = PlotSettings.from_config(num_uavs=len(final.genome), num_ues=len(ues))
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, item, title in zip(axes[0], (initial, final), ('Initial PSO best', 'Final PSO best')):
        draw_snapshot(ax, title=f'{title}; V={item.violation:.3g}',
                      genome=item.evaluated_genome, exact=item.exact_result,
                      ues=ues, phys=phys, settings=settings)
    calls = [row['milp_calls'] for row in history]
    axes[1, 0].step(calls, [row['feasible_served_ues'] for row in history], where='post')
    axes[1, 0].axhline(len(ues), color='gray', linestyle=':')
    axes[1, 0].set(xlabel='Unique MILP solves', ylabel='Best feasible served UEs')
    axes[1, 1].step(calls, [row['power_ratio'] for row in history], where='post', label='Power / total budget')
    axes[1, 1].step(calls, [row['prb_ratio'] for row in history], where='post', label='PRBs / system budget')
    axes[1, 1].set(xlabel='Unique MILP solves', ylabel='Reported incumbent resource load')
    axes[1, 1].legend()
    for ax in axes[1]:
        ax.grid(alpha=.25)
    figure.suptitle('PSO + shared fixed-placement MILP')
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def main(argv=None):
    defaults = GASettings()
    parser = argparse.ArgumentParser(description='Independent PSO-MILP baseline')
    parser.add_argument('--swarm', type=int, default=40)
    parser.add_argument('--milp-budget', type=int,
                        default=defaults.initial_milp_seeds + 4 * defaults.generations)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--map-seed', type=int, default=None)
    parser.add_argument('--num-ues', type=int, default=config.num_ues)
    parser.add_argument('--ue-distribution', choices=('uniform', 'hotspot'), default='uniform')
    parser.add_argument('--weight-method', choices=('contextual', 'fixed'), default='contextual')
    parser.add_argument('--inertia', type=float, default=.7298)
    parser.add_argument('--cognitive', type=float, default=1.49618)
    parser.add_argument('--social', type=float, default=1.49618)
    parser.add_argument('--velocity-limit', type=float, default=.2)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args(argv)
    budget = max(1, args.milp_budget)
    swarm = min(budget, max(2, args.swarm))
    map_seed = args.seed if args.map_seed is None else args.map_seed
    output = args.output_dir or ROOT / 'baselines' / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if (output / 'pso_result.json').exists():
        output = output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    phys, scenario = configured_scenario(seed=map_seed, ue_distribution=args.ue_distribution,
                                         num_ues=max(1, args.num_ues))
    evaluator, _ = make_exact_evaluator(scenario.uavs, scenario.ues, phys)
    scoring = build_ga(uavs=scenario.uavs, ues=scenario.ues, phys=phys,
                       evaluator=evaluator, population=defaults.population_size,
                       population_capacity=defaults.population_capacity,
                       generations=defaults.generations, seed=args.seed,
                       weight_method=args.weight_method)
    lower = np.asarray(scoring.bounds.lower)
    span = scoring.bounds.span
    rng = np.random.default_rng(args.seed)
    positions = rng.random((swarm, len(scenario.uavs), 4))
    positions[:, :, :2] = .5
    velocity_limit = abs(args.velocity_limit)
    velocities = rng.uniform(-velocity_limit, velocity_limit, positions.shape)
    memories = [[] for _ in range(swarm)]
    evaluated = {}
    history = []
    started = time.perf_counter()
    batch = 0
    epsilon0 = None
    initial = None
    stop_reason = 'budget_exhausted'
    result = {
        'method': 'PSO-MILP', 'status': 'running',
        'settings': {**vars(args), 'output_dir': str(output), 'map_seed': map_seed,
                     'swarm': swarm, 'milp_budget': budget},
        'environment': {'num_uavs': len(scenario.uavs), 'num_ues': len(scenario.ues),
                        'area_m': [config.length_m, config.width_m],
                        'physics': asdict(phys), 'a2g_environment': config.a2g_environment,
                        'system_prbs': config.system_max_prbs,
                        'orthogonal_prb_pool': config.global_orthogonal_prb_pool,
                        'uav_power_budgets_w': [u.max_power_w for u in scenario.uavs],
                        'uav_prb_bounds': [u.max_prbs for u in scenario.uavs],
                        'separation_enabled': defaults.include_separation_constraint,
                        'min_separation_m': defaults.min_uav_separation_m,
                        'ues': [asdict(ue) for ue in scenario.ues]},
        'initialization': 'shared centre x/y; uniform h/theta; random normalized velocity',
        'weight_learning': 'shared exact-pair learner; no surrogate prediction pairs',
        'epsilon_schedule': 'initial V quantile 0.8, quadratic decay over 70% of post-initial MILP budget',
        'history': history,
    }
    print(f'PSO: {swarm} particles, budget={budget} MILP solves, map seed={map_seed}', flush=True)
    print(f'Results: {output.resolve()}', flush=True)
    while True:
        scoring.current_generation = batch
        before = evaluator.unique_solves
        for index in range(swarm):
            genome = lower + positions[index] * span
            key = tuple(np.round(genome, 7).ravel())
            if key not in evaluated:
                if evaluator.unique_solves >= budget:
                    break
                item = Individual(genome, source='pso')
                scoring.evaluate_exact(item)
                evaluated[key] = item
                print(f'MILP {evaluator.unique_solves}/{budget}: '
                      f'served={len(item.coverage)}/{len(scenario.ues)}, V={item.violation:.4g}', flush=True)
            item = evaluated[key]
            if not any(previous is item for previous in memories[index]):
                memories[index].append(item)
        records = list(evaluated.values())
        if epsilon0 is None:
            epsilon0 = float(np.quantile([item.violation for item in records], .8))
        progress = (evaluator.unique_solves - swarm) / max(1, budget - swarm)
        scoring.current_eps = epsilon0 * max(0., 1. - progress / .7) ** 2
        if batch % max(1, defaults.update_period) == 0 or evaluator.unique_solves >= budget:
            scoring.update_contextual_model(f'PSO-{batch}', records)
        scoring._refresh_resource_scores(records)
        personal = [best_of(items, scoring._better) for items in memories]
        guide = best_of(records, scoring._better)
        final = best_of(records, scoring._better_exact_incumbent)
        if initial is None:
            initial = final.clone()
        feasible = [item for item in records if item.violation <= 1e-12]
        history.append({
            'iteration': batch, 'milp_calls': evaluator.unique_solves,
            'elapsed_seconds': time.perf_counter() - started,
            'epsilon': scoring.current_eps, 'violation': final.violation,
            'served_ues': len(final.coverage),
            'feasible_served_ues': max((len(item.coverage) for item in feasible), default=None),
            'service_score': final.service_score, 'power_w': final.power_est_w,
            'prbs': final.prbs_est, 'power_ratio': final.power_est_w / scoring.power_budget_w,
            'prb_ratio': final.prbs_est / scoring.prb_budget,
            'weights': scoring._resource_weights.tolist(),
        })
        result.update(final_genome=final.evaluated_genome.tolist(), final_milp=final.exact_result,
                      final_violation=final.violation, served_ues=len(final.coverage),
                      unique_milp_solves=evaluator.unique_solves,
                      elapsed_seconds=time.perf_counter() - started,
                      weight_learning_history=scoring.weight_learning_history)
        save_result(output / 'pso_result.json', result)
        print(f'PSO {batch}: best served={len(final.coverage)}/{len(scenario.ues)}, '
              f'V={final.violation:.4g}, PRB={final.prbs_est:g}, P={final.power_est_w:.3f} W', flush=True)
        if evaluator.unique_solves >= budget:
            break
        if batch > 0 and evaluator.unique_solves == before:
            stop_reason = 'no_new_placements'
            break
        pbest = np.asarray([(item.evaluated_genome - lower) / span for item in personal])
        gbest = (guide.evaluated_genome - lower) / span
        velocities = (args.inertia * velocities
                      + args.cognitive * rng.random(positions.shape) * (pbest - positions)
                      + args.social * rng.random(positions.shape) * (gbest - positions))
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        proposed = positions + velocities
        velocities[(proposed < 0.) | (proposed > 1.)] *= -1.
        positions = 1. - np.abs(np.mod(proposed, 2.) - 1.)
        batch += 1
    result.update(status='complete', stop_reason=stop_reason)
    save_result(output / 'pso_result.json', result)
    draw_result(history, initial, final, scenario.ues, phys, output / 'pso_convergence.png', not args.no_show)
    print(f'Saved: {output / "pso_result.json"}', flush=True)
    print(f'Saved: {output / "pso_convergence.png"}', flush=True)


if __name__ == '__main__':
    main()
