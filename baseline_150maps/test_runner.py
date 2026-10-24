import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from common import COUNTS, ExactJournal, RunLock, read, save, select_maps
from worker import load


class RunnerTests(unittest.TestCase):
    def test_plan_and_reuse_priority(self):
        rows = {f'ue_{n:03d}_map_{i:03d}.json': dict(num_ues=n, map_index=i)
                for n in COUNTS for i in range(1, 51)}
        chosen = select_maps(rows, {'ue_020_map_050.json': 5})
        self.assertEqual(len(chosen), 150)
        self.assertEqual(len(set(chosen)), 150)
        self.assertIn('ue_020_map_050.json', chosen)
        self.assertEqual(chosen, select_maps(rows, {'ue_020_map_050.json': 5}))
        for count, size in COUNTS.items():
            self.assertEqual(sum(rows[name]['num_ues'] == count for name in chosen), size)

    def test_committed_solves_survive_interruption(self):
        calls = []
        def base(x):
            calls.append(float(x[0, 0]))
            return dict(value=float(x[0, 0]), milp=dict(solver_optimal=True))
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / 'cache.sqlite'
            journal = ExactJournal(path, dict(map=1))
            evaluate = journal.wrap(base)
            a = evaluate(np.array([[1.]]))
            evaluate(np.array([[2.]]))
            journal.close()
            resumed = ExactJournal(path, dict(map=1))
            evaluate = resumed.wrap(base)
            self.assertEqual(evaluate(np.array([[1.]])), a)
            evaluate(np.array([[2.]]))
            evaluate(np.array([[2.]]))
            evaluate(np.array([[3.]]))
            self.assertEqual(calls, [1., 2., 3.])
            self.assertEqual(evaluate.unique_solves, 3)
            self.assertEqual(resumed.replayed, 2)
            resumed.close()
            with self.assertRaises(RuntimeError):
                ExactJournal(path, dict(map=2))

    def test_failed_solve_is_not_a_trusted_label(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            journal = ExactJournal(Path(directory) / 'cache.sqlite', 'test')
            evaluate = journal.wrap(lambda x: dict(milp=dict(solver_optimal=False)))
            evaluate(np.array([[1.]]))
            self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM solves').fetchone()[0], 0)
            journal.close()

    def test_lock_blocks_second_runner_and_releases(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / '.lock'
            with RunLock(path):
                with self.assertRaises(RuntimeError):
                    with RunLock(path):
                        pass
            with RunLock(path):
                pass

    def test_root_main_is_not_shadowed(self):
        launcher = load('test_study_launcher', 'baseline_150maps/main.py')
        import main
        self.assertEqual(Path(main.__file__).resolve(), ROOT / 'main.py')
        self.assertTrue(callable(main.build_parser))
        self.assertEqual(set(launcher.METHODS), {'random', 'ga', 'pso', 'zhong', 'grid'})

    def test_replay_matches_uninterrupted_classic_algorithms(self):
        from test_sweep_checkpoint import optimizer_and_evaluator
        runner = load('test_study_classic', 'classic_milp_baselines/main.py')
        row = dict(unique_milp_solves=8, algorithm_seed=17, num_ues=2)
        manifest = dict(ga_settings=dict(initial_milp_seeds=2))
        args = SimpleNamespace(population=2, report_every=2)
        for method in ('random', 'ga'):
            with self.subTest(method=method), tempfile.TemporaryDirectory(dir=HERE) as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                path = Path(directory)
                full, full_eval, _ = optimizer_and_evaluator()
                with patch.object(runner, 'save', lambda *_: None):
                    expected = runner.optimize(method, row, manifest, args, path/'full.json', full, full_eval)
                first, first_base, _ = optimizer_and_evaluator()
                journal = ExactJournal(path/'cache.sqlite', method)
                evaluated = journal.wrap(first_base)
                first.milp_evaluator = evaluated
                count = [0]
                def stop_after_two(*_):
                    count[0] += 1
                    if count[0] == 1:
                        raise KeyboardInterrupt()
                with patch.object(runner, 'save', stop_after_two):
                    runner.optimize(method, row, manifest, args, path/'part.json', first, evaluated)
                journal.close()
                resumed, base, _ = optimizer_and_evaluator()
                journal = ExactJournal(path/'cache.sqlite', method)
                evaluated = journal.wrap(base)
                resumed.milp_evaluator = evaluated
                with patch.object(runner, 'save', lambda *_: None):
                    actual = runner.optimize(method, row, manifest, args, path/'resumed.json', resumed, evaluated)
                self.assertEqual(actual['served_ues'], expected['served_ues'])
                self.assertEqual(actual['final_genome'], expected['final_genome'])
                self.assertEqual(actual['unique_milp_solves'], expected['unique_milp_solves'])
                self.assertGreater(journal.replayed, 0)
                for a, b in zip(actual['evaluations'], expected['evaluations']):
                    self.assertEqual({k: v for k, v in a.items() if k != 'elapsed_seconds'},
                                     {k: v for k, v in b.items() if k != 'elapsed_seconds'})
                journal.close()

    def test_pso_replay_preserves_search(self):
        from test_sweep_checkpoint import optimizer_and_evaluator
        import config
        import env
        runner = load('test_study_pso', 'baselines/run_pso.py')
        scene = SimpleNamespace(uavs=[env.UAV(0, 0., 0.), env.UAV(1, 0., 0.)],
                                ues=[env.UE(0, 500., 500., 5.), env.UE(1, 550., 500., 5.)])
        def scorer(**kwargs):
            model, _, _ = optimizer_and_evaluator()
            model.milp_evaluator = kwargs['evaluator']
            return model
        with tempfile.TemporaryDirectory(dir=HERE) as directory, contextlib.redirect_stdout(io.StringIO()):
            home = Path(directory)
            def execute(label, journal=None, stop=False):
                _, base, _ = optimizer_and_evaluator()
                evaluate = journal.wrap(base) if journal else base
                saves = [0]
                def save_result(path, result):
                    save(path, result)
                    saves[0] += 1
                    if stop and saves[0] == 1:
                        raise KeyboardInterrupt()
                with patch.object(runner, 'configured_scenario', lambda **_: (config.PhysConstant(), scene)), \
                        patch.object(runner, 'build_ga', scorer), \
                        patch.object(runner, 'make_exact_evaluator', lambda *_: (evaluate, {})), \
                        patch.object(runner, 'draw_result', lambda *_: None), \
                        patch.object(runner, 'save_result', save_result):
                    runner.main(['--swarm', '2', '--milp-budget', '8', '--seed', '17',
                                 '--output-dir', str(home/label), '--no-show'])
                return read(home/label/'pso_result.json')
            expected = execute('full')
            journal = ExactJournal(home/'cache.sqlite', 'pso')
            with self.assertRaises(KeyboardInterrupt):
                execute('partial', journal, True)
            journal.close()
            journal = ExactJournal(home/'cache.sqlite', 'pso')
            actual = execute('resumed', journal)
            self.assertEqual(actual['final_genome'], expected['final_genome'])
            self.assertEqual(actual['unique_milp_solves'], expected['unique_milp_solves'])
            self.assertGreater(journal.replayed, 0)
            journal.close()

    def test_zhong_replay_preserves_search(self):
        from test_sweep_checkpoint import optimizer_and_evaluator
        import config
        import env
        from dataclasses import asdict
        from ga import GASettings
        runner = load('test_study_zhong', 'zhong_ga/main.py')
        row = dict(unique_milp_solves=8, algorithm_seed=17, num_ues=2,
                   ues=[asdict(env.UE(0, 500., 500., 5.)), asdict(env.UE(1, 550., 500., 5.))])
        manifest = dict(area_m=[1000., 1000.], system_prbs=135, ga_settings=asdict(GASettings()))
        args = SimpleNamespace(population=2, crossover=.8, mutation=1., max_generations=20)
        with tempfile.TemporaryDirectory(dir=HERE) as directory, contextlib.redirect_stdout(io.StringIO()):
            home = Path(directory)
            def execute(journal=None, stop=False):
                _, base, _ = optimizer_and_evaluator()
                evaluate = journal.wrap(base) if journal else base
                saves = [0]
                def persist(*_):
                    saves[0] += 1
                    if stop and saves[0] == 1:
                        raise KeyboardInterrupt()
                with patch.object(runner, 'save', persist):
                    return runner.optimize(row, manifest, args, home/'result.json', evaluate,
                        [env.UAV(0, 0., 0.), env.UAV(1, 0., 0.)], config.PhysConstant())
            expected = execute()
            journal = ExactJournal(home/'cache.sqlite', 'zhong')
            execute(journal, True)
            journal.close()
            journal = ExactJournal(home/'cache.sqlite', 'zhong')
            actual = execute(journal)
            self.assertEqual(actual['best']['genome'], expected['best']['genome'])
            self.assertEqual(actual['unique_milp_solves'], expected['unique_milp_solves'])
            self.assertGreater(journal.replayed, 0)
            journal.close()

    def test_grid_saves_modes_and_solver_incumbent(self):
        import config
        import highspy
        from dataclasses import asdict
        from worker import run_grid
        built_models = []
        def build(*args):
            model = highspy.Highs()
            model.setOptionValue('output_flag', False)
            model.setOptionValue('threads', 1)
            x = model.addBinary()
            model.changeColCost(int(x), 1.)
            model.setMaximize()
            built_models.append(model)
            return model, [], dict(variables=1, constraints=0)
        model_code = SimpleNamespace(build_model=build,
            extract=lambda *args: dict(served_ues=1, violation=0., total_power_w=.1, total_prbs=1))
        class FakeModes:
            calls = 0
            def __init__(self, *args, **kwargs):
                self.service_modes = {}
            def precompute_service_modes(self):
                FakeModes.calls += 1
        with tempfile.TemporaryDirectory(dir=HERE) as directory, contextlib.redirect_stdout(io.StringIO()):
            home = Path(directory)
            destination = home/'grid/runs/map.json'
            destination.parent.mkdir(parents=True)
            (home/'grid/logs').mkdir()
            checkpoint = home/'grid/checkpoints/map'
            checkpoint.parent.mkdir()
            row = dict(ues=[dict(id=0, x_m=500., y_m=500., require_mbps=5.)], elapsed_seconds=100., num_ues=1)
            manifest = dict(physics=asdict(config.PhysConstant()), area_m=[1000., 1000.],
                num_uavs=1, per_uav_power_w=2., per_uav_prb_bound=135, system_prbs=135,
                ga_settings=dict(min_uav_separation_m=0., include_separation_constraint=False),
                solver=dict(num_threads=1, parallel_search=False))
            def persist(path, value):
                save(path, dict(value, elapsed_seconds=0.))
            with patch('worker.load', return_value=model_code), \
                    patch('milp.UAVMILP', FakeModes), patch('env.make_placements', return_value=[]):
                self.assertEqual(run_grid(home, row, manifest, 'test-grid', destination, checkpoint, persist, time_budget=100.), 0)
                self.assertEqual(run_grid(home, row, manifest, 'test-grid', destination, checkpoint, persist, time_budget=100.), 0)
            self.assertEqual(FakeModes.calls, 1)
            self.assertTrue(checkpoint.with_suffix('.solution.pkl').exists())
            self.assertEqual(read(destination)['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
