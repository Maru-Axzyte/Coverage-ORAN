import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('zhong_baseline', Path(__file__).with_name('main.py'))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.lower = np.array([0., 0., 250., 30.])
        self.upper = np.array([1000., 1000., 300., 60.])
        self.ues = [dict(id=i, x_m=float(i % 5 * 200), y_m=float(i // 5 * 200),
                         require_mbps=5.) for i in range(25)]

    def test_initialize_and_operators_preserve_geometry(self):
        rng = np.random.default_rng(7)
        population = baseline.initialize(12, 5, self.ues, self.lower, self.upper, 250., rng)
        self.assertIsNotNone(population)
        for _ in range(30):
            population, attempts, rejected = baseline.evolve(population, list(range(12)), rng,
                self.lower, self.upper, 250., .8, .5)
            self.assertLessEqual(rejected, attempts)
            for genome in population:
                self.assertLessEqual(baseline.geometry_violation(genome, self.lower, self.upper, 250.), 1e-12)

    def test_competitor_selection_not_deb(self):
        population = [np.full((2, 4), value) for value in (.2, .5, .8)]
        rng = np.random.default_rng(12)
        c = int(np.random.default_rng(12).integers(3))
        selected, _, _ = baseline.evolve(population, [0, 1, 2], rng,
                                         np.zeros(4), np.ones(4), 0, 0, 0)
        for i in range(3):
            np.testing.assert_array_equal(selected[i], population[c if i < c else i])

    def test_mock_budget_and_results(self):
        class Evaluator:
            unique_solves = 0
            solve_seconds = 0.

            def __call__(self, genome):
                self.unique_solves += 1
                return dict(served_ue_ids=[0, 1], total_power_w=.2, total_num_prbs=3,
                            milp=dict(solver_optimal=True, power_by_uav_w=[.1, .1]))

        args = SimpleNamespace(population=4, crossover=.8, mutation=1., max_generations=30)
        row = dict(algorithm_seed=2, num_ues=25, unique_milp_solves=9, ues=self.ues)
        manifest = dict(area_m=[1000, 1000], system_prbs=135,
                        ga_settings=dict(include_separation_constraint=True, min_uav_separation_m=250.))
        phys = SimpleNamespace(H_U=[250, 300], Theta_U=[30, 60])
        evaluator = Evaluator()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'result.json'
            result = baseline.optimize(row, manifest, args, path, evaluator,
                                       [SimpleNamespace(max_power_w=2.)] * 2, phys)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['stop_reason'], 'budget_exhausted')
            self.assertEqual(evaluator.unique_solves, 9)
            self.assertEqual(len(result['evaluations']), 9)
            self.assertEqual(result['best']['milp_calls'], 1)
            self.assertEqual(baseline.read(path)['best']['served_ues'], 2)

    def test_unknown_solver_result_not_marked_infeasible(self):
        class Evaluator:
            unique_solves = 0
            solve_seconds = 0.

            def __call__(self, genome):
                self.unique_solves += 1
                return dict(milp=dict(solver_optimal=False))

        args = SimpleNamespace(population=2, crossover=.8, mutation=.01, max_generations=1)
        row = dict(algorithm_seed=1, num_ues=25, unique_milp_solves=2, ues=self.ues)
        manifest = dict(area_m=[1000, 1000], ga_settings=dict(include_separation_constraint=False))
        phys = SimpleNamespace(H_U=[250, 300], Theta_U=[30, 60])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = baseline.optimize(row, manifest, args, Path(directory) / 'result.json',
                                       Evaluator(), [SimpleNamespace(max_power_w=2.)], phys)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['stop_reason'], 'unconfirmed_milp_optimum')
            self.assertEqual(result['evaluations'], [])


if __name__ == '__main__':
    unittest.main()
