import itertools
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import highspy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config
import env
from milp import ServiceMode, UAVMILP
from model import build_model, conflict_sets, extract, objective_weights


class GridMILPTests(unittest.TestCase):
    def setUp(self):
        self.settings = config.SolverSettings(log_to_console=False, num_threads=1)

    def test_compact_conflicts_match_every_pair(self):
        placements = env.make_placements([0., 200., 250., 500.], [0., 200.], [250., 300.], [30., 60.])
        for distance in (0., 40., 250.):
            constraints = list(conflict_sets(placements, distance))
            for a, b in itertools.combinations(range(len(placements)), 2):
                p, q = placements[a], placements[b]
                incompatible = math.dist((p.x_u, p.y_u, p.h_u), (q.x_u, q.y_u, q.h_u)) < distance - 1e-9
                actual = any(a in row and b in row for row in constraints)
                self.assertEqual(actual, incompatible, (a, b, distance))

    def test_joint_selection_matches_exhaustive_binary_search(self):
        placements = [env.solution(i, x, 0., 250., 60.) for i, x in enumerate([0., 100., 400.])]
        ues = [env.UE(i, i * 100., 0., demand) for i, demand in enumerate([5., 10., 15.])]
        modes = [ServiceMode(0, k, s, i, n, p, 20., 100.) for i, (s, k, n, p) in enumerate([
            (0, 0, 1, .6), (0, 1, 2, .6), (1, 0, 1, .4), (1, 2, 3, .5),
            (2, 1, 1, .5), (2, 2, 2, .7)])]
        model, _, _ = build_model(placements, modes, ues, 2, 1., 4, 4, 250., self.settings)
        model.run()
        self.assertEqual(model.getModelStatus(), highspy.HighsModelStatus.kOptimal)
        units, cw, dw = objective_weights(ues, 4, 2.)
        expected = -math.inf
        for sites in itertools.combinations(range(len(placements)), 2):
            if abs(placements[sites[0]].x_u - placements[sites[1]].x_u) < 250.:
                continue
            for bits in itertools.product((0, 1), repeat=len(modes)):
                chosen = [m for m, bit in zip(modes, bits) if bit]
                if any(m.placement not in sites for m in chosen):
                    continue
                if len({m.k for m in chosen}) != len(chosen) or sum(m.num_prbs for m in chosen) > 4:
                    continue
                if any(sum(m.power_w for m in chosen if m.placement == s) > 1. + 1e-9 for s in sites):
                    continue
                value = sum(cw + dw * units[m.k] - 2 * m.num_prbs - m.power_w / 2. for m in chosen)
                expected = max(expected, value)
        self.assertAlmostEqual(model.getObjectiveValue(), expected, places=6)

    def test_real_radio_modes_match_fixed_placement_oracle(self):
        phys = config.PhysConstant()
        placements = [env.solution(i, x, 0., 250., 60.) for i, x in enumerate([0., 300., 600.])]
        ues = [env.UE(i, x, 0., d) for i, (x, d) in enumerate([(10., 5.), (310., 10.), (590., 15.)])]
        uav = env.UAV(0, 0., 0., max_power_w=2., max_prbs=135)
        oracle = UAVMILP([uav], ues, placements, phys, self.settings,
                         system_prb_budget=135, orthogonal_prb_pool=True)
        oracle.precompute_service_modes()
        model, modes, _ = build_model(placements, oracle.service_modes.values(), ues, 1,
                                      2., 135, 135, 0., self.settings)
        model.run()
        result = extract(model, placements, modes, 1, ues, 2., 135, 135, 0., phys)
        self.assertIsNotNone(result)
        units, cw, dw = objective_weights(ues, 135, 2.)
        objectives = []
        for s in range(len(placements)):
            exact = UAVMILP([uav], ues, placements, phys, self.settings,
                            fixed_placement_indices=(s,), system_prb_budget=135,
                            orthogonal_prb_pool=True).solve()
            ids = {m['ue_id'] for m in exact['uav_mode_choices']}
            objectives.append(cw * len(ids) + dw * sum(units[k] for k in ids)
                              - 2 * exact['total_num_prbs'] - exact['total_power_w'] / 2.)
        self.assertAlmostEqual(model.getObjectiveValue(), max(objectives), places=6)
        self.assertLessEqual(result['total_power_w'], 2. + 1e-9)
        self.assertLessEqual(result['total_prbs'], 135)

    def test_saved_watt_physics_keeps_original_power_grid(self):
        from dataclasses import asdict
        values = asdict(config.PhysConstant())
        if values['power_level_step_db'] is None:
            values['P_min_dbm'] = values['P_max_dbm'] = None
        restored = config.PhysConstant(**values)
        self.assertEqual(restored.power_levels_w, config.PhysConstant().power_levels_w)

    def test_worker_saves_valid_incumbent_and_grid_bound(self):
        from dataclasses import asdict
        spec = importlib.util.spec_from_file_location('grid_runner', HERE / 'main.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        phys = config.PhysConstant(H_U=(250.,), Theta_U=(30., 60.))
        manifest = dict(physics=asdict(phys), num_uavs=2, per_uav_power_w=2.,
                        per_uav_prb_bound=135, system_prbs=135, area_m=[600., 600.],
                        ga_settings=dict(include_separation_constraint=True, min_uav_separation_m=250.),
                        solver=dict(num_threads=1, parallel_search=False))
        ues = [asdict(env.UE(i, x, y, 5.)) for i, (x, y) in enumerate([(20., 20.), (580., 580.)])]
        row = dict(ues=ues, served_ues=2, served_ue_ids=[0, 1], final_violation=0.,
                   map_index=1, map_seed=2, total_power_w=1., total_prbs=10, elapsed_seconds=10.)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory)
            (output / 'inputs').mkdir()
            (output / 'logs').mkdir()
            (output / 'selection.json').write_text(json.dumps(dict(method='test grid', maps=['map.json'],
                identity=dict(source_manifest=manifest, grid_intervals=1, time_limit=10.))))
            (output / 'inputs/map.json').write_text(json.dumps(row))
            self.assertEqual(runner.run_map('map.json', output), 0)
            result = json.loads((output / 'runs/map.json').read_text())
            self.assertTrue(result['solver_optimal'])
            self.assertEqual(result['best']['served_ues'], 2)
            self.assertEqual(result['grid_served_upper_bound'], 2)
            self.assertTrue(result['grid_served_count_proven'])
            self.assertGreaterEqual(result['best']['min_uav_distance_m'], 250.)
            runner.summarize(output, dict(maps=['map.json']))
            summary = json.loads((output / 'comparison.json').read_text())
            self.assertEqual(summary['completed'], 1)
            self.assertTrue(summary['rows'][0]['resource_comparable'])
            self.assertTrue(summary['rows'][0]['same_served_ue_set'])
            self.assertTrue((output / 'comparison.csv').exists())
            self.assertTrue((output / 'comparison.png').exists())


if __name__ == '__main__':
    unittest.main()
